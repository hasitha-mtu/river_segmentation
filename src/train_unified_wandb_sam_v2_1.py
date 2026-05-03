
import os
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
from src.utils.losses import get_loss_function
from src.utils.metrics import SegmentationMetrics
from src.foundation_models.sam.dataset import create_sam2_dataset
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─────────────────────────────────────────────────────────────────────────────
# Unified Trainer
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── GPU diagnostics ────────────────────────────────────────────────
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled   = True
            torch.backends.cudnn.benchmark = True
            print('\n' + '=' * 80)
            print('GPU MEMORY CHECK')
            print('=' * 80)
            torch.cuda.empty_cache()
            print(f'GPU   : {torch.cuda.get_device_name(0)}')
            print(f'Total : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
            print(f'Alloc : {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
            print(f'Rsrvd : {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB')
            print('=' * 80 + '\n')

        self.setup_directories()

        # ── wandb ─────────────────────────────────────────────────────────
        self.use_wandb = config['logging'].get('use_wandb', False)
        if self.use_wandb:
            self.init_wandb()

        # ── Model ─────────────────────────────────────────────────────────
        model_name = config['model']['name']
        variant    = config['model'].get('variant', None)
        print(f'\nInitializing {model_name}' + (f' ({variant})' if variant else '') + '…')

        ckpt_base = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam2'
        variant_map = {
            'sam2.1_hiera_tiny'      : ('configs/sam2.1/sam2.1_hiera_t.yaml',  'sam2.1_hiera_tiny.pt'),
            'sam2.1_hiera_small'     : ('configs/sam2.1/sam2.1_hiera_s.yaml',  'sam2.1_hiera_small.pt'),
            'sam2.1_hiera_base_plus' : ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'),
        }
        if variant not in variant_map:
            raise ValueError(f'Unknown SAM2 variant: {variant}. '
                             f'Choose from {list(variant_map.keys())}')

        cfg_file, ckpt_file = variant_map[variant]
        self.model = build_sam2(
            config_file = cfg_file,
            ckpt_path   = os.path.join(ckpt_base, ckpt_file),
            device      = 'cuda',
        )

        # ── Freeze / unfreeze modules ──────────────────────────────────────
        #
        # Image encoder: ALWAYS frozen.
        #   - It is a large pretrained Hiera backbone; fine-tuning it on a
        #     small UAV dataset risks catastrophic forgetting and uses ~8 GB
        #     of extra gradient memory.
        #   - Must be in eval() so its BN/dropout behave correctly while frozen.
        #
        # Prompt encoder + Mask decoder: trained.
        #   - These are the lightweight heads that learn the task-specific
        #     mapping from prompts → river water masks.
        #
        self.model.image_encoder.eval()
        self.model.image_encoder.requires_grad_(False)

        self.model.sam_prompt_encoder.train()
        self.model.sam_prompt_encoder.requires_grad_(True)

        self.model.sam_mask_decoder.train()
        self.model.sam_mask_decoder.requires_grad_(True)

        self.predictor = SAM2ImagePredictor(self.model)

        self.scaler = torch.amp.GradScaler('cuda')

        # ── Data loaders ───────────────────────────────────────────────────
        data_root       = config['data']['data_root']
        self.batch_size = config['training']['batch_size']
        self.img_size   = config['data']['image_size']

        print(f'Loading data from {data_root}…')
        self.train_dataset      = create_sam2_dataset(data_root, 'train', self.batch_size)
        self.validation_dataset = create_sam2_dataset(data_root, 'val',   self.batch_size)
        print(f'  train: {len(self.train_dataset)} samples  |  val: {len(self.validation_dataset)} samples')
        print(f'  Image size: {self.img_size}×{self.img_size}')

        # ── Loss ───────────────────────────────────────────────────────────
        # IMPORTANT: _forward() returns raw logits (no sigmoid applied).
        # Your combined loss must therefore use BCEWithLogitsLoss internally,
        # NOT BCELoss. Dice and boundary terms should also operate on logits
        # (apply sigmoid inside the loss function, not before passing in).
        loss_cfg = config['loss']
        print(f'Using {loss_cfg["type"]} loss (expects LOGITS, not probabilities)…')
        if loss_cfg['type'] == 'combined':
            self.criterion = get_loss_function(
                loss_cfg['type'],
                bce_weight      = loss_cfg.get('bce_weight',      1.0),
                dice_weight     = loss_cfg.get('dice_weight',     1.0),
                boundary_weight = loss_cfg.get('boundary_weight', 1.0),
                use_boundary    = loss_cfg.get('use_boundary',    True),
            )
        else:
            self.criterion = get_loss_function(loss_cfg['type'])

        # ── Optimizer ─────────────────────────────────────────────────────
        # Only the trainable modules (prompt encoder + mask decoder) are
        # registered. The frozen image encoder has no gradients so adding
        # it would be harmless but misleading.
        trainable_params = (
            list(self.model.sam_prompt_encoder.parameters()) +
            list(self.model.sam_mask_decoder.parameters())
        )

        opt_cfg = config['training']['optimizer']
        base_lr = opt_cfg['learning_rate']
        wd      = opt_cfg.get('weight_decay', 0.0)

        if opt_cfg['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                trainable_params, lr=base_lr,
                weight_decay=wd, betas=(0.9, 0.999),
            )
        elif opt_cfg['type'] == 'adam':
            self.optimizer = torch.optim.Adam(
                trainable_params, lr=base_lr, weight_decay=wd,
            )
        elif opt_cfg['type'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                trainable_params, lr=base_lr,
                momentum=opt_cfg.get('momentum', 0.9),
                weight_decay=wd,
            )
        else:
            raise ValueError(f'Unknown optimizer type: {opt_cfg["type"]}')

        # ── Scheduler ─────────────────────────────────────────────────────
        sched_cfg = config['training']['scheduler']
        if sched_cfg['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max   = config['training']['epochs'],
                eta_min = sched_cfg.get('min_lr', 1e-6),
            )
        elif sched_cfg['type'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size = sched_cfg.get('step_size', 30),
                gamma     = sched_cfg.get('gamma', 0.1),
            )
        elif sched_cfg['type'] == 'warmup_cosine':
            self.scheduler = self._setup_warmup_scheduler()
        else:
            self.scheduler = None

        # ── Logging ────────────────────────────────────────────────────────
        self.metrics = SegmentationMetrics(threshold=0.5)
        self.writer  = SummaryWriter(self.log_dir)

        # ── Training state ─────────────────────────────────────────────────
        self.start_epoch   = 0
        self.best_val_dice = 0.0
        self.best_val_iou  = 0.0

        self.es_patience  = config['training'].get('early_stopping_patience',  None)
        self.es_min_delta = config['training'].get('early_stopping_min_delta', 1e-4)
        self.es_counter   = 0
        self.es_best_dice = 0.0

        self.print_configuration()

        if self.use_wandb and config['logging'].get('watch_model', False):
            wandb.watch(self.model, log='all', log_freq=100)

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def init_wandb(self):
        model_name = self.config['model']['name']
        variant    = self.config['model'].get('variant', None)

        run_name = model_name
        if variant:
            run_name += f'_{variant}'
        run_name += f'_{self.config["loss"]["type"]}'
        run_name += f'_bs{self.config["training"]["batch_size"]}'
        run_name += f'_lr{self.config["training"]["optimizer"]["learning_rate"]}'

        tags = [
            model_name,
            variant if variant else 'no_variant',
            self.config['loss']['type'],
            f'img_{self.config["data"].get("image_size", 512)}',
        ]

        wandb.init(
            project = self.config['logging'].get('wandb_project', 'river-segmentation'),
            name    = run_name,
            config  = self.config,
            tags    = tags,
            notes   = self.config['logging'].get('wandb_notes', ''),
            dir     = self.exp_dir,
        )

        if torch.cuda.is_available():
            wandb.config.update({
                'gpu_name'      : torch.cuda.get_device_name(0),
                'gpu_memory_gb' : torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version'  : torch.version.cuda,
                'cudnn_version' : torch.backends.cudnn.version(),
            })

        print(f'\n✓ Weights & Biases initialized: {wandb.run.url}')

    def _setup_warmup_scheduler(self):
        warmup_epochs = self.config['training']['scheduler'].get('warmup_epochs', 5)
        total_epochs  = self.config['training']['epochs']

        def warmup_cosine(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_cosine)

    def setup_directories(self):
        model_name = self.config['model']['name']
        variant    = self.config['model'].get('variant', None)

        exp_name = f'{model_name}_{variant}' if variant else f'{model_name}'

        output_dir          = self.config['system']['output_dir']
        self.model_dir      = os.path.join(output_dir, model_name)
        self.exp_dir        = os.path.join(self.model_dir, exp_name)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir        = os.path.join(self.exp_dir, 'logs')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

        print(f'Experiment directory: {self.exp_dir}')

    def print_configuration(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())

        print(f'\n{"="*80}')
        print('TRAINING CONFIGURATION')
        print(f'{"="*80}')
        print(f'Device     : {self.device}')
        print(f'Model      : {self.config["model"]["name"]}', end='')
        if self.config['model'].get('variant'):
            print(f' ({self.config["model"]["variant"]})')
        else:
            print()
        print(f'Parameters : {total:,}  (trainable: {trainable:,}  frozen: {total - trainable:,})')
        print(f'Batch size : {self.config["training"]["batch_size"]}')
        print(f'LR         : {self.config["training"]["optimizer"]["learning_rate"]}')
        print(f'Optimizer  : {self.config["training"]["optimizer"]["type"]}')
        print(f'Scheduler  : {self.config["training"]["scheduler"]["type"]}')
        print(f'Loss       : {self.config["loss"]["type"]} (logit-space)')
        print(f'Epochs     : {self.config["training"]["epochs"]}')
        if self.es_patience:
            print(f'Early stop : patience={self.es_patience}, min_delta={self.es_min_delta}')
        else:
            print(f'Early stop : DISABLED (fixed epochs)')
        if self.use_wandb:
            print(f'W&B        : ENABLED ({wandb.run.name})')
        else:
            print(f'W&B        : DISABLED')
        print(f'{"="*80}\n')

    # ── Forward pass ──────────────────────────────────────────────────────────
    #
    # Design:
    #   1. Filter the batch for valid samples (non-null image/mask, valid points).
    #   2. Encode ALL valid images in ONE batched GPU forward pass via
    #      set_image_batch(). This is the key fix: instead of calling
    #      set_image() N times (N × CPU-preprocess → GPU-encode round trips),
    #      we pay the CPU preprocessing cost once per image but the GPU encoder
    #      cost only once for the whole batch.
    #   3. Per-sample: prompt encoding + mask decoding (these are lightweight
    #      and are inherently per-object, so a loop is correct here).
    #   4. Return RAW LOGITS — no sigmoid. Sigmoid is applied inside the loss
    #      function (BCEWithLogitsLoss) and explicitly for metric computation.
    #      Keeping logits in float32 / bfloat16 avoids the numerical instability
    #      of applying sigmoid before a log-loss.
    #
    def _forward(self, batch):

        # ── Step 1: filter valid samples ───────────────────────────────────
        valid_items = []
        for item in batch:
            if item['image'] is None or item['mask'] is None:
                continue
            if item['labels_size'] == 0:
                continue
            pts = item['points']
            if not isinstance(pts, np.ndarray) or pts.size == 0:
                continue
            valid_items.append(item)

        if not valid_items:
            return None, None

        # ── Step 2: batched image encoding (one GPU pass) ──────────────────
        # set_image_batch() runs CPU transforms for each image (resize/pad)
        # then calls the Hiera encoder once on the stacked batch tensor.
        # After this call:
        #   predictor._features["image_embed"]    : [B, C, H, W]
        #   predictor._features["high_res_feats"] : list of [B, C, H, W]
        #   predictor._orig_hw                    : list of (H_orig, W_orig)
        with torch.amp.autocast('cuda'):
            self.predictor.set_image_batch([item['image'] for item in valid_items])

        # ── Step 3: per-sample prompt encoding + mask decoding ─────────────
        prd_logits_list = []
        gt_masks_list   = []

        with torch.amp.autocast('cuda'):
            for i, item in enumerate(valid_items):
                input_point = item['points']                      # (N, 2) numpy
                num_masks   = item['labels_size']
                input_label = np.ones((num_masks, 1), dtype=np.int32)

                # Prepare prompt tensors.
                # img_idx=i tells _prep_prompts which image's (H, W) to use
                # when normalising point coordinates to [0, 1].
                mask_input, unnorm_coords, labels, _ = self.predictor._prep_prompts(
                    input_point, input_label,
                    box=None, mask_logits=None,
                    normalize_coords=True,
                    img_idx=i,
                )

                if unnorm_coords is None or unnorm_coords.shape[0] == 0:
                    continue

                # Slice this sample's features out of the batch.
                image_embed = self.predictor._features['image_embed'][i].unsqueeze(0)  # [1, C, H, W]
                high_res_feats = [
                    feat_level[i].unsqueeze(0)                                          # [1, C, H, W]
                    for feat_level in self.predictor._features['high_res_feats']
                ]

                # Prompt encoder: points → sparse & dense embeddings.
                sparse_emb, dense_emb = self.predictor.model.sam_prompt_encoder(
                    points = (unnorm_coords, labels),
                    boxes  = None,
                    masks  = None,
                )

                # Mask decoder: embeddings → low-res logit masks + IoU scores.
                batched_mode = unnorm_coords.shape[0] > 1
                low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                    image_embeddings        = image_embed,
                    image_pe                = self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings= sparse_emb,
                    dense_prompt_embeddings = dense_emb,
                    multimask_output        = True,
                    repeat_image            = batched_mode,
                    high_res_features       = high_res_feats,
                )

                # Upsample low-res logits back to original image resolution.
                # Result is still LOGITS (no sigmoid).
                prd_logits = self.predictor._transforms.postprocess_masks(
                    low_res_masks, self.predictor._orig_hw[i]
                )

                # Take the first mask channel (index 0 of multimask output).
                prd_logits_list.append(prd_logits[:, 0])  # [1, H, W]  logits

                gt_mask = torch.as_tensor(
                    item['mask'].astype(np.float32), device=self.device
                )
                gt_masks_list.append(gt_mask)

        if not prd_logits_list:
            return None, None

        return torch.stack(prd_logits_list), torch.stack(gt_masks_list)

    # ── Loss ─────────────────────────────────────────────────────────────────
    # main_out is LOGITS. Your combined loss must use BCEWithLogitsLoss
    # (not BCELoss) so it applies log-sum-exp internally for stability.

    def _compute_loss(self, main_out, masks):
        if self.config['loss']['type'] == 'combined':
            loss, loss_dict = self.criterion(main_out, masks, None)
        else:
            loss = self.criterion(main_out, masks)
            loss_dict = {'total': loss.item()}
        return loss, loss_dict

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int):
        # Keep image encoder frozen in eval mode throughout.
        self.model.image_encoder.eval()
        self.model.sam_prompt_encoder.train()
        self.model.sam_mask_decoder.train()

        total_loss      = 0.0
        running_metrics = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                           'intersection': 0.0, 'union': 0.0}

        lr   = self.optimizer.param_groups[0]['lr']
        pbar = tqdm(
            self.train_dataset,
            desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]} [train] LR={lr:.2e}',
        )

        clip_grad = self.config['training'].get('clip_grad', 0.0)

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            main_out, masks = self._forward(batch)
            if main_out is None:
                continue

            loss, loss_dict = self._compute_loss(main_out, masks)

            # ── AMP backward + grad clip + optimizer step ──────────────────
            # Correct order when using GradScaler:
            #   1. scale(loss).backward()   — accumulate scaled gradients
            #   2. unscale_(optimizer)      — convert gradients back to fp32
            #   3. clip_grad_norm_()        — clip BEFORE the step (not after)
            #   4. scaler.step(optimizer)   — update weights
            #   5. scaler.update()          — adjust scale factor for next iter
            self.scaler.scale(loss).backward()

            if clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.sam_prompt_encoder.parameters()) +
                    list(self.model.sam_mask_decoder.parameters()),
                    clip_grad,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # ── Metrics — apply sigmoid ONCE on logits ─────────────────────
            with torch.no_grad():
                probs        = torch.sigmoid(main_out)          # logits → [0,1]
                preds        = (probs > 0.5).float()
                masks_binary = masks.float()

                running_metrics['tp']           += (preds * masks_binary).sum().item()
                running_metrics['fp']           += (preds * (1 - masks_binary)).sum().item()
                running_metrics['fn']           += ((1 - preds) * masks_binary).sum().item()
                running_metrics['tn']           += ((1 - preds) * (1 - masks_binary)).sum().item()
                running_metrics['intersection'] += (preds * masks_binary).sum().item()
                running_metrics['union']        += (preds + masks_binary).clamp(0, 1).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            log_interval = self.config['system'].get('log_interval', 10)
            if batch_idx % log_interval == 0:
                step = epoch * len(self.train_dataset) + batch_idx
                self.writer.add_scalar('Batch/Loss', loss.item(), step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Batch/{key}', value, step)
                if self.use_wandb:
                    wandb_log = {'train/batch_loss': loss.item(), 'train/batch_step': step}
                    for key, value in loss_dict.items():
                        wandb_log[f'train/batch_{key}'] = value
                    wandb.log(wandb_log)

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        eps      = 1e-7
        avg_loss = total_loss / max(len(self.train_dataset), 1)
        tp, fp, fn = running_metrics['tp'], running_metrics['fp'], running_metrics['fn']
        dice      = (2 * tp) / (2 * tp + fp + fn + eps)
        iou       = running_metrics['intersection'] / (running_metrics['union'] + eps)
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)

        return avg_loss, {'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall}

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, epoch: int):
        self.model.image_encoder.eval()
        self.model.sam_prompt_encoder.eval()
        self.model.sam_mask_decoder.eval()

        total_loss      = 0.0
        running_metrics = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                           'intersection': 0.0, 'union': 0.0}

        sample_images, sample_masks, sample_preds = [], [], []
        log_samples = self.use_wandb and self.config['logging'].get('log_images', False)
        max_samples = 4

        with torch.no_grad():
            for batch in tqdm(self.validation_dataset, desc='Validation'):
                main_out, masks = self._forward(batch)
                if main_out is None:
                    continue

                loss, _ = self._compute_loss(main_out, masks)
                total_loss += loss.item()

                # ── Metrics — apply sigmoid ONCE on logits ─────────────────
                probs        = torch.sigmoid(main_out)          # logits → [0,1]
                preds        = (probs > 0.5).float()
                masks_binary = masks.float()

                running_metrics['tp']           += (preds * masks_binary).sum().item()
                running_metrics['fp']           += (preds * (1 - masks_binary)).sum().item()
                running_metrics['fn']           += ((1 - preds) * masks_binary).sum().item()
                running_metrics['tn']           += ((1 - preds) * (1 - masks_binary)).sum().item()
                running_metrics['intersection'] += (preds * masks_binary).sum().item()
                running_metrics['union']        += (preds + masks_binary).clamp(0, 1).sum().item()

                if log_samples and len(sample_images) < max_samples:
                    sample_images.append(batch[0]['image'])
                    sample_masks.append(masks[0].cpu())
                    sample_preds.append(preds[0].cpu())

        avg_loss = total_loss / max(len(self.validation_dataset), 1)
        eps      = 1e-7
        tp, fp, fn = running_metrics['tp'], running_metrics['fp'], running_metrics['fn']
        dice      = (2 * tp) / (2 * tp + fp + fn + eps)
        iou       = running_metrics['intersection'] / (running_metrics['union'] + eps)
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)

        if log_samples and sample_images:
            self._log_predictions_wandb(sample_images, sample_masks, sample_preds, epoch)

        return avg_loss, {'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall}

    # ── wandb image logging ───────────────────────────────────────────────────

    def _log_predictions_wandb(self, images, masks, preds, epoch):
        wandb_images = []
        for img, mask, pred in zip(images, masks, preds):
            img_np  = img
            mask_np = mask.squeeze().numpy()
            pred_np = pred.squeeze().numpy()
            if img_np.min() < 0:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            wandb_images.append(wandb.Image(
                img_np,
                masks={
                    'ground_truth': {'mask_data': mask_np,
                                     'class_labels': {0: 'background', 1: 'river'}},
                    'prediction':   {'mask_data': pred_np,
                                     'class_labels': {0: 'background', 1: 'river'}},
                },
                caption=f'epoch {epoch}',
            ))
        wandb.log({'val/predictions': wandb_images, 'epoch': epoch})

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch, val_dice, val_iou, is_best):
        checkpoint = {
            'epoch'               : epoch,
            'model_state_dict'    : self.predictor.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_dice'       : self.best_val_dice,
            'best_val_iou'        : self.best_val_iou,
            'config'              : self.config,
        }

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'  ✓ New best — Dice: {val_dice:.4f}  IoU: {val_iou:.4f}')
            if self.use_wandb and self.config['logging'].get('save_model_wandb', False):
                wandb.save(best_path)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        print(f'\nStarting training from epoch {self.start_epoch + 1}…')

        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            print(f'\n{"="*80}')
            print(f'Epoch {epoch + 1}/{self.config["training"]["epochs"]}')
            print(f'{"="*80}')

            torch.cuda.empty_cache()
            train_loss, train_metrics = self.train_epoch(epoch)

            torch.cuda.empty_cache()
            val_loss, val_metrics = self.validate(epoch)

            torch.cuda.empty_cache()

            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            print(f'\nEpoch {epoch + 1} Summary:')
            print(f'  LR     : {lr:.2e}')
            print(f'  Train  — Loss: {train_loss:.4f} | Dice: {train_metrics["dice"]:.4f} | IoU: {train_metrics["iou"]:.4f}')
            print(f'  Val    — Loss: {val_loss:.4f}   | Dice: {val_metrics["dice"]:.4f}   | IoU: {val_metrics["iou"]:.4f}')

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                print(f'  GPU    : {gpu_mem:.2f} GB allocated')

            # TensorBoard
            for tag, val in [
                ('Loss/train',      train_loss),
                ('Loss/val',        val_loss),
                ('Dice/train',      train_metrics['dice']),
                ('Dice/val',        val_metrics['dice']),
                ('IoU/train',       train_metrics['iou']),
                ('IoU/val',         val_metrics['iou']),
                ('Precision/train', train_metrics['precision']),
                ('Precision/val',   val_metrics['precision']),
                ('Recall/train',    train_metrics['recall']),
                ('Recall/val',      val_metrics['recall']),
                ('LR',              lr),
            ]:
                self.writer.add_scalar(tag, val, epoch)

            # W&B
            if self.use_wandb:
                wandb_log = {
                    'epoch'           : epoch,
                    'train/loss'      : train_loss,
                    'train/dice'      : train_metrics['dice'],
                    'train/iou'       : train_metrics['iou'],
                    'train/precision' : train_metrics['precision'],
                    'train/recall'    : train_metrics['recall'],
                    'val/loss'        : val_loss,
                    'val/dice'        : val_metrics['dice'],
                    'val/iou'         : val_metrics['iou'],
                    'val/precision'   : val_metrics['precision'],
                    'val/recall'      : val_metrics['recall'],
                    'learning_rate'   : lr,
                }
                if torch.cuda.is_available():
                    wandb_log['gpu_memory_gb'] = torch.cuda.memory_allocated(0) / 1024**3
                wandb.log(wandb_log)

            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.best_val_iou  = val_metrics['iou']
                if self.use_wandb:
                    wandb.run.summary['best_val_dice'] = self.best_val_dice
                    wandb.run.summary['best_val_iou']  = self.best_val_iou
                    wandb.run.summary['best_epoch']    = epoch

            self.save_checkpoint(epoch, val_metrics['dice'], val_metrics['iou'], is_best)

            # Early stopping
            if self.es_patience is not None:
                if val_metrics['dice'] > self.es_best_dice + self.es_min_delta:
                    self.es_best_dice = val_metrics['dice']
                    self.es_counter   = 0
                else:
                    self.es_counter += 1
                    print(f'  [EarlyStopping] {self.es_counter}/{self.es_patience} '
                          f'best={self.es_best_dice:.4f} current={val_metrics["dice"]:.4f}')
                if self.es_counter >= self.es_patience:
                    print(f'\n[EarlyStopping] Triggered at epoch {epoch + 1}. '
                          f'Best val Dice: {self.es_best_dice:.4f}')
                    if self.use_wandb:
                        wandb.run.summary['stopped_epoch']    = epoch + 1
                        wandb.run.summary['early_stopped']    = True
                        wandb.run.summary['es_best_val_dice'] = self.es_best_dice
                    self.writer.add_scalar('EarlyStopping/stopped_epoch', epoch + 1, epoch)
                    print('-' * 80)
                    break

            print('-' * 80)

        print('\n' + '=' * 80)
        print('TRAINING COMPLETE')
        print(f'Best Dice     : {self.best_val_dice:.4f}')
        print(f'Best IoU      : {self.best_val_iou:.4f}')
        print(f'Epochs run    : {epoch + 1} / {self.config["training"]["epochs"]}')
        if self.es_patience:
            print(f'Early stopped : {"YES" if self.es_counter >= self.es_patience else "NO"}')
        print('=' * 80 + '\n')

        self.writer.close()
        if self.use_wandb:
            wandb.finish()

def get_sam2_dataset(data_root, batch_size):
    dataset = create_sam2_dataset(data_root, 'test', batch_size)
    return dataset

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_default_config():
    return {
        'model': {
            'name'      : 'sam2',
            'variant'   : None,
            'n_channels': 3,
            'n_classes' : 1,
        },
        'data': {
            'data_root'    : r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/dataset/processed_512_resized/sequential',
            'image_size'   : 512,
            'augment_train': True,
        },
        'training': {
            'batch_size': 4,
            'epochs'    : 1,
            'clip_grad' : 1.0,
            'resume'    : False,
            'optimizer' : {
                'type'         : 'adam',
                'learning_rate': 1e-4,
                'weight_decay' : 0.01,
                'momentum'     : 0.9,
            },
            'scheduler' : {
                'type'         : 'cosine',
                'min_lr'       : 1e-6,
                'step_size'    : 30,
                'gamma'        : 0.1,
                'warmup_epochs': 5,
            },
        },
        'loss': {
            # IMPORTANT: combined loss must use BCEWithLogitsLoss internally.
            # _forward() returns logits; sigmoid must NOT be applied before
            # the loss function receives its input.
            'type'           : 'combined',
            'bce_weight'     : 1.0,
            'dice_weight'    : 1.0,
            'boundary_weight': 1.0,
            'use_boundary'   : True,
        },
        'logging': {
            'use_wandb'       : True,
            'wandb_project'   : 'river-segmentation',
            'wandb_notes'     : 'UAV river segmentation benchmark',
            'watch_model'     : False,
            'log_images'      : True,
            'save_model_wandb': True,
        },
        'system': {
            'seed'         : 42,
            'num_workers'  : 0,
            'output_dir'   : r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/experiments',
            'log_interval' : 10,
            'save_interval': 10,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training entry points
# ─────────────────────────────────────────────────────────────────────────────

def train_single_model(config: dict):
    torch.manual_seed(config['system']['seed'])
    np.random.seed(config['system']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['system']['seed'])
        torch.cuda.empty_cache()
        import gc; gc.collect()

    trainer = UnifiedTrainer(config)
    trainer.train()


def train_all_models(base_config: dict):
    """Train all SAM2 variants sequentially."""
    all_models = {
        'sam_v2_fine_tuned': ['sam2.1_hiera_tiny', 'sam2.1_hiera_small', 'sam2.1_hiera_base_plus'],
    }

    FOUNDATION_MODELS = {'sam', 'sam_fpn', 'sam_v1_fine_tuned', 'sam_v2_fine_tuned'}

    for model_name, variants in all_models.items():
        for variant in (variants or [None]):
            tag = f'{model_name}' + (f' - {variant}' if variant else '')
            print(f'\n{"="*80}\nTraining {tag}\n{"="*80}\n')

            config = {**base_config}
            config['model'] = {
                **base_config['model'],
                'name'   : model_name,
                'variant': variant,
            }

            if model_name in FOUNDATION_MODELS:
                config['training'] = {
                    **config['training'],
                    'early_stopping_patience' : 20,
                    'early_stopping_min_delta': 1e-4,
                }

            train_single_model(config)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    train_all_models(get_default_config())


if __name__ == '__main__':
    main()
