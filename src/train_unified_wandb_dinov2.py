# Training module for dinov2_Mask2Former only

import os
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from models import get_model
from src.utils.losses import get_loss_function
from src.dataset.dataset_loader import get_training_dataloaders
from src.utils.metrics import SegmentationMetrics
from src.foundation_models.dinov2.mask2former_head import DINOv2Mask2FormerSegmentation


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

        # ── Mode flags — must be set before any method that reads them ────
        model_name = config['model']['name']
        variant    = config['model'].get('variant', None)

        # ── Model ─────────────────────────────────────────────────────────
        print(f'\nInitializing {model_name}' + (f' ({variant})' if variant else '') + '…')

        self.model = get_model(
                model_name = model_name,
                variant    = variant,
                n_channels = config['model'].get('n_channels', 3),
                n_classes  = config['model'].get('n_classes', 1),
            ).to(self.device)

        # ── Data loaders ───────────────────────────────────────────────────
        data_root  = config['data']['data_root']
        batch_size = config['training']['batch_size']
        n_workers  = config['system']['num_workers']
        img_size   = config['data']['image_size']
        augment    = config['data'].get('augment_train', True)

        print(f'Loading data from {data_root}…')

        self.train_loader, self.val_loader = get_training_dataloaders(
                data_dir      = data_root,
                batch_size    = batch_size,
                num_workers   = n_workers,
                augment_train = augment,
                image_size    = img_size,
            )

        # ── Loss ───────────────────────────────────────────────────────────
        loss_cfg = config['loss']
        print(f'Using {loss_cfg["type"]} loss…')
        if loss_cfg['type'] == 'combined':
            self.criterion = get_loss_function(
                loss_cfg['type'],
                bce_weight      = loss_cfg.get('bce_weight', 1.0),
                dice_weight     = loss_cfg.get('dice_weight', 1.0),
                boundary_weight = loss_cfg.get('boundary_weight', 1.0),
                use_boundary    = loss_cfg.get('use_boundary', False),
            )
        else:
            self.criterion = get_loss_function(loss_cfg['type'])

        # Auxiliary loss weight for deep supervision on global/local branches
        self.aux_weight = config['loss'].get('aux_weight', 0.4)

        # ── Optimizer ─────────────────────────────────────────────────────
        opt_cfg = config['training']['optimizer']
        base_lr = opt_cfg['learning_rate']
        wd      = opt_cfg.get('weight_decay', 0.0)


        if hasattr(self.model, 'get_params_groups'):
            # Foundation / transformer models (DINOv2, SegFormer, etc.) expose
            # get_params_groups() for differential per-group learning rates.
            param_groups = self.model.get_params_groups(lr=base_lr)
            print(f'  [Optimizer] Differential LR groups for {model_name}:')
            for i, g in enumerate(param_groups):
                name = g.get('name', f'group_{i}')
                n_params = sum(p.numel() for p in g['params'])
                print(f'    {name:<16} LR={g["lr"]:.1e}  params={n_params:,}')
        else:
            param_groups = self.model.parameters()

        if opt_cfg['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                param_groups, lr=base_lr,
                weight_decay=wd, betas=(0.9, 0.999),
            )
        elif opt_cfg['type'] == 'adam':
            self.optimizer = torch.optim.Adam(
                param_groups, lr=base_lr, weight_decay=wd,
            )
        elif opt_cfg['type'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                param_groups, lr=base_lr,
                momentum=opt_cfg.get('momentum', 0.9),
                weight_decay=wd,
            )

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

        # Early stopping state.
        # Disabled by default (patience=None) — only active when the config
        # explicitly sets 'early_stopping_patience', which train_all_models()
        # does exclusively for foundation models (sam, dinov2).
        # All already-completed models ran with patience=None (full 100 epochs),
        # so this does not alter their training conditions.
        self.es_patience  = config['training'].get('early_stopping_patience',  None)
        self.es_min_delta = config['training'].get('early_stopping_min_delta', 1e-4)
        self.es_counter   = 0
        self.es_best_dice = 0.0

        if config['training'].get('resume', False):
            self.load_checkpoint()

        self.print_configuration()

        if self.use_wandb and config['logging'].get('watch_model', False):
            wandb.watch(self.model, log='all', log_freq=100)

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def init_wandb(self):
        """Initialize Weights & Biases run."""
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
            project = self.config['logging'].get('wandb_project', 'river-segmentation-v2'),
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
        variant = self.config['model'].get('variant', None)
        output_dir = self.config['system']['output_dir']

        if variant is None:
            self.model_dir = os.path.join(output_dir, model_name)
            self.exp_dir = self.model_dir
        else:
            self.model_dir = os.path.join(output_dir, model_name)
            self.exp_dir = os.path.join(self.model_dir, variant)

        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

        print(f'Experiment directory: {self.exp_dir}')

    def print_configuration(self):
        print(f'\n{"="*80}')
        print('TRAINING CONFIGURATION')
        print(f'{"="*80}')
        print(f'Device     : {self.device}')
        print(f'Model      : {self.config["model"]["name"]}', end='')
        if self.config['model'].get('variant'):
            print(f' ({self.config["model"]["variant"]})')
        else:
            print()
        total_params    = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Parameters : {total_params:,}  (trainable: {trainable_params:,})')

        print(f'Batch size : {self.config["training"]["batch_size"]}')
        print(f'LR         : {self.config["training"]["optimizer"]["learning_rate"]}')
        print(f'Optimizer  : {self.config["training"]["optimizer"]["type"]}')
        print(f'Scheduler  : {self.config["training"]["scheduler"]["type"]}')
        print(f'Loss       : {self.config["loss"]["type"]}')
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

    # ── Loss helpers ──────────────────────────────────────────────────────────

    def _compute_loss(self, main_out, masks, aux_out=None):
        """
        Unified loss computation.

        Three cases:
          1. Mask2Former : aux_out is a 0-dim scalar tensor (HF internal loss).
                           Return it directly — Hungarian matching already done.
          2. GlobalLocal : aux_out is (g_logits, l_logits).
          3. Standard    : aux_out is None.
        """
        # ── Case 1: Mask2Former pre-computed Hungarian-matching loss ──────────
        if isinstance(aux_out, torch.Tensor) and aux_out.dim() == 0:
            loss_dict = {
                'total': aux_out.item(),
                'm2f_loss': aux_out.item(),
            }
            return aux_out, loss_dict

        # ── Cases 2 & 3: existing logic — NO CHANGES ─────────────────────────
        if self.config['loss']['type'] == 'combined':
            main_loss, loss_dict = self.criterion(main_out, masks, None)
        else:
            main_loss = self.criterion(main_out, masks)
            loss_dict = {'total': main_loss.item()}

        if aux_out is not None:
            g_logits, l_logits = aux_out
            if self.config['loss']['type'] == 'combined':
                g_loss, _ = self.criterion(g_logits, masks, None)
                l_loss, _ = self.criterion(l_logits, masks, None)
            else:
                g_loss = self.criterion(g_logits, masks)
                l_loss = self.criterion(l_logits, masks)

            aux_loss = (g_loss + l_loss) / 2.0
            total = main_loss + self.aux_weight * aux_loss

            loss_dict['main_loss'] = main_loss.item()
            loss_dict['global_aux'] = g_loss.item()
            loss_dict['local_aux'] = l_loss.item()
            loss_dict['total'] = total.item()
            return total, loss_dict

        return main_loss, loss_dict

    def _forward(self, batch):
        """
        Mask2Former-specific forward pass.

        Training
        --------
        1. Converts (B, 1, H, W) GT masks to HF instance-label format.
           Each image has at most 1 instance (water region, class index 0).
           Empty-mask images produce zero-length tensors, which HF handles as
           all-no-object supervision for all queries.
        2. Calls model(images, mask_labels, class_labels).
           HF internally computes Hungarian matching + loss:
             L = sum over decoder layers of  class_CE + mask_BCE + mask_Dice
           Final layer weight = 1.0; intermediate layers = 0.5.
        3. Returns (seg_logits, m2f_loss, masks).
           m2f_loss is a 0-dim scalar tensor, detected in _compute_loss() by:
             isinstance(aux_out, torch.Tensor) and aux_out.dim() == 0

        Inference
        ---------
        No labels passed. Returns (seg_logits, None, masks).

        seg_logits (B, 1, H, W) are compatible with the trainer metrics path:
          preds = (torch.sigmoid(seg_logits) > 0.5).float()
        """

        images = batch['image'].to(self.device)  # (B, 3, H, W)
        masks = batch['mask'].to(self.device)  # (B, 1, H, W)  float 0/1

        if self.model.training:
            mask_labels, class_labels = (
                DINOv2Mask2FormerSegmentation.prepare_binary_labels(masks)
            )
            seg_logits, m2f_loss = self.model(
                pixel_values=images,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            return seg_logits, m2f_loss, masks
        else:
            seg_logits, _ = self.model(images)
            return seg_logits, None, masks

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int):
        self.model.train()

        total_loss = 0
        running_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                           'intersection': 0, 'union': 0}
        batch_losses = []

        lr   = self.optimizer.param_groups[0]['lr']
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]} [train] LR={lr:.2e}',
        )

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            main_out, aux_out, masks = self._forward(batch)
            loss, loss_dict          = self._compute_loss(main_out, masks, aux_out)

            loss.backward()

            clip_grad = self.config['training'].get('clip_grad', 0)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

            self.optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())

            with torch.no_grad():
                preds        = (torch.sigmoid(main_out) > 0.5).float()
                masks_binary = masks.float()

                tp           = (preds * masks_binary).sum().item()
                fp           = (preds * (1 - masks_binary)).sum().item()
                fn           = ((1 - preds) * masks_binary).sum().item()
                tn           = ((1 - preds) * (1 - masks_binary)).sum().item()
                intersection = (preds * masks_binary).sum().item()
                union        = (preds + masks_binary).clamp(0, 1).sum().item()

                running_metrics['tp']           += tp
                running_metrics['fp']           += fp
                running_metrics['fn']           += fn
                running_metrics['tn']           += tn
                running_metrics['intersection'] += intersection
                running_metrics['union']        += union

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            log_interval = self.config['system'].get('log_interval', 10)
            if batch_idx % log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
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

        # ── Epoch-level metrics ────────────────────────────────────────────
        eps  = 1e-7
        avg_loss  = total_loss / len(self.train_loader)
        dice      = (2 * running_metrics['tp']) / (2 * running_metrics['tp'] +
                     running_metrics['fp'] + running_metrics['fn'] + eps)
        iou       = running_metrics['intersection'] / (running_metrics['union'] + eps)
        precision = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fp'] + eps)
        recall    = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fn'] + eps)

        return avg_loss, {'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall}

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, epoch: int):
        self.model.eval()

        total_loss = 0
        running_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                           'intersection': 0, 'union': 0}

        sample_images, sample_masks, sample_preds = [], [], []
        log_samples = self.use_wandb and self.config['logging'].get('log_images', False)
        max_samples = 4

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                main_out, aux_out, masks = self._forward(batch)
                loss, _                  = self._compute_loss(main_out, masks, aux_out)

                total_loss += loss.item()

                preds        = (torch.sigmoid(main_out) > 0.5).float()
                masks_binary = masks.float()

                tp           = (preds * masks_binary).sum().item()
                fp           = (preds * (1 - masks_binary)).sum().item()
                fn           = ((1 - preds) * masks_binary).sum().item()
                tn           = ((1 - preds) * (1 - masks_binary)).sum().item()
                intersection = (preds * masks_binary).sum().item()
                union        = (preds + masks_binary).clamp(0, 1).sum().item()

                running_metrics['tp']           += tp
                running_metrics['fp']           += fp
                running_metrics['fn']           += fn
                running_metrics['tn']           += tn
                running_metrics['intersection'] += intersection
                running_metrics['union']        += union

                if log_samples and len(sample_images) < max_samples:
                    # Log local patch as the primary image
                    sample_images.append(batch['image'][0].cpu())
                    sample_masks.append(masks[0].cpu())
                    sample_preds.append(preds[0].cpu())

        avg_loss  = total_loss / len(self.val_loader)
        eps       = 1e-7
        dice      = (2 * running_metrics['tp']) / (2 * running_metrics['tp'] +
                     running_metrics['fp'] + running_metrics['fn'] + eps)
        iou       = running_metrics['intersection'] / (running_metrics['union'] + eps)
        precision = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fp'] + eps)
        recall    = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fn'] + eps)

        if log_samples and sample_images:
            self.log_predictions_wandb(
                sample_images, sample_masks, sample_preds, epoch, None,
            )

        return avg_loss, {'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall}

    # ── wandb image logging ───────────────────────────────────────────────────

    def log_predictions_wandb(self, images, masks, preds, epoch, global_images=None):
        """Log local patches (and optionally global frames) with mask overlays."""
        wandb_images  = []
        wandb_globals = []

        for i, (img, mask, pred) in enumerate(zip(images, masks, preds)):
            img_np  = img.permute(1, 2, 0).numpy()
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
                caption=f'Local patch — epoch {epoch}',
            ))

            if global_images and i < len(global_images):
                g_np = global_images[i].permute(1, 2, 0).numpy()
                if g_np.min() < 0:
                    g_np = (g_np - g_np.min()) / (g_np.max() - g_np.min() + 1e-8)
                wandb_globals.append(wandb.Image(
                    g_np, caption=f'Global context — epoch {epoch}',
                ))

        log_dict = {'val/local_predictions': wandb_images, 'epoch': epoch}
        if wandb_globals:
            log_dict['val/global_context'] = wandb_globals
        wandb.log(log_dict)

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch, val_dice, val_iou, is_best):
        checkpoint = {
            'epoch'               : epoch,
            'model_state_dict'    : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_dice'       : self.best_val_dice,
            'best_val_iou'        : self.best_val_iou,
            'config'              : self.config,
            'mode'                : 'standard',
        }

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'  ✓ New best — Dice: {val_dice:.4f}  IoU: {val_iou:.4f}')
            if self.use_wandb and self.config['logging'].get('save_model_wandb', False):
                wandb.save(best_path)

    def load_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, 'latest.pth')
        if os.path.exists(path):
            print(f'Loading checkpoint: {path}')
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.scheduler and ckpt['scheduler_state_dict']:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.start_epoch   = ckpt['epoch'] + 1
            self.best_val_dice = ckpt['best_val_dice']
            self.best_val_iou  = ckpt['best_val_iou']
            print(f'  Resumed from epoch {self.start_epoch}')
        else:
            print(f'No checkpoint found at {path}')

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
                ('Loss/train', train_loss),      ('Loss/val',   val_loss),
                ('Dice/train', train_metrics['dice']), ('Dice/val',   val_metrics['dice']),
                ('IoU/train',  train_metrics['iou']),  ('IoU/val',    val_metrics['iou']),
                ('Precision/train', train_metrics['precision']),
                ('Precision/val',   val_metrics['precision']),
                ('Recall/train',    train_metrics['recall']),
                ('Recall/val',      val_metrics['recall']),
                ('LR', lr),
            ]:
                self.writer.add_scalar(tag, val, epoch)

            # Wandb
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
                    wandb_log['gpu_memory_gb'] = gpu_mem
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

            # Early stopping — only active when es_patience is set in config.
            # Foundation models (sam, dinov2) have this enabled; all previously
            # completed models did not, preserving identical training conditions
            # for the benchmark comparison.
            if self.es_patience is not None:
                if val_metrics['dice'] > self.es_best_dice + self.es_min_delta:
                    self.es_best_dice = val_metrics['dice']
                    self.es_counter   = 0
                else:
                    self.es_counter += 1
                    print(f'  [EarlyStopping] No significant improvement for '
                          f'{self.es_counter}/{self.es_patience} epoch(s). '
                          f'Best={self.es_best_dice:.4f}, '
                          f'Current={val_metrics["dice"]:.4f}')
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


# ─────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_default_config():
    return {
        'model': {
            'name'      : 'unet',
            'variant'   : None,
            'n_channels': 3,
            'n_classes' : 1,
        },
        'data': {
            'data_root'    : r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/dataset/processed_512_resized',
            'image_size'   : 512,
            'augment_train': True
        },
        'training': {
            'batch_size': 4,
            'epochs'    : 100,
            'clip_grad' : 1.0,
            'resume'    : False,
            'optimizer' : {
                'type'         : 'adam',
                'learning_rate': 1e-4,
                'weight_decay' : 0.01,
                'momentum'     : 0.9,
            },
            'scheduler' : {
                'type'        : 'cosine',
                'min_lr'      : 1e-6,
                'step_size'   : 30,
                'gamma'       : 0.1,
                'warmup_epochs': 5,
            },
        },
        'loss': {
            'type'          : 'combined',
            'bce_weight'    : 1.0,
            'dice_weight'   : 1.0,
            'boundary_weight': 1.0,
            'use_boundary'  : False,
            'aux_weight'    : 0.4,  # GlobalLocal: weight for auxiliary branch losses
        },
        'logging': {
            'use_wandb'       : True,
            'wandb_project'   : 'river-segmentation',
            'wandb_notes'     : 'UAV river segmentation — global-local dual-branch',
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
    # all_models = {
    #     'dinov2_Mask2Former': ['vit_s', 'vit_b', 'vit_l' ],
    #     'dinov2_DPT'        : ['vit_s', 'vit_b', 'vit_l' ],
    # }

    all_models = {
        'dinov2_Mask2Former': ['vit_s' ],
        # 'dinov2_DPT'        : ['vit_s' ],
    }


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
    # dataset_variations  = ['sequential', 'stratified', 'alternative']
    dataset_variations  = ['stratified', 'alternative']
    for dataset_variation in dataset_variations:

        default_config = get_default_config()
        print(f'default_config: {default_config}')
        print(f'data_root: {default_config['data']['data_root']}')
        print(f'epochs: {default_config['training']['epochs']}')
        print(f'output_dir: {default_config['system']['output_dir']}')

        default_config['training']['epochs'] = 25
        default_config['logging']['use_wandb'] = False

        data_root = default_config['data']['data_root']
        data_root = f'{data_root}/{dataset_variation}'
        output_dir = default_config['system']['output_dir']
        output_dir = f'{output_dir}/{dataset_variation}'

        default_config['data']['data_root'] = data_root
        default_config['system']['output_dir'] = output_dir


        print(f'updated data_root: {default_config['data']['data_root']}')
        print(f'updated output_dir: {default_config['system']['output_dir']}')

        train_all_models(default_config)


if __name__ == '__main__':
    main()
