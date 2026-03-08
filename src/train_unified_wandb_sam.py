"""
train_unified_wandb.py
======================
Unified Training Script with Weights & Biases Integration.

Supports:
  - Standard models   : CNN baselines (UNet, UNet++, ResUNet++, DeepLabV3+),
                        Hybrid SOTA (ConvNeXt-UPerNet, HRNet-OCR),
                        Transformers (SegFormer, Swin-UNet)
  - Foundation models : DINOv2 (via get_model), SAM vit_b/vit_l/vit_h

All models use the same single dataset (processed_512_resized), the same
combined loss (BCE + Dice, boundary optional), Adam optimiser, and cosine
scheduler.  Foundation models additionally use early stopping (patience=20)
to prevent overfitting on the 274-image training set.
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from models import get_model
from src.utils.losses import get_loss_function
from src.dataset.dataset_loader import get_training_dataloaders
from src.utils.metrics import SegmentationMetrics

# SAM-specific imports — only required when training SAM variants.
try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print('[WARN] segment_anything not found — SAM training will not be available.')


# ─────────────────────────────────────────────────────────────────────────────
# SAM Dataset
# ─────────────────────────────────────────────────────────────────────────────

# SAM normalisation constants — match values hardcoded in sam_model.preprocess().
_SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28,  103.53 ]).view(3, 1, 1)
_SAM_PIXEL_STD  = torch.tensor([ 58.395,  57.12,   57.375]).view(3, 1, 1)


def _sam_preprocess(x: torch.Tensor, img_size: int = 1024) -> torch.Tensor:
    """
    Replicate sam_model.preprocess() without requiring the model object.
    Input  : float tensor [3, H, W], pixel values 0-255
    Output : float tensor [3, img_size, img_size], normalised + zero-padded
    """
    x = (x.float() - _SAM_PIXEL_MEAN) / _SAM_PIXEL_STD
    h, w = x.shape[-2:]
    pad_h = img_size - h
    pad_w = img_size - w
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
    return x


class SAMDataset(Dataset):
    """
    On-the-fly dataset for SAM fine-tuning.

    Loads images from the same processed_512_resized dataset used by every
    other model.  Adds SAM-specific preprocessing on top:
      1. Load image → resize to image_size × image_size
      2. Optional augmentation (flips, 90° rotations)
      3. ResizeLongestSide(1024) — square 512 input → 1024×1024
      4. SAM normalisation + zero-padding to 1024×1024

    Returned batch dict keys
    ------------------------
    input_tensor   : FloatTensor [3, 1024, 1024]  — SAM-preprocessed image
    input_size     : LongTensor  [2]              — (H', W') after ResizeLongestSide
    original_size  : LongTensor  [2]              — (image_size, image_size)
    mask           : FloatTensor [1, image_size, image_size]  — binary 0/1
    image_name     : str
    """

    SAM_IMG_SIZE = 1024

    def __init__(
        self,
        data_root : str,
        split     : str,
        image_size: int  = 512,
        augment   : bool = False,
    ):
        import cv2 as _cv2
        self._cv2       = _cv2
        self.image_size = image_size
        self.augment    = augment

        img_dir  = Path(data_root) / split / 'images'
        mask_dir = Path(data_root) / split / 'masks'

        img_paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        self.samples: list[tuple[Path, Path]] = []
        skipped = 0

        for img_path in img_paths:
            mask_path = mask_dir / f'{img_path.stem}.png'
            if not mask_path.exists():
                mask_path = mask_dir / f'{img_path.stem}.jpg'
            if not mask_path.exists():
                skipped += 1
                continue
            self.samples.append((img_path, mask_path))

        if skipped:
            print(f'  [SAMDataset/{split}] Skipped {skipped} images (no matching mask).')
        print(f'  [SAMDataset/{split}] {len(self.samples)} samples loaded.')

        if not SAM_AVAILABLE:
            raise RuntimeError(
                'segment_anything is not installed. '
                'Run: pip install git+https://github.com/facebookresearch/segment-anything.git'
            )
        self.resize_transform = ResizeLongestSide(self.SAM_IMG_SIZE)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.samples[idx]

        image = self._cv2.imread(str(img_path))
        image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        image = self._cv2.resize(
            image, (self.image_size, self.image_size),
            interpolation=self._cv2.INTER_LINEAR,
        )

        mask = self._cv2.imread(str(mask_path), self._cv2.IMREAD_GRAYSCALE)
        mask = self._cv2.resize(
            mask, (self.image_size, self.image_size),
            interpolation=self._cv2.INTER_NEAREST,
        )
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

        original_size = (image.shape[0], image.shape[1])

        resized_image = self.resize_transform.apply_image(image)
        input_size    = (resized_image.shape[0], resized_image.shape[1])

        img_tensor   = torch.as_tensor(resized_image).permute(2, 0, 1).float()
        input_tensor = _sam_preprocess(img_tensor, self.SAM_IMG_SIZE)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return {
            'input_tensor' : input_tensor,
            'input_size'   : torch.tensor(input_size,    dtype=torch.long),
            'original_size': torch.tensor(original_size, dtype=torch.long),
            'mask'         : mask_tensor,
            'image_name'   : img_path.stem,
        }

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        import random
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask  = np.fliplr(mask).copy()
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask  = np.flipud(mask).copy()
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask  = np.rot90(mask,  k).copy()
        return image, mask


# ─────────────────────────────────────────────────────────────────────────────
# Unified Trainer
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedTrainer:
    """
    Unified trainer for all benchmark models.

    Supports standard single-input models (CNN, Transformer, Hybrid) and SAM.
    All models share the same dataset (processed_512_resized), combined loss,
    Adam optimiser, and CosineAnnealing scheduler.
    """

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

        # ── Mode flag ─────────────────────────────────────────────────────
        model_name  = config['model']['name']
        variant     = config['model'].get('variant', None)
        self.is_sam = (model_name == 'sam')

        # ── Model ─────────────────────────────────────────────────────────
        print(f'\nInitializing {model_name}' + (f' ({variant})' if variant else '') + '…')

        if self.is_sam:
            if not SAM_AVAILABLE:
                raise RuntimeError(
                    'segment_anything is required for SAM training. '
                    'Install with: pip install git+https://github.com/facebookresearch/segment-anything.git'
                )
            sam_ckpt = (
                config.get('foundation', {})
                      .get('sam_checkpoints', {})
                      .get(variant, f'./checkpoints/sam_fpn/sam_{variant}.pth')
            )
            print(f'  [SAM] Loading {variant} checkpoint: {sam_ckpt}')
            self.model = sam_model_registry[variant](checkpoint=sam_ckpt).to(self.device)
            print(f'  [SAM] Pretrained weights loaded ✓')
        else:
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

        if self.is_sam:
            # SAM requires its own preprocessing (ResizeLongestSide + SAM
            # normalisation) that differs from the standard pipeline.
            train_ds = SAMDataset(data_root, 'train', image_size=img_size, augment=augment)
            val_ds   = SAMDataset(data_root, 'val',   image_size=img_size, augment=False)
            self.train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=n_workers, pin_memory=True, drop_last=False,
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=n_workers, pin_memory=True, drop_last=False,
            )
            print(f'  train: {len(train_ds)} samples  |  val: {len(val_ds)} samples')
            print(f'  Image size: {img_size}×{img_size}  →  SAM input: 1024×1024')
        else:
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
                bce_weight      = loss_cfg.get('bce_weight',      1.0),
                dice_weight     = loss_cfg.get('dice_weight',     1.0),
                boundary_weight = loss_cfg.get('boundary_weight', 1.0),
                use_boundary    = loss_cfg.get('use_boundary',    False),
            )
        else:
            self.criterion = get_loss_function(loss_cfg['type'])

        # ── Optimizer ─────────────────────────────────────────────────────
        opt_cfg = config['training']['optimizer']
        base_lr = opt_cfg['learning_rate']
        wd      = opt_cfg.get('weight_decay', 0.0)

        if self.is_sam:
            # Only the mask_decoder is optimised — image_encoder and
            # prompt_encoder are kept frozen via no_grad in _forward_sam().
            param_groups = [
                {'params': self.model.mask_decoder.parameters(),
                 'lr': base_lr, 'name': 'mask_decoder'},
            ]
            print(f'  [Optimizer] SAM: mask_decoder only  LR={base_lr:.1e}')
            print(f'    image_encoder  FROZEN  '
                  f'params={sum(p.numel() for p in self.model.image_encoder.parameters()):,}')
            print(f'    prompt_encoder FROZEN  '
                  f'params={sum(p.numel() for p in self.model.prompt_encoder.parameters()):,}')
            print(f'    mask_decoder   trainable  '
                  f'params={sum(p.numel() for p in self.model.mask_decoder.parameters()):,}')
        elif hasattr(self.model, 'get_params_groups'):
            param_groups = self.model.get_params_groups(lr=base_lr)
            print(f'  [Optimizer] Differential LR groups:')
            for i, g in enumerate(param_groups):
                name = g.get('name', f'group_{i}')
                n_p  = sum(p.numel() for p in g['params'])
                print(f'    {name:<16} LR={g["lr"]:.1e}  params={n_p:,}')
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

        # Early stopping — disabled by default (patience=None).
        # Enabled for foundation models only via train_all_models().
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
        loss_type  = self.config['loss']['type']

        exp_name = f'{model_name}_{variant}_{loss_type}' if variant else f'{model_name}_{loss_type}'

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
        print(f'\n{"="*80}')
        print('TRAINING CONFIGURATION')
        print(f'{"="*80}')
        print(f'Device     : {self.device}')
        print(f'Mode       : {"SAM fine-tuning (mask_decoder only)" if self.is_sam else "Standard"}')
        print(f'Model      : {self.config["model"]["name"]}', end='')
        if self.config['model'].get('variant'):
            print(f' ({self.config["model"]["variant"]})')
        else:
            print()
        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Parameters : {total_params:,}  (trainable: {trainable_params:,})')
        if self.is_sam:
            decoder_params = sum(p.numel() for p in self.model.mask_decoder.parameters())
            print(f'  mask_decoder: {decoder_params:,}')
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

    # ── Forward pass ─────────────────────────────────────────────────────────

    def _forward(self, batch):
        """
        Run the model and return (main_out, masks).
        Dispatches to _forward_sam() for SAM, otherwise standard pipeline.
        """
        if self.is_sam:
            return self._forward_sam(batch)

        images  = batch['image'].to(self.device)
        masks   = batch['mask'].to(self.device)
        outputs = self.model(images)
        # Some models return (main, aux) tuples — take main only
        main_out = outputs[0] if isinstance(outputs, tuple) else outputs
        return main_out, masks

    def _forward_sam(self, batch):
        """
        SAM-specific forward pass.

        Why per-image loop (not vectorised batch):
        -------------------------------------------
        SAM's mask_decoder.predict_masks() calls:
            src = torch.repeat_interleave(image_embedding, tokens.shape[0], dim=0)
        where tokens.shape[0] should be the number of prompt tokens per image
        (1 for promptless inference).  Passing a batched sparse_emb [B, N, C]
        makes tokens.shape[0] = B, producing a [B*B, ...] tensor and crashing.

        Fix: run image_encoder on the full batch (expensive ViT, worth batching),
        then loop prompt_encoder + mask_decoder per image (tiny, negligible overhead).

        Returns (logits [B,1,H,W], masks [B,1,H,W]).
        """
        input_tensor  = batch['input_tensor'].to(self.device)   # [B, 3, 1024, 1024]
        masks         = batch['mask'].to(self.device)            # [B, 1, img_size, img_size]
        B             = input_tensor.shape[0]

        input_size    = tuple(int(x) for x in batch['input_size'][0])
        original_size = tuple(int(x) for x in batch['original_size'][0])

        # ── image_encoder — batched, frozen ──────────────────────────────
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(input_tensor)
            # [B, 256, 64, 64]

        # ── prompt_encoder + mask_decoder — per image ────────────────────
        upscaled_list = []
        for i in range(B):
            emb_i = image_embeddings[i].unsqueeze(0)  # [1, 256, 64, 64]

            with torch.no_grad():
                sparse_emb, dense_emb = self.model.prompt_encoder(
                    points=None, boxes=None, masks=None,
                )
                # sparse_emb: [1, 0, 256]   dense_emb: [1, 256, 64, 64]

            low_res_i, _ = self.model.mask_decoder(
                image_embeddings         = emb_i,
                image_pe                 = self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sparse_emb,
                dense_prompt_embeddings  = dense_emb,
                multimask_output         = False,
            )
            # low_res_i: [1, 1, 256, 256]

            upscaled_i = self.model.postprocess_masks(
                low_res_i, input_size, original_size,
            )
            # upscaled_i: [1, 1, H_orig, W_orig]
            upscaled_list.append(upscaled_i)

        return torch.cat(upscaled_list, dim=0), masks   # [B,1,H,W], [B,1,H,W]

    # ── Loss ─────────────────────────────────────────────────────────────────

    def _compute_loss(self, main_out, masks):
        if self.config['loss']['type'] == 'combined':
            loss, loss_dict = self.criterion(main_out, masks, None)
        else:
            loss = self.criterion(main_out, masks)
            loss_dict = {'total': loss.item()}
        return loss, loss_dict

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int):
        self.model.train()

        total_loss      = 0.0
        running_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                           'intersection': 0, 'union': 0}

        lr   = self.optimizer.param_groups[0]['lr']
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]} [train] LR={lr:.2e}',
        )

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            main_out, masks = self._forward(batch)
            loss, loss_dict = self._compute_loss(main_out, masks)

            loss.backward()

            clip_grad = self.config['training'].get('clip_grad', 0)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

            self.optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                preds        = (torch.sigmoid(main_out) > 0.5).float()
                masks_binary = masks.float()

                running_metrics['tp']           += (preds * masks_binary).sum().item()
                running_metrics['fp']           += (preds * (1 - masks_binary)).sum().item()
                running_metrics['fn']           += ((1 - preds) * masks_binary).sum().item()
                running_metrics['tn']           += ((1 - preds) * (1 - masks_binary)).sum().item()
                running_metrics['intersection'] += (preds * masks_binary).sum().item()
                running_metrics['union']        += (preds + masks_binary).clamp(0, 1).sum().item()

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

        eps      = 1e-7
        avg_loss = total_loss / len(self.train_loader)
        tp, fp, fn = (running_metrics['tp'], running_metrics['fp'], running_metrics['fn'])
        dice      = (2 * tp) / (2 * tp + fp + fn + eps)
        iou       = running_metrics['intersection'] / (running_metrics['union'] + eps)
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)

        return avg_loss, {'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall}

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, epoch: int):
        self.model.eval()

        total_loss      = 0.0
        running_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                           'intersection': 0, 'union': 0}

        sample_images, sample_masks, sample_preds = [], [], []
        log_samples = self.use_wandb and self.config['logging'].get('log_images', False)
        max_samples = 4

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                main_out, masks = self._forward(batch)
                loss, _         = self._compute_loss(main_out, masks)

                total_loss += loss.item()

                preds        = (torch.sigmoid(main_out) > 0.5).float()
                masks_binary = masks.float()

                running_metrics['tp']           += (preds * masks_binary).sum().item()
                running_metrics['fp']           += (preds * (1 - masks_binary)).sum().item()
                running_metrics['fn']           += ((1 - preds) * masks_binary).sum().item()
                running_metrics['tn']           += ((1 - preds) * (1 - masks_binary)).sum().item()
                running_metrics['intersection'] += (preds * masks_binary).sum().item()
                running_metrics['union']        += (preds + masks_binary).clamp(0, 1).sum().item()

                if log_samples and len(sample_images) < max_samples:
                    # For SAM batches use input_tensor; standard batches use image
                    img_key = 'input_tensor' if self.is_sam else 'image'
                    sample_images.append(batch[img_key][0].cpu())
                    sample_masks.append(masks[0].cpu())
                    sample_preds.append(preds[0].cpu())

        avg_loss = total_loss / len(self.val_loader)
        eps      = 1e-7
        tp, fp, fn = (running_metrics['tp'], running_metrics['fp'], running_metrics['fn'])
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
                caption=f'epoch {epoch}',
            ))
        wandb.log({'val/predictions': wandb_images, 'epoch': epoch})

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
        }

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'  ✓ New best — Dice: {val_dice:.4f}  IoU: {val_iou:.4f}')
            if self.use_wandb and self.config['logging'].get('save_model_wandb', False):
                wandb.save(best_path)

        save_interval = self.config['system'].get('save_interval', 10)
        if epoch % save_interval == 0:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth'))

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

            # Early stopping — only active for foundation models (patience set
            # in train_all_models).  All other benchmark models run fixed epochs.
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


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
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
            'data_root'    : './dataset/processed_512_resized',
            'image_size'   : 512,
            'augment_train': True,
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
                'type'         : 'cosine',
                'min_lr'       : 1e-6,
                'step_size'    : 30,
                'gamma'        : 0.1,
                'warmup_epochs': 5,
            },
        },
        'loss': {
            'type'           : 'combined',
            'bce_weight'     : 1.0,
            'dice_weight'    : 1.0,
            'boundary_weight': 1.0,
            'use_boundary'   : False,
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
            'output_dir'   : './experiments',
            'log_interval' : 10,
            'save_interval': 10,
        },
        'foundation': {
            'pretrained'    : True,
            'freeze_encoder': False,
            # SAM checkpoints — download from Meta before training:
            #   vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
            #   vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
            #   vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
            'sam_checkpoints': {
                'vit_b': './checkpoints/sam/sam_vit_b_01ec64.pth',
                'vit_l': './checkpoints/sam/sam_vit_l_0b3195.pth',
                'vit_h': './checkpoints/sam/sam_vit_h_4b8939.pth',
            },
        },
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
    """Train all SAM variants (and optionally DINOv2) sequentially."""
    all_models = {
        # Uncomment when ready:
        # 'dinov2': ['vit_s', 'vit_b', 'vit_l'],
        'sam_fpn': ['vit_b', 'vit_l', 'vit_h'],
    }

    # Foundation models use early stopping — all other benchmark models ran
    # fixed 100 epochs.  This asymmetry is documented in the paper.
    FOUNDATION_MODELS = {'sam_fpn', 'dinov2'}

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
