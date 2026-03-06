"""
train_unified_wandb.py
======================
Unified Training Script with Weights & Biases Integration.

Supports:
  - Standard models  : CNN baselines, Transformers, Hybrid SOTA, Foundation Models
  - GlobalLocal mode : dual-branch model trained on paired resized + sliced datasets

GlobalLocal mode overview
-------------------------
Each training sample consists of:
  • global_image  — the full drone frame resized to 512×512 (scene context)
  • local_patch   — a 512×512 tile sliced from the original full-res image (fine detail)
  • mask          — the ground-truth mask for the local patch tile

The GlobalLocalWrapper's fusion head learns to gate between the two views:
  high alpha → trust global context  (useful under dense tree canopy occlusion)
  low  alpha → trust local detail    (clear riverbank edges and thin structures)
"""

import os
import re
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import wandb

from models import get_model, get_model_varient
from src.utils.losses import get_loss_function
from src.dataset.dataset_loader import get_training_dataloaders
from src.utils.metrics import SegmentationMetrics
from wrapper import GlobalLocalWrapper
from torchviz import make_dot
import inspect

# SAM-specific imports — only required when training SAM variants.
# If segment_anything is not installed and SAM is not being trained, these
# can be safely ignored.
try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print('[WARN] segment_anything not found — SAM training will not be available.')

# ─────────────────────────────────────────────────────────────────────────────
# Dual Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GlobalLocalDataset(Dataset):
    """
    Pairs every local patch with its corresponding resized global image.

    Directory layout expected:
        <local_root>/
            train/images/  DJI_xxx_patch_0000.jpg  ...
            train/masks/   DJI_xxx_patch_0000.png  ...
        <global_root>/
            train/images/  DJI_xxx.jpg  ...
            train/masks/   DJI_xxx.png  ...    (not used — local mask is the target)

    Pairing rule:
        Strip '_patch_NNNN' from the local filename stem to recover the source stem,
        then look up the matching file in the global directory.
        e.g. 'DJI_20230615120000_0042_V_patch_0007' → 'DJI_20230615120000_0042_V'

    Args:
        local_root  : Root of the sliced-patch dataset (e.g. 'dataset/processed_512_patch')
        global_root : Root of the resized dataset      (e.g. 'dataset/processed_512_resized')
        split       : One of 'train', 'val', 'test'
        image_size  : Resize both inputs to this square size (default 512)
        augment     : Apply random flips/rotations (training only)
    """

    # ImageNet normalisation shared by both branches
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        local_root: str,
        global_root: str,
        split: str = 'train',
        image_size: int = 512,
        augment: bool = False,
    ):
        self.local_img_dir  = Path(local_root)  / split / 'images'
        self.local_mask_dir = Path(local_root)  / split / 'masks'
        self.global_img_dir = Path(global_root) / split / 'images'
        self.image_size = image_size
        self.augment    = augment

        # Build index: [(local_img_path, local_mask_path, global_img_path), ...]
        self.samples: list[tuple[Path, Path, Path]] = []
        skipped = 0

        for local_img in sorted(self.local_img_dir.glob('*.jpg')):
            local_mask = self.local_mask_dir / f'{local_img.stem}.png'
            if not local_mask.exists():
                skipped += 1
                continue

            # Recover source stem by stripping '_patch_NNNN'
            source_stem = re.sub(r'_patch_\d+$', '', local_img.stem)
            global_img  = self.global_img_dir / f'{source_stem}.jpg'

            if not global_img.exists():
                skipped += 1
                continue

            self.samples.append((local_img, local_mask, global_img))

        if skipped:
            print(f'  [GlobalLocalDataset/{split}] Skipped {skipped} unmatched samples.')
        print(f'  [GlobalLocalDataset/{split}] {len(self.samples)} paired samples loaded.')

        # Transforms
        self.img_tf  = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])
        # Mask transform: always produces [1, H, W] float tensor in [0, 1]
        # - Resize with NEAREST to preserve hard label boundaries
        # - ToTensor on an 'L'-mode PIL image gives shape [1, H, W], values [0.0, 1.0]
        # - Lambda clamp handles edge cases where mask pixels are 0/255 or not perfectly binary
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),                          # [1, H, W], values in [0, 1]
            transforms.Lambda(lambda m: (m > 0.5).float()) # binarise: 0.0 or 1.0 only
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        local_img_path, local_mask_path, global_img_path = self.samples[idx]

        local_img  = Image.open(local_img_path).convert('RGB')
        global_img = Image.open(global_img_path).convert('RGB')
        mask       = Image.open(local_mask_path).convert('L')  # 'L' = grayscale → [1,H,W] after ToTensor

        # Consistent augmentation: apply the SAME geometric transform to all three
        if self.augment:
            local_img, global_img, mask = self._augment(local_img, global_img, mask)

        mask_tensor = self.mask_tf(mask)

        # Defensive shape check — catches regressions before they reach the loss
        assert mask_tensor.shape[0] == 1, (
            f"Mask must be single-channel [1,H,W], got {mask_tensor.shape}. "
            f"Source: {local_mask_path}"
        )
        assert mask_tensor.shape[1:] == torch.Size([self.image_size, self.image_size]), (
            f"Mask spatial size mismatch: expected [{self.image_size},{self.image_size}], "
            f"got {list(mask_tensor.shape[1:])}"
        )

        return {
            'local_image':  self.img_tf(local_img),
            'global_image': self.img_tf(global_img),
            'mask':         mask_tensor,
            # Keep paths for debugging
            'local_path':   str(local_img_path),
            'global_path':  str(global_img_path),
        }

    # ── augmentation ──────────────────────────────────────────────────────────

    def _augment(self, local_img, global_img, mask):
        """Apply identical random flips / 90° rotations to all three inputs."""
        import random

        # Random horizontal flip
        if random.random() > 0.5:
            local_img  = local_img.transpose(Image.FLIP_LEFT_RIGHT)
            global_img = global_img.transpose(Image.FLIP_LEFT_RIGHT)
            mask       = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() > 0.5:
            local_img  = local_img.transpose(Image.FLIP_TOP_BOTTOM)
            global_img = global_img.transpose(Image.FLIP_TOP_BOTTOM)
            mask       = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            angle = k * 90
            local_img  = local_img.rotate(angle)
            global_img = global_img.rotate(angle)
            mask       = mask.rotate(angle)

        return local_img, global_img, mask



# ─────────────────────────────────────────────────────────────────────────────
# SAM Dataset
# ─────────────────────────────────────────────────────────────────────────────

# SAM uses its own normalisation (pixel values 0-255, not 0-1).
# These match the values hardcoded inside sam_model.preprocess().
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

    Unlike the original train_sam.py (which pre-loads all images into a RAM
    dict), this dataset loads and preprocesses images on demand so it scales
    to any dataset size.

    Pipeline per sample
    -------------------
    1. Load image (JPEG) with PIL, convert to RGB.
    2. Resize to `image_size × image_size` (matching every other model's input
       so that masks are on a consistent grid).
    3. Optionally apply random flips / 90° rotations for training augmentation.
    4. Apply SAM's ResizeLongestSide(1024) — for a 512×512 square input this
       produces a 1024×1024 tensor (both sides equal the longest side).
    5. Convert to float tensor [3, H', W'] (pixel values 0-255).
    6. Apply SAM normalisation + zero-padding to 1024×1024.
    7. Load the corresponding mask (PNG), resize to `image_size`, binarise.

    Returned batch dict keys
    ------------------------
    input_tensor   : FloatTensor [3, 1024, 1024]  — SAM-preprocessed image
    input_size     : LongTensor  [2]              — (H', W') after ResizeLongestSide
    original_size  : LongTensor  [2]              — (image_size, image_size)
    mask           : FloatTensor [1, image_size, image_size]  — binary 0/1
    image_name     : str
    """

    SAM_IMG_SIZE = 1024  # SAM ViT input resolution (fixed for all variants)

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

        # Accept .jpg and .png images; sort for reproducibility
        img_paths  = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        self.samples: list[tuple[Path, Path]] = []
        skipped = 0

        for img_path in img_paths:
            # Match mask by stem regardless of extension
            mask_path = mask_dir / f'{img_path.stem}.png'
            if not mask_path.exists():
                # Fallback: try .jpg mask
                mask_path = mask_dir / f'{img_path.stem}.jpg'
            if not mask_path.exists():
                skipped += 1
                continue
            self.samples.append((img_path, mask_path))

        if skipped:
            print(f'  [SAMDataset/{split}] Skipped {skipped} images (no matching mask).')
        print(f'  [SAMDataset/{split}] {len(self.samples)} samples loaded.')

        if SAM_AVAILABLE:
            self.resize_transform = ResizeLongestSide(self.SAM_IMG_SIZE)
        else:
            raise RuntimeError(
                'segment_anything is not installed. '
                'Run: pip install git+https://github.com/facebookresearch/segment-anything.git'
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────────
        image = self._cv2.imread(str(img_path))
        image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        image = self._cv2.resize(
            image, (self.image_size, self.image_size),
            interpolation=self._cv2.INTER_LINEAR,
        )

        # ── Load mask ─────────────────────────────────────────────────────
        mask = self._cv2.imread(str(mask_path), self._cv2.IMREAD_GRAYSCALE)
        mask = self._cv2.resize(
            mask, (self.image_size, self.image_size),
            interpolation=self._cv2.INTER_NEAREST,
        )
        mask = (mask > 127).astype(np.float32)  # binary 0.0 / 1.0

        # ── Augmentation (same transform applied to image AND mask) ───────
        if self.augment:
            image, mask = self._augment(image, mask)

        original_size = (image.shape[0], image.shape[1])  # (512, 512)

        # ── SAM preprocessing ─────────────────────────────────────────────
        # ResizeLongestSide preserves aspect ratio; for square 512×512 input
        # this gives exactly 1024×1024.
        resized_image = self.resize_transform.apply_image(image)         # np uint8 [H', W', 3]
        input_size    = (resized_image.shape[0], resized_image.shape[1]) # (H', W')

        # Convert to float tensor [3, H', W'], then normalise + pad to 1024×1024
        img_tensor    = torch.as_tensor(resized_image).permute(2, 0, 1).float()
        input_tensor  = _sam_preprocess(img_tensor, self.SAM_IMG_SIZE)   # [3, 1024, 1024]

        # ── Mask tensor ───────────────────────────────────────────────────
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, 512, 512]

        return {
            'input_tensor' : input_tensor,                              # [3, 1024, 1024]
            'input_size'   : torch.tensor(input_size,    dtype=torch.long),  # [2]
            'original_size': torch.tensor(original_size, dtype=torch.long),  # [2]
            'mask'         : mask_tensor,                               # [1, 512, 512]
            'image_name'   : img_path.stem,
        }

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        """Apply identical random flips / 90° rotations to image and mask."""
        import random
        # Horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask  = np.fliplr(mask).copy()
        # Vertical flip
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask  = np.flipud(mask).copy()
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask  = np.rot90(mask,  k).copy()
        return image, mask


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
        self.is_sam  = (model_name == 'sam')

        # ── Model ─────────────────────────────────────────────────────────
        print(f'\nInitializing {model_name}' + (f' ({variant})' if variant else '') + '…')

        if self.is_sam:
            # SAM uses its own registry rather than the shared get_model() factory.
            # The checkpoint path is taken from config['foundation']['sam_checkpoints'].
            if not SAM_AVAILABLE:
                raise RuntimeError(
                    'segment_anything is required for SAM training. '
                    'Install with: pip install git+https://github.com/facebookresearch/segment-anything.git'
                )
            sam_ckpt = (
                config.get('foundation', {})
                      .get('sam_checkpoints', {})
                      .get(variant, f'./checkpoints/sam/sam_{variant}.pth')
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
            # SAM requires its own preprocessing pipeline (ResizeLongestSide +
            # SAM normalisation) that is incompatible with the standard
            # get_training_dataloaders() transforms.  SAMDataset handles this
            # on-the-fly so no images are pre-loaded into RAM.
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
            print(f'Loaded {len(train_ds)} samples from train set')
            print(f'  Image size: {img_size}x{img_size}  (SAM input: 1024×1024)')
            print(f'  Data augmentation: {"ENABLED" if augment else "DISABLED"}')
            print(f'Loaded {len(val_ds)} samples from val set')
            print(f'  Image size: {img_size}x{img_size}')
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

        if self.is_sam:
            # SAM fine-tuning strategy:
            #   image_encoder  — frozen via no_grad in _forward_sam (NOT in optimizer)
            #   prompt_encoder — frozen via no_grad in _forward_sam (NOT in optimizer)
            #   mask_decoder   — fully trainable at base_lr
            #
            # Only the mask_decoder is given to the optimizer.  This keeps the
            # effective LR identical to every other model in the benchmark
            # (base_lr = 1e-4) while preventing gradient updates to the large
            # pretrained encoder (91M–632M params).
            param_groups = [
                {'params': self.model.mask_decoder.parameters(),
                 'lr': base_lr, 'name': 'mask_decoder'},
            ]
            print(f'  [Optimizer] SAM differential LR:')
            print(f'    image_encoder   FROZEN  (no_grad in forward)  '
                  f'params={sum(p.numel() for p in self.model.image_encoder.parameters()):,}')
            print(f'    prompt_encoder  FROZEN  (no_grad in forward)  '
                  f'params={sum(p.numel() for p in self.model.prompt_encoder.parameters()):,}')
            print(f'    mask_decoder    LR={base_lr:.1e}  '
                  f'params={sum(p.numel() for p in self.model.mask_decoder.parameters()):,}')
        elif hasattr(self.model, 'get_params_groups'):
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

        output_dir           = self.config['system']['output_dir']
        self.model_dir       = os.path.join(output_dir, model_name)
        self.exp_dir         = os.path.join(self.model_dir, exp_name)
        self.checkpoint_dir  = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir         = os.path.join(self.exp_dir, 'logs')

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
        if self.is_sam:
            print(f'Mode       : SAM fine-tuning (mask_decoder only)')
        else:
            print(f'Mode       : Standard single-input')
        print(f'Model      : {self.config["model"]["name"]}', end='')
        if self.config['model'].get('variant'):
            print(f' ({self.config["model"]["variant"]})')
        else:
            print()
        total_params    = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Parameters : {total_params:,}  (trainable: {trainable_params:,})')
        if self.is_sam:
            decoder_params = sum(p.numel() for p in self.model.mask_decoder.parameters())
            print(f'  mask_decoder trainable: {decoder_params:,}')
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
        Unified loss computation for both standard and GlobalLocal modes.

        For GlobalLocal mode (aux_out is a tuple of (g_logits, l_logits)):
            total = main_loss + aux_weight * (global_aux_loss + local_aux_loss) / 2

        This applies deep supervision: the global and local branches are each
        independently penalised against the ground-truth mask, encouraging each
        branch to produce a meaningful segmentation on its own before fusion.
        """
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
            total    = main_loss + self.aux_weight * aux_loss

            loss_dict['main_loss']   = main_loss.item()
            loss_dict['global_aux']  = g_loss.item()
            loss_dict['local_aux']   = l_loss.item()
            loss_dict['total']       = total.item()
            return total, loss_dict

        return main_loss, loss_dict

    def _forward(self, batch):
        """
        Run the model forward pass and return (main_out, aux_out, masks).
        Handles standard, GlobalLocal, and SAM batch dicts transparently.
        """
        if self.is_sam:
            return self._forward_sam(batch)
        images   = batch['image'].to(self.device)
        masks    = batch['mask'].to(self.device)
        outputs  = self.model(images)
        main_out, aux_out = (outputs if isinstance(outputs, tuple) else (outputs, None))

        return main_out, aux_out, masks

    def _forward_sam(self, batch):
        """
        SAM-specific forward pass.

        Architecture:
          image_encoder   (frozen, no_grad) → image embeddings [B, 256, 64, 64]
          prompt_encoder  (frozen, no_grad) → sparse [B, 0, 256], dense [B, 256, 64, 64]
          mask_decoder    (trainable)        → low_res_masks [B, 1, 256, 256]
          postprocess_masks                  → logits [B, 1, H_orig, W_orig]

        The image_encoder and prompt_encoder are kept in no_grad to prevent
        gradient computation through the large pretrained ViT backbone
        (91M params for vit_b, 308M for vit_l, 632M for vit_h).
        Only mask_decoder parameters receive gradient updates.

        Returns (logits, None, masks) matching the (main_out, aux_out, masks)
        contract of _forward(), so train_epoch / validate are unchanged.
        """
        input_tensor  = batch['input_tensor'].to(self.device)   # [B, 3, 1024, 1024]
        masks         = batch['mask'].to(self.device)            # [B, 1, img_size, img_size]
        B             = input_tensor.shape[0]

        # All images in a batch are the same spatial size (512×512 → 1024×1024),
        # so we read sizes from the first sample in the batch.
        input_size    = tuple(int(x) for x in batch['input_size'][0])    # (H', W')
        original_size = tuple(int(x) for x in batch['original_size'][0]) # (H_orig, W_orig)

        # ── Frozen components (no gradient computation) ───────────────────
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(input_tensor)
            # [B, 256, 64, 64]

            # prompt_encoder with no prompts returns:
            #   sparse_embeddings: [1, 0, 256]  (empty — no point/box tokens)
            #   dense_embeddings : [1, 256, 64, 64]  (no-mask embedding)
            sparse_emb, dense_emb = self.model.prompt_encoder(
                points=None, boxes=None, masks=None,
            )
            # # Expand from single-sample [1, ...] to full batch [B, ...]
            # sparse_emb = sparse_emb.expand(B, -1, -1).contiguous()
            # dense_emb  = dense_emb.expand(B, -1, -1, -1).contiguous()

        # ── Trainable mask decoder ────────────────────────────────────────
        low_res_masks, _ = self.model.mask_decoder(
            image_embeddings         = image_embeddings,
            image_pe                 = self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_emb,
            dense_prompt_embeddings  = dense_emb,
            multimask_output         = False,   # single best mask per image
        )
        # low_res_masks: [B, 1, 256, 256] — raw logits at decoder resolution

        # ── Upscale to original image resolution ─────────────────────────
        # postprocess_masks: interpolates to 1024×1024, crops to input_size,
        # then interpolates to original_size.  Returns LOGITS (not sigmoid).
        upscaled_masks = self.model.postprocess_masks(
            low_res_masks, input_size, original_size,
        )
        # upscaled_masks: [B, 1, H_orig, W_orig] — logits ready for loss fn

        return upscaled_masks, None, masks

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
            'data_root'    : './dataset/processed_512_resized',
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
            'output_dir'   : './experiments',
            'log_interval' : 10,
            'save_interval': 10,
        },
        # ── Foundation model settings ──────────────────────────────────────────
        # Only read when config['model']['name'] is 'sam' or 'dinov2'.
        # get_model() must forward these to build_sam_segmentation /
        # build_dinov2_segmentation via the config dict.
        'foundation': {
            'pretrained'    : True,   # always True for benchmark validity
            'freeze_encoder': False,  # full fine-tuning (set True for linear probe)
            # SAM checkpoint paths — download from Meta before training:
            #   vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
            #   vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
            #   vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
            'sam_checkpoints': {
                'vit_b': './checkpoints/sam/sam_vit_b_01ec64.pth',
                'vit_l': './checkpoints/sam/sam_vit_l_0b3195.pth',
                'vit_h': './checkpoints/sam/sam_vit_h_4b8939.pth',
            },
            # DINOv2 is auto-downloaded via torch.hub — no checkpoint paths needed.
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
        # Uncomment dinov2 when ready to train:
        'dinov2': ['vit_s', 'vit_b', 'vit_l'],
        # 'sam'    : ['vit_b', 'vit_l', 'vit_h'],
    }

    # Foundation models use early stopping to prevent overfitting on the
    # 274-image training set given their large parameter counts (91M–632M).
    # All other benchmark models trained with fixed 100 epochs — this
    # asymmetry is documented in the paper's training details table.
    FOUNDATION_MODELS = {'sam', 'dinov2'}

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
