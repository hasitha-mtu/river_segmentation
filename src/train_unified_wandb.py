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


def get_global_local_dataloaders(config: dict):
    """
    Build train and val DataLoaders for GlobalLocal dual-branch training.

    Reads from config['data']['local_root'] and config['data']['global_root'].
    """
    data_cfg   = config['data']
    system_cfg = config['system']

    train_ds = GlobalLocalDataset(
        local_root  = data_cfg['local_root'],
        global_root = data_cfg['global_root'],
        split       = 'train',
        image_size  = data_cfg.get('image_size', 512),
        augment     = data_cfg.get('augment_train', True),
    )
    val_ds = GlobalLocalDataset(
        local_root  = data_cfg['local_root'],
        global_root = data_cfg['global_root'],
        split       = 'val',
        image_size  = data_cfg.get('image_size', 512),
        augment     = False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = config['training']['batch_size'],
        shuffle     = True,
        num_workers = system_cfg.get('num_workers', 0),
        pin_memory  = True,
        drop_last   = True,     # keeps batch stats stable for BatchNorm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config['training']['batch_size'],
        shuffle     = False,
        num_workers = system_cfg.get('num_workers', 0),
        pin_memory  = True,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Unified Trainer
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedTrainer:
    """
    Unified trainer supporting:
      • Standard single-input models (CNN, Transformer, Foundation)
      • GlobalLocal dual-branch model (global_local mode)

    Set config['model']['name'] = 'global_local' to activate dual-branch mode.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Detect operating mode early — used throughout __init__
        self.is_global_local = (config['model']['name'] == 'global_local')

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
        if self.is_global_local:
            gl_cfg = config['model'].get('global_local', {})

            # Global branch config — falls back to top-level model keys so
            # existing single-model configs also work as the global branch.
            global_name    = gl_cfg.get('global_model_name',
                                        config['model'].get('name', 'unet'))
            global_variant = gl_cfg.get('global_variant',
                                        config['model'].get('variant', None))

            # Local branch config — defaults to same as global (symmetric)
            local_name    = gl_cfg.get('local_model_name',  None)
            local_variant = gl_cfg.get('local_variant',     None)

            print(f'\nInitializing GlobalLocalWrapper (dual-branch)…')
            print(f'  Global : {global_name}' + (f' ({global_variant})' if global_variant else ''))
            print(f'  Local  : {local_name or global_name}' +
                  (f' ({local_variant or global_variant})' if (local_variant or global_variant) else ''))

            self.model = GlobalLocalWrapper(
                num_classes       = config['model'].get('n_classes', 1),
                n_channels        = config['model'].get('n_channels', 3),
                global_model_name = global_name,
                global_variant    = global_variant,
                local_model_name  = local_name,
                local_variant     = local_variant,
            ).to(self.device)
            print(f'  → {self.model.description()}')
        else:
            model_name = config['model']['name']
            variant    = config['model'].get('variant', None)
            print(f'\nInitializing {model_name}' + (f' ({variant})' if variant else '') + '…')
            self.model = get_model(
                model_name = model_name,
                variant    = variant,
                n_channels = config['model'].get('n_channels', 3),
                n_classes  = config['model'].get('n_classes', 1),
            ).to(self.device)

        # ── Data loaders ───────────────────────────────────────────────────
        if self.is_global_local:
            print(f'Loading dual-branch data…')
            print(f'  Local  (patches) : {config["data"]["local_root"]}')
            print(f'  Global (resized) : {config["data"]["global_root"]}')
            self.train_loader, self.val_loader = get_global_local_dataloaders(config)
        else:
            print(f'Loading data from {config["data"]["data_root"]}…')
            self.train_loader, self.val_loader = get_training_dataloaders(
                data_dir      = config['data']['data_root'],
                batch_size    = config['training']['batch_size'],
                num_workers   = config['system']['num_workers'],
                augment_train = config['data'].get('augment_train', True),
                image_size    = config['data']['image_size'],
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

        if self.is_global_local:
            # Use separate LR groups: backbone at base_lr, fusion at 5× base_lr
            param_groups = self.model.parameter_groups(
                base_lr              = base_lr,
                fusion_lr_multiplier = config['model'].get('global_local', {}).get(
                    'fusion_lr_multiplier', 5.0
                ),
            )
        else:
            param_groups = self.model.parameters()

        if opt_cfg['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                param_groups, lr=base_lr,
                weight_decay=opt_cfg['weight_decay'], betas=(0.9, 0.999),
            )
        elif opt_cfg['type'] == 'adam':
            self.optimizer = torch.optim.Adam(
                param_groups, lr=base_lr, weight_decay=opt_cfg['weight_decay'],
            )
        elif opt_cfg['type'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                param_groups, lr=base_lr,
                momentum=opt_cfg.get('momentum', 0.9),
                weight_decay=opt_cfg['weight_decay'],
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
        if self.is_global_local:
            gl_cfg = self.config['model'].get('global_local', {})
            tags.append('dual_branch')
            tags.append('global_local')
            tags.append(gl_cfg.get('global_model_name', 'unet'))
            local = gl_cfg.get('local_model_name', None)
            if local and local != gl_cfg.get('global_model_name'):
                tags.append(f'local_{local}')  # only add tag when asymmetric

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
        print(f'Mode       : {"GlobalLocal dual-branch" if self.is_global_local else "Standard single-input"}')
        print(f'Model      : {self.config["model"]["name"]}', end='')
        if self.config['model'].get('variant'):
            print(f' ({self.config["model"]["variant"]})')
        else:
            print()
        print(f'Parameters : {sum(p.numel() for p in self.model.parameters()):,}')
        print(f'Batch size : {self.config["training"]["batch_size"]}')
        print(f'LR         : {self.config["training"]["optimizer"]["learning_rate"]}')
        print(f'Optimizer  : {self.config["training"]["optimizer"]["type"]}')
        print(f'Scheduler  : {self.config["training"]["scheduler"]["type"]}')
        print(f'Loss       : {self.config["loss"]["type"]}')
        if self.is_global_local:
            print(f'Aux weight : {self.aux_weight}')
        print(f'Epochs     : {self.config["training"]["epochs"]}')
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

        if aux_out is not None and self.is_global_local:
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
        Handles both standard and GlobalLocal batch dicts transparently.
        """
        if self.is_global_local:
            global_img  = batch['global_image'].to(self.device)
            local_patch = batch['local_image'].to(self.device)
            masks       = batch['mask'].to(self.device)
            main_out, aux_out = self.model(global_img, local_patch, return_aux=True)
        else:
            images   = batch['image'].to(self.device)
            masks    = batch['mask'].to(self.device)
            outputs  = self.model(images)
            main_out, aux_out = (
                outputs if isinstance(outputs, tuple) else (outputs, None)
            )

        return main_out, aux_out, masks

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

        sample_images, sample_globals, sample_masks, sample_preds = [], [], [], []
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
                    if self.is_global_local:
                        sample_images.append(batch['local_image'][0].cpu())
                        sample_globals.append(batch['global_image'][0].cpu())
                    else:
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
                sample_images, sample_masks, sample_preds, epoch,
                global_images=sample_globals if self.is_global_local else None,
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
            'mode'                : 'global_local' if self.is_global_local else 'standard',
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
            print('-' * 80)

        print('\n' + '=' * 80)
        print('TRAINING COMPLETE')
        print(f'Best Dice : {self.best_val_dice:.4f}')
        print(f'Best IoU  : {self.best_val_iou:.4f}')
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
            # Standard mode
            'data_root'    : './dataset/processed_512_patch',
            'image_size'   : 512,
            'augment_train': True,
            # GlobalLocal mode (set when name='global_local')
            'local_root'   : './dataset/processed_512_patch',
            'global_root'  : './dataset/processed_512_resized',
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
            'use_boundary'  : True,
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
    }


def get_global_local_config(
    global_model_name: str = 'unet',
    global_variant:    str = None,
    local_model_name:  str = None,   # None → same as global (symmetric)
    local_variant:     str = None,
):
    """
    Return a config pre-filled for GlobalLocal dual-branch training.

    Args:
        global_model_name : Architecture for the global (resized) branch.
        global_variant    : Variant for the global branch (e.g. 'b2', 'tiny').
        local_model_name  : Architecture for the local (patch) branch.
                            Defaults to global_model_name (symmetric pairing).
        local_variant     : Variant for the local branch.

    Examples:
        # Symmetric — same model on both branches
        cfg = get_global_local_config('segformer', 'b2')

        # Asymmetric — heavyweight global, lightweight local
        cfg = get_global_local_config(
            global_model_name='convnext_upernet', global_variant='base',
            local_model_name='unet',
        )
    """
    config = get_default_config()
    config['model']['name']    = 'global_local'
    config['model']['variant'] = global_variant   # shown in W&B run name

    config['model']['global_local'] = {
        'global_model_name'   : global_model_name,
        'global_variant'      : global_variant,
        'local_model_name'    : local_model_name,   # None → symmetric
        'local_variant'       : local_variant,
        'fusion_lr_multiplier': 5.0,
    }

    config['data']['local_root']  = './dataset/processed_512_patch'
    config['data']['global_root'] = './dataset/processed_512_resized'
    config['data']['image_size']  = 512
    config['loss']['aux_weight']  = 0.4
    config['training']['batch_size'] = 4   # 2 × 512² per step — monitor VRAM
    return config


def train_all_global_local_models(base_config: dict, symmetric_only: bool = True):
    """
    Train GlobalLocalWrapper with every supported model architecture as branches.

    Args:
        base_config     : Base config dict (data paths, training hyper-params, etc.)
        symmetric_only  : If True (default), both branches use the same architecture.
                          If False, also trains a selection of asymmetric pairings.

    Symmetric run matrix (one experiment per model/variant):
        unet, unetpp, resunetpp, deeplabv3plus, deeplabv3plus_cbam,
        segformer/b0, segformer/b2, swin_unet/tiny,
        convnext_upernet/tiny … base, hrnet_ocr/w18 … w48,
        sam/vit_b … vit_h, dinov2/vit_s … vit_g

    Asymmetric pairings (added when symmetric_only=False):
        convnext_upernet/base (global) + unet (local)
        segformer/b2          (global) + unet (local)
        dinov2/vit_b          (global) + unet (local)
    """
    # All models with their variants — same registry as train_all_models()
    all_models = {
        'unet'              : [None],
        'unetpp'            : [None],
        'resunetpp'         : [None],
        'deeplabv3plus'     : [None],
        'deeplabv3plus_cbam': [None],
        'segformer'         : ['b0', 'b2'],
        'swin_unet'         : ['tiny'],
        'convnext_upernet'  : ['tiny', 'small', 'base'],
        'hrnet_ocr'         : ['w18', 'w32', 'w48'],
        'sam'               : ['vit_b', 'vit_l', 'vit_h'],
        'dinov2'            : ['vit_s', 'vit_b', 'vit_l', 'vit_g'],
    }

    experiments = []

    # ── Symmetric pairs ───────────────────────────────────────────────────────
    for model_name, variants in all_models.items():
        for variant in variants:
            experiments.append({
                'global_model_name': model_name,
                'global_variant':    variant,
                'local_model_name':  None,      # symmetric → mirrors global
                'local_variant':     None,
                'label':             f'gl_sym_{model_name}' +
                                     (f'_{variant}' if variant else ''),
            })

    # ── Asymmetric pairs (heavyweight global + lightweight local) ─────────────
    if not symmetric_only:
        asymmetric_pairs = [
            ('convnext_upernet', 'base', 'unet',     None),
            ('segformer',        'b2',   'unet',     None),
            ('dinov2',           'vit_b','unet',     None),
            ('hrnet_ocr',        'w48',  'segformer','b0'),
        ]
        for g_name, g_var, l_name, l_var in asymmetric_pairs:
            experiments.append({
                'global_model_name': g_name,
                'global_variant':    g_var,
                'local_model_name':  l_name,
                'local_variant':     l_var,
                'label':             f'gl_asym_{g_name}_{g_var}__local_{l_name}',
            })

    # ── Run experiments ───────────────────────────────────────────────────────
    for exp in experiments:
        print(f'\n{"="*80}')
        print(f'GlobalLocal experiment: {exp["label"]}')
        print(f'{"="*80}\n')

        cfg = get_global_local_config(
            global_model_name = exp['global_model_name'],
            global_variant    = exp['global_variant'],
            local_model_name  = exp['local_model_name'],
            local_variant     = exp['local_variant'],
        )

        # Inherit training hyper-params from base_config
        for key in ['training', 'loss', 'logging', 'system']:
            if key in base_config:
                cfg[key] = {**cfg[key], **base_config[key]}

        try:
            train_single_model(cfg)
        except Exception as e:
            print(f'  [ERROR] {exp["label"]} failed: {e}')
            print('  Skipping to next experiment…')
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc; gc.collect()


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
    """Train all standard models and their variants sequentially."""
    all_models = {
        # CNN baselines
        'unet'           : [],
        'unetpp'         : [],
        'resunetpp'      : [],
        'deeplabv3plus'  : [],
        'deeplabv3plus_cbam': [],
        # Transformers
        'segformer'      : ['b0', 'b2'],
        'swin_unet'      : ['tiny'],
        # Hybrid SOTA
        'convnext_upernet': ['tiny', 'small', 'base'],
        'hrnet_ocr'      : ['w18', 'w32', 'w48'],
        # Foundation models
        'sam'            : ['vit_b', 'vit_l', 'vit_h'],
        'dinov2'         : ['vit_s', 'vit_b', 'vit_l', 'vit_g'],
    }

    for model_name, variants in all_models.items():
        for variant in (variants or [None]):
            tag = f'{model_name}' + (f' - {variant}' if variant else '')
            print(f'\n{"="*80}\nTraining {tag}\n{"="*80}\n')
            config = {**base_config}
            config['model'] = {**base_config['model'],
                                'name': model_name, 'variant': variant}
            train_single_model(config)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Option A: Single standard model ──────────────────────────────────────
    # config = get_default_config()
    # config['model']['name']    = 'unet'
    # config['model']['variant'] = None
    # config['data']['data_root'] = './dataset/processed_512_patch'
    # config['data']['image_size'] = 512
    # config['training']['batch_size'] = 4
    # config['training']['epochs']     = 100
    # config['logging']['use_wandb']   = True
    # train_single_model(config)

    # ── Option B: GlobalLocal — symmetric (same arch on both branches) ────────
    gl_config = get_global_local_config(
        global_model_name = 'unet',   # ← change to any supported model
        global_variant    = None,
        # local_model_name / local_variant omitted → symmetric (mirrors global)
    )
    gl_config['training']['epochs']     = 100
    gl_config['training']['batch_size'] = 4
    gl_config['logging']['use_wandb']   = True
    gl_config['logging']['wandb_notes'] = (
        'GlobalLocal dual-branch: resized global context + sliced local detail. '
        'Attention-gated fusion. Deep supervision on both branches.'
    )
    # train_single_model(gl_config)

    # ── Option C: GlobalLocal — asymmetric (different arch per branch) ────────
    # gl_asym = get_global_local_config(
    #     global_model_name = 'convnext_upernet', global_variant = 'base',
    #     local_model_name  = 'unet',             local_variant  = None,
    # )
    # train_single_model(gl_asym)

    # ── Option D: All standard models ─────────────────────────────────────────
    train_all_models(get_default_config())

    # ── Option E: All GlobalLocal symmetric experiments ───────────────────────
    # train_all_global_local_models(get_default_config(), symmetric_only=True)

    # ── Option F: All GlobalLocal including asymmetric pairings ───────────────
    # train_all_global_local_models(get_default_config(), symmetric_only=False)


if __name__ == '__main__':
    main()
