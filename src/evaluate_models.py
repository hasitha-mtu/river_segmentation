"""
evaluate_models.py
==================
Comprehensive test-set evaluation and comparison of all trained segmentation
models for flash-flood forecasting (UAV river segmentation).

What this script does
---------------------
1. Auto-discovers every ``best.pth`` checkpoint under ``experiments/``.
2. Reconstructs each model using the config stored inside the checkpoint.
3. Runs full inference on the held-out **test** split.
4. Computes a rich metric suite per model:
      Dice · IoU · Precision · Recall · F1 · Specificity
      Boundary-F1  (river-edge accuracy — critical for water-level estimation)
      Inference time (ms / image)  ·  Parameter count
5. Saves:
      results/test_results.csv           — full results table
      results/test_results_summary.txt   — ranked leaderboard (console-friendly)
      results/predictions/<model>/       — overlay PNGs for visual inspection
6. Optionally logs a W&B comparison table.

Usage
-----
# Evaluate all discovered models (default paths):
    python evaluate_models.py

# Override paths or toggle options via CLI:
    python evaluate_models.py \\
        --experiments_dir ./experiments \\
        --data_root       ./dataset/processed_512_resized \\
        --output_dir      ./results \\
        --batch_size      4 \\
        --save_predictions         \\
        --num_pred_samples 8       \\
        --use_wandb                \\
        --wandb_project   river-segmentation

Notes
-----
* DINOv2 vit_g is automatically skipped if its checkpoint is absent.
* The script gracefully handles CUDA OOM by falling back to CPU for that model.
* All metrics are macro-averaged over the full test set (accumulated TP/FP/FN/TN).
* Boundary-F1 uses a 3-pixel dilation kernel — matches typical UAV GSD at 50 m AGL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ── Optional deps (graceful degradation) ────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── Project imports ──────────────────────────────────────────────────────────
# Adjust sys.path so the script can be run from any working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from models import get_model
from wrapper import GlobalLocalWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    """
    Simple single-input test dataset.
    Mirrors the transform pipeline used during training (ImageNet normalisation).
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, data_root: str, split: str = 'test', image_size: int = 512):
        root      = Path(data_root)
        img_dir   = root / split / 'images'
        mask_dir  = root / split / 'masks'

        if not img_dir.exists():
            raise FileNotFoundError(
                f'Test image directory not found: {img_dir}\n'
                f'Expected layout: {data_root}/{split}/images/'
            )

        self.samples: list[tuple[Path, Optional[Path]]] = []
        for img_path in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')):
            mask_path = mask_dir / f'{img_path.stem}.png'
            if not mask_path.exists():
                mask_path = mask_dir / f'{img_path.stem}.jpg'
            self.samples.append((img_path, mask_path if mask_path.exists() else None))

        if not self.samples:
            raise ValueError(f'No images found in {img_dir}')

        print(f'  [TestDataset] {len(self.samples)} samples in {split} split.')

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda m: (m > 0.5).float()),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask_tensor = self.mask_tf(mask)
        else:
            mask_tensor = torch.zeros(1, 512, 512)  # dummy if no mask

        return {
            'image'     : self.img_tf(image),
            'mask'      : mask_tensor,
            'image_path': str(img_path),
        }


class GlobalLocalTestDataset(Dataset):
    """
    Paired (global, local) test dataset for GlobalLocal models.
    Mirrors GlobalLocalDataset from train_unified_wandb.py.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        local_root: str,
        global_root: str,
        split: str = 'test',
        image_size: int = 512,
    ):
        import re
        local_img_dir  = Path(local_root)  / split / 'images'
        local_mask_dir = Path(local_root)  / split / 'masks'
        global_img_dir = Path(global_root) / split / 'images'

        self.samples: list[tuple[Path, Path, Path]] = []
        skipped = 0

        for local_img in sorted(local_img_dir.glob('*.jpg')):
            local_mask = local_mask_dir / f'{local_img.stem}.png'
            if not local_mask.exists():
                skipped += 1
                continue
            source_stem = re.sub(r'_patch_\d+$', '', local_img.stem)
            global_img  = global_img_dir / f'{source_stem}.jpg'
            if not global_img.exists():
                skipped += 1
                continue
            self.samples.append((local_img, local_mask, global_img))

        if skipped:
            print(f'  [GlobalLocalTestDataset/{split}] Skipped {skipped} unmatched samples.')
        print(f'  [GlobalLocalTestDataset/{split}] {len(self.samples)} paired samples.')

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda m: (m > 0.5).float()),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        local_img_path, local_mask_path, global_img_path = self.samples[idx]

        local_img  = self.img_tf(Image.open(local_img_path).convert('RGB'))
        global_img = self.img_tf(Image.open(global_img_path).convert('RGB'))
        mask       = self.mask_tf(Image.open(local_mask_path).convert('L'))

        return {
            'local_image' : local_img,
            'global_image': global_img,
            'mask'        : mask,
            'image_path'  : str(local_img_path),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class MetricAccumulator:
    """
    Accumulates per-batch TP/FP/FN/TN then computes final metrics.
    All metrics are macro-averaged at the pixel level over the whole test set.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.boundary_tp = self.boundary_fp = self.boundary_fn = 0
        self.total_images = 0

    def update(self, preds_prob: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds_prob : sigmoid probabilities  [B, 1, H, W]
            targets    : binary ground truth    [B, 1, H, W]
        """
        preds = (preds_prob > self.threshold).float()
        targets = targets.float()

        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()
        self.tn += ((1 - preds) * (1 - targets)).sum().item()
        self.total_images += preds.shape[0]

        # Boundary F1 — erode prediction and target, compare boundaries
        self._update_boundary(preds, targets)

    def _update_boundary(self, preds: torch.Tensor, targets: torch.Tensor, dilation: int = 3):
        """
        Compute boundary TP/FP/FN using morphological operations.
        Falls back to a pure-PyTorch dilation if cv2 is unavailable.
        """
        for b in range(preds.shape[0]):
            p = preds[b, 0].cpu().numpy().astype(np.uint8)
            t = targets[b, 0].cpu().numpy().astype(np.uint8)

            if CV2_AVAILABLE:
                kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
                p_border = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel)
                t_border = cv2.morphologyEx(t, cv2.MORPH_GRADIENT, kernel)
            else:
                # Lightweight fallback: detect boundaries via diff with eroded version
                p_border = self._border_torch(preds[b, 0], dilation)
                t_border = self._border_torch(targets[b, 0], dilation)

            self.boundary_tp += int((p_border * t_border).sum())
            self.boundary_fp += int((p_border * (1 - t_border)).sum())
            self.boundary_fn += int(((1 - p_border) * t_border).sum())

    @staticmethod
    def _border_torch(mask: torch.Tensor, dilation: int) -> np.ndarray:
        """Pure-PyTorch morphological gradient (boundary extractor)."""
        import torch.nn.functional as F
        m = mask.unsqueeze(0).unsqueeze(0).float()
        k = torch.ones(1, 1, dilation, dilation, device=mask.device)
        dilated = (F.conv2d(m, k, padding=dilation // 2) > 0).float()
        eroded  = (F.conv2d(1 - m, k, padding=dilation // 2) == 0).float()
        border  = (dilated - eroded).clamp(0, 1)
        return border.squeeze().cpu().numpy().astype(np.uint8)

    def compute(self) -> dict:
        eps = 1e-7
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        precision   = tp / (tp + fp + eps)
        recall      = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        f1          = 2 * precision * recall / (precision + recall + eps)
        iou         = tp / (tp + fp + fn + eps)
        dice        = 2 * tp / (2 * tp + fp + fn + eps)

        b_tp, b_fp, b_fn = self.boundary_tp, self.boundary_fp, self.boundary_fn
        b_prec    = b_tp / (b_tp + b_fp + eps)
        b_recall  = b_tp / (b_tp + b_fn + eps)
        boundary_f1 = 2 * b_prec * b_recall / (b_prec + b_recall + eps)

        return {
            'dice'        : round(dice,        4),
            'iou'         : round(iou,         4),
            'f1'          : round(f1,          4),
            'precision'   : round(precision,   4),
            'recall'      : round(recall,      4),
            'specificity' : round(specificity, 4),
            'boundary_f1' : round(boundary_f1, 4),
            'n_images'    : self.total_images,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_checkpoints(experiments_dir: str) -> list[Path]:
    """
    Walk the experiments directory and collect all ``best.pth`` files.
    Returns paths sorted alphabetically for reproducibility.
    """
    ckpts = sorted(Path(experiments_dir).rglob('checkpoints/best.pth'))
    print(f'\nDiscovered {len(ckpts)} checkpoint(s) under: {experiments_dir}')
    for p in ckpts:
        print(f'  {p.relative_to(experiments_dir)}')
    return ckpts


def load_checkpoint_config(ckpt_path: Path) -> dict:
    """Load the config dict embedded in the checkpoint (CPU-safe)."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'config' not in ckpt:
        raise KeyError(f'Checkpoint {ckpt_path} has no embedded config key.')
    return ckpt['config'], ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def build_model_from_config(config: dict, ckpt: dict, device: torch.device) -> nn.Module:
    """Reconstruct the model from its training config and load weights."""
    model_name = config['model']['name']
    variant    = config['model'].get('variant', None)
    n_channels = config['model'].get('n_channels', 3)
    n_classes  = config['model'].get('n_classes',  1)

    if model_name == 'global_local':
        gl = config['model']['global_local']
        model = GlobalLocalWrapper(
            num_classes       = n_classes,
            n_channels        = n_channels,
            global_model_name = gl['global_model_name'],
            global_variant    = gl.get('global_variant'),
            local_model_name  = gl.get('local_model_name'),
            local_variant     = gl.get('local_variant'),
        )
    else:
        model = get_model(
            model_name = model_name,
            variant    = variant,
            n_channels = n_channels,
            n_classes  = n_classes,
        )

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_display_name(config: dict) -> str:
    """Human-readable model label used in tables and filenames."""
    model_name = config['model']['name']
    variant    = config['model'].get('variant', None)

    if model_name == 'global_local':
        gl = config['model'].get('global_local', {})
        g  = gl.get('global_model_name', 'unet')
        gv = gl.get('global_variant', '')
        l  = gl.get('local_model_name', None)
        lv = gl.get('local_variant', '')
        g_label = f'{g}_{gv}' if gv else g
        if l and l != g:
            l_label = f'{l}_{lv}' if lv else l
            return f'gl_asym_{g_label}__{l_label}'
        return f'gl_{g_label}'

    return f'{model_name}_{variant}' if variant else model_name


def model_family(config: dict) -> str:
    """Assign each model to a display family for table grouping."""
    name = config['model']['name']
    if name == 'global_local':
        return 'GlobalLocal'
    if name in ('unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam'):
        return 'CNN Baseline'
    if name in ('segformer', 'swin_unet'):
        return 'Transformer'
    if name in ('convnext_upernet', 'hrnet_ocr'):
        return 'Hybrid SOTA'
    if name in ('sam', 'dinov2'):
        return 'Foundation'
    return 'Other'


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model      : nn.Module,
    loader     : DataLoader,
    device     : torch.device,
    is_global_local: bool = False,
    threshold  : float = 0.5,
) -> tuple[dict, list[dict]]:
    """
    Run full test-set inference and return aggregated + per-image metrics.

    Returns
    -------
    (aggregate_metrics, per_image_records)
        aggregate_metrics : dict of scalar metrics
        per_image_records : list of dicts (one per image) for per-sample analysis
    """
    acc       = MetricAccumulator(threshold=threshold)
    per_image = []
    timing    = []

    model.eval()

    for batch in tqdm(loader, desc='  Inference', leave=False, ncols=90):
        masks = batch['mask'].to(device)

        t0 = time.perf_counter()

        if is_global_local:
            global_img  = batch['global_image'].to(device)
            local_patch = batch['local_image'].to(device)
            outputs = model(global_img, local_patch, return_aux=False)
        else:
            images  = batch['image'].to(device)
            outputs = model(images)

        # Some models return a tuple (logits, aux); take the main output
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Ensure correct shape: [B, 1, H, W]
        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(1)

        probs = torch.sigmoid(outputs)

        t1 = time.perf_counter()
        timing.append((t1 - t0) / masks.shape[0])  # per-image seconds

        acc.update(probs.cpu(), masks.cpu())

        # Per-image metrics
        for i, img_path in enumerate(batch.get('image_path', [''] * masks.shape[0])):
            p = probs[i].cpu()
            m = masks[i].cpu()
            eps = 1e-7
            tp = (p > threshold).float() * m
            fp = (p > threshold).float() * (1 - m)
            fn = (1 - (p > threshold).float()) * m
            tp_sum = tp.sum().item()
            fp_sum = fp.sum().item()
            fn_sum = fn.sum().item()
            img_dice = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + eps)
            img_iou  = tp_sum / (tp_sum + fp_sum + fn_sum + eps)
            per_image.append({
                'image'    : os.path.basename(img_path),
                'dice'     : round(img_dice, 4),
                'iou'      : round(img_iou,  4),
                'pred_prob': round(p.mean().item(), 4),
                'gt_ratio' : round(m.mean().item(), 4),
            })

    agg = acc.compute()
    agg['inference_ms'] = round(np.mean(timing) * 1000, 2)
    return agg, per_image


# ─────────────────────────────────────────────────────────────────────────────
# Prediction visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction_overlays(
    model         : nn.Module,
    loader        : DataLoader,
    device        : torch.device,
    out_dir       : Path,
    is_global_local: bool = False,
    n_samples     : int = 8,
    threshold     : float = 0.5,
):
    """
    Save side-by-side overlay images: [Image | Ground Truth | Prediction | Overlay].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()

    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for batch in loader:
            if saved >= n_samples:
                break

            masks = batch['mask'].to(device)

            if is_global_local:
                outputs = model(
                    batch['global_image'].to(device),
                    batch['local_image'].to(device),
                    return_aux=False,
                )
                imgs_for_vis = batch['local_image']
            else:
                outputs = model(batch['image'].to(device))
                imgs_for_vis = batch['image']

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.dim() == 3:
                outputs = outputs.unsqueeze(1)
            probs = torch.sigmoid(outputs).cpu()
            preds = (probs > threshold).float()

            for i in range(min(len(batch['image_path']), n_samples - saved)):
                img_np   = imgs_for_vis[i].permute(1, 2, 0).numpy()
                img_np   = (img_np * STD + MEAN).clip(0, 1)  # de-normalise
                mask_np  = masks[i, 0].cpu().numpy()
                pred_np  = preds[i, 0].numpy()

                h, w = img_np.shape[:2]

                def to_uint8(arr):
                    return (arr * 255).astype(np.uint8)

                img_u8    = to_uint8(img_np)
                gt_u8     = np.stack([to_uint8(mask_np)] * 3, axis=2)
                pred_u8   = np.stack([to_uint8(pred_np)] * 3, axis=2)

                # Colour overlay: GT = green, Pred = red, overlap = yellow
                overlay   = img_u8.copy()
                gt_mask   = mask_np > 0.5
                pred_mask = pred_np > 0.5
                overlay[gt_mask & pred_mask]   = [255, 255,   0]   # yellow = correct
                overlay[gt_mask & ~pred_mask]  = [  0, 200,   0]   # green  = missed (FN)
                overlay[~gt_mask & pred_mask]  = [200,   0,   0]   # red    = false alarm (FP)

                strip = np.concatenate([img_u8, gt_u8, pred_u8, overlay], axis=1)
                fname = f'{saved:04d}_{Path(batch["image_path"][i]).stem}.png'
                Image.fromarray(strip).save(out_dir / fname)
                saved += 1


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(records: list[dict], path: Path):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f'  Saved: {path}')


def print_leaderboard(records: list[dict]) -> str:
    """
    Print and return a formatted leaderboard string, ranked by Dice.
    Groups models by family.
    """
    if not records:
        return ''

    sorted_records = sorted(records, key=lambda r: r['dice'], reverse=True)

    # Column widths
    col_model   = max(len(r['model']) for r in records) + 2
    col_family  = max(len(r['family']) for r in records) + 2
    col_w       = 10

    sep  = '─' * (col_model + col_family + col_w * 9 + 4)
    hdr  = (
        f'{"Model":<{col_model}}{"Family":<{col_family}}'
        f'{"Dice":>{col_w}}{"IoU":>{col_w}}{"F1":>{col_w}}'
        f'{"Precision":>{col_w}}{"Recall":>{col_w}}{"Specificity":>{col_w}}'
        f'{"BoundaryF1":>{col_w}}{"Infer(ms)":>{col_w}}{"Params(M)":>{col_w}}'
    )

    lines = ['\n' + '=' * len(sep), 'MODEL COMPARISON — TEST SET RESULTS (ranked by Dice)',
             '=' * len(sep), hdr, sep]

    prev_family = None
    for rank, r in enumerate(sorted_records, start=1):
        if r['family'] != prev_family:
            if prev_family is not None:
                lines.append('')
            lines.append(f'  ── {r["family"]} ──')
            prev_family = r['family']

        params_m = f'{r["params_M"]:.1f}'
        line = (
            f'{r["model"]:<{col_model}}{r["family"]:<{col_family}}'
            f'{r["dice"]:>{col_w}.4f}{r["iou"]:>{col_w}.4f}{r["f1"]:>{col_w}.4f}'
            f'{r["precision"]:>{col_w}.4f}{r["recall"]:>{col_w}.4f}'
            f'{r["specificity"]:>{col_w}.4f}{r["boundary_f1"]:>{col_w}.4f}'
            f'{r["inference_ms"]:>{col_w}.1f}{params_m:>{col_w}}'
        )
        marker = ' ◄ BEST' if rank == 1 else ''
        lines.append(line + marker)

    lines.append(sep)
    lines.append(f'  Evaluated on {sorted_records[0]["n_images"]} test images.')
    lines.append('  Boundary-F1: 3-pixel dilation kernel | Threshold: 0.5')
    lines.append('=' * len(sep) + '\n')

    text = '\n'.join(lines)
    print(text)
    return text


def log_to_wandb(records: list[dict], project: str):
    """Log a W&B comparison table."""
    if not WANDB_AVAILABLE:
        print('  [W&B] wandb not installed — skipping.')
        return

    wandb.init(project=project, job_type='evaluation', name='test_comparison')
    table = wandb.Table(
        columns=list(records[0].keys()),
        data=[[r[k] for k in records[0].keys()] for r in records],
    )
    wandb.log({'test_results': table})
    wandb.finish()
    print('  [W&B] Comparison table logged.')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate and compare all trained segmentation models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Examples:
              python evaluate_models.py
              python evaluate_models.py --experiments_dir ./experiments \\
                  --data_root ./dataset/processed_512_resized \\
                  --save_predictions --num_pred_samples 8
        '''),
    )
    p.add_argument('--experiments_dir', default='./experiments',
                   help='Root directory of saved experiment checkpoints.')
    p.add_argument('--data_root', default='./dataset/processed_512_resized',
                   help='Root of the standard (resized) dataset.')
    p.add_argument('--local_root', default='./dataset/processed_512_patch',
                   help='Root of the patched dataset (GlobalLocal models).')
    p.add_argument('--output_dir', default='./results',
                   help='Directory for output CSV, summaries, and prediction images.')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Binarisation threshold for predictions.')
    p.add_argument('--save_predictions', action='store_true',
                   help='Save overlay prediction images for visual inspection.')
    p.add_argument('--num_pred_samples', type=int, default=8,
                   help='Number of prediction overlay images to save per model.')
    p.add_argument('--use_wandb', action='store_true',
                   help='Log comparison table to Weights & Biases.')
    p.add_argument('--wandb_project', default='river-segmentation',
                   help='W&B project name for the comparison run.')
    p.add_argument('--no_cuda', action='store_true', help='Disable CUDA.')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cpu') if args.no_cuda or not torch.cuda.is_available() \
             else torch.device('cuda')

    print('\n' + '=' * 80)
    print('RIVER SEGMENTATION MODEL EVALUATION')
    print('=' * 80)
    print(f'Device          : {device}')
    print(f'Experiments dir : {args.experiments_dir}')
    print(f'Data root       : {args.data_root}')
    print(f'Output dir      : {args.output_dir}')
    print(f'Threshold       : {args.threshold}')

    # ── Discover checkpoints ──────────────────────────────────────────────────
    ckpt_paths = discover_checkpoints(args.experiments_dir)
    if not ckpt_paths:
        print('\n[ERROR] No best.pth checkpoints found.')
        print('Make sure training has completed and the experiments directory is correct.')
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results     : list[dict] = []
    per_image_all   : list[dict] = []
    failed_models   : list[str]  = []

    # ── Evaluate each checkpoint ──────────────────────────────────────────────
    for ckpt_path in ckpt_paths:
        print(f'\n{"─" * 80}')
        print(f'Checkpoint: {ckpt_path}')

        try:
            config, ckpt = load_checkpoint_config(ckpt_path)
        except Exception as e:
            print(f'  [SKIP] Could not load config: {e}')
            failed_models.append(str(ckpt_path))
            continue

        display_name    = model_display_name(config)
        family          = model_family(config)
        is_global_local = (config['model']['name'] == 'global_local')

        print(f'Model : {display_name}  [{family}]')

        # ── Build dataloader ──────────────────────────────────────────────────
        image_size = config['data'].get('image_size', 512)
        try:
            if is_global_local:
                local_root  = config['data'].get('local_root',  args.local_root)
                global_root = config['data'].get('global_root', args.data_root)
                test_ds = GlobalLocalTestDataset(
                    local_root  = local_root,
                    global_root = global_root,
                    split       = 'test',
                    image_size  = image_size,
                )
            else:
                data_root = config['data'].get('data_root', args.data_root)
                test_ds   = TestDataset(
                    data_root  = data_root,
                    split      = 'test',
                    image_size = image_size,
                )
        except FileNotFoundError as e:
            print(f'  [SKIP] Dataset not found: {e}')
            failed_models.append(display_name)
            continue

        test_loader = DataLoader(
            test_ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            pin_memory  = (device.type == 'cuda'),
        )

        # ── Build model ───────────────────────────────────────────────────────
        try:
            model = build_model_from_config(config, ckpt, device)
        except Exception as e:
            print(f'  [SKIP] Model construction failed: {e}')
            failed_models.append(display_name)
            continue

        params = model_param_count(model)
        print(f'  Parameters: {params / 1e6:.2f} M')

        # ── Inference ─────────────────────────────────────────────────────────
        try:
            agg_metrics, per_img = evaluate_model(
                model           = model,
                loader          = test_loader,
                device          = device,
                is_global_local = is_global_local,
                threshold       = args.threshold,
            )
        except torch.cuda.OutOfMemoryError:
            print('  [WARN] CUDA OOM — retrying on CPU…')
            torch.cuda.empty_cache()
            model = model.cpu()
            device_cpu = torch.device('cpu')
            agg_metrics, per_img = evaluate_model(
                model           = model,
                loader          = test_loader,
                device          = device_cpu,
                is_global_local = is_global_local,
                threshold       = args.threshold,
            )
            model = model.to(device)

        # ── Save prediction overlays ──────────────────────────────────────────
        if args.save_predictions:
            pred_dir = output_dir / 'predictions' / display_name
            print(f'  Saving {args.num_pred_samples} prediction overlays → {pred_dir}')
            try:
                save_prediction_overlays(
                    model           = model,
                    loader          = test_loader,
                    device          = device,
                    out_dir         = pred_dir,
                    is_global_local = is_global_local,
                    n_samples       = args.num_pred_samples,
                    threshold       = args.threshold,
                )
            except Exception as e:
                print(f'  [WARN] Could not save overlays: {e}')

        # ── Collect result record ─────────────────────────────────────────────
        result = {
            'model'       : display_name,
            'family'      : family,
            'params_M'    : round(params / 1e6, 2),
            **agg_metrics,
            'checkpoint'  : str(ckpt_path),
            'epoch'       : ckpt.get('epoch', -1),
        }
        all_results.append(result)

        for rec in per_img:
            rec['model'] = display_name
        per_image_all.extend(per_img)

        print(
            f'  Dice: {agg_metrics["dice"]:.4f} | '
            f'IoU: {agg_metrics["iou"]:.4f} | '
            f'BoundaryF1: {agg_metrics["boundary_f1"]:.4f} | '
            f'Infer: {agg_metrics["inference_ms"]:.1f} ms/img'
        )

        # Free VRAM between models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final reporting ───────────────────────────────────────────────────────
    if not all_results:
        print('\n[ERROR] No models were successfully evaluated.')
        sys.exit(1)

    leaderboard_text = print_leaderboard(all_results)

    # Save CSVs
    save_csv(all_results,   output_dir / 'test_results.csv')
    save_csv(per_image_all, output_dir / 'test_results_per_image.csv')

    # Save summary text
    summary_path = output_dir / 'test_results_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(leaderboard_text)
    print(f'  Saved: {summary_path}')

    # Report failures
    if failed_models:
        print(f'\n[WARNING] The following models were skipped:')
        for m in failed_models:
            print(f'  {m}')

    # W&B table
    if args.use_wandb:
        log_to_wandb(all_results, args.wandb_project)

    # ── Best model highlight ──────────────────────────────────────────────────
    best = max(all_results, key=lambda r: r['dice'])
    print('\n' + '★' * 60)
    print(f'  BEST MODEL  :  {best["model"]}')
    print(f'  Family      :  {best["family"]}')
    print(f'  Dice        :  {best["dice"]:.4f}')
    print(f'  IoU         :  {best["iou"]:.4f}')
    print(f'  Boundary-F1 :  {best["boundary_f1"]:.4f}   ← river-edge accuracy')
    print(f'  Inference   :  {best["inference_ms"]:.1f} ms / image')
    print(f'  Parameters  :  {best["params_M"]:.1f} M')
    print('★' * 60 + '\n')

    print('Evaluation complete.')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
