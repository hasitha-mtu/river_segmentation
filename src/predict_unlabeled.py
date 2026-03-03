"""
predict_unlabeled.py
====================
Run all trained segmentation models on a set of images that have **no ground-truth
masks** and save visually inspectable overlays.

Designed for manual verification of model predictions across images collected from
multiple field locations (e.g. different reaches of the Lee River catchment).

What this script does
---------------------
1. Auto-discovers every ``best.pth`` checkpoint under ``experiments/``.
2. Reconstructs each model from its embedded config (same as evaluate_models.py).
3. Runs inference on every image found under ``--images_dir`` (flat or nested).
4. Saves a 3-panel PNG per image per model:
       [  Original Image  |  Probability Heatmap  |  Prediction Overlay  ]
5. Preserves the source folder hierarchy:
       <output_dir>/<model_name>/<location_subfolder>/<image_stem>_pred.png
6. Writes a summary CSV:
       <output_dir>/prediction_summary.csv
       Columns: model, location, image, water_fraction_pct, max_prob, mean_prob

Folder layouts supported
------------------------
Flat (all images in one directory):
    images_dir/
        img_001.jpg
        img_002.png
        ...

Nested by location (recommended for multi-site surveys):
    images_dir/
        crookstown/
            frame_0001.jpg
            ...
        lee_valley/
            frame_0001.jpg
            ...

Usage
-----
    python predict_unlabeled.py \\
        --images_dir   ./field_images \\
        --experiments_dir ./experiments \\
        --output_dir   ./predictions_unlabeled \\
        --image_size   512 \\
        --threshold    0.5 \\
        --batch_size   4 \\
        --heatmap_cmap jet

    # To skip GlobalLocal models (useful if paired global images are unavailable):
        --skip_global_local

Notes
-----
* Images are resized to ``--image_size`` for inference, then the overlay is saved
  at the ORIGINAL resolution so field details are not lost.
* The probability heatmap uses a colour map (default: jet) where blue = low
  confidence, red = high confidence water prediction.
* DINOv2 / large foundation models are automatically skipped on CUDA OOM.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ── Optional deps ────────────────────────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ── Project imports ──────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from models import get_model
from wrapper import GlobalLocalWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


class UnlabeledImageDataset(Dataset):
    """
    Loads images from a flat or nested directory.  No masks required.

    Each sample returns:
        image       : normalised tensor  [3, H, W]
        image_path  : absolute path string
        location    : subfolder name (or 'root' if images are at top level)
        orig_size   : (width, height) of the original image before resizing
    """

    def __init__(self, images_dir: str, image_size: int = 512):
        root = Path(images_dir)
        if not root.exists():
            raise FileNotFoundError(f'Images directory not found: {root}')

        # Collect all image paths — support one level of nesting (location folders)
        self.samples: list[tuple[Path, str]] = []  # (path, location_name)

        # Check if the root itself contains images (flat layout)
        top_level_images = [
            p for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
        if top_level_images:
            for p in top_level_images:
                self.samples.append((p, 'root'))

        # Also collect from immediate subdirectories (nested layout)
        for subdir in sorted(root.iterdir()):
            if subdir.is_dir():
                sub_images = [
                    p for p in sorted(subdir.iterdir())
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                ]
                for p in sub_images:
                    self.samples.append((p, subdir.name))

        if not self.samples:
            raise ValueError(
                f'No images found in {root}\n'
                f'Supported extensions: {SUPPORTED_EXTS}'
            )

        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(),
                                 std=IMAGENET_STD.tolist()),
        ])

        # Report what was found
        locations = sorted({loc for _, loc in self.samples})
        print(f'\n  [UnlabeledImageDataset] Found {len(self.samples)} image(s) '
              f'across {len(locations)} location(s): {locations}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, location = self.samples[idx]
        pil_img  = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size

        return {
            'image'     : self.transform(pil_img),
            'image_path': str(img_path),
            'location'  : location,
            'orig_w'    : orig_w,
            'orig_h'    : orig_h,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (mirrors evaluate_models.py)
# ─────────────────────────────────────────────────────────────────────────────

def discover_checkpoints(experiments_dir: str) -> list[Path]:
    ckpts = sorted(Path(experiments_dir).rglob('checkpoints/best.pth'))
    print(f'\nDiscovered {len(ckpts)} checkpoint(s) under: {experiments_dir}')
    for p in ckpts:
        print(f'  {p.relative_to(experiments_dir)}')
    return ckpts


def load_checkpoint_config(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'config' not in ckpt:
        raise KeyError(f'Checkpoint {ckpt_path} has no embedded config key.')
    return ckpt['config'], ckpt


def build_model_from_config(config: dict, ckpt: dict, device: torch.device) -> nn.Module:
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


def model_display_name(config: dict) -> str:
    model_name = config['model']['name']
    variant    = config['model'].get('variant', None)
    if model_name == 'global_local':
        gl  = config['model'].get('global_local', {})
        g   = gl.get('global_model_name', 'unet')
        gv  = gl.get('global_variant', '')
        l   = gl.get('local_model_name', None)
        lv  = gl.get('local_variant', '')
        g_label = f'{g}_{gv}' if gv else g
        if l and l != g:
            l_label = f'{l}_{lv}' if lv else l
            return f'gl_asym_{g_label}__{l_label}'
        return f'gl_{g_label}'
    return f'{model_name}_{variant}' if variant else model_name


def model_family(config: dict) -> str:
    name = config['model']['name']
    if name == 'global_local':         return 'GlobalLocal'
    if name in ('unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam'):
        return 'CNN Baseline'
    if name in ('segformer', 'swin_unet'):  return 'Transformer'
    if name in ('convnext_upernet', 'hrnet_ocr'): return 'Hybrid SOTA'
    if name in ('sam', 'dinov2'):       return 'Foundation'
    return 'Other'


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model     : nn.Module,
    loader    : DataLoader,
    device    : torch.device,
    threshold : float = 0.5,
) -> list[dict]:
    """
    Run inference and return a list of per-image result dicts:
        image_path, location, prob_map (H×W float32 numpy), pred_mask (H×W bool),
        orig_w, orig_h, inference_ms
    """
    model.eval()
    results = []

    for batch in tqdm(loader, desc='  Inference', leave=False, ncols=90):
        images = batch['image'].to(device)

        t0 = time.perf_counter()
        outputs = model(images)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.perf_counter()

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(1)

        probs = torch.sigmoid(outputs).cpu().numpy()   # [B, 1, H, W]
        ms_per_img = (t1 - t0) / images.shape[0] * 1000

        for i in range(images.shape[0]):
            prob_map = probs[i, 0]                     # [H, W] float32
            results.append({
                'image_path'   : batch['image_path'][i],
                'location'     : batch['location'][i],
                'orig_w'       : int(batch['orig_w'][i]),
                'orig_h'       : int(batch['orig_h'][i]),
                'prob_map'     : prob_map,
                'pred_mask'    : prob_map > threshold,
                'inference_ms' : round(ms_per_img, 2),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def apply_colormap(prob_map: np.ndarray, cmap_name: str = 'jet') -> np.ndarray:
    """Convert a [H,W] float32 probability map to a [H,W,3] uint8 RGB image."""
    if MPL_AVAILABLE:
        cmap  = mpl_cm.get_cmap(cmap_name)
        rgb   = cmap(prob_map)[:, :, :3]          # drop alpha
        return (rgb * 255).astype(np.uint8)
    else:
        # Fallback: greyscale
        grey = (prob_map * 255).astype(np.uint8)
        return np.stack([grey, grey, grey], axis=2)


def make_overlay_panel(
    img_np   : np.ndarray,   # [H, W, 3] uint8  original image
    pred_mask: np.ndarray,   # [H, W]    bool   predicted water pixels
    alpha    : float = 0.45,
    color    : tuple = (0, 180, 255),   # cyan-blue water colour
) -> np.ndarray:
    """
    Blend a semi-transparent coloured mask onto the original image.
    Returns [H, W, 3] uint8.
    """
    overlay = img_np.copy().astype(np.float32)
    water   = pred_mask.astype(bool)

    colour_layer          = np.zeros_like(overlay)
    colour_layer[water]   = color

    blended               = overlay.copy()
    blended[water]        = (
        (1 - alpha) * overlay[water] + alpha * colour_layer[water]
    )

    # Draw a thin contour around the prediction if cv2 is available
    if CV2_AVAILABLE:
        mask_u8 = pred_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blended_u8  = blended.astype(np.uint8)
        cv2.drawContours(blended_u8, contours, -1, (255, 255, 0), 1)   # yellow border
        return blended_u8

    return blended.astype(np.uint8)


def save_result_image(
    result       : dict,
    out_path     : Path,
    model_name   : str,
    threshold    : float,
    heatmap_cmap : str = 'jet',
):
    """
    Save a 3-panel PNG:
        [ Original image | Probability heatmap | Prediction overlay ]

    The panels are composited at the ORIGINAL image resolution so fine river
    features remain visible.
    """
    img_path   = result['image_path']
    orig_w     = result['orig_w']
    orig_h     = result['orig_h']
    prob_map   = result['prob_map']    # inference resolution (image_size × image_size)
    pred_mask  = result['pred_mask']

    # ── Load original image at its native resolution ──────────────────────────
    orig_pil = Image.open(img_path).convert('RGB')
    orig_np  = np.array(orig_pil)     # [orig_h, orig_w, 3] uint8

    # ── Resize prob_map and pred_mask back to original resolution ─────────────
    prob_pil  = Image.fromarray((prob_map * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.BILINEAR)
    prob_full = np.array(prob_pil).astype(np.float32) / 255.0

    mask_pil  = Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
        (orig_w, orig_h), Image.NEAREST)
    mask_full = np.array(mask_pil) > 127

    # ── Build panels ──────────────────────────────────────────────────────────
    heatmap_np = apply_colormap(prob_full, heatmap_cmap)
    overlay_np = make_overlay_panel(orig_np, mask_full)

    water_pct  = mask_full.mean() * 100

    # ── Add text labels using PIL ─────────────────────────────────────────────
    def label_panel(arr: np.ndarray, text: str) -> np.ndarray:
        """Write a label banner at the top of a panel."""
        pil = Image.fromarray(arr)
        # Create a dark banner
        banner_h = max(22, orig_h // 30)
        banner   = Image.new('RGB', (orig_w, banner_h), color=(20, 20, 20))
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(banner)
            draw.text((4, 3), text, fill=(240, 240, 240))
        except Exception:
            pass
        combined = Image.new('RGB', (orig_w, orig_h + banner_h))
        combined.paste(banner, (0, 0))
        combined.paste(pil,    (0, banner_h))
        return np.array(combined)

    panel1 = label_panel(orig_np,   'Original')
    panel2 = label_panel(heatmap_np, f'Probability  (threshold={threshold:.2f})')
    panel3 = label_panel(overlay_np,
                         f'Prediction  |  water={water_pct:.2f}%  |  {model_name}')

    # Make sure all panels have the same height (orig_h + banner_h)
    h      = panel1.shape[0]
    strip  = np.concatenate([panel1, panel2, panel3], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(out_path, quality=92)


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_csv(records: list[dict], path: Path):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['model', 'family', 'location', 'image',
                  'water_fraction_pct', 'mean_prob', 'max_prob', 'inference_ms']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(f'  Saved summary: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Predict segmentation masks for unlabeled field images using all trained models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Examples:
              # Basic usage (images in flat directory):
              python predict_unlabeled.py --images_dir ./field_images

              # Multi-site nested folder with custom threshold:
              python predict_unlabeled.py \\
                  --images_dir   ./field_images \\
                  --experiments_dir ./experiments \\
                  --output_dir   ./predictions_unlabeled \\
                  --threshold    0.4 \\
                  --heatmap_cmap turbo

              # Skip GlobalLocal models:
              python predict_unlabeled.py \\
                  --images_dir ./field_images \\
                  --skip_global_local
        '''),
    )
    p.add_argument('--images_dir', required=True,
                   help='Directory containing field images (flat or one-level nested by location).')
    p.add_argument('--experiments_dir', default='./experiments',
                   help='Root directory of saved experiment checkpoints.')
    p.add_argument('--output_dir', default='./predictions_unlabeled',
                   help='Root directory for output overlays and summary CSV.')
    p.add_argument('--image_size', type=int, default=512,
                   help='Resolution used for model inference (should match training).')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Binarisation threshold for the prediction mask.')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--heatmap_cmap', default='jet',
                   choices=['jet', 'turbo', 'plasma', 'viridis', 'hot'],
                   help='Matplotlib colourmap for the probability heatmap panel.')
    p.add_argument('--skip_global_local', action='store_true',
                   help='Skip GlobalLocal models (useful when paired global images are unavailable).')
    p.add_argument('--no_cuda', action='store_true',
                   help='Force CPU inference.')
    return p.parse_args()


def main():
    args   = parse_args()
    device = (torch.device('cpu')
              if args.no_cuda or not torch.cuda.is_available()
              else torch.device('cuda'))

    print('\n' + '=' * 80)
    print('RIVER SEGMENTATION — UNLABELED IMAGE PREDICTION')
    print('=' * 80)
    print(f'Device          : {device}')
    print(f'Images dir      : {args.images_dir}')
    print(f'Experiments dir : {args.experiments_dir}')
    print(f'Output dir      : {args.output_dir}')
    print(f'Image size      : {args.image_size}')
    print(f'Threshold       : {args.threshold}')
    print(f'Heatmap cmap    : {args.heatmap_cmap}')
    if args.skip_global_local:
        print('  [INFO] Skipping GlobalLocal models.')

    # ── Load dataset ──────────────────────────────────────────────────────────
    try:
        dataset = UnlabeledImageDataset(args.images_dir, image_size=args.image_size)
    except (FileNotFoundError, ValueError) as e:
        print(f'\n[ERROR] {e}')
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device.type == 'cuda'),
    )

    # ── Discover checkpoints ──────────────────────────────────────────────────
    ckpt_paths = discover_checkpoints(args.experiments_dir)
    if not ckpt_paths:
        print('\n[ERROR] No best.pth checkpoints found.')
        sys.exit(1)

    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary  : list[dict] = []
    failed_models: list[str]  = []

    # ── Per-model inference ───────────────────────────────────────────────────
    for ckpt_path in ckpt_paths:
        print(f'\n{"─" * 80}')
        print(f'Checkpoint: {ckpt_path}')

        try:
            config, ckpt = load_checkpoint_config(ckpt_path)
        except Exception as e:
            print(f'  [SKIP] Could not load config: {e}')
            failed_models.append(str(ckpt_path))
            continue

        is_global_local = (config['model']['name'] == 'global_local')
        display_name    = model_display_name(config)
        family          = model_family(config)

        if is_global_local and args.skip_global_local:
            print(f'  [SKIP] GlobalLocal model skipped: {display_name}')
            continue

        if is_global_local:
            print(f'  [SKIP] {display_name} requires paired global images — '
                  f'not supported in unlabeled mode. Use --skip_global_local to suppress this.')
            failed_models.append(display_name)
            continue

        print(f'Model : {display_name}  [{family}]')

        # ── Build model ───────────────────────────────────────────────────────
        try:
            model = build_model_from_config(config, ckpt, device)
        except Exception as e:
            print(f'  [SKIP] Model construction failed: {e}')
            failed_models.append(display_name)
            continue

        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'  Parameters: {params_m:.2f} M')

        # ── Inference ─────────────────────────────────────────────────────────
        try:
            results = run_inference(model, loader, device, threshold=args.threshold)
        except torch.cuda.OutOfMemoryError:
            print('  [WARN] CUDA OOM — retrying on CPU…')
            torch.cuda.empty_cache()
            model   = model.cpu()
            results = run_inference(model, loader, torch.device('cpu'),
                                    threshold=args.threshold)
            model   = model.to(device)
        except Exception as e:
            print(f'  [SKIP] Inference failed: {e}')
            failed_models.append(display_name)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # ── Save overlays and collect summary ─────────────────────────────────
        model_out_dir = output_dir / display_name
        n_saved       = 0

        for res in results:
            img_stem  = Path(res['image_path']).stem
            location  = res['location']
            out_path  = model_out_dir / location / f'{img_stem}_pred.png'

            try:
                save_result_image(
                    result       = res,
                    out_path     = out_path,
                    model_name   = display_name,
                    threshold    = args.threshold,
                    heatmap_cmap = args.heatmap_cmap,
                )
                n_saved += 1
            except Exception as e:
                print(f'  [WARN] Could not save overlay for {img_stem}: {e}')

            water_pct = float(res['pred_mask'].mean() * 100)
            all_summary.append({
                'model'             : display_name,
                'family'            : family,
                'location'          : location,
                'image'             : Path(res['image_path']).name,
                'water_fraction_pct': round(water_pct, 4),
                'mean_prob'         : round(float(res['prob_map'].mean()), 4),
                'max_prob'          : round(float(res['prob_map'].max()),  4),
                'inference_ms'      : res['inference_ms'],
            })

        avg_water = np.mean([r['water_fraction_pct'] for r in all_summary
                             if r['model'] == display_name])
        print(f'  Saved {n_saved} overlay(s) → {model_out_dir}')
        print(f'  Avg predicted water fraction: {avg_water:.3f}%')

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save summary CSV ──────────────────────────────────────────────────────
    save_summary_csv(all_summary, output_dir / 'prediction_summary.csv')

    # ── Report skipped models ─────────────────────────────────────────────────
    if failed_models:
        print(f'\n[WARNING] The following models were skipped:')
        for m in failed_models:
            print(f'  {m}')

    # ── Output structure summary ──────────────────────────────────────────────
    models_run = sorted({r['model'] for r in all_summary})
    print('\n' + '=' * 80)
    print(f'PREDICTION COMPLETE  —  {len(models_run)} model(s) processed')
    print('=' * 80)
    print(f'Output directory: {output_dir}')
    print('\nOutput structure:')
    print(f'  {output_dir}/')
    print(f'  ├── prediction_summary.csv       ← all models × all images')
    for m in models_run:
        locs = sorted({r['location'] for r in all_summary if r['model'] == m})
        print(f'  ├── {m}/')
        for loc in locs:
            n = sum(1 for r in all_summary if r['model'] == m and r['location'] == loc)
            print(f'  │   └── {loc}/   ({n} overlay PNG(s))')
    print('\nEach PNG contains 3 panels:')
    print('  [ Original Image | Probability Heatmap | Prediction Overlay ]')
    print('  Overlay key: cyan = predicted water  |  yellow border = water edge')
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
