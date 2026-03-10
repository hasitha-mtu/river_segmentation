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

try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print('[WARN] segment_anything not installed — SAM inference unavailable.')


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


class SAMUnlabeledDataset(Dataset):
    """
    Unlabeled dataset for SAM models — no masks required.

    MUST use SAM's own preprocessing (ResizeLongestSide + SAM normalisation),
    NOT ImageNet normalisation.  Using UnlabeledImageDataset for SAM would
    silently corrupt predictions because the model was never trained with
    ImageNet statistics.

    Preprocessing matches SAMDataset in train_unified_wandb_sam.py exactly:
      1. Load image → cv2 resize to image_size × image_size
      2. ResizeLongestSide(1024) — square 512 input → 1024×1024
      3. SAM normalisation  (mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375])
      4. Zero-pad to 1024×1024

    Returned batch dict keys:
        input_tensor  : FloatTensor [3, 1024, 1024]
        input_size    : LongTensor  [2]   — (H', W') after ResizeLongestSide
        original_size : LongTensor  [2]   — (image_size, image_size)
        image_path    : str
        location      : str
        orig_w        : int   — native image width  (for overlay visualisation)
        orig_h        : int   — native image height (for overlay visualisation)
    """

    SAM_IMG_SIZE = 1024
    _PIXEL_MEAN  = torch.tensor([123.675, 116.28,  103.53]).view(3, 1, 1)
    _PIXEL_STD   = torch.tensor([ 58.395,  57.12,   57.375]).view(3, 1, 1)

    def __init__(self, images_dir: str, image_size: int = 512):
        if not SAM_AVAILABLE:
            raise RuntimeError(
                'segment_anything is required for SAM inference.\n'
                'pip install git+https://github.com/facebookresearch/segment-anything.git'
            )
        if not CV2_AVAILABLE:
            raise RuntimeError('cv2 (opencv-python) is required for SAMUnlabeledDataset.')

        root = Path(images_dir)
        if not root.exists():
            raise FileNotFoundError(f'Images directory not found: {root}')

        self.samples: list[tuple[Path, str]] = []

        top_level_images = [
            p for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
        for p in top_level_images:
            self.samples.append((p, 'root'))

        for subdir in sorted(root.iterdir()):
            if subdir.is_dir():
                for p in sorted(subdir.iterdir()):
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                        self.samples.append((p, subdir.name))

        if not self.samples:
            raise ValueError(
                f'No images found in {root}\n'
                f'Supported extensions: {SUPPORTED_EXTS}'
            )

        self.image_size       = image_size
        self.resize_transform = ResizeLongestSide(self.SAM_IMG_SIZE)

        locations = sorted({loc for _, loc in self.samples})
        print(f'\n  [SAMUnlabeledDataset] Found {len(self.samples)} image(s) '
              f'across {len(locations)} location(s): {locations}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, location = self.samples[idx]

        # Load at native resolution first to capture orig_w / orig_h
        raw = cv2.imread(str(img_path))
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        orig_h_native, orig_w_native = raw.shape[:2]

        # Resize to model's training resolution
        image = cv2.resize(
            raw, (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        original_size = (image.shape[0], image.shape[1])  # (image_size, image_size)

        # SAM preprocessing: ResizeLongestSide then normalise + zero-pad
        resized    = self.resize_transform.apply_image(image)
        input_size = (resized.shape[0], resized.shape[1])

        img_t = torch.as_tensor(resized).permute(2, 0, 1).float()
        img_t = (img_t - self._PIXEL_MEAN) / self._PIXEL_STD
        h, w  = img_t.shape[-2:]
        img_t = torch.nn.functional.pad(
            img_t, (0, self.SAM_IMG_SIZE - w, 0, self.SAM_IMG_SIZE - h)
        )  # [3, 1024, 1024]

        return {
            'input_tensor' : img_t,
            'input_size'   : torch.tensor(input_size,    dtype=torch.long),
            'original_size': torch.tensor(original_size, dtype=torch.long),
            'image_path'   : str(img_path),
            'location'     : location,
            'orig_w'       : orig_w_native,
            'orig_h'       : orig_h_native,
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

    if model_name == 'sam':
        # Training saves a raw sam_model_registry[variant] object.
        # State-dict keys: image_encoder.*, prompt_encoder.*, mask_decoder.*
        # models.py's SAMEncoderDecoder uses different key prefixes — using
        # get_model() here causes a key-mismatch crash.
        if not SAM_AVAILABLE:
            raise RuntimeError(
                'segment_anything is not installed — required for SAM inference.\n'
                'pip install git+https://github.com/facebookresearch/segment-anything.git'
            )
        sam_ckpt = (
            config.get('foundation', {})
                  .get('sam_checkpoints', {})
                  .get(variant, f'./checkpoints/sam/sam_{variant}.pth')
        )
        print(f'  [SAM] Rebuilding {variant} from: {sam_ckpt}')
        model = sam_model_registry[variant](checkpoint=sam_ckpt)
        model.load_state_dict(ckpt['model_state_dict'])

    elif model_name == 'global_local':
        from wrapper import GlobalLocalWrapper
        gl = config['model']['global_local']
        model = GlobalLocalWrapper(
            num_classes       = n_classes,
            n_channels        = n_channels,
            global_model_name = gl['global_model_name'],
            global_variant    = gl.get('global_variant'),
            local_model_name  = gl.get('local_model_name'),
            local_variant     = gl.get('local_variant'),
        )
        model.load_state_dict(ckpt['model_state_dict'])

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
    if name in ('sam', 'sam_fpn', 'dinov2'):       return 'Foundation'
    return 'Other'


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_sam(
    model  : nn.Module,
    batch  : dict,
    device : torch.device,
) -> torch.Tensor:
    """
    SAM forward pass for unlabeled inference.

    Mirrors _forward_sam() in train_unified_wandb_sam.py exactly.
    image_encoder runs batched (frozen); prompt_encoder + mask_decoder
    run per-image because SAM's internal repeat_interleave crashes with
    batch > 1 when no spatial prompts are given.

    Returns logits [B, 1, H, W] (sigmoid applied by caller).
    """
    input_tensor  = batch['input_tensor'].to(device)   # [B, 3, 1024, 1024]
    B             = input_tensor.shape[0]
    input_size    = tuple(int(x) for x in batch['input_size'][0])
    original_size = tuple(int(x) for x in batch['original_size'][0])

    with torch.no_grad():
        image_embeddings = model.image_encoder(input_tensor)  # [B, 256, 64, 64]

    upscaled_list = []
    for i in range(B):
        emb_i = image_embeddings[i].unsqueeze(0)  # [1, 256, 64, 64]

        with torch.no_grad():
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None, boxes=None, masks=None,
            )

        low_res_i, _ = model.mask_decoder(
            image_embeddings         = emb_i,
            image_pe                 = model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_emb,
            dense_prompt_embeddings  = dense_emb,
            multimask_output         = False,
        )  # [1, 1, 256, 256]

        upscaled_i = model.postprocess_masks(
            low_res_i, input_size, original_size,
        )  # [1, 1, image_size, image_size]
        upscaled_list.append(upscaled_i)

    return torch.cat(upscaled_list, dim=0)  # [B, 1, H, W] logits


@torch.no_grad()
def run_inference(
    model     : nn.Module,
    loader    : DataLoader,
    device    : torch.device,
    threshold : float = 0.5,
    is_sam    : bool  = False,
) -> list[dict]:
    """
    Run inference and return a list of per-image result dicts:
        image_path, location, prob_map (H×W float32 numpy), pred_mask (H×W bool),
        orig_w, orig_h, inference_ms
    """
    model.eval()
    results = []

    for batch in tqdm(loader, desc='  Inference', leave=False, ncols=90):
        t0 = time.perf_counter()

        if is_sam:
            # SAM needs its own forward: batched image_encoder + per-image
            # prompt_encoder/mask_decoder.  model(batch['image']) would call
            # the wrong class entirely (SAMEncoderDecoder vs raw SAM registry).
            outputs = _forward_sam(model, batch, device)
        else:
            images  = batch['image'].to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(1)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        probs = torch.sigmoid(outputs).cpu().numpy()   # [B, 1, H, W]
        B     = probs.shape[0]
        ms_per_img = (t1 - t0) / B * 1000

        for i in range(B):
            prob_map = probs[i, 0]   # [H, W] float32
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
    jpeg_quality : int = 85,
    max_panel_w  : int = 1920,
):
    """
    Save a 3-panel JPEG:
        [ Original image | Probability heatmap | Prediction overlay ]

    Why JPEG, not PNG
    -----------------
    The 3-panel strip is rendered at full drone resolution (5472×3648 typical),
    producing a ~16000×3648 canvas.  PNG is lossless — each file can exceed
    100 MB, and a full benchmark run (20 models × hundreds of images) reaches
    hundreds of GB.  JPEG at quality=85 keeps each file under ~3–5 MB with no
    visible degradation for visual inspection.

    Note: passing ``quality`` to PIL's PNG save is silently ignored — PNG has
    no quality setting.  This was the original bug causing the bloated files.

    Resolution cap
    --------------
    ``max_panel_w`` caps each panel's width before compositing.  The default
    (1920 px) means the full strip is at most 5760 px wide — enough to inspect
    fine river features while being practical to open and store.  Set to 0 to
    render at full original resolution.
    """
    img_path  = result['image_path']
    orig_w    = result['orig_w']
    orig_h    = result['orig_h']
    prob_map  = result['prob_map']
    pred_mask = result['pred_mask']

    # ── Determine display resolution (cap width, preserve aspect ratio) ───────
    if max_panel_w > 0 and orig_w > max_panel_w:
        scale  = max_panel_w / orig_w
        disp_w = max_panel_w
        disp_h = max(1, int(orig_h * scale))
    else:
        disp_w, disp_h = orig_w, orig_h

    # ── Load original image at display resolution ─────────────────────────────
    orig_pil = Image.open(img_path).convert('RGB')
    if (disp_w, disp_h) != (orig_w, orig_h):
        orig_pil = orig_pil.resize((disp_w, disp_h), Image.LANCZOS)
    orig_np = np.array(orig_pil)

    # ── Resize prob_map and pred_mask to display resolution ───────────────────
    prob_pil  = Image.fromarray((prob_map * 255).astype(np.uint8)).resize(
        (disp_w, disp_h), Image.BILINEAR)
    prob_full = np.array(prob_pil).astype(np.float32) / 255.0

    mask_pil  = Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
        (disp_w, disp_h), Image.NEAREST)
    mask_full = np.array(mask_pil) > 127

    # ── Build panels ──────────────────────────────────────────────────────────
    heatmap_np = apply_colormap(prob_full, heatmap_cmap)
    overlay_np = make_overlay_panel(orig_np, mask_full)

    water_pct = mask_full.mean() * 100

    # ── Add text labels ───────────────────────────────────────────────────────
    def label_panel(arr: np.ndarray, text: str) -> np.ndarray:
        pil      = Image.fromarray(arr)
        banner_h = max(22, disp_h // 30)
        banner   = Image.new('RGB', (disp_w, banner_h), color=(20, 20, 20))
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(banner)
            draw.text((4, 3), text, fill=(240, 240, 240))
        except Exception:
            pass
        combined = Image.new('RGB', (disp_w, disp_h + banner_h))
        combined.paste(banner, (0, 0))
        combined.paste(pil,    (0, banner_h))
        return np.array(combined)

    panel1 = label_panel(orig_np,    'Original')
    panel2 = label_panel(heatmap_np, f'Probability  (threshold={threshold:.2f})')
    panel3 = label_panel(overlay_np, f'Prediction  |  water={water_pct:.2f}%  |  {model_name}')

    strip = np.concatenate([panel1, panel2, panel3], axis=1)

    # ── Save as JPEG ──────────────────────────────────────────────────────────
    # Force .jpg extension — the original .png extension caused PIL to write a
    # lossless file where the quality= argument is silently ignored.
    out_path = out_path.with_suffix('.jpg')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(str(out_path), format='JPEG',
                                quality=jpeg_quality, optimize=True)


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
    p.add_argument('--jpeg_quality', type=int, default=85,
                   help='JPEG quality for saved overlays (1–95). Default 85.')
    p.add_argument('--max_panel_w', type=int, default=1920,
                   help='Cap each panel width to this many pixels before compositing. '
                        'Set 0 for full original resolution (warning: very large files). '
                        'Default 1920.')
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
        is_sam          = (config['model']['name'] == 'sam')
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

        # ── Build per-model loader (SAM needs different preprocessing) ────────
        image_size = config['data'].get('image_size', 512)
        if is_sam:
            # SAM was trained with ResizeLongestSide(1024) + SAM normalisation.
            # Re-using the shared loader (ImageNet norm) would silently corrupt
            # predictions — build a dedicated SAM loader per model instead.
            try:
                sam_dataset = SAMUnlabeledDataset(
                    images_dir = args.images_dir,
                    image_size = image_size,
                )
            except Exception as e:
                print(f'  [SKIP] Could not build SAM dataset: {e}')
                failed_models.append(display_name)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            active_loader = DataLoader(
                sam_dataset,
                batch_size  = args.batch_size,
                shuffle     = False,
                num_workers = args.num_workers,
                pin_memory  = (device.type == 'cuda'),
            )
        else:
            active_loader = loader

        # ── Inference ─────────────────────────────────────────────────────────
        try:
            results = run_inference(model, active_loader, device,
                                    threshold=args.threshold, is_sam=is_sam)
        except torch.cuda.OutOfMemoryError:
            print('  [WARN] CUDA OOM — retrying on CPU…')
            torch.cuda.empty_cache()
            model   = model.cpu()
            results = run_inference(model, active_loader, torch.device('cpu'),
                                    threshold=args.threshold, is_sam=is_sam)
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
            out_path  = model_out_dir / location / f'{img_stem}_pred.jpg'

            try:
                save_result_image(
                    result       = res,
                    out_path     = out_path,
                    model_name   = display_name,
                    threshold    = args.threshold,
                    heatmap_cmap = args.heatmap_cmap,
                    jpeg_quality = args.jpeg_quality,
                    max_panel_w  = args.max_panel_w,
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
