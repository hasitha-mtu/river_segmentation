"""
predict_masks.py
================
Generate and save predicted binary masks (and optionally probability maps)
for every test-set image across all trained segmentation models.

What this script does
---------------------
1. Auto-discovers every ``best.pth`` checkpoint under ``experiments/``.
2. Reconstructs each model from the config embedded in the checkpoint.
3. Runs inference on every image in the test split (default: all 64).
4. Saves per-model, per-image outputs:
      results/masks/<model_name>/<image_stem>.png   — binary mask  (0 / 255)
      results/probs/<model_name>/<image_stem>.npy   — float32 probability
                                                       (only with --save_probs)

Output masks are saved at the model's inference resolution (512×512 by default).
Pass ``--original_size`` to upsample each mask back to its source image
dimensions using nearest-neighbour interpolation.

Usage
-----
# Predict masks for all models, save at 512×512:
    python predict_masks.py

# Also save probability maps and resize masks to original image dimensions:
    python predict_masks.py \\
        --experiments_dir ./experiments \\
        --data_root       ./dataset/processed_512_resized \\
        --output_dir      ./results \\
        --save_probs \\
        --original_size \\
        --threshold 0.5

Notes
-----
* SAM models use ResizeLongestSide(1024) + SAM normalisation — exactly as in
  training.  Standard ImageNet normalisation is never applied to SAM inputs.
* SAM-FPN models (model_name == 'sam_fpn') use the standard TestDataset
  (ImageNet normalisation) because their FPN decoder expects dense feature maps,
  not the SAM prompt pipeline.
* Models are processed sequentially; VRAM is freed between each model.
* If a model fails (OOM, missing checkpoint config, etc.) it is skipped and
  reported at the end — other models continue.
* The script is fully self-contained; it does not import evaluate_models.py.
"""

from __future__ import annotations

import argparse
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


# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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
    print('[WARN] segment_anything not installed — SAM mask prediction unavailable.')


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    """
    Standard test dataset with ImageNet normalisation.
    Used by all models except SAM (prompt-decoder variant).
    Returns original image dimensions so masks can be resized back if needed.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, data_root: str, split: str = 'test', image_size: int = 512):
        root     = Path(data_root)
        img_dir  = root / split / 'images'
        mask_dir = root / split / 'masks'

        if not img_dir.exists():
            raise FileNotFoundError(
                f'Image directory not found: {img_dir}\n'
                f'Expected layout: {data_root}/{split}/images/'
            )

        self.samples: list[tuple[Path, Optional[Path]]] = []
        for p in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')):
            mp = mask_dir / f'{p.stem}.png'
            if not mp.exists():
                mp = mask_dir / f'{p.stem}.jpg'
            self.samples.append((p, mp if mp.exists() else None))

        if not self.samples:
            raise ValueError(f'No images found in {img_dir}')

        print(f'  [TestDataset/{split}] {len(self.samples)} images.')

        self.image_size = image_size
        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, _ = self.samples[idx]
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size   # PIL size is (W, H)

        return {
            'image'       : self.img_tf(pil_img),
            'image_path'  : str(img_path),
            'orig_h'      : orig_h,
            'orig_w'      : orig_w,
        }


class SAMTestDataset(Dataset):
    """
    Test dataset for vanilla SAM (prompt-decoder) models.
    Preprocessing must exactly mirror SAMDataset in train_unified_wandb_sam.py:
      cv2 load → resize to image_size × image_size
      → ResizeLongestSide(1024)
      → SAM pixel normalisation (0–255 space)
      → zero-pad to 1024×1024

    Using ImageNet normalisation here would silently corrupt all predictions.
    """

    SAM_IMG_SIZE = 1024
    _PIXEL_MEAN  = torch.tensor([123.675, 116.28,  103.53]).view(3, 1, 1)
    _PIXEL_STD   = torch.tensor([ 58.395,  57.12,   57.375]).view(3, 1, 1)

    def __init__(self, data_root: str, split: str = 'test', image_size: int = 512):
        if not SAM_AVAILABLE:
            raise RuntimeError(
                'segment_anything is required for SAM prediction.\n'
                'pip install git+https://github.com/facebookresearch/segment-anything.git'
            )
        if not CV2_AVAILABLE:
            raise RuntimeError('cv2 is required for SAM preprocessing. pip install opencv-python')

        self.image_size      = image_size
        self.resize_transform = ResizeLongestSide(self.SAM_IMG_SIZE)

        root    = Path(data_root)
        img_dir = root / split / 'images'
        if not img_dir.exists():
            raise FileNotFoundError(f'Image directory not found: {img_dir}')

        self.samples = []
        for p in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')):
            self.samples.append(p)

        if not self.samples:
            raise ValueError(f'No images found in {img_dir}')

        print(f'  [SAMTestDataset/{split}] {len(self.samples)} images.')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h_src, orig_w_src = image.shape[:2]

        # Resize to model inference size (512×512 matches training)
        image = cv2.resize(
            image, (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        original_size = (image.shape[0], image.shape[1])  # (image_size, image_size)

        # SAM-specific preprocessing
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
            'orig_h'       : orig_h_src,
            'orig_w'       : orig_w_src,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def discover_checkpoints(experiments_dir: str) -> list[Path]:
    ckpts = sorted(Path(experiments_dir).rglob('checkpoints/best.pth'))
    print(f'\nDiscovered {len(ckpts)} checkpoint(s) under: {experiments_dir}')
    for p in ckpts:
        print(f'  {p.relative_to(experiments_dir)}')
    return ckpts


def load_checkpoint_config(ckpt_path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'config' not in ckpt:
        raise KeyError(f'Checkpoint {ckpt_path} has no embedded "config" key.')
    return ckpt['config'], ckpt


def model_display_name(config: dict) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict, ckpt: dict, device: torch.device) -> nn.Module:
    """
    Reconstruct model from training config and load weights.
    SAM (prompt-decoder) is rebuilt via sam_model_registry to match saved key
    structure exactly — using get_model() would cause a key-mismatch crash.
    """
    model_name = config['model']['name']
    variant    = config['model'].get('variant', None)
    n_channels = config['model'].get('n_channels', 3)
    n_classes  = config['model'].get('n_classes',  1)

    if model_name == 'sam':
        if not SAM_AVAILABLE:
            raise RuntimeError('segment_anything not installed.')
        sam_ckpt = (
            config.get('foundation', {})
                  .get('sam_checkpoints', {})
                  .get(variant, f'./checkpoints/sam/sam_{variant}.pth')
        )
        print(f'  [SAM] Rebuilding {variant} from: {sam_ckpt}')
        model = sam_model_registry[variant](checkpoint=sam_ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model = get_model(
            model_name = model_name,
            variant    = variant,
            n_channels = n_channels,
            n_classes  = n_classes,
        )
        model.load_state_dict(ckpt['model_state_dict'])

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# SAM forward pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_sam(
    model  : nn.Module,
    batch  : dict,
    device : torch.device,
) -> torch.Tensor:
    """
    Batched-encoder + per-image-decoder SAM forward pass.

    SAM's mask_decoder.predict_masks() calls:
        src = torch.repeat_interleave(image_embedding, tokens.shape[0], dim=0)
    When no spatial prompts are given, tokens.shape[0] == 1.  Passing a batched
    sparse_emb [B, N, C] makes tokens.shape[0] = B → [B*B, …] intermediate and
    crashes.  Fix: encode all images together (the expensive ViT step), then loop
    the cheap prompt_encoder + mask_decoder per image.

    Returns sigmoid probabilities [B, 1, H, W] at original_size resolution.
    """
    input_tensor  = batch['input_tensor'].to(device)     # [B, 3, 1024, 1024]
    B             = input_tensor.shape[0]
    input_size    = tuple(int(x) for x in batch['input_size'][0])
    original_size = tuple(int(x) for x in batch['original_size'][0])

    image_embeddings = model.image_encoder(input_tensor)  # [B, 256, 64, 64]

    probs_list = []
    for i in range(B):
        emb_i = image_embeddings[i].unsqueeze(0)          # [1, 256, 64, 64]

        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=None, masks=None,
        )

        low_res_i, _ = model.mask_decoder(
            image_embeddings         = emb_i,
            image_pe                 = model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_emb,
            dense_prompt_embeddings  = dense_emb,
            multimask_output         = False,
        )  # [1, 1, 256, 256] logits

        upscaled_i = model.postprocess_masks(
            low_res_i, input_size, original_size,
        )  # [1, 1, original_size[0], original_size[1]]

        probs_list.append(torch.sigmoid(upscaled_i))

    return torch.cat(probs_list, dim=0)  # [B, 1, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Mask prediction and saving
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_and_save(
    model        : nn.Module,
    loader       : DataLoader,
    device       : torch.device,
    mask_dir     : Path,
    prob_dir     : Optional[Path],
    is_sam       : bool,
    threshold    : float,
    original_size: bool,
):
    """
    Run inference over the full loader and write outputs to disk.

    Output layout
    -------------
    mask_dir/<image_stem>.png     — 8-bit grayscale binary mask (0 = background,
                                    255 = water), at model resolution or original
                                    image resolution if original_size=True.
    prob_dir/<image_stem>.npy     — float32 probability map [H, W], same
                                    resolution as the mask. Only written when
                                    prob_dir is not None.
    """
    mask_dir.mkdir(parents=True, exist_ok=True)
    if prob_dir is not None:
        prob_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for batch in tqdm(loader, desc='  Predicting', leave=False, ncols=90):

        if is_sam:
            probs = _forward_sam(model, batch, device)   # [B,1,H,W] already sigmoid
        else:
            images  = batch['image'].to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.dim() == 3:
                outputs = outputs.unsqueeze(1)
            probs = torch.sigmoid(outputs)               # [B,1,H,W]

        probs = probs.cpu()
        preds = (probs > threshold).float()              # [B,1,H,W] binary

        img_paths = batch['image_path']
        orig_hs   = batch.get('orig_h', [None] * len(img_paths))
        orig_ws   = batch.get('orig_w', [None] * len(img_paths))

        for i in range(len(img_paths)):
            stem = Path(img_paths[i]).stem

            prob_np = probs[i, 0].numpy()   # [H, W]  float32
            pred_np = preds[i, 0].numpy()   # [H, W]  0.0 / 1.0

            # ── Optionally resize to original image resolution ────────────────
            if original_size and orig_hs[i] is not None:
                oh = int(orig_hs[i])
                ow = int(orig_ws[i])
                if (oh, ow) != pred_np.shape:
                    prob_pil = Image.fromarray(prob_np)
                    prob_np  = np.array(
                        prob_pil.resize((ow, oh), Image.BILINEAR)
                    )
                    pred_np = (prob_np > threshold).astype(np.float32)

            # ── Binary mask PNG (0 / 255) ─────────────────────────────────────
            mask_u8 = (pred_np * 255).astype(np.uint8)
            Image.fromarray(mask_u8, mode='L').save(mask_dir / f'{stem}.png')

            # ── Probability map .npy (optional) ──────────────────────────────
            if prob_dir is not None:
                np.save(prob_dir / f'{stem}.npy', prob_np.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Predict binary masks for every test image across all trained models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Examples
            --------
            # Predict masks at model resolution (512x512, default):
                python predict_masks.py

            # Save probability maps and resize masks to original image dimensions:
                python predict_masks.py \\
                    --experiments_dir ./experiments \\
                    --data_root ./dataset/processed_512_resized \\
                    --output_dir ./results \\
                    --save_probs \\
                    --original_size \\
                    --threshold 0.5
        '''),
    )
    p.add_argument('--experiments_dir', default='./experiments',
                   help='Root directory containing experiment checkpoints.')
    p.add_argument('--data_root', default='./dataset/processed_512_resized',
                   help='Dataset root (contains test/images/ and test/masks/).')
    p.add_argument('--output_dir', default='./results',
                   help='Root output directory. Masks written to <output_dir>/masks/<model>/')
    p.add_argument('--split', default='test',
                   help='Dataset split to predict on (default: test).')
    p.add_argument('--batch_size', type=int, default=4,
                   help='Inference batch size (default: 4).')
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader worker processes (default: 0 for Windows/WSL2).')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Sigmoid threshold for binarising the probability map (default: 0.5).')
    p.add_argument('--save_probs', action='store_true',
                   help='Also save float32 probability maps as .npy files.')
    p.add_argument('--original_size', action='store_true',
                   help='Resize output masks to each image\'s original (source) dimensions.')
    p.add_argument('--no_cuda', action='store_true',
                   help='Force CPU inference.')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cpu') if args.no_cuda or not torch.cuda.is_available() \
             else torch.device('cuda')

    print('\n' + '=' * 70)
    print('UAV RIVER SEGMENTATION — MASK PREDICTION')
    print('=' * 70)
    print(f'Device          : {device}')
    print(f'Experiments dir : {args.experiments_dir}')
    print(f'Data root       : {args.data_root}')
    print(f'Output dir      : {args.output_dir}')
    print(f'Split           : {args.split}')
    print(f'Threshold       : {args.threshold}')
    print(f'Save probs      : {args.save_probs}')
    print(f'Original size   : {args.original_size}')

    # ── Discover checkpoints ──────────────────────────────────────────────────
    ckpt_paths = discover_checkpoints(args.experiments_dir)
    if not ckpt_paths:
        print('\n[ERROR] No best.pth checkpoints found. Check --experiments_dir.')
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_models: list[str] = []
    t_start = time.perf_counter()

    for ckpt_path in ckpt_paths:
        print(f'\n{"─" * 70}')
        print(f'Checkpoint : {ckpt_path}')

        # ── Load config ───────────────────────────────────────────────────────
        try:
            config, ckpt = load_checkpoint_config(ckpt_path)
        except Exception as e:
            print(f'  [SKIP] Could not load config: {e}')
            failed_models.append(str(ckpt_path))
            continue

        display_name = model_display_name(config)
        model_name   = config['model']['name']
        is_sam       = (model_name == 'sam')        # prompt-decoder SAM only
        image_size   = config['data'].get('image_size', 512)

        print(f'Model      : {display_name}')

        # ── Skip if already done ──────────────────────────────────────────────
        mask_out_dir = output_dir / 'masks' / display_name
        if mask_out_dir.exists() and any(mask_out_dir.glob('*.png')):
            print(f'  [SKIP] Masks already exist in {mask_out_dir} — delete to rerun.')
            continue

        # ── Build dataset ─────────────────────────────────────────────────────
        try:
            data_root = config['data'].get('data_root', args.data_root)

            if is_sam:
                # SAM (prompt-decoder) requires its own preprocessing pipeline.
                dataset = SAMTestDataset(
                    data_root  = data_root,
                    split      = args.split,
                    image_size = image_size,
                )
            else:
                # All other models (CNN, Hybrid, Transformer, SAM-FPN, DINOv2)
                # use standard ImageNet normalisation.
                dataset = TestDataset(
                    data_root  = data_root,
                    split      = args.split,
                    image_size = image_size,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f'  [SKIP] Dataset error: {e}')
            failed_models.append(display_name)
            continue

        loader = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            pin_memory  = (device.type == 'cuda'),
        )

        # ── Build model ───────────────────────────────────────────────────────
        try:
            model = build_model(config, ckpt, device)
        except Exception as e:
            print(f'  [SKIP] Model construction failed: {e}')
            failed_models.append(display_name)
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Parameters : {n_params / 1e6:.2f} M')

        # ── Predict and save ──────────────────────────────────────────────────
        prob_out_dir = (output_dir / 'probs' / display_name) if args.save_probs else None

        try:
            predict_and_save(
                model         = model,
                loader        = loader,
                device        = device,
                mask_dir      = mask_out_dir,
                prob_dir      = prob_out_dir,
                is_sam        = is_sam,
                threshold     = args.threshold,
                original_size = args.original_size,
            )
        except torch.cuda.OutOfMemoryError:
            print('  [WARN] CUDA OOM — retrying on CPU …')
            torch.cuda.empty_cache()
            model = model.cpu()
            predict_and_save(
                model         = model,
                loader        = loader,
                device        = torch.device('cpu'),
                mask_dir      = mask_out_dir,
                prob_dir      = prob_out_dir,
                is_sam        = is_sam,
                threshold     = args.threshold,
                original_size = args.original_size,
            )
        except Exception as e:
            print(f'  [SKIP] Inference failed: {e}')
            failed_models.append(display_name)
            # Remove any partial output so the skip-check above works correctly
            import shutil
            if mask_out_dir.exists():
                shutil.rmtree(mask_out_dir)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        n_masks = len(list(mask_out_dir.glob('*.png')))
        print(f'  Saved {n_masks} masks → {mask_out_dir}')
        if prob_out_dir:
            print(f'  Saved {n_masks} prob maps → {prob_out_dir}')

        # ── Free VRAM ─────────────────────────────────────────────────────────
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print('\n' + '=' * 70)
    print(f'Prediction complete in {elapsed:.1f} s')
    print(f'Output directory: {output_dir / "masks"}')

    if failed_models:
        print(f'\n[WARNING] The following models were skipped ({len(failed_models)}):')
        for m in failed_models:
            print(f'  {m}')

    # Summarise output structure
    masks_root = output_dir / 'masks'
    if masks_root.exists():
        model_dirs = sorted(masks_root.iterdir())
        print(f'\nOutput summary ({len(model_dirs)} model(s)):')
        for d in model_dirs:
            n = len(list(d.glob('*.png')))
            print(f'  {d.name:<40} {n:>3} masks')

    print('=' * 70)


if __name__ == '__main__':
    main()
