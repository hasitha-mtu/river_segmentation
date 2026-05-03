from __future__ import annotations

import argparse
import csv
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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.train_unified_wandb_sam_v2_1 import get_sam2_dataset

try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print('[WARN] segment_anything not installed — SAM evaluation unavailable.')


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


class SAMTestDataset(Dataset):
    """
    Test dataset for SAM models.

    MUST mirror the preprocessing in SAMDataset (train_unified_wandb_sam_v1.py)
    exactly — using ImageNet normalisation here would silently corrupt every
    metric because the model was never exposed to those statistics.

    Preprocessing steps (identical to training):
      1. cv2 load → resize to image_size × image_size
      2. ResizeLongestSide(1024) → square 512 → 1024×1024
      3. SAM normalisation  (mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375])
      4. Zero-pad to 1024×1024

    Batch-dict keys returned:
        input_tensor  : FloatTensor [3, 1024, 1024]   — SAM-preprocessed image
        input_size    : LongTensor  [2]               — (H', W') after ResizeLongestSide
        original_size : LongTensor  [2]               — (image_size, image_size)
        mask          : FloatTensor [1, image_size, image_size]
        image_path    : str
    """

    SAM_IMG_SIZE = 1024
    _PIXEL_MEAN  = torch.tensor([123.675, 116.28,  103.53]).view(3, 1, 1)
    _PIXEL_STD   = torch.tensor([ 58.395,  57.12,   57.375]).view(3, 1, 1)

    def __init__(self, data_root: str, split: str = 'test', image_size: int = 512):
        if not SAM_AVAILABLE:
            raise RuntimeError(
                'segment_anything is required for SAM evaluation.\n'
                'pip install git+https://github.com/facebookresearch/segment-anything.git'
            )
        import cv2 as _cv2
        self._cv2       = _cv2
        self.image_size = image_size

        root     = Path(data_root)
        img_dir  = root / split / 'images'
        mask_dir = root / split / 'masks'

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

        print(f'  [SAMTestDataset] {len(self.samples)} samples in {split} split.')
        self.resize_transform = ResizeLongestSide(self.SAM_IMG_SIZE)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        # Load and resize to model's training resolution
        image = self._cv2.imread(str(img_path))
        image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        image = self._cv2.resize(
            image, (self.image_size, self.image_size),
            interpolation=self._cv2.INTER_LINEAR,
        )
        original_size = (image.shape[0], image.shape[1])   # (image_size, image_size)

        # SAM preprocessing: ResizeLongestSide then normalise + zero-pad
        resized    = self.resize_transform.apply_image(image)
        input_size = (resized.shape[0], resized.shape[1])

        img_t = torch.as_tensor(resized).permute(2, 0, 1).float()  # [3, H', W']
        img_t = (img_t - self._PIXEL_MEAN) / self._PIXEL_STD
        h, w  = img_t.shape[-2:]
        img_t = torch.nn.functional.pad(
            img_t, (0, self.SAM_IMG_SIZE - w, 0, self.SAM_IMG_SIZE - h)
        )  # [3, 1024, 1024]

        # Mask
        if mask_path:
            mask_raw = self._cv2.imread(str(mask_path), self._cv2.IMREAD_GRAYSCALE)
            mask_raw = self._cv2.resize(
                mask_raw, (self.image_size, self.image_size),
                interpolation=self._cv2.INTER_NEAREST,
            )
            mask_t = torch.from_numpy((mask_raw > 127).astype(np.float32)).unsqueeze(0)
        else:
            mask_t = torch.zeros(1, self.image_size, self.image_size)

        return {
            'input_tensor' : img_t,
            'input_size'   : torch.tensor(input_size,    dtype=torch.long),
            'original_size': torch.tensor(original_size, dtype=torch.long),
            'mask'         : mask_t,
            'image_path'   : str(img_path),
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

def build_model_from_config(config: dict, ckpt: dict, device: torch.device, ckpt_path) -> nn.Module:
    """Reconstruct the model from its training config and load weights."""
    model_name = config['model']['name']
    variant    = config['model'].get('variant', None)
    n_channels = config['model'].get('n_channels', 3)
    n_classes  = config['model'].get('n_classes',  1)
    predictor = None
    print(f'Build model_name : {model_name}')
    if model_name == 'sam':
        # The training script (train_unified_wandb_sam_v1.py) saves a raw
        # sam_model_registry[variant] object.  Its state-dict keys are:
        #   image_encoder.*, prompt_encoder.*, mask_decoder.*
        #
        # models.py's build_sam_segmentation() builds SAMEncoderDecoder whose
        # keys are sam.image_encoder.*, decoder.* — a completely different
        # structure.  Using get_model() here causes the key-mismatch crash.
        # We must rebuild via sam_model_registry to match saved keys exactly.
        sam_ckpt = (
            config.get('foundation', {})
                  .get('sam_checkpoints', {})
                  .get(variant, f'./checkpoints/sam/sam_{variant}.pth')
        )
        print(f'  [SAM] Rebuilding {variant} from: {sam_ckpt}')
        model = sam_model_registry[variant](checkpoint=sam_ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
    elif model_name == 'sam_v2_fine_tuned':
        ckpt_base = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam2'
        variant_map = {
            'sam2.1_hiera_tiny': ('configs/sam2.1/sam2.1_hiera_t.yaml', 'sam2.1_hiera_tiny.pt'),
            'sam2.1_hiera_small': ('configs/sam2.1/sam2.1_hiera_s.yaml', 'sam2.1_hiera_small.pt'),
            'sam2.1_hiera_base_plus': ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'),
        }
        cfg_file, ckpt_file = variant_map[variant]
        print(f'Build model from config file: {cfg_file}')
        # model = build_sam2(
        #     config_file=cfg_file,
        #     ckpt_path=os.path.join(ckpt_base, ckpt_file),
        #     device='cuda',
        # )
        # print(f'Build predictor for model: {model_name}')
        # predictor = SAM2ImagePredictor(model)
        # print(f'Loading weights from checkpoint: {ckpt_path}')
        # predictor = predictor.model.load_state_dict(torch.load(ckpt_path))

        # FINE_TUNED_MODEL_WEIGHTS = r'C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\experiments\sam2\sam2_sam2.1_hiera_tiny\checkpoints\best.pth'
        # sam2_checkpoint = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam2/sam2.1_hiera_tiny.pt'
        # model_cfg = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam2/sam2.1_hiera_t.yaml'

        # Build net and load weights
        model = build_sam2(config_file=cfg_file,
                           ckpt_path=os.path.join(ckpt_base, ckpt_file),
                           device="cuda")  # load model
        predictor = SAM2ImagePredictor(model)
        predictor.model.load_state_dict(ckpt['model_state_dict'])

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
    return model, predictor


def model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_display_name(config: dict) -> str:
    """Human-readable model label used in tables and filenames."""
    model_name = config['model']['name']
    variant    = config['model'].get('variant', None)
    return f'{model_name}_{variant}' if variant else model_name


def model_family(config: dict) -> str:
    """Assign each model to a display family for table grouping."""
    name = config['model']['name']
    if name in ('unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam'):
        return 'CNN Baseline'
    if name in ('segformer', 'swin_unet'):
        return 'Transformer'
    if name in ('convnext_upernet', 'hrnet_ocr'):
        return 'Hybrid SOTA'
    if name in ('sam', 'sam_fpn', 'sam_v2_fine_tuned', 'dinov2'):
        return 'Foundation'
    return 'Other'


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_sam_eval(
    model  : nn.Module,
    batch  : dict,
    device : torch.device,
) -> torch.Tensor:
    """
    Standalone SAM forward pass for evaluation.

    Mirrors _forward_sam() in train_unified_wandb_sam_v1.py exactly so that
    the inference path at eval time is identical to training.

    Why the per-image loop is required
    ------------------------------------
    SAM's mask_decoder.predict_masks() calls:
        src = torch.repeat_interleave(image_embedding, tokens.shape[0], dim=0)
    When no spatial prompts are given, tokens.shape[0] == 1 (the iou_token).
    Passing a batched sparse_emb [B, N, C] makes tokens.shape[0] = B, which
    produces a [B*B, ...] intermediate and crashes.

    Fix: encode all images together (the expensive ViT), then loop
    prompt_encoder + mask_decoder per image (trivially cheap).

    Returns logits [B, 1, H, W]  (no sigmoid — evaluate_model handles that).
    """
    input_tensor  = batch['input_tensor'].to(device)   # [B, 3, 1024, 1024]
    B             = input_tensor.shape[0]
    input_size    = tuple(int(x) for x in batch['input_size'][0])
    original_size = tuple(int(x) for x in batch['original_size'][0])

    # Batched encode (frozen image encoder — the expensive step)
    with torch.no_grad():
        image_embeddings = model.image_encoder(input_tensor)  # [B, 256, 64, 64]

    # Per-image decode
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
        )  # [1, 1, original_size[0], original_size[1]]
        upscaled_list.append(upscaled_i)

    return torch.cat(upscaled_list, dim=0)   # [B, 1, H, W] logits

@torch.no_grad()
def _forward_sam2_eval(
    model  : nn.Module,
    batch  : dict,
    device : torch.device,
    config : dict,
    predictor  : SAM2ImagePredictor = None):
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

    with torch.amp.autocast('cuda'):
        predictor.set_image_batch([item['image'] for item in valid_items])

    prd_logits_list = []
    gt_masks_list = []

    with torch.amp.autocast('cuda'):
        for i, item in enumerate(valid_items):
            input_point = item['points']  # (N, 2) numpy
            num_masks = item['labels_size']
            input_label = np.ones((num_masks, 1), dtype=np.int32)

            # Prepare prompt tensors.
            # img_idx=i tells _prep_prompts which image's (H, W) to use
            # when normalising point coordinates to [0, 1].
            mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                input_point, input_label,
                box=None, mask_logits=None,
                normalize_coords=True,
                img_idx=i,
            )

            if unnorm_coords is None or unnorm_coords.shape[0] == 0:
                continue

            # Slice this sample's features out of the batch.
            image_embed = predictor._features['image_embed'][i].unsqueeze(0)  # [1, C, H, W]
            high_res_feats = [
                feat_level[i].unsqueeze(0)  # [1, C, H, W]
                for feat_level in predictor._features['high_res_feats']
            ]

            # Prompt encoder: points → sparse & dense embeddings.
            sparse_emb, dense_emb = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=None,
                masks=None,
            )

            # Mask decoder: embeddings → low-res logit masks + IoU scores.
            batched_mode = unnorm_coords.shape[0] > 1
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_feats,
            )

            # Upsample low-res logits back to original image resolution.
            # Result is still LOGITS (no sigmoid).
            prd_logits = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[i]
            )

            # Take the first mask channel (index 0 of multimask output).
            prd_logits_list.append(prd_logits[:, 0])  # [1, H, W]  logits
            gt_mask = torch.as_tensor(
                item['mask'].astype(np.float32), device=device
            )
            gt_masks_list.append(gt_mask)

    if not prd_logits_list:
        return None, None

    return torch.cat(prd_logits_list, dim=0), torch.stack(gt_masks_list)   # [B, 1, H, W] logits

@torch.no_grad()
def evaluate_model(
    model      : nn.Module,
    loader     : DataLoader,
    device     : torch.device,
    config     : dict,
    threshold  : float = 0.5,
    predictor  : SAM2ImagePredictor = None,
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
    model_name = config['model']['name']
    model.eval()

    for batch in tqdm(loader, desc='  Inference', leave=False, ncols=90):
        t0 = time.perf_counter()
        if model_name == 'sam':
            # SAM needs its own forward: batched encoder + per-image decoder.
            # Using model(batch['image']) would call the wrong class entirely.
            outputs = _forward_sam_eval(model, batch, device)
            per_image, acc = _per_image_metrics(per_image, acc, timing, outputs, batch, device, t0, threshold)
        elif model_name == 'sam_v2_fine_tuned':
            outputs, masks = _forward_sam2_eval(model, batch, device, config, predictor)
            per_image, acc = _per_image_metrics_sam2(per_image, acc, timing, outputs, masks, batch, device, t0, threshold)
        else:
            images  = batch['image'].to(device)
            outputs = model(images)
            per_image, acc = _per_image_metrics(per_image, acc, timing, outputs, batch, device, t0, threshold)
    agg = acc.compute()
    agg['inference_ms'] = round(np.mean(timing) * 1000, 2)
    return agg, per_image

def _per_image_metrics_sam2(per_image, acc, timing, outputs, masks, batch, device, t0, threshold):
    # ── Metrics — apply sigmoid ONCE on logits ─────────────────
    probs = torch.sigmoid(outputs)  # logits → [0,1]

    t1 = time.perf_counter()
    timing.append((t1 - t0) / masks.shape[0])  # per-image seconds

    acc.update(probs.cpu(), masks.cpu())

    for i in range(len(batch)):
        p = probs[i].cpu()
        m = masks[i].cpu()
        batch_item = batch[i]
        eps = 1e-7
        tp = (p > threshold).float() * m
        fp = (p > threshold).float() * (1 - m)
        fn = (1 - (p > threshold).float()) * m
        tp_sum = tp.sum().item()
        fp_sum = fp.sum().item()
        fn_sum = fn.sum().item()
        img_dice = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + eps)
        img_iou = tp_sum / (tp_sum + fp_sum + fn_sum + eps)
        per_image.append({
            'image': os.path.basename(batch_item['image_path']),
            'dice': round(img_dice, 4),
            'iou': round(img_iou, 4),
            'pred_prob': round(p.mean().item(), 4),
            'gt_ratio': round(m.mean().item(), 4),
        })
    return per_image, acc

def _per_image_metrics(per_image, acc, timing, outputs, batch, device, t0, threshold):
    # Some models return a tuple (logits, aux); take the main output
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    # Ensure correct shape: [B, 1, H, W]
    if outputs.dim() == 3:
        outputs = outputs.unsqueeze(1)

    probs = torch.sigmoid(outputs)

    t1 = time.perf_counter()
    masks = batch['mask'].to(device)
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
        img_iou = tp_sum / (tp_sum + fp_sum + fn_sum + eps)
        per_image.append({
            'image': os.path.basename(img_path),
            'dice': round(img_dice, 4),
            'iou': round(img_iou, 4),
            'pred_prob': round(p.mean().item(), 4),
            'gt_ratio': round(m.mean().item(), 4),
        })
    return per_image, acc

# ─────────────────────────────────────────────────────────────────────────────
# Prediction visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction_overlays(
    model         : nn.Module,
    loader        : DataLoader,
    device        : torch.device,
    out_dir       : Path,
    is_sam         : bool = False,
    n_samples     : int = 8,
    threshold     : float = 0.5,
):
    """
    Save side-by-side overlay images: [Image | Ground Truth | Prediction | Overlay].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for batch in loader:
            if saved >= n_samples:
                break

            masks = batch['mask'].to(device)

            if is_sam:
                outputs      = _forward_sam_eval(model, batch, device)
                imgs_for_vis = None   # SAM: load from image_path (see below)
            else:
                outputs = model(batch['image'].to(device))
                imgs_for_vis = batch['image']

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.dim() == 3:
                outputs = outputs.unsqueeze(1)
            probs = torch.sigmoid(outputs).cpu()
            preds = (probs > threshold).float()

            img_paths = batch.get('image_path', [''] * masks.shape[0])
            for i in range(min(len(img_paths), n_samples - saved)):
                if is_sam:
                    # ── SAM: load the original image directly from disk ───────
                    # Do NOT use batch['input_tensor'] for visualisation.
                    #
                    # Why: SAMTestDataset applies ResizeLongestSide(1024) to the
                    # image_size×image_size source image before storing it in
                    # input_tensor.  For a square 512×512 source, this scales
                    # both sides 2× to 1024×1024.  Cropping input_tensor to
                    # [:orig_h, :orig_w] = [:512, :512] then yields only the
                    # top-left quarter of the 2× upscaled image — appearing
                    # "zoomed in" relative to the 512×512 mask/prediction.
                    #
                    # The correct image for overlay is the 512×512 source image,
                    # which is what mask and prediction both reference.
                    orig_h = int(batch['original_size'][i][0])
                    orig_w = int(batch['original_size'][i][1])
                    img_pil = Image.open(img_paths[i]).convert('RGB').resize(
                        (orig_w, orig_h), Image.BILINEAR
                    )
                    img_np = np.array(img_pil).astype(np.float32) / 255.0
                else:
                    img_np = imgs_for_vis[i].permute(1, 2, 0).numpy()
                    img_np = (img_np * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)

                mask_np = masks[i, 0].cpu().numpy()
                pred_np = preds[i, 0].numpy()

                def to_uint8(arr):
                    return (arr * 255).astype(np.uint8)

                img_u8    = to_uint8(img_np)
                gt_u8     = np.stack([to_uint8(mask_np)] * 3, axis=2)
                pred_u8   = np.stack([to_uint8(pred_np)] * 3, axis=2)

                overlay   = img_u8.copy()
                gt_mask   = mask_np > 0.5
                pred_mask = pred_np > 0.5
                overlay[gt_mask & pred_mask]   = [255, 255,   0]   # yellow = TP
                overlay[gt_mask & ~pred_mask]  = [  0, 200,   0]   # green  = FN
                overlay[~gt_mask & pred_mask]  = [200,   0,   0]   # red    = FP

                strip = np.concatenate([img_u8, gt_u8, pred_u8, overlay], axis=1)
                fname = f'{saved:04d}_{Path(img_paths[i]).stem}.png'
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
    p.add_argument('--experiments_dir', default=r'C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\experiments',
                   help='Root directory of saved experiment checkpoints.')
    p.add_argument('--data_root', default=r'C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_512_resized\sequential',
                   help='Root of the standard (resized) dataset.')
    p.add_argument('--output_dir', default=r'C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\results',
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
        is_sam          = (config['model']['name'] == 'sam')

        print(f'Model : {display_name}  [{family}]')

        # ── Build dataloader ──────────────────────────────────────────────────
        image_size = config['data'].get('image_size', 512)
        try:
            if config['model']['name'] == 'sam':
                # SAM was trained with ResizeLongestSide(1024) + SAM normalisation.
                # Using standard TestDataset (ImageNet norm) here would silently
                # corrupt all metrics.
                data_root = config['data'].get('data_root', args.data_root)
                test_ds   = SAMTestDataset(
                    data_root  = data_root,
                    split      = 'test',
                    image_size = image_size,
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == 'cuda'),
                )
            elif config['model']['name'] == 'sam_v2_fine_tuned':
                data_root = config['data'].get('data_root', args.data_root)
                test_loader = get_sam2_dataset(data_root, args.batch_size)
            else:
                data_root = config['data'].get('data_root', args.data_root)
                test_ds   = TestDataset(
                    data_root  = data_root,
                    split      = 'test',
                    image_size = image_size,
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == 'cuda'),
                )
        except FileNotFoundError as e:
            print(f'  [SKIP] Dataset not found: {e}')
            failed_models.append(display_name)
            continue

        # ── Build model ───────────────────────────────────────────────────────
        model, predictor = build_model_from_config(config, ckpt, device, ckpt_path)
        # try:
        #     model, predictor = build_model_from_config(config, ckpt, device, ckpt_path)
        # except Exception as e:
        #     print(f'  [SKIP] Model construction failed: {e}')
        #     failed_models.append(display_name)
        #     continue

        params = model_param_count(model)
        print(f'  Parameters: {params / 1e6:.2f} M')

        # ── Inference ─────────────────────────────────────────────────────────
        try:
            agg_metrics, per_img = evaluate_model(
                model           = model,
                loader          = test_loader,
                device          = device,
                config          = config,
                threshold       = args.threshold,
                predictor       = predictor
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
                config          = config,
                threshold       = args.threshold,
                predictor       = predictor
            )
            model = model.to(device)

        # ── Save prediction overlays ──────────────────────────────────────────
        pred_dir = output_dir / 'predictions' / display_name
        print(f'  Saving {args.num_pred_samples} prediction overlays → {pred_dir}')
        try:
            save_prediction_overlays(
                model=model,
                loader=test_loader,
                device=device,
                out_dir=pred_dir,
                is_sam=is_sam,
                n_samples=args.num_pred_samples,
                threshold=args.threshold,
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
