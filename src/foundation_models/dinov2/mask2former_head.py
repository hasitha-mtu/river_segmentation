"""
mask2former_head.py
===================
DINOv2 + Mask2Former binary segmentation using HuggingFace Transformers.

No Detectron2. No custom CUDA compilation. Pure PyTorch + HF transformers.

Architecture
------------
  DINOv2 ViT backbone  (frozen, pretrained from HF)
      │  4 feature levels via out_indices — all at stride 14 (ViT patch size)
      ▼
  Mask2Former Pixel Decoder  (FPN-style lateral upsampling, pure PyTorch)
      │  multi-scale features + high-res mask features
      ▼
  Mask2Former Transformer Decoder  (9-layer masked cross-attention)
      │  N=20 learnable queries → class logits + mask embeddings
      ▼
  Bipartite matching loss  (Hungarian, with auxiliary losses per decoder layer)
      +
  Binary segmentation logits  (B, 1, H, W)

Loss (training only)
--------------------
  HF Mask2Former computes the full Hungarian-matching loss internally:
    L = class_CE  +  mask_BCE  +  mask_Dice
  Summed across all 9 decoder layers (layer 9 × 1.0, layers 1-8 × 0.5).
  This is returned as a scalar tensor from forward() and handled by a
  specialised _forward_mask2former() path in UnifiedTrainer.

Supported variants
------------------
  'vit_s' → facebook/dinov2-small  (384-dim,  12 blocks)
  'vit_b' → facebook/dinov2-base   (768-dim,  12 blocks)
  'vit_l' → facebook/dinov2-large  (1024-dim, 24 blocks)

Input / output
--------------
  Input  : (B, 3, H, W)  RGB, normalised, any size divisible by 14
  Output : (B, 1, H, W)  raw logits — apply sigmoid + threshold for mask

Integration
-----------
  model_name == 'dinov2_Mask2Former' in models/get_model()
  Uses _forward_mask2former() + _compute_loss() override in UnifiedTrainer.
  See TRAINER PATCH section at the bottom of this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    Dinov2Config,
    Dinov2Model,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Variant registry
# ─────────────────────────────────────────────────────────────────────────────

_VARIANTS: dict[str, dict] = {
    'vit_s': {
        'hf_backbone': 'facebook/dinov2-small',
        'embed_dim'  : 384,
        'num_blocks' : 12,
        # Last 4 blocks: good balance of localisation vs. semantics
        'out_indices': (-4, -3, -2, -1),
    },
    'vit_b': {
        'hf_backbone': 'facebook/dinov2-base',
        'embed_dim'  : 768,
        'num_blocks' : 12,
        'out_indices': (-4, -3, -2, -1),
    },
    'vit_l': {
        'hf_backbone': 'facebook/dinov2-large',
        'embed_dim'  : 1024,
        'num_blocks' : 24,
        # Spread across deeper half of the 24-block ViT-L
        'out_indices': (-4, -3, -2, -1),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2Mask2FormerSegmentation(nn.Module):
    """
    Binary semantic segmentation: DINOv2 (frozen) + Mask2Former decoder (trainable).

    Designed to slot into UnifiedTrainer via the _forward_mask2former() path.
    The forward() method has the signature:

        seg_logits, loss = model(pixel_values)                     # eval
        seg_logits, loss = model(pixel_values, mask_labels,        # train
                                 class_labels)

    seg_logits : (B, 1, H, W)  — raw logits, compatible with trainer metrics
    loss        : scalar Tensor — HF Mask2Former Hungarian-matching loss,
                  or None during eval / inference
    """

    def __init__(
        self,
        num_classes     : int = 1,
        encoder_variant : str = 'vit_b',
        num_queries     : int = 20,
    ):
        super().__init__()

        if encoder_variant not in _VARIANTS:
            raise ValueError(
                f'Unknown variant {encoder_variant!r}. '
                f'Choose from: {sorted(_VARIANTS)}'
            )

        vcfg = _VARIANTS[encoder_variant]

        # ── 1. DINOv2 backbone config ────────────────────────────────────────
        #
        # out_indices:           4 feature levels from the last 4 ViT blocks.
        #                        All levels are at stride 14 (ViT patch size).
        # reshape_hidden_states: return (B, C, H/14, W/14) — spatial maps,
        #                        not raw patch sequences. Required by M2F FPN.
        backbone_cfg = Dinov2Config.from_pretrained(
            vcfg['hf_backbone'],
            out_indices           = list(vcfg['out_indices']),
            reshape_hidden_states = True,
        )

        # ── 2. Mask2Former config ────────────────────────────────────────────
        #
        # feature_strides = [14,14,14,14]:
        #   DINOv2 ViT has no spatial downsampling between blocks — all 4
        #   selected feature maps are at the same stride-14 resolution.
        #   The M2F pixel decoder projects + upsamples from there.
        #   This differs from the original Swin/ResNet backbone where strides
        #   are [4,8,16,32], but the FPN still learns meaningful multi-head
        #   feature fusion from the depth-varying DINOv2 activations.
        #
        # num_queries = 20:
        #   100 queries (COCO default) is wasteful for binary segmentation.
        #   20 gives sufficient capacity for a single foreground class.
        #
        # use_auxiliary_loss = True:
        #   Computes loss at every decoder layer — critical for convergence
        #   on small datasets (421 images). Each layer's loss backpropagates
        #   directly to its parameters; intermediate layers don't rely solely
        #   on gradients flowing from the final layer.
        m2f_cfg = Mask2FormerConfig(
            backbone_config         = backbone_cfg,
            num_labels              = num_classes,
            num_queries             = num_queries,
            feature_size            = 256,
            mask_feature_size       = 256,
            hidden_dim              = 256,
            encoder_layers          = 6,
            decoder_layers          = 9,
            num_attention_heads     = 8,
            dropout                 = 0.0,
            dim_feedforward         = 2048,
            pre_norm                = False,
            use_auxiliary_loss      = True,
            # Hungarian matching cost weights
            class_weight            = 2.0,
            mask_weight             = 5.0,
            dice_weight             = 5.0,
            # Point sampling for mask loss (default 12544 ≈ 112×112)
            train_num_points        = 12544,
            oversample_ratio        = 3.0,
            importance_sample_ratio = 0.75,
            # All DINOv2 feature levels are at stride 14
            feature_strides         = [14, 14, 14, 14],
        )

        # ── 3. Instantiate full M2F model ────────────────────────────────────
        self.model = Mask2FormerForUniversalSegmentation(m2f_cfg)

        # ── 4. Load pretrained DINOv2 backbone weights ───────────────────────
        self._load_pretrained_backbone(vcfg['hf_backbone'])

        # ── 5. Freeze backbone ───────────────────────────────────────────────
        encoder = self.model.model.pixel_level_module.encoder
        for param in encoder.parameters():
            param.requires_grad_(False)

        # ── Bookkeeping ──────────────────────────────────────────────────────
        self.num_classes = num_classes
        self.variant     = encoder_variant

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f'[DINOv2Mask2Former] variant={encoder_variant} '
            f'| embed={vcfg["embed_dim"]} '
            f'| queries={num_queries} '
            f'| params={total:,} (trainable={trainable:,})'
        )

    # ── Weight loading ────────────────────────────────────────────────────────

    def _load_pretrained_backbone(self, hf_name: str) -> None:
        """
        Copy pretrained DINOv2 weights into the M2F backbone.

        Mask2FormerForUniversalSegmentation initialises the backbone randomly
        when constructed from a config dict (not from_pretrained). We manually
        load the matching keys from the standalone Dinov2Model checkpoint.

        Missing keys are expected for the 'feature_map_*' output projection
        layers that Mask2Former adds on top of the vanilla DINOv2 encoder.
        """
        print(f'  Loading pretrained DINOv2 from {hf_name} …')
        pretrained = Dinov2Model.from_pretrained(hf_name)
        encoder    = self.model.model.pixel_level_module.encoder

        missing, unexpected = encoder.load_state_dict(
            pretrained.state_dict(), strict=False
        )
        # Unexpected = keys in pretrained not in M2F backbone — should be empty.
        # Missing    = M2F-specific projection layers — expected.
        if unexpected:
            print(f'  [Warning] Unexpected keys: {unexpected[:5]} …')
        print(
            f'  ✓ DINOv2 backbone loaded '
            f'({len(missing)} new projection layers randomly init)'
        )
        del pretrained

    # ── Label preparation ─────────────────────────────────────────────────────

    @staticmethod
    def prepare_binary_labels(
        masks: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Convert binary mask batch to HF Mask2Former label format.

        Args:
            masks : (B, 1, H, W) — binary float tensor {0, 1}

        Returns:
            mask_labels  : list[B] of (num_instances, H, W) float tensors
            class_labels : list[B] of (num_instances,) long tensors

        For binary water segmentation each image has at most 1 instance
        (the water region). Empty masks produce zero-length instance tensors,
        which HF handles correctly as "no object" supervision.
        """
        B, _, H, W = masks.shape
        ml, cl = [], []

        for b in range(B):
            m = masks[b, 0]  # (H, W)
            if m.sum() > 0:
                ml.append(m.unsqueeze(0).float())                              # (1, H, W)
                cl.append(torch.tensor([0], dtype=torch.long, device=m.device))
            else:
                # Empty image: no foreground instances
                ml.append(torch.zeros(0, H, W, dtype=torch.float, device=m.device))
                cl.append(torch.zeros(0,       dtype=torch.long,  device=m.device))
        return ml, cl

    # ── Output post-processing ────────────────────────────────────────────────

    @staticmethod
    def _hf_outputs_to_logits(
        hf_outputs,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """
        Aggregate Mask2Former query predictions into a single binary logit map.

        Method (mirrors HF's post_process_semantic_segmentation):
          1.  Softmax class logits  → foreground probability per query
          2.  Sigmoid mask logits   → per-pixel confidence per query
          3.  Weighted sum over queries
          4.  Upsample to target resolution
          5.  Probability → logit  (for compatibility with trainer's sigmoid)

        Args:
            hf_outputs : Mask2FormerForUniversalSegmentationOutput
            target_hw  : (H, W) of the input image

        Returns:
            (B, 1, H, W) raw logits
        """
        # class_queries_logits : (B, Q, num_labels + 1)
        #   last dim = "no object" class
        class_probs = hf_outputs.class_queries_logits.softmax(dim=-1)
        fg_probs    = class_probs[..., :-1]   # (B, Q, num_labels) = (B, Q, 1)

        # masks_queries_logits : (B, Q, H/4, W/4) — raw logits at decoder res
        mask_probs = hf_outputs.masks_queries_logits.sigmoid()  # (B, Q, h, w)

        # Weighted aggregation: Σ_q fg_prob(q) · mask_prob(q, h, w)
        # Result shape: (B, num_labels, H_dec, W_dec) = (B, 1, h, w)
        seg_prob = torch.einsum('bqhw, bqc -> bchw', mask_probs, fg_probs)

        # Upsample to input resolution
        seg_prob = F.interpolate(
            seg_prob,
            size          = target_hw,
            mode          = 'bilinear',
            align_corners = False,
        )

        # Probability → logit.  Clamped to avoid log(0).
        seg_logit = torch.logit(seg_prob.clamp(1e-6, 1.0 - 1e-6))

        return seg_logit  # (B, 1, H, W)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values : torch.Tensor,
        mask_labels  : list[torch.Tensor] | None = None,
        class_labels : list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            pixel_values : (B, 3, H, W)
            mask_labels  : HF-format instance masks — supply during training
            class_labels : HF-format class indices  — supply during training

        Returns:
            seg_logits : (B, 1, H, W) — raw logits for sigmoid + threshold
            loss       : scalar Tensor (training) | None (inference)

        The HF Mask2Former loss aggregates:
          - Per-layer Hungarian matching loss (class CE + mask BCE + mask Dice)
          - Final layer weight × 1.0;  intermediate layers × 0.5  (aux weight)
        This is backpropagated through the M2F transformer decoder only;
        the DINOv2 backbone is frozen (no grad).
        """
        H, W = pixel_values.shape[-2:]

        if mask_labels is not None and class_labels is not None:
            outputs = self.model(
                pixel_values = pixel_values,
                mask_labels  = mask_labels,
                class_labels = class_labels,
            )
            loss = outputs.loss  # scalar, includes all auxiliary layer losses
        else:
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.model(pixel_values=pixel_values)
            loss = None

        seg_logits = self._hf_outputs_to_logits(outputs, target_hw=(H, W))
        return seg_logits, loss

    # ── Differential learning rates ───────────────────────────────────────────

    def get_params_groups(self, lr: float) -> list[dict]:
        """
        Three trainable parameter groups (backbone is frozen, excluded).

          pixel_decoder        : lr      — FPN projection + lateral connections
          transformer_decoder  : lr      — masked cross-attention + FFN
          class_predictor      : lr × 2  — small head, benefits from higher LR
        """
        return [
            {
                'name'  : 'pixel_decoder',
                'params': list(
                    self.model.model.pixel_level_module.decoder.parameters()
                ),
                'lr'    : lr,
            },
            {
                'name'  : 'transformer_decoder',
                'params': list(
                    self.model.model.transformer_module.parameters()
                ),
                'lr'    : lr,
            },
            {
                'name'  : 'class_predictor',
                'params': list(self.model.class_predictor.parameters()),
                'lr'    : lr * 2.0,
            },
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Factory — mirrors build_dinov2_segmentation() interface
# ─────────────────────────────────────────────────────────────────────────────

def build_dinov2_mask2former_segmentation(
    num_classes : int = 1,
    variant     : str = 'vit_b',
    num_queries : int = 20,
) -> DINOv2Mask2FormerSegmentation:
    """
    Factory for DINOv2Mask2FormerSegmentation.

    Called by get_model() when model_name == 'dinov2_Mask2Former':

        elif model_name == 'dinov2_Mask2Former':
            return build_dinov2_mask2former_segmentation(
                num_classes = n_classes,
                variant     = variant,
            )

    Args:
        num_classes : Foreground classes — 1 for binary water segmentation
        variant     : 'vit_s' | 'vit_b' | 'vit_l'
        num_queries : Learnable queries (20 is appropriate for binary seg)
    """
    return DINOv2Mask2FormerSegmentation(
        num_classes     = num_classes,
        encoder_variant = variant,
        num_queries     = num_queries,
    )


# ─────────────────────────────────────────────────────────────────────────────
#
#  TRAINER PATCH — changes required in train_unified_wandb_dinov2.py
#
# ─────────────────────────────────────────────────────────────────────────────
#
#
# ── PATCH 1: __init__ — add is_mask2former flag (near is_sam, ~line 72) ──────
#
#     model_name = config['model']['name']
#     variant    = config['model'].get('variant', None)
#
#     self.is_sam          = model_name in ('sam', 'sam2')
#     self.is_mask2former  = model_name == 'dinov2_Mask2Former'   # ← ADD THIS
#
#
# ── PATCH 2: _forward — add M2F routing (at the top of the method) ───────────
#
#     def _forward(self, batch):
#         if self.is_sam:
#             return self._forward_sam(batch)
#         if self.is_mask2former:                                  # ← ADD THIS
#             return self._forward_mask2former(batch)              # ← ADD THIS
#         images  = batch['image'].to(self.device)
#         ...
#
#
# ── PATCH 3: new _forward_mask2former method — add after _forward_sam ────────
#
#     def _forward_mask2former(self, batch):
#         """
#         Mask2Former forward pass.
#
#         During training:
#           - Converts GT masks to HF instance-label format
#           - Passes labels to the model so HF computes Hungarian matching loss
#           - Returns (seg_logits, m2f_loss, masks)
#
#         During validation:
#           - No labels passed; model runs in pure-inference mode
#           - Returns (seg_logits, None, masks)
#
#         The m2f_loss (scalar tensor) is detected in _compute_loss() and
#         returned directly, bypassing the standard BCE/Dice path.
#         """
#         from mask2former_head import DINOv2Mask2FormerSegmentation
#
#         images = batch['image'].to(self.device)   # (B, 3, H, W)
#         masks  = batch['mask'].to(self.device)    # (B, 1, H, W)
#
#         if self.model.training:
#             mask_labels, class_labels = (
#                 DINOv2Mask2FormerSegmentation.prepare_binary_labels(masks)
#             )
#             seg_logits, m2f_loss = self.model(images, mask_labels, class_labels)
#             # m2f_loss: scalar tensor (HF Hungarian matching + aux layer losses)
#             return seg_logits, m2f_loss, masks
#         else:
#             seg_logits, _ = self.model(images)
#             return seg_logits, None, masks
#
#
# ── PATCH 4: _compute_loss — handle pre-computed M2F loss ────────────────────
#
#     def _compute_loss(self, main_out, masks, aux_out=None):
#
#         # ── Mask2Former: aux_out IS the loss (scalar tensor from HF) ─────────
#         # Detect this case: aux_out is a 0-dim tensor (scalar), not a tuple
#         # of branch logits (GlobalLocal) or None (standard).
#         if isinstance(aux_out, torch.Tensor) and aux_out.dim() == 0:
#             loss_dict = {
#                 'total'    : aux_out.item(),
#                 'm2f_loss' : aux_out.item(),
#             }
#             return aux_out, loss_dict
#
#         # ── Standard / GlobalLocal path — unchanged ──────────────────────────
#         if self.config['loss']['type'] == 'combined':
#             main_loss, loss_dict = self.criterion(main_out, masks, None)
#         else:
#             ...  # existing code continues unchanged
#
#
# ─────────────────────────────────────────────────────────────────────────────
