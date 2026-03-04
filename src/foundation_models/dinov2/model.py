"""
DINOv2 Segmentation Model
==========================

Complete model combining pretrained DINOv2 encoder and FPN decoder for binary
river water segmentation.

Changes from the original from-scratch version
-----------------------------------------------
* Encoder now loads real pretrained DINOv2 weights from torch.hub.
* ``pretrained`` and ``freeze_encoder`` arguments added so you can choose between:
    - Full fine-tuning  (pretrained=True,  freeze_encoder=False)  ← recommended
    - Linear probing    (pretrained=True,  freeze_encoder=True)
    - Random baseline   (pretrained=False, freeze_encoder=False)
* ``num_classes`` defaults to 1 (binary segmentation with BCEWithLogitsLoss),
  matching your training setup.
* ``get_params_groups()`` now applies 10× lower LR to the backbone and normal
  LR to projection convs + decoder, which is the standard fine-tuning recipe.
* ``img_size`` argument is kept for API compatibility but DINOv2 handles
  arbitrary input sizes internally via positional embedding interpolation.

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import List

from .dinov2_encoder import DINOv2Encoder, build_dinov2_encoder, DINOV2_CONFIGS
from .dinov2_decoder import FPNDecoder


class DINOv2Segmentation(nn.Module):
    """
    DINOv2-based binary segmentation model.

    Architecture:
        Input (B, 3, H, W)
            │
            ▼  DINOv2 ViT backbone  (pretrained, fine-tuned at low LR)
            │  + 1×1 projection convs  (random init, trained at full LR)
            │
            ▼  FPN decoder  (random init, trained at full LR)
            │
            ▼  Output logits (B, num_classes, H, W)   [no sigmoid — use BCEWithLogitsLoss]

    Args:
        in_channels:     Input channels (must be 3 for pretrained backbone).
        num_classes:     Output channels — 1 for binary segmentation.
        encoder_variant: DINOv2 size ('vit_s' | 'vit_b' | 'vit_l' | 'vit_g').
        img_size:        Kept for API compatibility; DINOv2 handles arbitrary sizes.
        fpn_channels:    Intermediate channels in the FPN decoder.
        pretrained:      Load Meta pretrained DINOv2 weights (strongly recommended).
        freeze_encoder:  Freeze backbone weights (linear probing mode).

    Example:
        >>> model = DINOv2Segmentation(num_classes=1, encoder_variant='vit_b')
        >>> x = torch.randn(2, 3, 512, 512)
        >>> logits = model(x)          # (2, 1, 512, 512) — raw logits
        >>> probs  = torch.sigmoid(logits)
    """

    def __init__(
        self,
        in_channels:     int  = 3,
        num_classes:     int  = 1,
        encoder_variant: str  = 'vit_b',
        img_size:        int  = 512,
        fpn_channels:    int  = 256,
        pretrained:      bool = True,
        freeze_encoder:  bool = False,
    ):
        super().__init__()

        self.in_channels     = in_channels
        self.num_classes     = num_classes
        self.encoder_variant = encoder_variant

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = build_dinov2_encoder(
            variant        = encoder_variant,
            img_size       = img_size,        # passed through for API compat
            in_channels    = in_channels,
            pretrained     = pretrained,
            freeze_encoder = freeze_encoder,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # in_channels_list comes directly from the encoder so decoder dimensions
        # are always consistent regardless of which variant is chosen.
        self.decoder = FPNDecoder(
            in_channels_list = self.encoder.out_channels,
            fpn_channels     = fpn_channels,
            num_classes      = num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: RGB images (B, 3, H, W), ImageNet-normalised.
               H and W do not need to be multiples of 14 — the encoder pads
               internally and the decoder upsamples back to the original size.

        Returns:
            Raw logits (B, num_classes, H, W).
            Apply torch.sigmoid() for probabilities,
            or pass directly to nn.BCEWithLogitsLoss.
        """
        input_size = (x.shape[2], x.shape[3])   # (H, W) for decoder upsampling

        features = self.encoder(x)               # list of 4 multi-scale tensors
        logits   = self.decoder(features, input_size)

        return logits

    # def get_params_groups(self, lr: float) -> List[dict]:
    #     """
    #     Parameter groups for differential learning rates.

    #     The pretrained backbone gets 10× lower LR than the randomly-initialised
    #     decoder and projection convolutions — standard practice for fine-tuning
    #     large pretrained transformers.

    #     Args:
    #         lr: Base learning rate (applied to decoder).

    #     Returns:
    #         List of dicts for an optimizer, e.g.:
    #             optimizer = AdamW(model.get_params_groups(lr=1e-4))
    #     """
    #     return [
    #         # Pretrained backbone — gentle fine-tuning
    #         {
    #             'params' : self.encoder.backbone.parameters(),
    #             'lr'     : lr * 0.1,
    #             'name'   : 'backbone',
    #         },
    #         # Projection convs — bridge pretrained features to FPN
    #         {
    #             'params' : self.encoder.proj_convs.parameters(),
    #             'lr'     : lr,
    #             'name'   : 'proj_convs',
    #         },
    #         # FPN decoder — fully task-specific
    #         {
    #             'params' : self.decoder.parameters(),
    #             'lr'     : lr,
    #             'name'   : 'decoder',
    #         },
    #     ]
    
    def get_params_groups(self, lr: float):
        return [
            {
                'params': self.encoder.backbone.parameters(),
                'lr'    : lr * 0.1,   # 1e-5 — gentle fine-tuning of pretrained weights
            },
            {
                'params': self.encoder.proj_convs.parameters(),
                'lr'    : lr * 10.0,  # 1e-3 — was 1e-4, needs to learn fast from scratch
            },
            {
                'params': self.decoder.parameters(),
                'lr'    : lr * 10.0,  # 1e-3 — was 1e-4, same reason
            },
        ]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_dinov2_segmentation(
    variant:        str  = 'vit_b',
    in_channels:    int  = 3,
    num_classes:    int  = 1,
    img_size:       int  = 512,
    fpn_channels:   int  = 256,
    pretrained:     bool = True,
    freeze_encoder: bool = False,
) -> DINOv2Segmentation:
    """
    Factory function for DINOv2 segmentation models.

    Args:
        variant:        'vit_s' | 'vit_b' | 'vit_l' | 'vit_g'
        in_channels:    Must be 3 for pretrained weights.
        num_classes:    1 for binary segmentation (water / no-water).
        img_size:       Kept for API compatibility.
        fpn_channels:   FPN intermediate channels (256 is standard).
        pretrained:     Load Meta pretrained weights.
        freeze_encoder: Freeze backbone (linear probing).

    Recommended usage for your benchmark:
        >>> # Full fine-tuning (best accuracy, standard approach)
        >>> model = build_dinov2_segmentation('vit_b')

        >>> # Linear probing (fast baseline, tests feature quality alone)
        >>> model = build_dinov2_segmentation('vit_b', freeze_encoder=True)
    """
    if variant not in DINOV2_CONFIGS:
        raise ValueError(
            f"Unknown variant: '{variant}'. "
            f"Choose from {list(DINOV2_CONFIGS.keys())}"
        )

    return DINOv2Segmentation(
        in_channels     = in_channels,
        num_classes     = num_classes,
        encoder_variant = variant,
        img_size        = img_size,
        fpn_channels    = fpn_channels,
        pretrained      = pretrained,
        freeze_encoder  = freeze_encoder,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("DINOv2 Segmentation Model — Pretrained")
    print("=" * 70)

    model = build_dinov2_segmentation(
        variant     = 'vit_b',
        in_channels = 3,
        num_classes = 1,       # binary: water vs background
        img_size    = 512,
        pretrained  = True,
    )

    x = torch.randn(2, 3, 512, 512)
    print(f"\nInput : {x.shape}")

    with torch.no_grad():
        logits = model(x)
        probs  = torch.sigmoid(logits)

    print(f"Logits: {logits.shape}")
    print(f"Probs : {probs.shape}  (min={probs.min():.3f}, max={probs.max():.3f})")

    # Parameter breakdown
    backbone_p = sum(p.numel() for p in model.encoder.backbone.parameters())
    proj_p     = sum(p.numel() for p in model.encoder.proj_convs.parameters())
    decoder_p  = sum(p.numel() for p in model.decoder.parameters())
    total_p    = backbone_p + proj_p + decoder_p

    print(f"\nParameter breakdown:")
    print(f"  Backbone (pretrained, LR×0.1) : {backbone_p:>12,}  ({backbone_p/total_p*100:.1f}%)")
    print(f"  Projections (random init)     : {proj_p:>12,}  ({proj_p/total_p*100:.1f}%)")
    print(f"  FPN Decoder (random init)     : {decoder_p:>12,}  ({decoder_p/total_p*100:.1f}%)")
    print(f"  Total                         : {total_p:>12,}  ({total_p*4/1024/1024:.1f} MB fp32)")

    # Verify differential LR groups
    print("\nParameter groups (for AdamW):")
    for g in model.get_params_groups(lr=1e-4):
        n = sum(p.numel() for p in g['params'])
        print(f"  {g['name']:15s}  LR={g['lr']:.1e}  params={n:,}")

    print("\n✓ Forward pass OK")
