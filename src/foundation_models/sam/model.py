"""
SAM Segmentation Model
======================

Complete model combining pretrained SAM encoder and FPN decoder for binary
river water segmentation.

Changes from the original from-scratch version
-----------------------------------------------
* Encoder now loads real pretrained SAM weights from a local checkpoint file.
* ``checkpoint_path`` argument is required when pretrained=True.
  Download checkpoints from:
      vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
      vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
      vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
* ``pretrained`` and ``freeze_encoder`` arguments control fine-tuning strategy:
    - Full fine-tuning  (pretrained=True,  freeze_encoder=False)  ← recommended
    - Linear probing    (pretrained=True,  freeze_encoder=True)
    - Random baseline   (pretrained=False, freeze_encoder=False)
* ``get_params_groups()`` now returns three groups matching DINOv2:
    backbone    → lr × 0.1   (pretrained ViT, fine-tune gently)
    proj_convs  → lr × 1.0   (randomly initialised, train at full LR)
    decoder     → lr × 1.0   (randomly initialised, task-specific)

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .sam_encoder import SAMImageEncoder, build_sam_encoder, SAM_CONFIGS
from .sam_decoder import FPNDecoder


class SAMSegmentation(nn.Module):
    """
    SAM-based binary segmentation model.

    Architecture:
        Input (B, 3, H, W)
            │
            ▼  SAM ViT image encoder  (pretrained, fine-tuned at low LR)
            │  + 1×1 projection convs  (random init, trained at full LR)
            │
            ▼  FPN decoder  (random init, trained at full LR)
            │
            ▼  Output logits (B, num_classes, H, W)  [no sigmoid — use BCEWithLogitsLoss]

    Args:
        in_channels:      Input channels (must be 3 for pretrained backbone).
        num_classes:      Output channels — 1 for binary segmentation.
        encoder_variant:  SAM size ('vit_b' | 'vit_l' | 'vit_h').
        img_size:         Kept for API compatibility; SAM handles arbitrary sizes.
        fpn_channels:     Intermediate channels in the FPN decoder.
        pretrained:       Load Meta pretrained SAM weights.
        freeze_encoder:   Freeze backbone weights (linear probing mode).
        checkpoint_path:  Path to local .pth file. Required when pretrained=True.

    Example:
        >>> model = SAMSegmentation(
        ...     num_classes=1,
        ...     encoder_variant='vit_b',
        ...     checkpoint_path='./checkpoints/sam/sam_vit_b_01ec64.pth'
        ... )
        >>> logits = model(torch.randn(2, 3, 512, 512))  # (2, 1, 512, 512)
        >>> probs  = torch.sigmoid(logits)
    """

    def __init__(
        self,
        in_channels:     int            = 3,
        num_classes:     int            = 1,
        encoder_variant: str            = 'vit_b',
        img_size:        int            = 512,
        fpn_channels:    int            = 256,
        pretrained:      bool           = True,
        freeze_encoder:  bool           = False,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        self.in_channels     = in_channels
        self.num_classes     = num_classes
        self.encoder_variant = encoder_variant

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = build_sam_encoder(
            variant         = encoder_variant,
            img_size        = img_size,
            in_channels     = in_channels,
            checkpoint_path = checkpoint_path,
            pretrained      = pretrained,
            freeze_encoder  = freeze_encoder,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # in_channels_list flows directly from encoder.out_channels so decoder
        # dimensions are always consistent across variants.
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

        Returns:
            Raw logits (B, num_classes, H, W).
            Apply torch.sigmoid() for probabilities,
            or pass directly to nn.BCEWithLogitsLoss.
        """
        input_size = (x.shape[2], x.shape[3])

        features = self.encoder(x)
        logits   = self.decoder(features, input_size)

        return logits

    # def get_params_groups(self, lr: float) -> List[dict]:
    #     """
    #     Parameter groups for differential learning rates.

    #     The pretrained SAM backbone gets 10× lower LR than the randomly-
    #     initialised projection convolutions and decoder — matching the standard
    #     fine-tuning recipe used for DINOv2.

    #     Args:
    #         lr: Base learning rate (applied to decoder and projection convs).

    #     Returns:
    #         List of dicts for AdamW, e.g.:
    #             optimizer = AdamW(model.get_params_groups(lr=1e-4))
    #     """
    #     return [
    #         # Pretrained SAM image encoder — gentle fine-tuning
    #         {
    #             'params' : self.encoder.backbone.parameters(),
    #             'lr'     : lr * 0.1,
    #             'name'   : 'backbone',
    #         },
    #         # Projection convs — bridge SAM features to FPN channel dimensions
    #         {
    #             'params' : self.encoder.proj_convs.parameters(),
    #             'lr'     : lr,
    #             'name'   : 'proj_convs',
    #         },
    #         # FPN decoder — fully task-specific, trained from scratch
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

def build_sam_segmentation(
    variant:        str            = 'vit_b',
    in_channels:    int            = 3,
    num_classes:    int            = 1,
    img_size:       int            = 512,
    fpn_channels:   int            = 256,
    pretrained:     bool           = True,
    freeze_encoder: bool           = False
) -> SAMSegmentation:
    """
    Factory function for SAM segmentation models.

    Args:
        variant:         'vit_b' | 'vit_l' | 'vit_h'
        in_channels:     Must be 3 for pretrained weights.
        num_classes:     1 for binary segmentation (water / no-water).
        img_size:        Kept for API compatibility.
        fpn_channels:    FPN intermediate channels (256 is standard).
        pretrained:      Load Meta pretrained SAM weights.
        freeze_encoder:  Freeze backbone (linear probing).
        checkpoint_path: Path to downloaded SAM .pth file.

    Recommended usage:
        >>> # Full fine-tuning (best accuracy)
        >>> model = build_sam_segmentation(
        ...     'vit_b', checkpoint_path='./checkpoints/sam/sam_vit_b_01ec64.pth'
        ... )

        >>> # Linear probing (tests SAM feature quality alone)
        >>> model = build_sam_segmentation(
        ...     'vit_b', freeze_encoder=True,
        ...     checkpoint_path='./checkpoints/sam/sam_vit_b_01ec64.pth'
        ... )
    """
    if variant not in SAM_CONFIGS:
        raise ValueError(
            f"Unknown variant: '{variant}'. "
            f"Choose from {list(SAM_CONFIGS.keys())}"
        )
    
    if variant=='vit_b':
        checkpoint_path = './checkpoints/sam/sam_vit_b_01ec64.pth'
    elif variant=='vit_l':
        checkpoint_path = './checkpoints/sam/sam_vit_l_0b3195.pth'
    elif variant=='vit_h':
        checkpoint_path = './checkpoints/sam/sam_vit_h_4b8939.pth'

    return SAMSegmentation(
        in_channels     = in_channels,
        num_classes     = num_classes,
        encoder_variant = variant,
        img_size        = img_size,
        fpn_channels    = fpn_channels,
        pretrained      = pretrained,
        freeze_encoder  = freeze_encoder,
        checkpoint_path = checkpoint_path,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=" * 70)
    print("SAM Segmentation Model — Pretrained")
    print("=" * 70)

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/sam/sam_vit_b_01ec64.pth"

    model = build_sam_segmentation(
        variant         = 'vit_b',
        in_channels     = 3,
        num_classes     = 1,
        img_size        = 512,
        pretrained      = True,
        checkpoint_path = ckpt,
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

    print("\nParameter groups (for AdamW):")
    for g in model.get_params_groups(lr=1e-4):
        n = sum(p.numel() for p in g['params'])
        print(f"  {g['name']:15s}  LR={g['lr']:.1e}  params={n:,}")

    print("\n✓ Forward pass OK")
