"""
DINOv2 Encoder for Semantic Segmentation
=========================================

Wraps Meta's pretrained DINOv2 ViT backbone and extracts multi-scale spatial
features suitable for the FPNDecoder.

Key difference from the previous from-scratch version
------------------------------------------------------
This module loads REAL pretrained DINOv2 weights (trained by Meta on 142M images
with self-supervised DINO + iBOT objectives).  The previous version initialised
all weights randomly and therefore had none of the representation quality that
makes DINOv2 a "foundation model".

How feature extraction works
-----------------------------
Meta's DINOv2 exposes ``get_intermediate_layers(x, blocks, return_class_token)``.
We tap the transformer at four evenly-spaced block indices to obtain features at
different semantic levels (shallow = fine detail, deep = high semantics), then:
  1. Remove the CLS token  →  patch-token sequence  (B, N, D)
  2. Reshape to 2D spatial grid  →  (B, D, H/14, W/14)
  3. Project channels via 1×1 conv  →  (B, out_ch[i], H/14, W/14)
  4. Bilinear resize to mimic stride-{4,8,16,32} pyramid for the FPN decoder

Author: Hasitha
Date: December 2025
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Variant configuration ─────────────────────────────────────────────────────

DINOV2_CONFIGS = {
    # variant : (hub_name,        embed_dim, depth, out_channels)
    'vit_s': ('dinov2_vits14',  384,  12, [48,  96,  192,  384]),
    'vit_b': ('dinov2_vitb14',  768,  12, [96,  192, 384,  768]),
    'vit_l': ('dinov2_vitl14', 1024,  24, [128, 256, 512, 1024]),
    'vit_g': ('dinov2_vitg14', 1536,  40, [192, 384, 768, 1536]),
}

PATCH_SIZE = 14   # DINOv2 always uses 14×14 patches


# ── Encoder ───────────────────────────────────────────────────────────────────

class DINOv2Encoder(nn.Module):
    """
    Pretrained DINOv2 encoder with multi-scale FPN-ready feature outputs.

    Args:
        variant:       DINOv2 size — 'vit_s' | 'vit_b' | 'vit_l' | 'vit_g'
        in_channels:   Number of input image channels (must be 3 for pretrained)
        pretrained:    Load Meta pretrained weights via torch.hub (recommended: True)
        freeze_encoder: Freeze all encoder weights (useful for linear probing)

    Attributes:
        out_channels:  List of output channels per scale, used by FPNDecoder
    """

    def __init__(
        self,
        variant:        str  = 'vit_b',
        in_channels:    int  = 3,
        pretrained:     bool = True,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        if variant not in DINOV2_CONFIGS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from {list(DINOV2_CONFIGS.keys())}"
            )
        if in_channels != 3:
            raise ValueError(
                "Pretrained DINOv2 expects 3-channel RGB input. "
                f"Got in_channels={in_channels}. "
                "If you need single-channel input, convert to 3-channel first."
            )

        hub_name, embed_dim, depth, out_channels = DINOV2_CONFIGS[variant]

        self.variant      = variant
        self.embed_dim    = embed_dim
        self.depth        = depth
        self.out_channels = out_channels   # consumed by FPNDecoder

        # ── Block indices at which we tap intermediate features ────────────────
        # Four evenly-spaced taps: ~25%, 50%, 75%, 100% depth
        # These are 0-indexed block numbers passed to get_intermediate_layers()
        self.extract_indices: List[int] = [
            depth // 4 - 1,
            depth // 2 - 1,
            depth * 3 // 4 - 1,
            depth - 1,
        ]

        # ── Load pretrained backbone ──────────────────────────────────────────
        if pretrained:
            print(f"  [DINOv2Encoder] Loading pretrained '{hub_name}' from torch.hub …")
            try:
                self.backbone = torch.hub.load(
                    'facebookresearch/dinov2',
                    hub_name,
                    pretrained=True,
                )
                print(f"  [DINOv2Encoder] Pretrained weights loaded ✓")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load pretrained DINOv2 '{hub_name}' from torch.hub.\n"
                    f"Ensure you have internet access and the facebookresearch/dinov2 "
                    f"repo is accessible.\n"
                    f"Original error: {e}"
                ) from e
        else:
            # Random-init backbone — only useful for architecture testing
            print(f"  [DINOv2Encoder] WARNING: loading '{hub_name}' WITHOUT pretrained "
                  f"weights. Features will be meaningless.")
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2',
                hub_name,
                pretrained=False,
            )

        # ── Optionally freeze backbone ─────────────────────────────────────────
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("  [DINOv2Encoder] Backbone frozen (linear probing mode).")

        # ── 1×1 projection convs: embed_dim → out_channels[i] ─────────────────
        # Trainable even when backbone is frozen — these adapt DINOv2 features to
        # the segmentation task.
        self.proj_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for out_ch in out_channels
        ])

        self._init_proj_weights()

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_proj_weights(self):
        for m in self.proj_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale spatial features from a batch of RGB images.

        Args:
            x: (B, 3, H, W) — H and W must be divisible by 14.
                              Images should use ImageNet normalisation.

        Returns:
            List of 4 feature maps at stride-{4,8,16,32} scales:
                [  (B, out_ch[0], H/4,  W/4 )   ← finest, most detail
                   (B, out_ch[1], H/8,  W/8 )
                   (B, out_ch[2], H/16, W/16)
                   (B, out_ch[3], H/32, W/32)  ]  ← coarsest, most semantic
        """
        B, _, H, W = x.shape

        # ── Pad input so H and W are multiples of 14 ─────────────────────────
        # DINOv2's patch size is always 14; inputs that aren't exact multiples
        # cause a shape mismatch in the patch embedding convolution.
        pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, H_pad, W_pad = x.shape
        grid_h = H_pad // PATCH_SIZE
        grid_w = W_pad // PATCH_SIZE

        # ── Extract intermediate patch-token features ─────────────────────────
        # get_intermediate_layers returns a tuple of tensors, one per requested
        # block.  Each tensor: (B, N, embed_dim) where N = grid_h * grid_w.
        # return_class_token=False drops the CLS token — we only need spatial tokens.
        raw_features = self.backbone.get_intermediate_layers(
            x,
            n=self.extract_indices,     # list of 0-indexed block numbers
            return_class_token=False,   # only patch tokens → (B, N, D)
        )
        # raw_features is a tuple of 4 tensors: each (B, N, embed_dim)

        # ── Reshape patch tokens to 2D spatial grids ─────────────────────────
        spatial_features = []
        for feat in raw_features:
            # (B, N, D) → (B, D, grid_h, grid_w)
            feat = feat.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
            spatial_features.append(feat)

        # ── Project channels and resize to FPN pyramid scales ─────────────────
        # Target scales: stride 4, 8, 16, 32 relative to original (unpadded) input
        outputs = []
        for i, (feat, proj) in enumerate(zip(spatial_features, self.proj_convs)):
            # Project: embed_dim → out_channels[i]
            feat = proj(feat)

            # Resize to stride-{4,8,16,32} of the ORIGINAL (unpadded) input size
            stride     = 4 * (2 ** i)        # 4, 8, 16, 32
            target_h   = math.ceil(H / stride)
            target_w   = math.ceil(W / stride)
            feat = F.interpolate(
                feat,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            )
            outputs.append(feat)

        return outputs


# ── Factory ───────────────────────────────────────────────────────────────────

def build_dinov2_encoder(
    variant:        str  = 'vit_b',
    img_size:       int  = 512,     # kept for API compatibility, not used internally
    in_channels:    int  = 3,
    pretrained:     bool = True,
    freeze_encoder: bool = False,
) -> DINOv2Encoder:
    """
    Build a pretrained DINOv2 encoder.

    Args:
        variant:       'vit_s' | 'vit_b' | 'vit_l' | 'vit_g'
        img_size:      Kept for API compatibility with the old from-scratch version.
                       Not used — DINOv2 accepts arbitrary sizes via position
                       embedding interpolation built into Meta's implementation.
        in_channels:   Must be 3 for pretrained weights.
        pretrained:    Load Meta pretrained weights (strongly recommended).
        freeze_encoder: Freeze backbone weights.

    Returns:
        DINOv2Encoder
    """
    return DINOv2Encoder(
        variant=variant,
        in_channels=in_channels,
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing pretrained DINOv2 Encoder")

    encoder = build_dinov2_encoder('vit_b', pretrained=True)
    x = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        features = encoder(x)

    print(f"\nInput : {x.shape}")
    for i, f in enumerate(features):
        print(f"Scale {i+1} (stride {4 * 2**i:>2d}): {f.shape}")

    backbone_params = sum(p.numel() for p in encoder.backbone.parameters())
    proj_params     = sum(p.numel() for p in encoder.proj_convs.parameters())
    print(f"\nBackbone parameters : {backbone_params:,}  (pretrained, fine-tuned at low LR)")
    print(f"Projection parameters: {proj_params:,}  (randomly init, trained at full LR)")
