"""
SAM Image Encoder for Semantic Segmentation
============================================

Wraps Meta's pretrained SAM image encoder (ViT backbone) and extracts
multi-scale spatial features suitable for the FPNDecoder.

Key difference from the previous from-scratch version
------------------------------------------------------
The previous version built a plain ViT from scratch with random weights, giving
none of the representation quality that makes SAM a "foundation model".

This version loads Meta's pretrained SAM image encoder from a local checkpoint
using the official `segment_anything` package, then uses forward hooks to tap
the transformer at four evenly-spaced depths and build an FPN-compatible
multi-scale feature pyramid.

SAM checkpoint download
-----------------------
SAM weights are NOT auto-downloadable via torch.hub — they must be downloaded
manually from Meta's GitHub release page:

    vit_b (~375 MB) : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    vit_l (~1.2 GB) : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    vit_h (~2.4 GB) : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Place these in a directory (e.g. ./checkpoints/sam/) and pass the path to
``build_sam_encoder(checkpoint_path=...)``.

Installation
------------
    pip install segment-anything
    # or: pip install git+https://github.com/facebookresearch/segment-anything.git

Author: Hasitha
Date: December 2025
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Variant configuration ─────────────────────────────────────────────────────

SAM_CONFIGS = {
    # variant : (sam_registry_key,  embed_dim, depth, num_heads, out_channels)
    'vit_b': ('vit_b', 768,  12, 12, [96,  192, 384,  768]),
    'vit_l': ('vit_l', 1024, 24, 16, [128, 256, 512, 1024]),
    'vit_h': ('vit_h', 1280, 32, 16, [160, 320, 640, 1280]),
}

PATCH_SIZE = 16   # SAM always uses 16×16 patches


# ── Encoder ───────────────────────────────────────────────────────────────────

class SAMImageEncoder(nn.Module):
    """
    Pretrained SAM image encoder with multi-scale FPN-ready feature outputs.

    Loads Meta's SAM ViT backbone and taps it at four evenly-spaced block
    depths using forward hooks to produce a feature pyramid compatible with
    the existing FPNDecoder.

    Args:
        variant:          SAM variant — 'vit_b' | 'vit_l' | 'vit_h'
        checkpoint_path:  Path to the downloaded .pth checkpoint file.
                          Required when pretrained=True.
        pretrained:       Load Meta pretrained SAM weights (strongly recommended).
        freeze_encoder:   Freeze all SAM backbone weights (linear probing mode).

    Attributes:
        out_channels:  List of output channels per scale, used by FPNDecoder.

    Example:
        >>> encoder = SAMImageEncoder('vit_b', checkpoint_path='./checkpoints/sam_vit_b_01ec64.pth')
        >>> features = encoder(torch.randn(2, 3, 512, 512))
        >>> [f.shape for f in features]
        # [(2,96,128,128), (2,192,64,64), (2,384,32,32), (2,768,16,16)]
    """

    def __init__(
        self,
        variant:         str            = 'vit_b',
        checkpoint_path: Optional[str] = None,
        pretrained:      bool           = True,
        freeze_encoder:  bool           = False,
    ):
        super().__init__()

        if variant not in SAM_CONFIGS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from {list(SAM_CONFIGS.keys())}"
            )

        registry_key, embed_dim, depth, num_heads, out_channels = SAM_CONFIGS[variant]

        self.variant      = variant
        self.embed_dim    = embed_dim
        self.depth        = depth
        self.out_channels = out_channels   # consumed by FPNDecoder

        # ── Block indices at which we tap intermediate features ────────────────
        # Four evenly-spaced taps: ~25%, 50%, 75%, 100% depth (0-indexed)
        self.extract_indices: List[int] = [
            depth // 4 - 1,
            depth // 2 - 1,
            depth * 3 // 4 - 1,
            depth - 1,
        ]

        # ── Load SAM backbone ─────────────────────────────────────────────────
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError(
                "The `segment_anything` package is required for SAM models.\n"
                "Install it with:\n"
                "    pip install segment-anything\n"
                "or: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        if pretrained:
            if checkpoint_path is None:
                raise ValueError(
                    "SAM pretrained weights require an explicit checkpoint_path.\n"
                    "Download from Meta:\n"
                    "  vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
                    "  vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
                    "  vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                )
            print(f"  [SAMImageEncoder] Loading pretrained SAM '{variant}' from {checkpoint_path} …")
            sam_model = sam_model_registry[registry_key](checkpoint=checkpoint_path)
            print(f"  [SAMImageEncoder] Pretrained weights loaded ✓")
        else:
            print(f"  [SAMImageEncoder] WARNING: loading SAM '{variant}' WITHOUT pretrained "
                  f"weights. Features will be meaningless.")
            sam_model = sam_model_registry[registry_key](checkpoint=None)

        # We only need the image encoder, not the prompt encoder or mask decoder
        self.backbone = sam_model.image_encoder

        # ── Optionally freeze backbone ─────────────────────────────────────────
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("  [SAMImageEncoder] Backbone frozen (linear probing mode).")

        # ── Storage for hooked intermediate features ──────────────────────────
        # Each hook writes its output here; cleared at the start of every forward().
        self._hooked_features: List[torch.Tensor] = []

        # ── Register forward hooks on transformer blocks ──────────────────────
        # SAM's image encoder exposes its transformer blocks as backbone.blocks
        self._hook_handles = []
        for idx in self.extract_indices:
            handle = self.backbone.blocks[idx].register_forward_hook(
                self._make_hook()
            )
            self._hook_handles.append(handle)

        # ── 1×1 projection convs: embed_dim → out_channels[i] ─────────────────
        # These are trainable even when the backbone is frozen.
        # BN + ReLU stabilise early training when projecting from frozen features.
        self.proj_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for out_ch in out_channels
        ])

        self._init_proj_weights()

    # ── Destructor — clean up hooks when encoder is deleted ───────────────────

    def __del__(self):
        for handle in getattr(self, '_hook_handles', []):
            handle.remove()

    # ── Hook factory ──────────────────────────────────────────────────────────

    def _make_hook(self):
        """
        Returns a hook function that appends the block's output to
        self._hooked_features.

        SAM's transformer blocks output tensors of shape (B, H', W', D)
        (spatial-first layout, unlike the sequence-first layout of DINOv2).
        We permute to (B, D, H', W') immediately for consistency.
        """
        def hook(module, input, output):
            # SAM ViT blocks: output is (B, H', W', D)  ← spatial-first
            # Permute to     (B, D, H', W')             ← channel-first for convs
            self._hooked_features.append(output.permute(0, 3, 1, 2).contiguous())
        return hook

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_proj_weights(self):
        for m in self.proj_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale spatial features from a batch of RGB images.

        Args:
            x: (B, 3, H, W) — ImageNet-normalised RGB images.
               Any spatial size is accepted; the input is internally resized to
               SAM_INPUT_SIZE before the backbone (SAM positional embedding is
               fixed for 1024x1024 and cannot be interpolated like DINOv2).

        Returns:
            List of 4 feature maps at stride-{4,8,16,32} scales relative to the
            ORIGINAL input size HxW:
                [  (B, out_ch[0], H/4,  W/4 )
                   (B, out_ch[1], H/8,  W/8 )
                   (B, out_ch[2], H/16, W/16)
                   (B, out_ch[3], H/32, W/32)  ]
        """
        B, _, H, W = x.shape

        # ── Resize to SAM fixed input resolution ──────────────────────────────
        # SAM image_encoder has pos_embed shape (1, 64, 64, embed_dim), fixed
        # for 1024x1024 input (1024 / 16 patch_size = 64). It does a plain
        # tensor addition with no interpolation, so any other spatial size
        # causes: RuntimeError: size of tensor a must match tensor b
        # Fix: always resize to 1024x1024, then map hooked features back to
        # FPN pyramid scales computed from the ORIGINAL H x W.
        SAM_INPUT_SIZE = 1024
        x_sam = F.interpolate(
            x,
            size=(SAM_INPUT_SIZE, SAM_INPUT_SIZE),
            mode='bilinear',
            align_corners=False,
        )

        # ── Clear hook storage before each forward pass ───────────────────────
        self._hooked_features.clear()

        # ── Run backbone (hooks populate self._hooked_features) ───────────────
        # Hooked features will be at (B, embed_dim, 64, 64).
        # We ignore the backbone final neck output.
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        ):
            _ = self.backbone(x_sam)

        assert len(self._hooked_features) == 4, (
            f"Expected 4 hooked features, got {len(self._hooked_features)}. "
            f"Check that extract_indices {self.extract_indices} are valid for "
            f"a {self.depth}-block backbone."
        )

        # ── Project channels and resize to FPN pyramid scales ─────────────────
        # Target sizes are from the ORIGINAL H x W so the decoder output
        # is always at the correct scale relative to the input image.
        outputs = []
        for i, (feat, proj) in enumerate(zip(self._hooked_features, self.proj_convs)):
            feat = proj(feat)   # (B, out_channels[i], 64, 64)

            stride   = 4 * (2 ** i)
            target_h = math.ceil(H / stride)
            target_w = math.ceil(W / stride)
            feat = F.interpolate(
                feat,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            )
            outputs.append(feat)

        return outputs


# ── Factory ───────────────────────────────────────────────────────────────────

def build_sam_encoder(
    variant:         str            = 'vit_b',
    img_size:        int            = 512,    # kept for API compatibility
    in_channels:     int            = 3,
    checkpoint_path: Optional[str] = None,
    pretrained:      bool           = True,
    freeze_encoder:  bool           = False,
) -> SAMImageEncoder:
    """
    Build a pretrained SAM image encoder.

    Args:
        variant:          'vit_b' | 'vit_l' | 'vit_h'
        img_size:         Kept for API compatibility. SAM handles arbitrary sizes.
        in_channels:      Must be 3 (pretrained on RGB).
        checkpoint_path:  Path to .pth checkpoint file downloaded from Meta.
                          Required when pretrained=True.
        pretrained:       Load Meta pretrained SAM weights.
        freeze_encoder:   Freeze backbone weights.

    Returns:
        SAMImageEncoder
    """
    if in_channels != 3:
        raise ValueError(
            f"Pretrained SAM expects 3-channel RGB input. Got in_channels={in_channels}."
        )
    return SAMImageEncoder(
        variant         = variant,
        checkpoint_path = checkpoint_path,
        pretrained      = pretrained,
        freeze_encoder  = freeze_encoder,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("Testing pretrained SAM Image Encoder")
    print("NOTE: requires segment_anything package and a valid checkpoint_path")

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/sam/sam_vit_b_01ec64.pth"
    encoder = build_sam_encoder('vit_b', checkpoint_path=ckpt, pretrained=True)

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        features = encoder(x)

    print(f"\nInput : {x.shape}")
    for i, f in enumerate(features):
        print(f"Scale {i+1} (stride {4 * 2**i:>2d}): {f.shape}")

    backbone_params = sum(p.numel() for p in encoder.backbone.parameters())
    proj_params     = sum(p.numel() for p in encoder.proj_convs.parameters())
    print(f"\nBackbone parameters  : {backbone_params:,}  (pretrained)")
    print(f"Projection parameters: {proj_params:,}  (randomly init)")
