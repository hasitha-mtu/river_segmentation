"""
wrapper.py
==========
GlobalLocalWrapper — dual-branch segmentation model for UAV river imagery.

Both branches are constructed via get_model(), so every architecture supported
by the unified training system works as a branch:

  CNN       : unet, unetpp, resunetpp, deeplabv3plus, deeplabv3plus_cbam
  Transformer: segformer (b0/b2), swin_unet (tiny)
  Hybrid    : convnext_upernet (tiny/small/base), hrnet_ocr (w18/w32/w48)
  Foundation: sam (vit_b/vit_l/vit_h), dinov2 (vit_s/vit_b/vit_l/vit_g)

The two branches can be the SAME architecture (symmetric) or DIFFERENT
architectures (asymmetric — e.g. a lightweight model for local detail and a
heavier model for global context).

Outputs
-------
Training  : (fused_logits, (g_logits, l_logits))  — aux heads for deep supervision
Inference : fused_logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_model


# ─────────────────────────────────────────────────────────────────────────────
# Attention-gated fusion head
# ─────────────────────────────────────────────────────────────────────────────

class AttentionFusion(nn.Module):
    """
    Spatial attention-gated fusion of global and local logit maps.

    Learns a per-pixel gate α ∈ (0, 1):
        fused = α * global_feat + (1 - α) * local_feat

    α is jointly conditioned on both streams, so the network learns WHERE to
    trust global context (dense canopy occlusion) vs. local detail (clear edges).

    Args:
        in_channels : Channels in each branch output (= num_classes).
                      Hidden dim is clamped to ≥ 4 to avoid BatchNorm(0)
                      when num_classes = 1 (binary segmentation).
    """

    _MIN_HIDDEN = 4   # guard: in_channels // 2 == 0 when num_classes == 1

    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(self._MIN_HIDDEN, in_channels // 2)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),                           # α ∈ (0, 1) per spatial location
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        alpha   = self.gate(torch.cat([global_feat, local_feat], dim=1))  # [B, 1, H, W]
        blended = alpha * global_feat + (1.0 - alpha) * local_feat        # [B, C, H, W]
        return self.project(blended)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GlobalLocalWrapper(nn.Module):
    """
    Dual-branch segmentation wrapper that wraps ANY two models supported by
    get_model() into a global-context + local-detail pair, fused via learned
    spatial attention gating.

    Args:
        num_classes          : Output channels (1 = binary river segmentation).
        n_channels           : Input image channels (default 3).
        global_model_name    : Architecture name for the global branch.
        global_variant       : Variant string for the global branch (or None).
        local_model_name     : Architecture name for the local branch.
                               Defaults to global_model_name (symmetric).
        local_variant        : Variant string for the local branch (or None).
                               Defaults to global_variant when symmetric.

    Symmetric usage (same arch for both branches):
        model = GlobalLocalWrapper(
            num_classes       = 1,
            global_model_name = 'segformer',
            global_variant    = 'b2',
        )

    Asymmetric usage (different arch per branch):
        model = GlobalLocalWrapper(
            num_classes       = 1,
            global_model_name = 'convnext_upernet', global_variant = 'base',
            local_model_name  = 'unet',             local_variant  = None,
        )

    Training:
        fused, (g_logits, l_logits) = model(global_img, local_patch, return_aux=True)

    Inference:
        pred = model(global_img, local_patch, return_aux=False)
    """

    def __init__(
        self,
        num_classes:       int = 1,
        n_channels:        int = 3,
        global_model_name: str = 'unet',
        global_variant:    str = None,
        local_model_name:  str = None,   # None → mirror global
        local_variant:     str = None,   # None → mirror global_variant
    ):
        super().__init__()

        # Resolve symmetric defaults
        _local_name    = local_model_name if local_model_name is not None else global_model_name
        _local_variant = local_variant    if local_model_name is not None else global_variant

        # Store for parameter_groups() naming and description()
        self.global_model_name = global_model_name
        self.global_variant    = global_variant
        self.local_model_name  = _local_name
        self.local_variant     = _local_variant

        # ── Build branches via the shared model factory ───────────────────
        print(f'  [Wrapper] Global branch : {global_model_name}'
              + (f' ({global_variant})' if global_variant else ''))
        self.global_branch: nn.Module = get_model(
            model_name = global_model_name,
            variant    = global_variant,
            n_channels = n_channels,
            n_classes  = num_classes,
        )

        print(f'  [Wrapper] Local  branch : {_local_name}'
              + (f' ({_local_variant})' if _local_variant else ''))
        self.local_branch: nn.Module = get_model(
            model_name = _local_name,
            variant    = _local_variant,
            n_channels = n_channels,
            n_classes  = num_classes,
        )

        # ── Attention-gated fusion ────────────────────────────────────────
        self.fusion = AttentionFusion(in_channels=num_classes)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        global_img:  torch.Tensor,
        local_patch: torch.Tensor,
        return_aux:  bool = True,
    ):
        """
        Args:
            global_img  : [B, C, H, W] resized full drone image  (scene context)
            local_patch : [B, C, H, W] sliced high-res tile       (fine detail)
            return_aux  : Return branch logits for deep supervision during training.

        Returns:
            return_aux=True  → (fused_logits, (g_logits, l_logits))
            return_aux=False → fused_logits
        """
        g_raw = self.global_branch(global_img)
        l_raw = self.local_branch(local_patch)

        # Some models (e.g. HRNet-OCR) return (main_logits, aux_logits) tuples
        g_logits = g_raw[0] if isinstance(g_raw, tuple) else g_raw
        l_logits = l_raw[0] if isinstance(l_raw, tuple) else l_raw

        # Align spatial dims — different decoder architectures may differ by 1-2px
        if g_logits.shape[2:] != l_logits.shape[2:]:
            g_logits = F.interpolate(
                g_logits, size=l_logits.shape[2:], mode='bilinear', align_corners=False
            )

        fused = self.fusion(g_logits, l_logits)     # [B, num_classes, H, W]

        if return_aux:
            return fused, (g_logits, l_logits)
        return fused

    # ── Optimiser parameter groups ────────────────────────────────────────────

    def parameter_groups(self, base_lr: float, fusion_lr_multiplier: float = 5.0) -> list:
        """
        Three optimiser groups with separate learning rates.

        The fusion head trains faster (starts from scratch) while branch
        backbones fine-tune slowly from their pretrained weights.

        Args:
            base_lr              : LR for both branches.
            fusion_lr_multiplier : LR multiplier for the fusion head (default 5×).
        """
        return [
            {
                'params': self.global_branch.parameters(),
                'lr':     base_lr,
                'name':   f'global_{self.global_model_name}',
            },
            {
                'params': self.local_branch.parameters(),
                'lr':     base_lr,
                'name':   f'local_{self.local_model_name}',
            },
            {
                'params': self.fusion.parameters(),
                'lr':     base_lr * fusion_lr_multiplier,
                'name':   'fusion_head',
            },
        ]

    # ── Description ───────────────────────────────────────────────────────────

    def description(self) -> str:
        g = self.global_model_name + (f'/{self.global_variant}' if self.global_variant else '')
        l = self.local_model_name  + (f'/{self.local_variant}'  if self.local_variant  else '')
        mode = 'symmetric' if g == l else 'asymmetric'
        return f'GlobalLocalWrapper [{mode}]  global={g}  local={l}'
