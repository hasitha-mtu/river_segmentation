"""
wrapper.py
==========
GlobalLocalWrapper — dual-branch segmentation model for UAV river imagery.

Architecture overview
---------------------
                    ┌─────────────────────┐
  global_img ──────►│  Global Branch      │──► g_logits  (full river path context)
  (resized 512²)    │  (any smp encoder)  │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
  local_patch ─────►│  Local Branch       │──► l_logits  (fine edges, thin structures)
  (sliced 512²)     │  (any smp encoder)  │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Attention Fusion   │──► fused_logits
                    │  (spatial gating)   │
                    └─────────────────────┘

Why attention-gated fusion?
  A plain 1×1 conv treats global and local contributions as globally fixed.
  A spatial attention gate lets the model learn WHERE to trust global context
  (e.g., under dense canopy where local features are occluded) vs. local
  detail (e.g., clear riverbank edges).

Outputs
-------
Training  : (fused_logits, (g_logits, l_logits))  — aux heads for deep supervision
Inference : fused_logits

The training output tuple is compatible with the existing combined loss which
already handles (main_out, aux_out) from the UnifiedTrainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class AttentionFusion(nn.Module):
    """
    Spatial attention-gated fusion of global and local feature maps.

    Learns a per-pixel gate α ∈ (0, 1):
        fused = α * global + (1 - α) * local

    where α is predicted from the concatenation of both feature maps.
    This is more expressive than a fixed 1×1 conv because the gate is
    conditioned on both streams jointly.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # Gate predictor: concat(global, local) → 1-channel attention map
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),                      # α ∈ (0, 1) per spatial location
        )
        # Final projection after weighted blend
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
        # Predict spatial attention gate from both streams
        alpha = self.gate(torch.cat([global_feat, local_feat], dim=1))  # [B, 1, H, W]
        # Weighted blend: high alpha → trust global more (useful under canopy)
        blended = alpha * global_feat + (1.0 - alpha) * local_feat      # [B, C, H, W]
        return self.project(blended)


class GlobalLocalWrapper(nn.Module):
    """
    Dual-branch segmentation wrapper combining a global context branch with
    a local detail branch, fused via learned spatial attention gating.

    Args:
        num_classes  : Number of output channels (1 for binary river segmentation)
        encoder_name : SMP encoder backbone (default: 'resnet34')
        encoder_weights: Pre-trained weights (default: 'imagenet')
        aux_loss     : Return auxiliary branch logits for deep supervision
                       during training (default: True)

    Usage:
        model = GlobalLocalWrapper(num_classes=1)

        # Training:
        fused, (g_aux, l_aux) = model(global_img, local_patch, return_aux=True)

        # Inference:
        pred = model(global_img, local_patch, return_aux=False)
    """

    def __init__(
        self,
        num_classes: int = 1,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
    ):
        super().__init__()

        shared_kwargs = dict(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,         # raw logits — sigmoid/softmax applied in loss
        )

        # Global branch: sees the full scene context (river path, surrounding terrain)
        self.global_branch = smp.Unet(**shared_kwargs)

        # Local branch: sees high-resolution tile details (thin edges, fine texture)
        self.local_branch = smp.Unet(**shared_kwargs)

        # Attention-gated fusion of both branch outputs
        self.fusion = AttentionFusion(in_channels=num_classes)

    def forward(
        self,
        global_img: torch.Tensor,
        local_patch: torch.Tensor,
        return_aux: bool = True,
    ):
        """
        Args:
            global_img  : [B, 3, H, W] — resized full drone image (context)
            local_patch : [B, 3, H, W] — sliced high-res tile (detail)
            return_aux  : Return auxiliary branch logits for deep supervision

        Returns:
            Training (return_aux=True)  : (fused_logits, (g_logits, l_logits))
            Inference (return_aux=False): fused_logits
        """
        g_logits = self.global_branch(global_img)    # [B, C, H, W]
        l_logits = self.local_branch(local_patch)    # [B, C, H, W]

        # Ensure spatial dimensions match before fusion (handles edge-case mismatches)
        if g_logits.shape != l_logits.shape:
            g_logits = F.interpolate(
                g_logits, size=l_logits.shape[2:], mode='bilinear', align_corners=False
            )

        fused = self.fusion(g_logits, l_logits)      # [B, C, H, W]

        if return_aux:
            return fused, (g_logits, l_logits)
        return fused

    def parameter_groups(self, base_lr: float, fusion_lr_multiplier: float = 5.0):
        """
        Return parameter groups with different learning rates.

        The fusion head is trained at a higher LR because it starts from scratch,
        while the encoder backbones benefit from slower fine-tuning of ImageNet weights.

        Args:
            base_lr              : Learning rate for backbone branches
            fusion_lr_multiplier : LR multiplier for the fusion head (default: 5×)

        Returns:
            List of dicts suitable for torch.optim.* constructors.
        """
        return [
            {'params': self.global_branch.parameters(), 'lr': base_lr,
             'name': 'global_branch'},
            {'params': self.local_branch.parameters(),  'lr': base_lr,
             'name': 'local_branch'},
            {'params': self.fusion.parameters(),        'lr': base_lr * fusion_lr_multiplier,
             'name': 'fusion'},
        ]
