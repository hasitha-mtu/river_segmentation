"""
UPerNet Decoder Module
======================

Implements Unified Perceptual Parsing Network (UPerNet) decoder.
UPerNet provides powerful multi-scale feature fusion for dense prediction.

Key components:
- Pyramid Pooling Module (PPM) for multi-scale context
- Feature Pyramid Network (FPN) for top-down feature fusion
- Multi-scale feature aggregation

Reference:
    Xiao et al. "Unified Perceptual Parsing for Scene Understanding" ECCV 2018

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) for multi-scale context aggregation.
    
    Captures context at multiple scales using adaptive pooling at different
    grid sizes. Originally introduced in PSPNet, widely used in segmentation.
    
    The idea is to pool features at multiple scales (1×1, 2×2, 3×3, 6×6),
    process them through 1×1 convs, then upsample and concatenate.
    This captures both global context (1×1) and local patterns (6×6).
    
    Architecture:
        Input Features (C, H, W)
            ├─→ AdaptiveAvgPool(1×1) → Conv1x1 → Upsample → (C', H, W)
            ├─→ AdaptiveAvgPool(2×2) → Conv1x1 → Upsample → (C', H, W)
            ├─→ AdaptiveAvgPool(3×3) → Conv1x1 → Upsample → (C', H, W)
            └─→ AdaptiveAvgPool(6×6) → Conv1x1 → Upsample → (C', H, W)
        
        Concatenate all → Conv → Output
    
    Args:
        in_channels: Input feature channels
        pool_scales: Pooling grid sizes (e.g., [1, 2, 3, 6])
        channels: Intermediate channels after each pooling
    """
    def __init__(
        self,
        in_channels: int,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
        channels: int = 512
    ):
        super().__init__()
        
        # Create pooling stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for scale in pool_scales
        ])
        
        # Final fusion convolution
        # Input: original features + all pooled features
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        
        Returns:
            Context-enriched features (B, C, H, W)
        """
        size = x.size()[2:]  # (H, W)
        
        # Multi-scale pooling
        pyramid_features = [x]  # Start with original features
        
        for stage in self.stages:
            # Pool to specific grid size
            pooled = stage(x)
            
            # Upsample back to original size
            upsampled = F.interpolate(
                pooled,
                size=size,
                mode='bilinear',
                align_corners=False
            )
            pyramid_features.append(upsampled)
        
        # Concatenate all features and fuse
        x = torch.cat(pyramid_features, dim=1)
        x = self.bottleneck(x)
        
        return x


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network (FPN) Decoder for UPerNet.
    
    FPN creates a feature pyramid with strong semantics at all scales
    through a top-down pathway and lateral connections.
    
    The key idea:
    1. Start from deepest (semantically strong) features
    2. Progressively upsample and add to higher-resolution features
    3. Refine with 3×3 convolutions
    
    This creates a pyramid where all levels have rich semantic information.
    
    Architecture:
        C4 (deepest) ──→ Lateral Conv ──→ P4
                              ↓ ⊕ Upsample
        C3 ──────────→ Lateral Conv ──→ P3
                              ↓ ⊕ Upsample
        C2 ──────────→ Lateral Conv ──→ P2
                              ↓ ⊕ Upsample
        C1 ──────────→ Lateral Conv ──→ P1
        
        Where ⊕ is element-wise addition
    
    Args:
        in_channels_list: List of channels for each encoder feature [C1, C2, C3, C4]
        out_channels: Unified channel dimension for all FPN features
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256
    ):
        super().__init__()
        
        # Lateral convs: Reduce encoder feature channels to uniform dimension
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, kernel_size=1)
            for in_c in in_channels_list
        ])
        
        # FPN convs: Refine features after fusion
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(in_channels_list))
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Top-down feature fusion with lateral connections.
        
        Args:
            features: Multi-scale features from encoder [C1, C2, C3, C4]
                     Ordered from high-resolution to low-resolution
        
        Returns:
            FPN features [P1, P2, P3, P4] with uniform channels
        """
        # Apply lateral convs to all features
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway (from deep to shallow)
        # Start from the deepest feature and work backwards
        for i in range(len(laterals) - 1, 0, -1):
            # Get size of the higher-resolution feature
            size = laterals[i - 1].shape[2:]
            
            # Upsample deeper feature and add to lateral
            upsampled = F.interpolate(
                laterals[i],
                size=size,
                mode='bilinear',
                align_corners=False
            )
            laterals[i - 1] = laterals[i - 1] + upsampled
        
        # Apply FPN convs for refinement
        fpn_features = [
            fpn_conv(laterals[i])
            for i, fpn_conv in enumerate(self.fpn_convs)
        ]
        
        return fpn_features


class UPerNetDecoder(nn.Module):
    """
    Complete UPerNet decoder combining PPM and FPN.
    
    UPerNet decoder strategy:
    1. Apply PPM to deepest features (C4) for multi-scale context
    2. Use FPN to fuse features across all scales
    3. Upsample all FPN features to common resolution
    4. Concatenate and fuse for final prediction
    
    This design provides:
    - Multi-scale context (PPM)
    - Top-down semantic propagation (FPN)
    - High-resolution details (feature fusion)
    
    Args:
        in_channels_list: Channels from encoder [C1, C2, C3, C4]
        fpn_channels: Unified FPN channel dimension
        num_classes: Number of segmentation classes
        ppm_pool_scales: PPM pooling scales
        ppm_channels: PPM intermediate channels
    """
    def __init__(
        self,
        in_channels_list: List[int],
        fpn_channels: int = 256,
        num_classes: int = 2,
        ppm_pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
        ppm_channels: int = 512
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Pyramid Pooling Module on deepest features (C4)
        self.ppm = PyramidPoolingModule(
            in_channels=in_channels_list[-1],
            pool_scales=ppm_pool_scales,
            channels=ppm_channels
        )
        
        # FPN Decoder for multi-scale fusion
        self.fpn_decoder = FPNDecoder(
            in_channels_list=in_channels_list,
            out_channels=fpn_channels
        )
        
        # Feature fusion: Concatenate all FPN features
        self.fusion = nn.Sequential(
            nn.Conv2d(
                fpn_channels * 4,  # Concatenate 4 FPN levels
                fpn_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        encoder_features: List[torch.Tensor],
        input_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Decode encoder features to segmentation map.
        
        Args:
            encoder_features: Multi-scale features [C1, C2, C3, C4]
            input_size: Original input size (H, W) for final upsampling
        
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # 1. Apply PPM to deepest features for multi-scale context
        encoder_features[-1] = self.ppm(encoder_features[-1])
        
        # 2. FPN fusion across all scales
        fpn_features = self.fpn_decoder(encoder_features)  # [P1, P2, P3, P4]
        
        # 3. Upsample all FPN features to same size (highest resolution)
        target_size = fpn_features[0].size()[2:]  # Size of P1
        upsampled_features = []
        
        for feat in fpn_features:
            if feat.size()[2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            upsampled_features.append(feat)
        
        # 4. Concatenate and fuse
        fused = torch.cat(upsampled_features, dim=1)
        fused = self.fusion(fused)
        
        # 5. Generate segmentation logits
        out = self.head(fused)
        
        # 6. Upsample to original input resolution
        out = F.interpolate(
            out,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )
        
        return out


if __name__ == "__main__":
    # Test UPerNet decoder
    print("Testing UPerNet Decoder...")
    
    # Simulate encoder features
    batch_size = 2
    encoder_features = [
        torch.randn(batch_size, 96, 128, 128),   # C1: 1/4 resolution
        torch.randn(batch_size, 192, 64, 64),    # C2: 1/8 resolution
        torch.randn(batch_size, 384, 32, 32),    # C3: 1/16 resolution
        torch.randn(batch_size, 768, 16, 16),    # C4: 1/32 resolution
    ]
    
    decoder = UPerNetDecoder(
        in_channels_list=[96, 192, 384, 768],
        fpn_channels=256,
        num_classes=2
    )
    
    output = decoder(encoder_features, input_size=(512, 512))
    
    print(f"\nEncoder features:")
    for i, feat in enumerate(encoder_features):
        print(f"  C{i+1}: {feat.shape}")
    
    print(f"\nOutput shape: {output.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {params:,}")
