"""
DINOv2 Segmentation Decoder
============================

FPN decoder for semantic segmentation using DINOv2 encoder features.

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network decoder for multi-scale feature fusion.
    
    Args:
        in_channels_list: List of input channels from encoder
        fpn_channels: Channels for FPN layers
        num_classes: Number of output classes
    """
    def __init__(
        self,
        in_channels_list: List[int],
        fpn_channels: int = 256,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.fpn_channels = fpn_channels
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        # Top-down pathway
        self.fpn_convs = nn.ModuleList([
            ConvBlock(fpn_channels, fpn_channels)
            for _ in range(len(in_channels_list))
        ])
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            ConvBlock(fpn_channels * len(in_channels_list), fpn_channels),
            ConvBlock(fpn_channels, fpn_channels),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
        )
    
    def forward(
        self,
        features: List[torch.Tensor],
        input_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: List of features from encoder
            input_size: Target output size (H, W)
        
        Returns:
            Segmentation output (B, num_classes, H, W)
        """
        # Lateral connections
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]
        
        # Top-down pathway
        fpn_features = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                fpn_feat = laterals[i]
            else:
                upsampled = F.interpolate(
                    fpn_feat, size=laterals[i].shape[2:],
                    mode='bilinear', align_corners=False
                )
                fpn_feat = laterals[i] + upsampled
            
            fpn_feat = self.fpn_convs[i](fpn_feat)
            fpn_features.append(fpn_feat)
        
        fpn_features = fpn_features[::-1]
        
        # Upsample to highest resolution
        target_size = fpn_features[0].shape[2:]
        upsampled_features = []
        for feat in fpn_features:
            upsampled = F.interpolate(
                feat, size=target_size,
                mode='bilinear', align_corners=False
            )
            upsampled_features.append(upsampled)
        
        # Concatenate
        combined = torch.cat(upsampled_features, dim=1)
        
        # Segmentation head
        output = self.seg_head(combined)
        
        # Upsample to input size
        output = F.interpolate(
            output, size=input_size,
            mode='bilinear', align_corners=False
        )
        
        return output


if __name__ == "__main__":
    print("Testing DINOv2 Decoder")
    
    features = [
        torch.randn(2, 96, 128, 128),
        torch.randn(2, 192, 64, 64),
        torch.randn(2, 384, 32, 32),
        torch.randn(2, 768, 16, 16),
    ]
    
    decoder = FPNDecoder(
        in_channels_list=[96, 192, 384, 768],
        fpn_channels=256,
        num_classes=2
    )
    
    output = decoder(features, input_size=(518, 518))
    print(f"Output: {output.shape}")
    
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {params:,}")
