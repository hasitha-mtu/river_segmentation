"""
DINOv2 Segmentation Model
==========================

Complete model combining DINOv2 encoder and FPN decoder for semantic segmentation.

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Union

from .dinov2_encoder import DINOv2Encoder, build_dinov2_encoder, DINOV2_CONFIGS
from .dinov2_decoder import FPNDecoder


class DINOv2Segmentation(nn.Module):
    """
    DINOv2-based semantic segmentation model.
    
    Combines:
    - DINOv2 ViT encoder (self-supervised)
    - FPN decoder for segmentation
    
    Args:
        in_channels: Input image channels (3 for RGB)
        num_classes: Number of segmentation classes
        encoder_variant: DINOv2 encoder size ('vit_s', 'vit_b', 'vit_l', 'vit_g')
        img_size: Input image size
        fpn_channels: FPN intermediate channels
    
    Example:
        >>> model = DINOv2Segmentation(in_channels=3, num_classes=2, encoder_variant='vit_b')
        >>> x = torch.randn(2, 3, 518, 518)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([2, 2, 518, 518])
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        encoder_variant: str = 'vit_b',
        img_size: int = 512,
        fpn_channels: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_variant = encoder_variant
        
        # Build encoder
        self.encoder = build_dinov2_encoder(
            variant=encoder_variant,
            img_size=img_size,
            in_channels=in_channels
        )
        
        # Build decoder
        self.decoder = FPNDecoder(
            in_channels_list=self.encoder.out_channels,
            fpn_channels=fpn_channels,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Segmentation output (B, num_classes, H, W)
        """
        input_size = x.size()[2:]
        
        # Extract features
        features = self.encoder(x)
        
        # Decode to segmentation
        output = self.decoder(features, input_size)
        
        return output
    
    def get_params_groups(self, lr: float):
        """
        Get parameter groups for differential learning rates.
        
        Args:
            lr: Base learning rate
        
        Returns:
            List of parameter groups
        """
        return [
            {'params': self.encoder.parameters(), 'lr': lr * 0.1},
            {'params': self.decoder.parameters(), 'lr': lr}
        ]


def build_dinov2_segmentation(
    variant: str = 'vit_b',
    in_channels: int = 3,
    num_classes: int = 1,
    img_size: int = 512,
    fpn_channels: int = 256
) -> DINOv2Segmentation:
    """
    Factory function to build DINOv2 segmentation variants.
    
    Args:
        variant: Model size ('vit_s', 'vit_b', 'vit_l', 'vit_g')
            - vit_s: Small model, ~22M params, fast inference
            - vit_b: Base model, ~86M params, good accuracy
            - vit_l: Large model, ~304M params, better accuracy
            - vit_g: Giant model, ~1.1B params, best accuracy
        in_channels: Input channels (3 for RGB)
        num_classes: Number of segmentation classes
        img_size: Input image size (default 518 for DINOv2)
        fpn_channels: FPN channels
    
    Returns:
        DINOv2Segmentation model
    
    Example:
        >>> model = build_dinov2_segmentation('vit_b', in_channels=3, num_classes=2)
        >>> x = torch.randn(2, 3, 518, 518)
        >>> output = model(x)
    """
    if variant not in DINOV2_CONFIGS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Choose from {list(DINOV2_CONFIGS.keys())}"
        )
    
    model = DINOv2Segmentation(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_variant=variant,
        img_size=img_size,
        fpn_channels=fpn_channels
    )
    
    return model


if __name__ == "__main__":
    print("=" * 80)
    print("DINOv2 Segmentation Model")
    print("=" * 80)
    
    # Create model
    model = build_dinov2_segmentation(
        variant='vit_b',
        in_channels=3,
        num_classes=1,
        img_size=512
    )
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\nParameter breakdown:")
    print(f"  Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    print(f"  Total:   {total_params:,}")
    print(f"  Size:    {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test all variants
    print("\n" + "=" * 80)
    print("All DINOv2 Variants")
    print("=" * 80)
    
    for variant in ['vit_s', 'vit_b', 'vit_l', 'vit_g']:
        model = build_dinov2_segmentation(variant=variant, in_channels=3, num_classes=2, img_size=518)
        params = sum(p.numel() for p in model.parameters())
        print(f"{variant.upper():10s}: {params:,} parameters "
              f"({params * 4 / 1024 / 1024:.1f} MB)")
