"""
SAM Segmentation Model
======================

Complete model combining SAM encoder and FPN decoder for semantic segmentation.

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Union

from .sam_encoder import SAMImageEncoder, build_sam_encoder, SAM_CONFIGS
from .sam_decoder import FPNDecoder


class SAMSegmentation(nn.Module):
    """
    SAM-based semantic segmentation model.
    
    Combines:
    - SAM ViT encoder for feature extraction
    - FPN decoder for multi-scale segmentation
    
    Args:
        in_channels: Input image channels (3 for RGB)
        num_classes: Number of segmentation classes
        encoder_variant: SAM encoder size ('vit_b', 'vit_l', 'vit_h')
        img_size: Input image size
        fpn_channels: FPN intermediate channels
    
    Example:
        >>> model = SAMSegmentation(in_channels=3, num_classes=2, encoder_variant='vit_b')
        >>> x = torch.randn(2, 3, 512, 512)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([2, 2, 512, 512])
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        encoder_variant: str = 'vit_b',
        img_size: int = 1024,
        fpn_channels: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_variant = encoder_variant
        
        # Build encoder
        self.encoder = build_sam_encoder(
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
            {'params': self.encoder.parameters(), 'lr': lr * 0.1},  # 10x smaller lr
            {'params': self.decoder.parameters(), 'lr': lr}
        ]


def build_sam_segmentation(
    variant: str = 'vit_b',
    in_channels: int = 3,
    num_classes: int = 1,
    img_size: int = 512,
    fpn_channels: int = 256
) -> SAMSegmentation:
    """
    Factory function to build SAM segmentation variants.
    
    Args:
        variant: Model size ('vit_b', 'vit_l', 'vit_h')
            - vit_b: Base model, ~90M params, good accuracy
            - vit_l: Large model, ~310M params, better accuracy
            - vit_h: Huge model, ~640M params, best accuracy
        in_channels: Input channels (3 for RGB)
        num_classes: Number of segmentation classes
        img_size: Input image size
        fpn_channels: FPN channels
    
    Returns:
        SAMSegmentation model
    
    Example:
        >>> model = build_sam_segmentation('vit_b', in_channels=3, num_classes=2)
        >>> x = torch.randn(2, 3, 512, 512)
        >>> output = model(x)
    """
    if variant not in SAM_CONFIGS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Choose from {list(SAM_CONFIGS.keys())}"
        )
    
    model = SAMSegmentation(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_variant=variant,
        img_size=img_size,
        fpn_channels=fpn_channels
    )
    
    return model


if __name__ == "__main__":
    print("=" * 80)
    print("SAM Segmentation Model")
    print("=" * 80)
    
    # Create model
    model = build_sam_segmentation(
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
    print("All SAM Variants")
    print("=" * 80)
    
    for variant in ['vit_b', 'vit_l', 'vit_h']:
        model = build_sam_segmentation(variant=variant, in_channels=3, num_classes=2, img_size=512)
        params = sum(p.numel() for p in model.parameters())
        print(f"{variant.upper():10s}: {params:,} parameters "
              f"({params * 4 / 1024 / 1024:.1f} MB)")
