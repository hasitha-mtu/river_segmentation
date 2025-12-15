"""
ConvNeXt-UPerNet Model
======================

Main model combining ConvNeXt encoder and UPerNet decoder for semantic segmentation.

This hybrid architecture brings together:
- ConvNeXt: Modern CNN with transformer-inspired design
- UPerNet: Powerful multi-scale decoder with PPM and FPN

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import List

from .convnext_encoder import ConvNeXtEncoder, build_convnext_encoder, CONVNEXT_CONFIGS
from .upernet_decoder import UPerNetDecoder


class ConvNeXtUPerNet(nn.Module):
    """
    ConvNeXt-UPerNet: CNN-Transformer Fusion for Semantic Segmentation.
    
    Combines ConvNeXt's efficient CNN design (with transformer principles)
    and UPerNet's powerful multi-scale decoder for superior segmentation.
    
    Architecture Overview:
        Input Image
            ↓
        ConvNeXt Encoder (4 stages)
            ├─→ C1 (1/4 resolution)
            ├─→ C2 (1/8 resolution)
            ├─→ C3 (1/16 resolution)
            └─→ C4 (1/32 resolution)
            ↓
        UPerNet Decoder
            ├─→ PPM (multi-scale context on C4)
            ├─→ FPN (top-down fusion)
            ├─→ Feature aggregation
            └─→ Segmentation head
            ↓
        Segmentation Map
    
    Key Advantages:
    - Efficient feature extraction (ConvNeXt)
    - Multi-scale context (PPM)
    - Strong semantic propagation (FPN)
    - Good accuracy-speed trade-off
    
    Args:
        in_channels: Input image channels (3 for RGB, 1 for luminance)
        num_classes: Number of segmentation classes
        encoder_variant: ConvNeXt size ('tiny', 'small', 'base', 'large', 'xlarge')
        fpn_channels: FPN output channels
        use_pretrained: Load pretrained encoder weights (not implemented)
    
    Input:
        x: (B, C, H, W) - Batch of images
    
    Output:
        logits: (B, num_classes, H, W) - Segmentation logits
    
    Example:
        >>> model = ConvNeXtUPerNet(in_channels=1, num_classes=2, encoder_variant='tiny')
        >>> x = torch.randn(2, 1, 512, 512)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([2, 2, 512, 512])
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        encoder_variant: str = 'tiny',
        fpn_channels: int = 256,
        use_pretrained: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_variant = encoder_variant
        
        # Build ConvNeXt encoder
        self.encoder = build_convnext_encoder(
            variant=encoder_variant,
            in_channels=in_channels
        )
        
        # Build UPerNet decoder
        self.decoder = UPerNetDecoder(
            in_channels_list=self.encoder.out_channels,
            fpn_channels=fpn_channels,
            num_classes=num_classes
        )
        
        if use_pretrained:
            print(f"Warning: Pretrained weights not implemented for {encoder_variant}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        input_size = x.size()[2:]
        
        # Extract multi-scale features
        encoder_features = self.encoder(x)
        
        # Decode to segmentation map
        output = self.decoder(encoder_features, input_size)
        
        return output
    
    def get_params_groups(self, lr: float):
        """
        Get parameter groups for differential learning rates.
        
        Useful for fine-tuning: lower lr for encoder, higher lr for decoder.
        
        Args:
            lr: Base learning rate
        
        Returns:
            List of parameter groups with different learning rates
        """
        return [
            {'params': self.encoder.parameters(), 'lr': lr * 0.1},  # 10x smaller lr
            {'params': self.decoder.parameters(), 'lr': lr}
        ]


def build_convnext_upernet(
    variant: str = 'tiny',
    in_channels: int = 3,
    num_classes: int = 2,
    fpn_channels: int = 256,
    pretrained: bool = False
) -> ConvNeXtUPerNet:
    """
    Factory function to build ConvNeXt-UPerNet variants.
    
    Args:
        variant: Model size ('tiny', 'small', 'base', 'large', 'xlarge')
        in_channels: Input channels (3 for RGB, 1 for luminance)
        num_classes: Number of segmentation classes
        fpn_channels: FPN feature channels
        pretrained: Load pretrained weights
    
    Returns:
        ConvNeXtUPerNet model
    
    Variants:
        - tiny:   ~28M params, fastest inference
        - small:  ~50M params, balanced
        - base:   ~89M params, good accuracy
        - large:  ~197M params, high accuracy
        - xlarge: ~350M params, best accuracy
    
    Example:
        >>> # For your research: luminance-only water segmentation
        >>> model = build_convnext_upernet(
        ...     variant='tiny',
        ...     in_channels=1,  # Luminance only (103.9% contribution)
        ...     num_classes=2   # Water vs. non-water
        ... )
    """
    if variant not in CONVNEXT_CONFIGS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Choose from {list(CONVNEXT_CONFIGS.keys())}"
        )
    
    model = ConvNeXtUPerNet(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_variant=variant,
        fpn_channels=fpn_channels,
        use_pretrained=pretrained
    )
    
    return model


if __name__ == "__main__":
    print("=" * 80)
    print("ConvNeXt-UPerNet Model")
    print("=" * 80)
    
    # Create model for water segmentation (your research)
    model = build_convnext_upernet(
        variant='tiny',
        in_channels=1,  # Luminance-only based on your findings
        num_classes=2   # Water vs. non-water
    )
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 512, 512)
    
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
    print("All ConvNeXt-UPerNet Variants")
    print("=" * 80)
    
    for variant in ['tiny', 'small', 'base']:
        model = build_convnext_upernet(variant=variant, in_channels=1, num_classes=2)
        params = sum(p.numel() for p in model.parameters())
        print(f"{variant.capitalize():10s}: {params:,} parameters "
              f"({params * 4 / 1024 / 1024:.1f} MB)")
