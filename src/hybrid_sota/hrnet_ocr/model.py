"""
HRNet-OCR Model
===============

Main model combining HRNet backbone and OCR module for semantic segmentation.

This hybrid architecture brings together:
- HRNet: High-resolution representations throughout the network
- OCR: Object-aware context modeling for enhanced segmentation

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import Union, Tuple

from .hrnet_backbone import HRNetBackbone, build_hrnet_backbone, HRNET_CONFIGS
from .ocr_module import OCRHead


class HRNetOCR(nn.Module):
    """
    HRNet-OCR: High-Resolution Network with Object Contextual Representations.
    
    Combines HRNet's high-resolution representation learning with OCR's
    object-aware context modeling for superior semantic segmentation.
    
    Architecture:
        Input Image
            ↓
        HRNet Backbone (4 parallel branches)
            ├─→ Branch 1 (1/4 resolution, HR)
            ├─→ Branch 2 (1/8 resolution)
            ├─→ Branch 3 (1/16 resolution)
            └─→ Branch 4 (1/32 resolution, LR)
            ↓
        Feature Aggregation
            ↓
        OCR Module
            ├─→ Soft object regions
            ├─→ Pixel-region attention
            └─→ Context augmentation
            ↓
        Segmentation Head
            ↓
        Outputs:
            - Main segmentation
            - Auxiliary segmentation (training only)
    
    Key advantages:
        - Maintains spatial details (HRNet)
        - Object-aware context (OCR)
        - Efficient multi-scale fusion
        - State-of-the-art accuracy
    
    Args:
        in_channels: Input image channels (3 for RGB, 1 for luminance)
        num_classes: Number of segmentation classes
        backbone_variant: HRNet size ('w18', 'w32', 'w48')
        ocr_mid_channels: OCR intermediate channels
        ocr_key_channels: OCR attention channels
        use_pretrained: Load pretrained backbone (not implemented)
    
    Input:
        x: (B, C, H, W) - Batch of images
    
    Output:
        During training (return_aux=True): (main_out, aux_out)
        During inference (return_aux=False): main_out
        Shape: (B, num_classes, H, W)
    
    Example:
        >>> model = HRNetOCR(in_channels=1, num_classes=2, backbone_variant='w48')
        >>> x = torch.randn(2, 1, 512, 512)
        >>> 
        >>> # Training mode
        >>> model.train()
        >>> main_out, aux_out = model(x)
        >>> 
        >>> # Inference mode
        >>> model.eval()
        >>> output = model(x)
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        backbone_variant: str = 'w48',
        ocr_mid_channels: int = 512,
        ocr_key_channels: int = 256,
        use_pretrained: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_variant = backbone_variant
        
        # Build HRNet backbone
        self.backbone = build_hrnet_backbone(
            variant=backbone_variant,
            in_channels=in_channels
        )
        
        # Build OCR head
        self.ocr_head = OCRHead(
            in_channels_list=self.backbone.out_channels,
            hidden_channels=self.backbone.out_channels[0],  # Use HR branch channels
            ocr_mid_channels=ocr_mid_channels,
            ocr_key_channels=ocr_key_channels,
            num_classes=num_classes
        )
        
        if use_pretrained:
            print(f"Warning: Pretrained weights not implemented for {backbone_variant}")
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input images (B, C, H, W)
            return_aux: Return auxiliary outputs (auto-detects from training mode if None)
        
        Returns:
            If training or return_aux=True: (main_output, aux_output)
            Otherwise: main_output only
        """
        input_size = x.size()[2:]
        
        # Extract multi-resolution features from HRNet
        features = self.backbone(x)
        
        # Determine whether to return auxiliary output
        if self.training:
            return_aux = True
        else:
            return_aux = False
        
        # Apply OCR head
        if return_aux:
            main_out, aux_out = self.ocr_head(features, input_size, return_aux=True)
            return main_out, aux_out
        else:
            main_out = self.ocr_head(features, input_size, return_aux=False)
            return main_out
    
    def get_params_groups(self, lr: float):
        """
        Get parameter groups for differential learning rates.
        
        Useful for fine-tuning: lower lr for backbone, higher lr for OCR head.
        
        Args:
            lr: Base learning rate
        
        Returns:
            List of parameter groups with different learning rates
        """
        return [
            {'params': self.backbone.parameters(), 'lr': lr * 0.1},  # 10x smaller lr
            {'params': self.ocr_head.parameters(), 'lr': lr}
        ]


def build_hrnet_ocr(
    variant: str = 'w48',
    in_channels: int = 3,
    num_classes: int = 2,
    ocr_mid_channels: int = 512,
    ocr_key_channels: int = 256,
    pretrained: bool = False
) -> HRNetOCR:
    """
    Factory function to build HRNet-OCR variants.
    
    Args:
        variant: Model size ('w18', 'w32', 'w48')
            - w18: Lightweight, ~22M params, faster inference
            - w32: Balanced, ~42M params, good accuracy
            - w48: Best accuracy, ~66M params, state-of-the-art
        in_channels: Input channels (3 for RGB, 1 for luminance)
        num_classes: Number of segmentation classes
        ocr_mid_channels: OCR module intermediate channels
        ocr_key_channels: OCR attention key/query channels
        pretrained: Load pretrained weights
    
    Returns:
        HRNetOCR model
    
    Example:
        >>> # For your research: luminance-only water segmentation
        >>> model = build_hrnet_ocr(
        ...     variant='w32',
        ...     in_channels=1,  # Luminance only (103.9% contribution)
        ...     num_classes=2   # Water vs. non-water
        ... )
        >>> 
        >>> # Training with auxiliary loss
        >>> model.train()
        >>> main_out, aux_out = model(images)
        >>> main_loss = criterion(main_out, masks)
        >>> aux_loss = criterion(aux_out, masks)
        >>> total_loss = main_loss + 0.4 * aux_loss
    """
    if variant not in HRNET_CONFIGS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Choose from {list(HRNET_CONFIGS.keys())}"
        )
    
    # Adjust OCR channels based on variant
    config = HRNET_CONFIGS[variant]
    
    if variant == 'w18':
        # Smaller model - reduce OCR channels
        ocr_mid_channels = min(ocr_mid_channels, 256)
        ocr_key_channels = min(ocr_key_channels, 128)
    elif variant == 'w32':
        # Medium model - standard OCR channels
        ocr_mid_channels = min(ocr_mid_channels, 512)
        ocr_key_channels = min(ocr_key_channels, 256)
    # w48 uses full specified channels
    
    model = HRNetOCR(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone_variant=variant,
        ocr_mid_channels=ocr_mid_channels,
        ocr_key_channels=ocr_key_channels,
        use_pretrained=pretrained
    )
    
    return model


if __name__ == "__main__":
    print("=" * 80)
    print("HRNet-OCR Model")
    print("=" * 80)
    
    # Create model for water segmentation (your research)
    model = build_hrnet_ocr(
        variant='w48',
        in_channels=1,  # Luminance-only based on your findings
        num_classes=2   # Water vs. non-water
    )
    
    # Test forward pass - training mode
    model.train()
    batch_size = 2
    x = torch.randn(batch_size, 1, 512, 512)
    
    print(f"\nInput shape: {x.shape}")
    print("\nTraining mode:")
    
    with torch.no_grad():
        main_out, aux_out = model(x)
    
    print(f"  Main output shape: {main_out.shape}")
    print(f"  Auxiliary output shape: {aux_out.shape}")
    
    # Test inference mode
    model.eval()
    print("\nInference mode:")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    ocr_params = sum(p.numel() for p in model.ocr_head.parameters())
    
    print(f"\nParameter breakdown:")
    print(f"  Backbone: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
    print(f"  OCR Head: {ocr_params:,} ({ocr_params/total_params*100:.1f}%)")
    print(f"  Total:    {total_params:,}")
    print(f"  Size:     {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test all variants
    print("\n" + "=" * 80)
    print("All HRNet-OCR Variants")
    print("=" * 80)
    
    for variant in ['w18', 'w32', 'w48']:
        model = build_hrnet_ocr(variant=variant, in_channels=1, num_classes=2)
        params = sum(p.numel() for p in model.parameters())
        print(f"{variant.upper():10s}: {params:,} parameters "
              f"({params * 4 / 1024 / 1024:.1f} MB)")
