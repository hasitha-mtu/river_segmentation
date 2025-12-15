"""
ConvNeXt-UPerNet Package
========================

CNN-Transformer hybrid for semantic segmentation.

Components:
- convnext_encoder: ConvNeXt backbone for feature extraction
- upernet_decoder: UPerNet decoder with PPM and FPN
- model: Complete ConvNeXt-UPerNet architecture

Author: Hasitha
Date: December 2025
"""

from .convnext_encoder import (
    ConvNeXtEncoder,
    ConvNeXtBlock,
    LayerNorm2d,
    DropPath,
    build_convnext_encoder,
    CONVNEXT_CONFIGS
)

from .upernet_decoder import (
    UPerNetDecoder,
    PyramidPoolingModule,
    FPNDecoder
)

from .model import (
    ConvNeXtUPerNet,
    build_convnext_upernet
)

__all__ = [
    # Encoder components
    'ConvNeXtEncoder',
    'ConvNeXtBlock',
    'LayerNorm2d',
    'DropPath',
    'build_convnext_encoder',
    'CONVNEXT_CONFIGS',
    
    # Decoder components
    'UPerNetDecoder',
    'PyramidPoolingModule',
    'FPNDecoder',
    
    # Main model
    'ConvNeXtUPerNet',
    'build_convnext_upernet'
]

__version__ = '1.0.0'
