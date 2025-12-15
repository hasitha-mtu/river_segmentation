"""
HRNet-OCR Package
=================

High-Resolution Network with Object Contextual Representations for semantic segmentation.

Components:
- hrnet_backbone: HRNet backbone with parallel multi-resolution branches
- ocr_module: Object Contextual Representations for context modeling
- model: Complete HRNet-OCR architecture

Author: Hasitha
Date: December 2025
"""

from .hrnet_backbone import (
    HRNetBackbone,
    HighResolutionModule,
    BasicBlock,
    Bottleneck,
    build_hrnet_backbone,
    HRNET_CONFIGS
)

from .ocr_module import (
    SpatialOCRModule,
    ObjectContextBlock,
    OCRHead
)

from .model import (
    HRNetOCR,
    build_hrnet_ocr
)

__all__ = [
    # Backbone components
    'HRNetBackbone',
    'HighResolutionModule',
    'BasicBlock',
    'Bottleneck',
    'build_hrnet_backbone',
    'HRNET_CONFIGS',
    
    # OCR components
    'SpatialOCRModule',
    'ObjectContextBlock',
    'OCRHead',
    
    # Main model
    'HRNetOCR',
    'build_hrnet_ocr'
]

__version__ = '1.0.0'
