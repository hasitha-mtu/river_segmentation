"""
SAM Segmentation Module
=======================

Segment Anything Model (SAM) adapted for semantic segmentation.
"""

from .sam_encoder import SAMImageEncoder, build_sam_encoder
from .sam_decoder import FPNDecoder
from .model import SAMSegmentation, build_sam_segmentation

__all__ = [
    'SAMImageEncoder',
    'build_sam_encoder',
    'FPNDecoder',
    'SAMSegmentation',
    'build_sam_segmentation'
]
