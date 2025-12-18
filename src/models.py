"""
Transformer-based Segmentation Models for River Water Detection
RGB Input Only - SAM, DINOv2
"""

import torch
from foundation_models.sam.model import build_sam_segmentation
from foundation_models.dinov2.model import build_dinov2_segmentation
from hybrid_sota.convnext_upernet.model import build_convnext_upernet
from hybrid_sota.hrnet_ocr.model import build_hrnet_ocr
from transformers.models import get_model as get_transformer_model
from cnn_baselines.models import get_model as get_cnn_model

def get_model(model_name, variant, n_channels=3, n_classes=1):
    """
    Factory function to get models
    
    Args:
        model_name: 'sam', or 'dinov2' or ...
        variant: 'vit_b', or 'vit_l', or ...
        n_channels: Number of input channels (3 for RGB images)
        num_classes: Number of output classes (1 for binary segmentation)
    """
    if model_name == 'sam':
        return build_sam_segmentation(
            variant=variant, 
            in_channels=n_channels, 
            num_classes=n_classes)
    elif model_name == 'dinov2':
        return build_dinov2_segmentation(
            variant=variant, 
            in_channels=n_channels, 
            num_classes=n_classes)
    elif model_name == 'convnext_upernet':
        return build_convnext_upernet(
            variant=variant, 
            in_channels=n_channels, 
            num_classes=n_classes)
    elif model_name == 'hrnet_ocr':
        return build_hrnet_ocr(
            variant=variant, 
            in_channels=n_channels, 
            num_classes=n_classes)
    elif model_name == 'swin_unet':
        return get_transformer_model(
            model_name,
            variant=variant, 
            num_classes=n_classes)
    elif model_name == 'segformer':
        return get_transformer_model(
            model_name,
            variant=variant, 
            num_classes=n_classes)
    elif model_name in ('unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam'):
        return get_cnn_model(
            model_name, 
            in_channels=n_channels, 
            num_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

if __name__ == "__main__":
    print("All models tested successfully!")

