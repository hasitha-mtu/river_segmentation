"""
Transformer-based Segmentation Models for River Water Detection
RGB Input Only - ConvNeXtUPerNet, HRNetOCR
"""

import torch
from convnext_upernet.model import build_convnext_upernet
from hrnet_ocr.model import build_hrnet_ocr

def get_model(model_name, varient, n_channels=3, n_classes=1):
    """
    Factory function to get models
    
    Args:
        model_name: 'convnext_upernet', or 'hrnet_ocr'
        n_channels: Number of input channels (3 for RGB images)
        num_classes: Number of output classes (1 for binary segmentation)
    """
    if model_name == 'convnext_upernet':
        return build_convnext_upernet(
            variant=varient, 
            in_channels=n_channels, 
            num_classes=n_classes)
    elif model_name == 'hrnet_ocr':
        return build_hrnet_ocr(
            variant=varient, 
            in_channels=n_channels, 
            num_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_model_varient(model_name):
    if model_name == 'hrnet_ocr':
        return  ['w48', 'w32', 'w18']
    else:
        return [ 'tiny', 'small', 'base' ]

if __name__ == "__main__":
    # Test models
    print("Testing Transformer Models (RGB Only)")
    print("=" * 60)
    
    x = torch.randn(2, 3, 512, 512)  # RGB input
    
    models = ['convnext_upernet', 'hrnet_ocr']
    
    for name in models:
        print(f"\nTesting {name}...")
        model = get_model(name)
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  Input:  {x.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Params: {params:,} ({params/1e6:.1f}M)")
    
    print("\n" + "=" * 60)
    print("All models tested successfully!")
