"""
Transformer-based Segmentation Models for River Water Detection
RGB Input Only - SAM, DINOv2
"""

import torch
from sam.model import build_sam_segmentation
from dinov2.model import build_dinov2_segmentation

def get_model(model_name, varient='vit_b', n_channels=3, n_classes=1):
    """
    Factory function to get models
    
    Args:
        model_name: 'sam', or 'dinov2'
        n_channels: Number of input channels (3 for RGB images)
        num_classes: Number of output classes (1 for binary segmentation)
    """
    if model_name == 'sam':
        return build_sam_segmentation(
            variant=varient, 
            in_channels=n_channels, 
            num_classes=n_classes)
    elif model_name == 'dinov2':
        return build_dinov2_segmentation(
            variant=varient, 
            in_channels=n_channels, 
            num_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def get_model_varient(model_name):
    if model_name == 'sam':
        return  ['vit_b', 'vit_l', 'vit_h']
    else:
        return ['vit_s', 'vit_b', 'vit_l', 'vit_g']

if __name__ == "__main__":
    # Test models
    print("Testing Foundation Models (RGB Only)")
    print("=" * 60)
    
    x = torch.randn(1, 3, 512, 512)  # RGB input
    
    models = ['sam', 'dinov2']
    
    for name in models:
        print(f"\nTesting {name}...")
        for varient in get_model_varient(name):
            model = get_model(name, varient)
            model.eval()
                
            with torch.no_grad():
                output = model(x)
                
            params = sum(p.numel() for p in model.parameters())
            print(f"  Input:  {x.shape}")
            print(f"  Output: {output.shape}")
            print(f"  Params: {params:,} ({params/1e6:.1f}M)")
    
    print("\n" + "=" * 60)
    print("All models tested successfully!")
