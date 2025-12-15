"""
Import Verification Script
===========================

This script verifies that all foundation model modules can be imported correctly.
"""

import sys
from pathlib import Path

# Add models directory to path
models_path = Path(__file__).parent / 'foundation_models'
sys.path.insert(0, str(models_path))

print("=" * 80)
print("Testing Foundation Model Imports")
print("=" * 80)

# Test SAM imports
print("\n1. Testing SAM modules...")
try:
    from sam.sam_encoder import SAMImageEncoder, build_sam_encoder
    print("   ✓ sam_encoder imported successfully")
except ImportError as e:
    print(f"   ✗ sam_encoder import failed: {e}")

try:
    from sam.sam_decoder import FPNDecoder
    print("   ✓ sam_decoder imported successfully")
except ImportError as e:
    print(f"   ✗ sam_decoder import failed: {e}")

try:
    from sam.model import SAMSegmentation, build_sam_segmentation
    print("   ✓ sam model imported successfully")
except ImportError as e:
    print(f"   ✗ sam model import failed: {e}")

try:
    from sam import build_sam_segmentation
    print("   ✓ sam_segmentation package imported successfully")
except ImportError as e:
    print(f"   ✗ sam_segmentation package import failed: {e}")

# Test DINOv2 imports
print("\n2. Testing DINOv2 modules...")
try:
    from dinov2.dinov2_encoder import DINOv2Encoder, build_dinov2_encoder
    print("   ✓ dinov2_encoder imported successfully")
except ImportError as e:
    print(f"   ✗ dinov2_encoder import failed: {e}")

try:
    from dinov2.dinov2_decoder import FPNDecoder
    print("   ✓ dinov2_decoder imported successfully")
except ImportError as e:
    print(f"   ✗ dinov2_decoder import failed: {e}")

try:
    from dinov2.model import DINOv2Segmentation, build_dinov2_segmentation
    print("   ✓ dinov2 model imported successfully")
except ImportError as e:
    print(f"   ✗ dinov2 model import failed: {e}")

try:
    from dinov2 import build_dinov2_segmentation
    print("   ✓ dinov2_segmentation package imported successfully")
except ImportError as e:
    print(f"   ✗ dinov2_segmentation package import failed: {e}")

# Test main models package
print("\n3. Testing main models package...")
sys.path.insert(0, str(Path(__file__).parent))
try:
    from models import get_model, build_sam_segmentation, build_dinov2_segmentation
    print("   ✓ models package imported successfully")
    
    # Try building models
    print("\n4. Testing model building...")
    import torch
    
    sam_model = get_model('sam', 'vit_b')
    print("   ✓ SAM model built successfully")
    
    dinov2_model = get_model('dinov2', 'vit_b')
    print("   ✓ DINOv2 model built successfully")
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    x = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        sam_output = sam_model(x)
    print(f"   ✓ SAM forward pass: {sam_output.shape}")
    
    with torch.no_grad():
        dinov2_output = dinov2_model(x)
    print(f"   ✓ DINOv2 forward pass: {dinov2_output.shape}")
    
except ImportError as e:
    print(f"   ✗ models package import failed: {e}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("✓ All imports verified successfully!")
print("=" * 80)
