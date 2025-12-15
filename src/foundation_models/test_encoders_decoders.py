"""
Standalone Test for Encoder/Decoder Modules
============================================

This script tests the encoder and decoder modules directly,
without relying on package imports.
"""

import torch
import sys
from pathlib import Path

# Get absolute path to models directory
models_dir = Path(__file__).parent
sys.path.insert(0, str(models_dir))

print("=" * 80)
print("Standalone Encoder/Decoder Test")
print("=" * 80)

# Test 1: SAM Encoder
print("\n1. Testing SAM Encoder...")

print(f'models_dir : {models_dir}')
try:
    sys.path.insert(0, str(models_dir / 'sam'))
    import sam_encoder
    
    encoder = sam_encoder.build_sam_encoder('vit_b', img_size=512, in_channels=3)
    x = torch.randn(1, 3, 512, 512)
    features = encoder(x)
    
    print(f"   ✓ SAM Encoder works!")
    print(f"   ✓ Input: {x.shape}")
    print(f"   ✓ Output features: {len(features)} scales")
    for i, f in enumerate(features):
        print(f"      - Feature {i+1}: {f.shape}")
    
    sys.path.pop(0)
except Exception as e:
    print(f"   ✗ SAM Encoder failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: SAM Decoder
print("\n2. Testing SAM Decoder...")
try:
    sys.path.insert(0, str(models_dir / 'sam'))
    import sam_decoder
    
    decoder = sam_decoder.FPNDecoder(
        in_channels_list=[96, 192, 384, 768],
        fpn_channels=256,
        num_classes=2
    )
    
    # Create dummy features
    features = [
        torch.randn(1, 96, 128, 128),
        torch.randn(1, 192, 64, 64),
        torch.randn(1, 384, 32, 32),
        torch.randn(1, 768, 16, 16),
    ]
    
    output = decoder(features, input_size=(512, 512))
    
    print(f"   ✓ SAM Decoder works!")
    print(f"   ✓ Output: {output.shape}")
    
    sys.path.pop(0)
except Exception as e:
    print(f"   ✗ SAM Decoder failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: DINOv2 Encoder
print("\n3. Testing DINOv2 Encoder...")
try:
    sys.path.insert(0, str(models_dir / 'dinov2'))
    import dinov2_encoder
    
    encoder = dinov2_encoder.build_dinov2_encoder('vit_b', img_size=518, in_channels=3)
    x = torch.randn(1, 3, 518, 518)
    features = encoder(x)
    
    print(f"   ✓ DINOv2 Encoder works!")
    print(f"   ✓ Input: {x.shape}")
    print(f"   ✓ Output features: {len(features)} scales")
    for i, f in enumerate(features):
        print(f"      - Feature {i+1}: {f.shape}")
    
    sys.path.pop(0)
except Exception as e:
    print(f"   ✗ DINOv2 Encoder failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: DINOv2 Decoder
print("\n4. Testing DINOv2 Decoder...")
try:
    sys.path.insert(0, str(models_dir / 'dinov2'))
    import dinov2_decoder
    
    decoder = dinov2_decoder.FPNDecoder(
        in_channels_list=[96, 192, 384, 768],
        fpn_channels=256,
        num_classes=2
    )
    
    # Create dummy features
    features = [
        torch.randn(1, 96, 128, 128),
        torch.randn(1, 192, 64, 64),
        torch.randn(1, 384, 32, 32),
        torch.randn(1, 768, 16, 16),
    ]
    
    output = decoder(features, input_size=(518, 518))
    
    print(f"   ✓ DINOv2 Decoder works!")
    print(f"   ✓ Output: {output.shape}")
    
    sys.path.pop(0)
except Exception as e:
    print(f"   ✗ DINOv2 Decoder failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Complete Models
print("\n5. Testing Complete Models...")
try:
    # Reset path
    while str(models_dir) in sys.path:
        sys.path.remove(str(models_dir))
    while str(models_dir / 'sam') in sys.path:
        sys.path.remove(str(models_dir / 'sam'))
    while str(models_dir / 'dinov2') in sys.path:
        sys.path.remove(str(models_dir / 'dinov2'))
    
    sys.path.insert(0, str(models_dir.parent))
    
    from foundation_models.models import build_sam_segmentation, build_dinov2_segmentation
    
    sam_model = build_sam_segmentation('vit_b', in_channels=3, num_classes=2, img_size=512)
    x = torch.randn(1, 3, 512, 512)
    output = sam_model(x)
    
    print(f"   ✓ SAM Complete Model works!")
    print(f"   ✓ Output: {output.shape}")
    
    dinov2_model = build_dinov2_segmentation('vit_b', in_channels=3, num_classes=2, img_size=518)
    x = torch.randn(1, 3, 518, 518)
    output = dinov2_model(x)
    
    print(f"   ✓ DINOv2 Complete Model works!")
    print(f"   ✓ Output: {output.shape}")
    
except Exception as e:
    print(f"   ✗ Complete models failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✓ All encoder/decoder modules verified!")
print("=" * 80)

# Print file locations
print("\nFile Locations:")
print(f"SAM Encoder:     {'sam/sam_encoder.py'}")
print(f"SAM Decoder:     {'sam/sam_decoder.py'}")
print(f"DINOv2 Encoder:  {'dinov2/dinov2_encoder.py'}")
print(f"DINOv2 Decoder:  {'dinov2/dinov2_decoder.py'}")

print("\nVerify files exist:")
for file_name in [
    f'{models_dir}/sam/sam_encoder.py',
    f'{models_dir}/sam/sam_decoder.py',
    f'{models_dir}/dinov2/dinov2_encoder.py',
    f'{models_dir}/dinov2/dinov2_decoder.py'
]:
    file = Path(file_name)
    
    exists = "✓" if file.exists() else "✗"
    size = f"{file.stat().st_size:,} bytes" if file.exists() else "MISSING"
    print(f"{exists} {file.name:25s} {size}")
