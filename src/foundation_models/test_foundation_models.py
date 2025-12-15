"""
Test Script for Foundation Models
==================================

Verify SAM and DINOv2 implementations work correctly.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'models'))

from sam.model import build_sam_segmentation
from dinov2.model import build_dinov2_segmentation


def test_sam():
    """Test all SAM variants."""
    print("=" * 80)
    print("Testing SAM Segmentation Models")
    print("=" * 80)
    
    variants = ['vit_b', 'vit_l', 'vit_h']
    img_size = 512  # Using 512 instead of 1024 for faster testing
    
    for variant in variants:
        print(f"\nTesting SAM-{variant.upper()}...")
        
        try:
            # Build model
            model = build_sam_segmentation(
                variant=variant,
                in_channels=3,
                num_classes=1,
                img_size=img_size
            )
            model.eval()
            
            # Test input
            batch_size = 2
            x = torch.randn(batch_size, 3, img_size, img_size)
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            # Verify shape
            expected_shape = (batch_size, 1, img_size, img_size)
            assert output.shape == expected_shape, \
                f"Expected {expected_shape}, got {output.shape}"
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            
            print(f"  ✓ {variant.upper()} passed!")
            print(f"    Input:  {tuple(x.shape)}")
            print(f"    Output: {tuple(output.shape)}")
            print(f"    Params: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")
            
        except Exception as e:
            print(f"  ✗ {variant.upper()} failed!")
            print(f"    Error: {e}")
            return False
    
    return True


def test_dinov2():
    """Test all DINOv2 variants."""
    print("\n" + "=" * 80)
    print("Testing DINOv2 Segmentation Models")
    print("=" * 80)
    
    variants = ['vit_s', 'vit_b', 'vit_l']  # Skip vit_g for speed
    img_size = 518  # DINOv2 default
    
    for variant in variants:
        print(f"\nTesting DINOv2-{variant.upper()}...")
        
        try:
            # Build model
            model = build_dinov2_segmentation(
                variant=variant,
                in_channels=3,
                num_classes=1,
                img_size=img_size
            )
            model.eval()
            
            # Test input
            batch_size = 2
            x = torch.randn(batch_size, 3, img_size, img_size)
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            # Verify shape
            expected_shape = (batch_size, 1, img_size, img_size)
            assert output.shape == expected_shape, \
                f"Expected {expected_shape}, got {output.shape}"
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            
            print(f"  ✓ {variant.upper()} passed!")
            print(f"    Input:  {tuple(x.shape)}")
            print(f"    Output: {tuple(output.shape)}")
            print(f"    Params: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")
            
        except Exception as e:
            print(f"  ✗ {variant.upper()} failed!")
            print(f"    Error: {e}")
            return False
    
    return True


def test_differential_lr():
    """Test differential learning rate groups."""
    print("\n" + "=" * 80)
    print("Testing Differential Learning Rates")
    print("=" * 80)
    
    # SAM
    model = build_sam_segmentation('vit_b', in_channels=3, num_classes=1, img_size=512)
    param_groups = model.get_params_groups(lr=1e-4)
    
    print(f"\nSAM ViT-B:")
    print(f"  Encoder LR: {param_groups[0]['lr']:.2e}")
    print(f"  Decoder LR: {param_groups[1]['lr']:.2e}")
    print(f"  Ratio: {param_groups[1]['lr'] / param_groups[0]['lr']:.1f}x")
    
    # DINOv2
    model = build_dinov2_segmentation('vit_b', in_channels=3, num_classes=1, img_size=518)
    param_groups = model.get_params_groups(lr=1e-4)
    
    print(f"\nDINOv2 ViT-B:")
    print(f"  Encoder LR: {param_groups[0]['lr']:.2e}")
    print(f"  Decoder LR: {param_groups[1]['lr']:.2e}")
    print(f"  Ratio: {param_groups[1]['lr'] / param_groups[0]['lr']:.1f}x")
    
    return True


def test_different_sizes():
    """Test with different input sizes."""
    print("\n" + "=" * 80)
    print("Testing Different Input Sizes")
    print("=" * 80)
    
    sizes = [(256, 256), (512, 512), (768, 768)]
    
    # Test SAM
    print("\nSAM ViT-B:")
    
    for h, w in sizes:
        model = build_sam_segmentation('vit_b', in_channels=3, num_classes=1, img_size=h)
        model.eval()
        try:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 1, h, w)
            print(f"  ✓ Size {h}×{w} works")
        except Exception as e:
            print(f"  ✗ Size {h}×{w} failed: {e}")
            return False
    
    # Test DINOv2
    print("\nDINOv2 ViT-S:")
    
    for h, w in sizes:
        model = build_dinov2_segmentation('vit_s', in_channels=3, num_classes=1, img_size=h)
        model.eval()
        try:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 1, h, w)
            print(f"  ✓ Size {h}×{w} works")
        except Exception as e:
            print(f"  ✗ Size {h}×{w} failed: {e}")
            return False
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Foundation Models Test Suite")
    print("=" * 80)
    
    # Run tests
    test1 = test_sam()
    test2 = test_dinov2()
    test3 = test_differential_lr()
    test4 = test_different_sizes()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"SAM models test:       {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"DINOv2 models test:    {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"Differential LR test:  {'✓ PASS' if test3 else '✗ FAIL'}")
    print(f"Input sizes test:      {'✓ PASS' if test4 else '✗ FAIL'}")
    
    if test1 and test2 and test3 and test4:
        print("\n🎉 All tests passed! Foundation models are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
