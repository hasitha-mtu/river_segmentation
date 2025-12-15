"""
Example Usage: Hybrid Segmentation Models
==========================================

This script demonstrates how to use the modular ConvNeXt-UPerNet and HRNet-OCR models.

Examples include:
1. Basic model creation
2. Forward pass
3. Training loop
4. Inference
5. Model components (encoder/decoder separation)

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import models
from models import get_model, list_models, build_convnext_upernet, build_hrnet_ocr

# Import individual components for advanced usage
from models.convnext_upernet import ConvNeXtEncoder, UPerNetDecoder
from models.hrnet_ocr import HRNetBackbone, SpatialOCRModule


def example_1_basic_usage():
    """Example 1: Basic model creation and forward pass."""
    print("=" * 80)
    print("Example 1: Basic Model Creation")
    print("=" * 80)
    
    # List available models
    print("\nAvailable models:")
    models = list_models()
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  - {model}")
    
    # Create ConvNeXt-Tiny-UPerNet for water segmentation
    print("\n" + "-" * 80)
    print("Creating ConvNeXt-Tiny-UPerNet (luminance-only water segmentation)")
    print("-" * 80)
    
    model = get_model(
        'convnext-tiny',
        in_channels=1,  # Luminance only (based on your 103.9% contribution finding)
        num_classes=2   # Water vs. non-water
    )
    
    # Create dummy input
    x = torch.randn(2, 1, 512, 512)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")
    
    # Create HRNet-W32-OCR for comparison
    print("\n" + "-" * 80)
    print("Creating HRNet-W32-OCR (luminance-only water segmentation)")
    print("-" * 80)
    
    model = get_model(
        'hrnet-w32',
        in_channels=1,
        num_classes=2
    )
    
    # Forward pass (training mode returns main + aux)
    model.train()
    with torch.no_grad():
        main_out, aux_out = model(x)
    
    print(f"Main output shape: {main_out.shape}")
    print(f"Auxiliary output shape: {aux_out.shape}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")


def example_2_training_loop():
    """Example 2: Complete training loop."""
    print("\n" + "=" * 80)
    print("Example 2: Training Loop")
    print("=" * 80)
    
    # Create model
    model = build_convnext_upernet(
        variant='tiny',
        in_channels=1,
        num_classes=2
    )
    
    # Setup training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Dummy training data
    images = torch.randn(4, 1, 512, 512).to(device)
    masks = torch.randint(0, 2, (4, 512, 512)).to(device)
    
    print(f"\nTraining on {device}")
    print(f"Batch size: {images.shape[0]}")
    
    # Training step
    model.train()
    
    # Forward
    outputs = model(images)
    loss = criterion(outputs, masks)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    
    # For HRNet-OCR with auxiliary loss
    print("\n" + "-" * 80)
    print("Training HRNet-OCR with Auxiliary Loss")
    print("-" * 80)
    
    model = build_hrnet_ocr(
        variant='w32',
        in_channels=1,
        num_classes=2
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    
    # Forward (returns main + aux)
    main_out, aux_out = model(images)
    
    # Compute weighted loss
    main_loss = criterion(main_out, masks)
    aux_loss = criterion(aux_out, masks)
    total_loss = main_loss + 0.4 * aux_loss  # 40% weight on auxiliary loss
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Main loss: {main_loss.item():.4f}")
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")


def example_3_inference():
    """Example 3: Inference pipeline."""
    print("\n" + "=" * 80)
    print("Example 3: Inference Pipeline")
    print("=" * 80)
    
    # Create model
    model = build_convnext_upernet('tiny', in_channels=1, num_classes=2)
    
    # Load pretrained weights (example)
    # model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Dummy image
    image = torch.randn(1, 1, 512, 512).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1)
    
    print(f"Input shape: {image.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Prediction shape: {prediction.shape}")
    
    # Calculate water coverage (for flood forecasting)
    water_pixels = (prediction == 1).sum().item()
    total_pixels = prediction.numel()
    water_coverage = (water_pixels / total_pixels) * 100
    
    print(f"\nFlood Metrics:")
    print(f"Water coverage: {water_coverage:.2f}%")
    print(f"Flooding detected: {water_coverage > 30.0}")  # Example threshold


def example_4_component_usage():
    """Example 4: Using individual components."""
    print("\n" + "=" * 80)
    print("Example 4: Individual Component Usage")
    print("=" * 80)
    
    # ConvNeXt encoder only
    print("\n" + "-" * 80)
    print("ConvNeXt Encoder (for feature extraction)")
    print("-" * 80)
    
    from models.convnext_upernet import build_convnext_encoder
    
    encoder = build_convnext_encoder('tiny', in_channels=1)
    x = torch.randn(2, 1, 512, 512)
    
    features = encoder(x)
    print(f"Input: {x.shape}")
    for i, feat in enumerate(features):
        print(f"  C{i+1} (1/{4*2**i} res): {feat.shape}")
    
    # HRNet backbone only
    print("\n" + "-" * 80)
    print("HRNet Backbone (parallel multi-resolution)")
    print("-" * 80)
    
    from models.hrnet_ocr import build_hrnet_backbone
    
    backbone = build_hrnet_backbone('w48', in_channels=1)
    
    features = backbone(x)
    print(f"Input: {x.shape}")
    for i, feat in enumerate(features):
        print(f"  Branch {i+1} (1/{4*2**i} res): {feat.shape}")
    
    # Custom decoder on encoder features
    print("\n" + "-" * 80)
    print("Custom Decoder Setup")
    print("-" * 80)
    
    encoder = build_convnext_encoder('tiny', in_channels=1)
    decoder = UPerNetDecoder(
        in_channels_list=encoder.out_channels,
        fpn_channels=256,
        num_classes=2
    )
    
    # Extract features
    encoder_features = encoder(x)
    
    # Decode
    output = decoder(encoder_features, input_size=(512, 512))
    
    print(f"Encoder features: {[f.shape for f in encoder_features]}")
    print(f"Decoder output: {output.shape}")


def example_5_differential_learning_rates():
    """Example 5: Differential learning rates for fine-tuning."""
    print("\n" + "=" * 80)
    print("Example 5: Differential Learning Rates")
    print("=" * 80)
    
    model = build_convnext_upernet('tiny', in_channels=1, num_classes=2)
    
    # Get parameter groups with different learning rates
    param_groups = model.get_params_groups(lr=1e-4)
    
    optimizer = optim.Adam(param_groups)
    
    print("Parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        params = sum(p.numel() for p in group['params'])
        print(f"  Group {i+1}: {params:,} params, lr={group['lr']}")
    
    print("\nThis is useful for fine-tuning:")
    print("  - Encoder: Lower learning rate (preserve pretrained features)")
    print("  - Decoder: Higher learning rate (learn task-specific features)")


def example_6_model_comparison():
    """Example 6: Compare different model variants."""
    print("\n" + "=" * 80)
    print("Example 6: Model Variant Comparison")
    print("=" * 80)
    
    # ConvNeXt-UPerNet variants
    print("\nConvNeXt-UPerNet Variants:")
    print(f"{'Variant':<15} {'Parameters':<15} {'Size (MB)':<12}")
    print("-" * 45)
    
    for variant in ['tiny', 'small', 'base']:
        model = build_convnext_upernet(variant, in_channels=1, num_classes=2)
        params = sum(p.numel() for p in model.parameters())
        size_mb = params * 4 / 1024 / 1024
        print(f"{variant:<15} {params:>13,} {size_mb:>10.1f}")
    
    # HRNet-OCR variants
    print("\nHRNet-OCR Variants:")
    print(f"{'Variant':<15} {'Parameters':<15} {'Size (MB)':<12}")
    print("-" * 45)
    
    for variant in ['w18', 'w32', 'w48']:
        model = build_hrnet_ocr(variant, in_channels=1, num_classes=2)
        params = sum(p.numel() for p in model.parameters())
        size_mb = params * 4 / 1024 / 1024
        print(f"{variant:<15} {params:>13,} {size_mb:>10.1f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("HYBRID MODELS - COMPREHENSIVE EXAMPLES")
    print("ConvNeXt-UPerNet & HRNet-OCR")
    print("=" * 80)
    
    example_1_basic_usage()
    example_2_training_loop()
    example_3_inference()
    example_4_component_usage()
    example_5_differential_learning_rates()
    example_6_model_comparison()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully! ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
