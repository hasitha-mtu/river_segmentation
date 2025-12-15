"""
Model Comparison Example
=========================

Compare all 4 model families: ConvNeXt-UPerNet, HRNet-OCR, SAM, and DINOv2

This script demonstrates:
1. Building all model types
2. Forward pass comparison
3. Parameter counting
4. Inference speed benchmarking
"""

import torch
import time
from models import get_model, list_models


def benchmark_model(model, input_size=(512, 512), batch_size=2, num_runs=20):
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        input_size: Input image size
        batch_size: Batch size
        num_runs: Number of runs for averaging
    
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Warm up
    x = torch.randn(batch_size, 3, *input_size).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000 / batch_size)
    
    return sum(times) / len(times)


def main():
    print("=" * 80)
    print("Semantic Segmentation Models Comparison")
    print("=" * 80)
    
    # Configuration
    num_classes = 2
    input_size = (512, 512)
    batch_size = 2
    
    # Models to compare
    models_to_test = {
        'Hybrid Models': [
            ('convnext-tiny', 'ConvNeXt-Tiny-UPerNet'),
            ('hrnet-w32', 'HRNet-W32-OCR'),
        ],
        'Foundation Models': [
            ('sam-vit-b', 'SAM-ViT-B'),
            ('dinov2-vit-b', 'DINOv2-ViT-B'),
        ]
    }
    
    results = []
    
    # Test each model
    for category, models in models_to_test.items():
        print(f"\n{category}")
        print("-" * 80)
        
        for model_name, display_name in models:
            print(f"\nTesting {display_name}...")
            
            try:
                # Build model
                model = get_model(model_name, in_channels=3, num_classes=num_classes)
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Test forward pass
                x = torch.randn(batch_size, 3, *input_size)
                with torch.no_grad():
                    output = model(x)
                    # Handle auxiliary output from HRNet-OCR
                    if isinstance(output, tuple):
                        output = output[0]
                
                # Benchmark speed (skip if no GPU)
                if torch.cuda.is_available():
                    inference_time = benchmark_model(model, input_size, batch_size)
                    speed_str = f"{inference_time:.1f} ms"
                else:
                    speed_str = "N/A (CPU)"
                
                # Store results
                results.append({
                    'name': display_name,
                    'params': total_params,
                    'output_shape': tuple(output.shape),
                    'speed': speed_str
                })
                
                print(f"  ✓ Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
                print(f"  ✓ Output shape: {tuple(output.shape)}")
                print(f"  ✓ Inference speed: {speed_str}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results.append({
                    'name': display_name,
                    'params': 'ERROR',
                    'output_shape': 'ERROR',
                    'speed': 'ERROR'
                })
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Model':<25} {'Parameters':<15} {'Memory (MB)':<15} {'Speed':<15}")
    print("-" * 80)
    
    for result in results:
        if result['params'] != 'ERROR':
            params = result['params']
            memory = params * 4 / 1024 / 1024
            print(f"{result['name']:<25} {params:>13,}   {memory:>13.1f}   {result['speed']:<15}")
        else:
            print(f"{result['name']:<25} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")
    
    # Print all available models
    print("\n" + "=" * 80)
    print("All Available Models")
    print("=" * 80)
    
    all_models = list_models()
    for category, models in all_models.items():
        print(f"\n{category}:")
        for model in models:
            print(f"  - {model}")
    
    # Usage examples
    print("\n" + "=" * 80)
    print("Usage Examples")
    print("=" * 80)
    
    print("""
# Train any model with same script
python train.py --model convnext-tiny --epochs 100
python train.py --model hrnet-w32 --epochs 100
python train.py --model sam-vit-b --epochs 100 --batch-size 4
python train.py --model dinov2-vit-b --epochs 100 --batch-size 4

# Test any model
python test.py --model sam-vit-b --checkpoint checkpoints/best_model.pth
python test.py --model dinov2-vit-b --checkpoint checkpoints/best_model.pth

# In Python
from models import get_model

# Any model with same interface
model = get_model('convnext-tiny', in_channels=3, num_classes=2)
model = get_model('hrnet-w32', in_channels=3, num_classes=2)
model = get_model('sam-vit-b', in_channels=3, num_classes=2)
model = get_model('dinov2-vit-b', in_channels=3, num_classes=2)

# All models work the same way
x = torch.randn(2, 3, 512, 512)
output = model(x)  # (2, 2, 512, 512)
""")


if __name__ == "__main__":
    main()
