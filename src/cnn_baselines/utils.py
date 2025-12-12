"""
Utility script for model testing, comparison, and quick experiments
"""

import torch
import time
import numpy as np
from models import get_model
from losses import get_loss_function
from metrics import SegmentationMetrics


def test_model_architecture(model_name, input_size=(512, 512), n_channels=3, device='cuda'):
    """Test a single model architecture"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()}")
    print('='*60)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(model_name, n_channels=n_channels, n_classes=1, pretrained=False)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size:      {total_params * 4 / (1024**2):.2f} MB (FP32)")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, n_channels, input_size[0], input_size[1]).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        # Warm up
        _ = model(x)
        
        # Time inference
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        n_runs = 10
        
        for _ in range(n_runs):
            output = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / n_runs
    
    if isinstance(output, list):
        print(f"Output shapes (deep supervision): {[o.shape for o in output]}")
        output = output[-1]
    else:
        print(f"Output shape: {output.shape}")
    
    print(f"\nInference time:")
    print(f"  Per batch ({batch_size} images): {elapsed*1000:.2f} ms")
    print(f"  Per image: {elapsed*1000/batch_size:.2f} ms")
    print(f"  FPS (batch): {batch_size/elapsed:.2f}")
    
    # Memory usage
    if device.type == 'cuda':
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        memory_reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"\nGPU Memory:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved:  {memory_reserved:.2f} MB")
    
    return model


def compare_all_models(input_size=(512, 512), n_channels=3, device='cuda'):
    """Compare all available models"""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*80)
    
    models = ['unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam']
    
    results = []
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name.upper()}")
        print('='*80)
        
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        try:
            # Create model
            model = get_model(model_name, n_channels=n_channels, n_classes=1, pretrained=False)
            model = model.to(device_obj)
            model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Test inference
            batch_size = 2
            x = torch.randn(batch_size, n_channels, input_size[0], input_size[1]).to(device_obj)
            
            with torch.no_grad():
                # Warm up
                _ = model(x)
                
                # Time inference
                if device_obj.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                output = model(x)
                
                if device_obj.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start
            
            if isinstance(output, list):
                output = output[-1]
            
            # Memory
            memory_mb = 0
            if device_obj.type == 'cuda':
                torch.cuda.synchronize()
                memory_mb = torch.cuda.memory_allocated() / (1024**2)
                torch.cuda.empty_cache()
            
            results.append({
                'model': model_name,
                'params': total_params,
                'inference_ms': elapsed * 1000 / batch_size,
                'memory_mb': memory_mb,
                'output_shape': output.shape
            })
            
            print(f"✓ {model_name} tested successfully")
            
        except Exception as e:
            print(f"✗ Error testing {model_name}: {str(e)}")
            results.append({
                'model': model_name,
                'params': 0,
                'inference_ms': 0,
                'memory_mb': 0,
                'output_shape': None
            })
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Model':<25} {'Parameters':<15} {'Inference (ms)':<18} {'Memory (MB)':<15}")
    print("-"*80)
    
    for r in results:
        if r['params'] > 0:
            print(f"{r['model']:<25} {r['params']:>12,}   {r['inference_ms']:>14.2f}   {r['memory_mb']:>12.2f}")
        else:
            print(f"{r['model']:<25} {'ERROR':>12}   {'ERROR':>14}   {'ERROR':>12}")
    
    print("-"*80)
    
    # Find best models
    valid_results = [r for r in results if r['params'] > 0]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x['inference_ms'])
        smallest = min(valid_results, key=lambda x: x['params'])
        most_efficient = min(valid_results, key=lambda x: x['memory_mb'])
        
        print("\nRecommendations:")
        print(f"  Fastest:         {fastest['model']} ({fastest['inference_ms']:.2f} ms/image)")
        print(f"  Smallest:        {smallest['model']} ({smallest['params']:,} params)")
        print(f"  Most efficient:  {most_efficient['model']} ({most_efficient['memory_mb']:.2f} MB)")
    
    return results


def test_loss_functions():
    """Test all loss functions"""
    print("\n" + "="*60)
    print("TESTING LOSS FUNCTIONS")
    print("="*60)
    
    # Create dummy data
    batch_size, height, width = 4, 128, 128
    predictions = torch.randn(batch_size, 1, height, width)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    losses = ['bce', 'dice', 'iou', 'focal', 'boundary', 'combined', 'tversky', 'combo']
    
    print(f"\nInput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets:     {targets.shape}")
    print("\n" + "-"*60)
    
    for loss_name in losses:
        try:
            if loss_name == 'combined':
                loss_fn = get_loss_function(loss_name, use_boundary=True)
                loss_value, loss_dict = loss_fn(predictions, targets)
                print(f"\n{loss_name.upper()} Loss:")
                for k, v in loss_dict.items():
                    print(f"  {k}: {v:.4f}")
            else:
                loss_fn = get_loss_function(loss_name)
                loss_value = loss_fn(predictions, targets)
                print(f"\n{loss_name.upper()} Loss: {loss_value.item():.4f}")
        except Exception as e:
            print(f"\n{loss_name.upper()} Loss: ERROR - {str(e)}")
    
    print("\n" + "="*60)


def test_metrics():
    """Test metric computation"""
    print("\n" + "="*60)
    print("TESTING METRICS")
    print("="*60)
    
    # Create dummy data with known properties
    batch_size, height, width = 2, 128, 128
    
    # Perfect prediction
    perfect_pred = torch.ones(batch_size, 1, height, width)
    perfect_target = torch.ones(batch_size, 1, height, width)
    
    # Random prediction
    random_pred = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    random_target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    metrics_calc = SegmentationMetrics(threshold=0.5)
    
    print("\n1. Perfect Prediction:")
    print("-"*60)
    metrics = metrics_calc.compute_metrics(perfect_pred, perfect_target)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n2. Random Prediction:")
    print("-"*60)
    metrics = metrics_calc.compute_metrics(random_pred, random_target)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n" + "="*60)


def benchmark_training_speed(model_name='unet', batch_size=8, num_iterations=10, device='cuda'):
    """Benchmark training speed"""
    print("\n" + "="*60)
    print(f"BENCHMARKING TRAINING SPEED: {model_name.upper()}")
    print("="*60)
    
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(model_name, n_channels=3, n_classes=1, pretrained=False)
    model = model.to(device_obj)
    model.train()
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = get_loss_function('dice')
    
    # Create dummy data
    x = torch.randn(batch_size, 3, 512, 512).to(device_obj)
    y = torch.randint(0, 2, (batch_size, 1, 512, 512)).float().to(device_obj)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Image size: 512x512")
    print(f"Iterations: {num_iterations}")
    
    # Warm up
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    if device_obj.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Per iteration: {elapsed/num_iterations:.2f} seconds")
    print(f"  Per image: {elapsed/(num_iterations*batch_size):.2f} seconds")
    print(f"  Images/second: {(num_iterations*batch_size)/elapsed:.2f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and compare segmentation models')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['compare', 'single', 'losses', 'metrics', 'benchmark'],
                       help='Test mode')
    parser.add_argument('--model', type=str, default='unet',
                       help='Model name for single model test')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for testing')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for benchmarking')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_all_models(device=args.device)
    elif args.mode == 'single':
        test_model_architecture(args.model, device=args.device)
    elif args.mode == 'losses':
        test_loss_functions()
    elif args.mode == 'metrics':
        test_metrics()
    elif args.mode == 'benchmark':
        benchmark_training_speed(args.model, batch_size=args.batch_size, device=args.device)
    
    print("\n✓ All tests completed successfully!")
