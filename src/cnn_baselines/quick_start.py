"""
Quick Start Example Script
Demonstrates basic usage of the river segmentation framework
"""

import torch
from models import get_model
from losses import get_loss_function
from dataset import get_dataloaders
from metrics import SegmentationMetrics
import numpy as np


def example_1_model_initialization():
    """Example 1: Initialize and test different models"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Model Initialization")
    print("="*60)
    
    # Initialize different models
    models = {
        'U-Net': get_model('unet', n_channels=3, n_classes=1),
        'U-Net++': get_model('unetpp', n_channels=3, n_classes=1),
        'ResUNet++': get_model('resunetpp', n_channels=3, n_classes=1),
        'DeepLabV3+': get_model('deeplabv3plus', n_channels=3, n_classes=1, pretrained=True),
        'DeepLabV3+ + CBAM': get_model('deeplabv3plus_cbam', n_channels=3, n_classes=1, pretrained=True),
    }
    
    # Test with dummy input
    x = torch.randn(2, 3, 512, 512)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(x)
            if isinstance(output, list):
                output = output[-1]
        
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {output.shape}")


def example_2_loss_functions():
    """Example 2: Using different loss functions"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Loss Functions")
    print("="*60)
    
    # Create dummy predictions and targets
    predictions = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    # Test different losses
    losses = {
        'BCE': get_loss_function('bce'),
        'Dice': get_loss_function('dice'),
        'IoU': get_loss_function('iou'),
        'Focal': get_loss_function('focal', alpha=0.25, gamma=2.0),
        'Boundary': get_loss_function('boundary'),
        'Combined': get_loss_function('combined', use_boundary=True, 
                                     bce_weight=1.0, dice_weight=1.0, boundary_weight=1.0),
    }
    
    for name, loss_fn in losses.items():
        if name == 'Combined':
            loss_value, loss_dict = loss_fn(predictions, targets)
            print(f"\n{name} Loss:")
            for k, v in loss_dict.items():
                print(f"  {k}: {v:.4f}")
        else:
            loss_value = loss_fn(predictions, targets)
            print(f"\n{name} Loss: {loss_value.item():.4f}")


def example_3_metrics():
    """Example 3: Computing evaluation metrics"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Evaluation Metrics")
    print("="*60)
    
    # Create dummy predictions and targets
    predictions = torch.randint(0, 2, (4, 1, 256, 256)).float()
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    # Compute metrics
    metrics_calc = SegmentationMetrics(threshold=0.5)
    metrics = metrics_calc.compute_metrics(predictions, targets)
    
    print("\nSegmentation Metrics:")
    print("-"*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:,}")


def example_4_training_loop():
    """Example 4: Basic training loop structure"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Training Loop Structure")
    print("="*60)
    
    print("\nBasic training loop structure:")
    print("""
    # Setup
    model = get_model('unet', n_channels=3, n_classes=1)
    model = model.to(device)
    
    criterion = get_loss_function('combined', use_boundary=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                # Compute metrics...
    """)
    
    print("\nFor full implementation, see train.py")


def example_5_inference():
    """Example 5: Model inference"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Model Inference")
    print("="*60)
    
    print("\nBasic inference structure:")
    print("""
    # Load model
    checkpoint = torch.load('path/to/checkpoint.pth')
    model = get_model('unet', n_channels=3, n_classes=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prepare image
    image = cv2.imread('path/to/image.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output)
        mask = (prob > 0.5).float()
    
    # Convert back to numpy
    mask = mask.squeeze().cpu().numpy()
    """)
    
    print("\nFor full implementation, see test.py")


def example_6_multispectral():
    """Example 6: Multi-spectral inputs"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Multi-Spectral Inputs")
    print("="*60)
    
    print("\nSupported channel configurations:")
    print("  - RGB (3 channels): Standard color image")
    print("  - YCbCr (3 channels): Luminance + Chrominance")
    print("  - Y/Luminance (1 channel): Luminance only")
    print("  - CbCr (2 channels): Chrominance only")
    
    print("\nUsage examples:")
    
    # RGB
    print("\n1. RGB (standard):")
    model_rgb = get_model('unet', n_channels=3, n_classes=1)
    print(f"   Input channels: 3 (RGB)")
    
    # Luminance only
    print("\n2. Luminance only:")
    model_y = get_model('unet', n_channels=1, n_classes=1)
    print(f"   Input channels: 1 (Y)")
    
    # CbCr
    print("\n3. Chrominance only:")
    model_cbcr = get_model('unet', n_channels=2, n_classes=1)
    print(f"   Input channels: 2 (CbCr)")
    
    print("\nFor data loading, use MultiSpectralDataset:")
    print("""
    from dataset import MultiSpectralDataset
    
    dataset = MultiSpectralDataset(
        data_root='./data',
        channels='luminance',  # or 'rgb', 'ycbcr', 'cbcr'
        image_size=(512, 512),
        normalize=True,
        augment=True
    )
    """)


def example_7_custom_training():
    """Example 7: Custom training configurations"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Custom Training Configurations")
    print("="*60)
    
    print("\n1. Training with Boundary Loss:")
    print("""
    python train.py \\
        --model deeplabv3plus_cbam \\
        --data-root ./data \\
        --loss combined \\
        --use-boundary \\
        --bce-weight 1.0 \\
        --dice-weight 1.0 \\
        --boundary-weight 1.0
    """)
    
    print("\n2. Training with Luminance Features:")
    print("""
    python train.py \\
        --model unet \\
        --data-root ./data \\
        --channels luminance \\
        --loss dice
    """)
    
    print("\n3. Training with Pretrained Weights:")
    print("""
    python train.py \\
        --model deeplabv3plus \\
        --data-root ./data \\
        --pretrained \\
        --learning-rate 1e-5
    """)
    
    print("\n4. Resuming Training:")
    print("""
    python train.py \\
        --model unet \\
        --data-root ./data \\
        --resume experiments/unet_combined_20241210/checkpoints/latest.pth
    """)


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("RIVER WATER SEGMENTATION - QUICK START EXAMPLES")
    print("="*80)
    
    examples = [
        example_1_model_initialization,
        example_2_loss_functions,
        example_3_metrics,
        example_4_training_loop,
        example_5_inference,
        example_6_multispectral,
        example_7_custom_training,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {str(e)}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Prepare your data in the correct format (see README.md)
2. Test the models: python utils.py --mode compare
3. Start training: python train.py --model unet --data-root ./your_data
4. Monitor with TensorBoard: tensorboard --logdir experiments/
5. Evaluate: python test.py --checkpoint path/to/checkpoint.pth --mode evaluate
    
For detailed documentation, see README.md
For model comparison, run: python utils.py --mode compare
For quick tests, run: python utils.py --mode single --model unet
    """)
    print("="*80)


if __name__ == "__main__":
    main()
