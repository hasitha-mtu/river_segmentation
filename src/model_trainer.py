import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train Transformer Models (RGB Only)')
    
    # Model
    parser.add_argument('--model', type=str, default='sam',
                       choices=['sam', 'dinov2'],
                       help='Model: SAM, DINOv2')
    
    # Data
    parser.add_argument('--data-root', type=str, required=True,
                       help='Data root with images/ and masks/')
    parser.add_argument('--image-size', type=int, default=1024,
                       help='Image size (default: 1024, saves memory vs 512)')
    parser.add_argument('--train-split', type=float, default=0.9,
                       help='Train/val split (default: 0.9)')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Epochs (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=6e-5,
                       help='Learning rate (default: 6e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                       help='Minimum LR (default: 1e-6)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs (default: 5)')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping (default: 1.0)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='Enable gradient checkpointing (saves ~40%% memory, 20%% slower)')
    
    # Loss
    parser.add_argument('--loss', type=str, default='combined',
                       choices=['bce', 'dice', 'iou', 'focal', 'boundary', 'combined'],
                       help='Loss function (default: combined)')
    parser.add_argument('--use-boundary', action='store_true',
                       help='Use boundary loss in combined')
    parser.add_argument('--bce-weight', type=float, default=1.0)
    parser.add_argument('--dice-weight', type=float, default=1.0)
    parser.add_argument('--boundary-weight', type=float, default=1.0)
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Data workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='./experiments',
                       help='Output directory (default: ./experiments)')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log interval (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save interval (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # Optimizer & Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw'],
                       help='Optimizer')
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f'args: {args}')

if __name__ == '__main__':
    main()

