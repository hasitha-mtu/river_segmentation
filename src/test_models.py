"""
Test Script for Trained Models
===============================

Test trained models (CNN, Transformer, Foundation) on test dataset.
Computes metrics and generates visualizations.

Author: Hasitha
Date: December 2025
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models import get_model
from dataset.dataset_loader import SegmentationDataset, get_dataloaders
from utils.metrics import StreamingMetrics
from utils.losses import get_loss_function
from improved_argparse import add_model_arguments, validate_args_post_parse


class ModelTester:
    """Test trained models on test dataset"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.setup_paths()
        
        # Load model
        print(f"Loading model from {args.checkpoint}...")
        self.model = self.load_model()
        
        # Setup test data
        print(f"Loading test data from {args.test_data}...")
        self.test_loader = self.get_test_loader()
        
        # Setup metrics
        self.metrics = StreamingMetrics(threshold=args.threshold)
        
        # Setup loss (optional)
        if args.compute_loss:
            self.criterion = get_loss_function(args.loss)
        
        print(f"\nTest setup complete:")
        print(f"  Device: {self.device}")
        print(f"  Model: {args.model}")
        print(f"  Test images: {len(self.test_loader.dataset)}")
        print(f"  Batch size: {args.batch_size}")
    
    def setup_paths(self):
        """Setup output directories"""
        self.output_dir = Path(os.path.join(self.args.output_dir, self.args.model, self.args.variant))
        # self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # self.predictions_dir = self.output_dir / "predictions"
        # self.predictions_dir.mkdir(exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def load_model(self):
        """Load trained model from checkpoint"""
        # Create model
        model = get_model(self.args.model, self.args.variant)
        model = model.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best Dice: {checkpoint.get('best_val_dice', 'N/A'):.4f}")
            print(f"  Best IoU: {checkpoint.get('best_val_iou', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded model state dict")
        
        model.eval()
        return model
    
    def get_test_loader(self):
        loader = get_dataloaders(
            data_dir=self.args.test_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            data_loader_type='test'
        )
        
        return loader
    
    @torch.no_grad()
    def test(self):
        """Run testing on test dataset"""
        print("\n" + "=" * 80)
        print("Testing Model")
        print("=" * 80)
        
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0
        num_batches = 0
        
        # Store predictions for visualization
        all_results = []
        
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Testing')):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss if requested
            if self.args.compute_loss:
                if self.args.loss == 'combined':
                    loss, _ = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                num_batches += 1
            
            # Get predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > self.args.threshold).float()
            
            # Update metrics
            self.metrics.update(preds, masks)
            
            # Store results for visualization
            if batch_idx < self.args.max_viz:
                for i in range(images.size(0)):
                    result = {
                        'image': images[i].cpu(),
                        'mask': masks[i].cpu(),
                        'pred': preds[i].cpu(),
                        'prob': probs[i].cpu(),
                        'filename': batch.get('filename', [f'img_{batch_idx}_{i}'])[i] if 'filename' in batch else f'img_{batch_idx}_{i}'
                    }
                    all_results.append(result)
        
        # Compute final metrics
        metrics = self.metrics.get_metrics()
        
        # Add loss if computed
        if self.args.compute_loss:
            metrics['loss'] = total_loss / num_batches
        
        # Print results
        self.print_results(metrics)
        
        # Save results
        self.save_results(metrics)
        
        # Generate visualizations
        if self.args.visualize:
            print("\nGenerating visualizations...")
            self.visualize_results(all_results)
        
        return metrics
    
    def print_results(self, metrics):
        """Print test results"""
        print("\n" + "=" * 80)
        print("Test Results")
        print("=" * 80)
        
        print(f"\nSegmentation Metrics:")
        print(f"  Dice Coefficient: {metrics['dice']:.4f}")
        print(f"  IoU Score:        {metrics['iou']:.4f}")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1 Score:         {metrics['f1']:.4f}")
        print(f"  Specificity:      {metrics['specificity']:.4f}")
        
        if 'loss' in metrics:
            print(f"\nLoss: {metrics['loss']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['TP']:,}")
        print(f"  True Negatives:  {metrics['TN']:,}")
        print(f"  False Positives: {metrics['FP']:,}")
        print(f"  False Negatives: {metrics['FN']:,}")
        
        print("=" * 80)
    
    def save_results(self, metrics):
        """Save test results to JSON"""
        results = {
            'model': self.args.model,
            'variant': self.args.variant,
            'checkpoint': str(self.args.checkpoint),
            'test_data': str(self.args.test_data),
            'num_images': len(self.test_loader.dataset),
            'image_size': self.args.image_size,
            'threshold': self.args.threshold,
            'timestamp': datetime.now().isoformat(),
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else int(v) if isinstance(v, np.integer) else v 
                       for k, v in metrics.items()}
        }
        
        # Save to JSON
        output_file = self.output_dir / 'test_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
    
    def visualize_results(self, results):
        """Generate visualizations of predictions"""
        print(f"Creating visualizations for {len(results)} images...")
        
        for idx, result in enumerate(tqdm(results, desc='Visualizing')):
            self.visualize_single(result, idx)
        
        # Create summary visualization
        self.create_summary_viz(results[:min(16, len(results))])
        
        print(f"✓ Visualizations saved to {self.viz_dir}")
    
    def visualize_single(self, result, idx):
        """Create visualization for single image"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Denormalize image for display
        image = result['image'].numpy().transpose(1, 2, 0)
        image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        image = np.clip(image, 0, 1)
        
        mask = result['mask'].squeeze().numpy()
        pred = result['pred'].squeeze().numpy()
        prob = result['prob'].squeeze().numpy()
        
        # 1. Original Image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Ground Truth
        axes[0, 1].imshow(image)
        axes[0, 1].imshow(mask, alpha=0.5, cmap='Blues')
        axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Prediction
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(pred, alpha=0.5, cmap='Reds')
        axes[1, 0].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Probability Map
        im = axes[1, 1].imshow(prob, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 1].set_title('Probability Map', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Calculate metrics for this image
        intersection = (pred * mask).sum()
        union = pred.sum() + mask.sum() - intersection
        dice = 2 * intersection / (pred.sum() + mask.sum() + 1e-7)
        iou = intersection / (union + 1e-7)
        
        # Add title with metrics
        filename = result.get('filename', f'image_{idx}')
        fig.suptitle(
            f"{filename}\nDice: {dice:.4f} | IoU: {iou:.4f}",
            fontsize=16, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'{idx:04d}_{filename}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_summary_viz(self, results):
        """Create summary grid visualization"""
        n = len(results)
        cols = 4
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, result in enumerate(results):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Denormalize image
            image = result['image'].numpy().transpose(1, 2, 0)
            image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            image = np.clip(image, 0, 1)
            
            mask = result['mask'].squeeze().numpy()
            pred = result['pred'].squeeze().numpy()
            
            # Create overlay
            ax.imshow(image)
            
            # Show ground truth in blue, prediction in red, overlap in purple
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask == 1] = [0, 0, 1, 0.3]  # Blue for GT
            overlay[pred == 1] = [1, 0, 0, 0.3]  # Red for pred
            overlay[(mask == 1) & (pred == 1)] = [0.5, 0, 0.5, 0.5]  # Purple for overlap
            
            ax.imshow(overlay)
            
            # Calculate Dice
            intersection = (pred * mask).sum()
            dice = 2 * intersection / (pred.sum() + mask.sum() + 1e-7)
            
            ax.set_title(f'Image {idx+1}\nDice: {dice:.3f}', fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='blue', alpha=0.3, label='Ground Truth'),
            mpatches.Patch(color='red', alpha=0.3, label='Prediction'),
            mpatches.Patch(color='purple', alpha=0.5, label='Overlap (Correct)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                  ncol=3, fontsize=12, frameon=True)
        
        plt.suptitle(f'Test Results Summary - {self.args.model}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.viz_dir / 'summary.png', dpi=150, bbox_inches='tight')
        plt.close()


def parse_args1():
    parser = argparse.ArgumentParser(description='Test trained segmentation model')
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                       help='Model type (e.g., sam, dinov2, unet, swin-unet)')
    parser.add_argument('--variant', type=str, default=None,
                       help='Model variant (e.g., vit_b, vit_l)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (best.pth)')
    
    # Data
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size (default: 512)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers (default: 4)') 
    
    # Testing
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold (default: 0.5)')
    parser.add_argument('--compute-loss', action='store_true',
                       help='Compute loss during testing')
    parser.add_argument('--loss', type=str, default='combined',
                       help='Loss function (default: combined)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations (default: True)')
    parser.add_argument('--max-viz', type=int, default=50,
                       help='Maximum images to visualize (default: 50)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory (default: test_results)')
    
    return parser.parse_args()

def parse_args():
    """Parse command line arguments with smart model-variant validation"""
    parser = argparse.ArgumentParser(
        description='Test trained segmentation model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add model arguments (with automatic validation)
    parser = add_model_arguments(parser)
    
    # Testing arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (best.pth)')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size (default: 512)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold (default: 0.5)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers (default: 4)') 
    parser.add_argument('--compute-loss', action='store_true',
                       help='Compute loss during testing')
    parser.add_argument('--loss', type=str, default='combined',
                       help='Loss function (default: combined)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations (default: True)')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory (default: test_results)')
    parser.add_argument('--max-viz', type=int, default=50,
                       help='Maximum images to visualize (default: 50)')
    
    # Parse and validate
    args = parser.parse_args()
    args = validate_args_post_parse(args)
    
    return args


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Model Testing")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test data: {args.test_data}")
    print(f"  Image size: {args.image_size}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Output: {args.output_dir}")
    
    # Create tester
    tester = ModelTester(args)
    
    # Run testing
    metrics = tester.test()
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
