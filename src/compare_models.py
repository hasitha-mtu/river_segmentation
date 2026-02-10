"""
Compare Multiple Trained Models
================================

Test and compare multiple trained models (CNN, Transformer, Foundation).
Generates comparison tables, charts, and visualizations.

Author: Hasitha
Date: December 2025
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models import get_model
from dataset import RiverSegmentationDataset
from utils.metrics import StreamingMetrics


class ModelComparator:
    """Compare multiple trained models"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.setup_paths()
        
        # Load model configurations
        self.models_config = self.load_models_config()
        
        # Setup test data
        print(f"Loading test data from {args.test_data}...")
        self.test_loader = self.get_test_loader()
        
        print(f"\nComparison setup complete:")
        print(f"  Device: {self.device}")
        print(f"  Models to compare: {len(self.models_config)}")
        print(f"  Test images: {len(self.test_loader.dataset)}")
    
    def setup_paths(self):
        """Setup output directories"""
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.comparison_dir = self.output_dir / "comparisons"
        self.comparison_dir.mkdir(exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def load_models_config(self):
        """Load model configurations from JSON file"""
        with open(self.args.models_config, 'r') as f:
            config = json.load(f)
        
        print(f"\nLoaded {len(config['models'])} model configurations:")
        for model_cfg in config['models']:
            print(f"  - {model_cfg['name']}: {model_cfg['model_type']}")
        
        return config['models']
    
    def get_test_loader(self):
        """Create test dataloader"""
        dataset = RiverSegmentationDataset(
            data_root=self.args.test_data,
            image_size=(self.args.image_size, self.args.image_size),
            augment=False
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return loader
    
    @torch.no_grad()
    def test_model(self, model_config):
        """Test a single model"""
        print(f"\nTesting {model_config['name']}...")
        
        # Load model
        model = get_model(
            model_config['model_type'],
            variant=model_config.get('variant', None)
        )
        model = model.to(self.device)
        
        checkpoint_path = model_config['checkpoint']
        print(f'Model checkpoint path: {checkpoint_path}')
        if os.path.isfile(checkpoint_path):

            # Load checkpoint
            checkpoint = torch.load(model_config['checkpoint'], map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            # Test
            metrics_calc = StreamingMetrics(threshold=self.args.threshold)
            metrics_calc.reset()
            
            all_predictions = []
            
            for batch in tqdm(self.test_loader, desc=f'Testing {model_config["name"]}'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > self.args.threshold).float()
                
                metrics_calc.update(preds, masks)
                
                # Store for comparison visualizations
                all_predictions.append({
                    'preds': preds.cpu(),
                    'masks': masks.cpu(),
                    'images': images.cpu()
                })
            
            metrics = metrics_calc.get_metrics()
            
            # Add model info
            result = {
                'name': model_config['name'],
                'model_type': model_config['model_type'],
                'variant': model_config.get('variant', 'N/A'),
                'checkpoint': model_config['checkpoint'],
                'metrics': metrics,
                'predictions': all_predictions
            }
            
            print(f"  Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}")
            
            return result
        else:
            print(f'Model checkpoint path: {checkpoint_path} not found')
            return None
    
    def compare_all(self):
        """Test and compare all models"""
        print("\n" + "=" * 80)
        print("Comparing Models")
        print("=" * 80)
        
        all_results = []
        
        for model_config in self.models_config:
            result = self.test_model(model_config)
            all_results.append(result)
        
        # Create comparison table
        self.create_comparison_table(all_results)
        
        # Create comparison charts
        self.create_comparison_charts(all_results)
        
        # Create side-by-side visualizations
        if self.args.visualize:
            self.create_comparison_visualizations(all_results)
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def create_comparison_table(self, results):
        """Create comparison table"""
        print("\n" + "=" * 80)
        print("Model Comparison Results")
        print("=" * 80)
        
        # Create DataFrame
        data = []
        for result in results:
            metrics = result['metrics']
            row = {
                'Model': result['name'],
                'Type': result['model_type'],
                'Variant': result['variant'],
                'Dice': metrics['dice'],
                'IoU': metrics['iou'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'Specificity': metrics['specificity']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Print table
        print("\n" + df.to_string(index=False))
        
        # Find best model for each metric
        print("\n" + "-" * 80)
        print("Best Models:")
        for metric in ['Dice', 'IoU', 'Accuracy', 'Precision', 'Recall', 'F1']:
            best_idx = df[metric].idxmax()
            best_model = df.loc[best_idx, 'Model']
            best_value = df.loc[best_idx, metric]
            print(f"  {metric:12s}: {best_model:20s} ({best_value:.4f})")
        
        # Save to CSV
        csv_file = self.output_dir / 'comparison_table.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Table saved to {csv_file}")
        
        return df
    
    def create_comparison_charts(self, results):
        """Create comparison charts"""
        print("\nGenerating comparison charts...")
        
        # Prepare data
        models = [r['name'] for r in results]
        metrics_dict = {
            'Dice': [r['metrics']['dice'] for r in results],
            'IoU': [r['metrics']['iou'] for r in results],
            'Accuracy': [r['metrics']['accuracy'] for r in results],
            'Precision': [r['metrics']['precision'] for r in results],
            'Recall': [r['metrics']['recall'] for r in results],
            'F1': [r['metrics']['f1'] for r in results]
        }
        
        # 1. Grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        width = 0.12
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))
        
        for idx, (metric, values) in enumerate(metrics_dict.items()):
            offset = (idx - len(metrics_dict) / 2) * width
            ax.bar(x + offset, values, width, label=metric, color=colors[idx])
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / 'comparison_bars.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_dict), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_dict.keys())
        ax.set_ylim(0, 1)
        ax.grid(True)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for idx, result in enumerate(results):
            values = [metrics_dict[metric][idx] for metric in metrics_dict.keys()]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=result['name'], 
                   color=colors[idx], alpha=0.7)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_title('Model Performance Radar Chart', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / 'comparison_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_matrix = np.array([[metrics_dict[metric][idx] for metric in metrics_dict.keys()] 
                                for idx in range(len(results))])
        
        sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=metrics_dict.keys(),
                   yticklabels=models,
                   cbar_kws={'label': 'Score'},
                   ax=ax, vmin=0, vmax=1)
        
        ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.comparison_dir / 'comparison_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison charts saved to {self.comparison_dir}")
    
    def create_comparison_visualizations(self, results):
        """Create side-by-side comparison visualizations"""
        print("\nCreating comparison visualizations...")
        
        # Select a few sample images
        num_samples = min(self.args.num_viz_samples, len(results[0]['predictions']))
        
        for sample_idx in tqdm(range(num_samples), desc='Creating comparisons'):
            self.visualize_sample_comparison(results, sample_idx)
        
        print(f"✓ Comparison visualizations saved to {self.comparison_dir}")
    
    def visualize_sample_comparison(self, results, sample_idx):
        """Create visualization comparing all models on one sample"""
        num_models = len(results)
        fig, axes = plt.subplots(2, num_models + 1, figsize=(4 * (num_models + 1), 8))
        
        # Get sample data from first model (same for all)
        sample_data = results[0]['predictions'][sample_idx]
        image = sample_data['images'][0].numpy().transpose(1, 2, 0)
        image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        image = np.clip(image, 0, 1)
        mask = sample_data['masks'][0].squeeze().numpy()
        
        # First column: Original + Ground Truth
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(mask, alpha=0.5, cmap='Blues')
        axes[1, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Other columns: Model predictions
        for idx, result in enumerate(results):
            pred = result['predictions'][sample_idx]['preds'][0].squeeze().numpy()
            
            # Calculate Dice for this prediction
            intersection = (pred * mask).sum()
            dice = 2 * intersection / (pred.sum() + mask.sum() + 1e-7)
            
            # Top row: Prediction overlay
            axes[0, idx + 1].imshow(image)
            axes[0, idx + 1].imshow(pred, alpha=0.5, cmap='Reds')
            axes[0, idx + 1].set_title(f"{result['name']}\nDice: {dice:.3f}", 
                                      fontsize=10, fontweight='bold')
            axes[0, idx + 1].axis('off')
            
            # Bottom row: Error map (FP in red, FN in blue, correct in green)
            error_map = np.zeros((*mask.shape, 3))
            # True Positives - Green
            error_map[(mask == 1) & (pred == 1)] = [0, 1, 0]
            # True Negatives - Black (background)
            error_map[(mask == 0) & (pred == 0)] = [0, 0, 0]
            # False Positives - Red
            error_map[(mask == 0) & (pred == 1)] = [1, 0, 0]
            # False Negatives - Blue
            error_map[(mask == 1) & (pred == 0)] = [0, 0, 1]
            
            axes[1, idx + 1].imshow(error_map)
            axes[1, idx + 1].set_title('Error Map', fontsize=10, fontweight='bold')
            axes[1, idx + 1].axis('off')
        
        plt.suptitle(f'Sample {sample_idx + 1} - Model Comparison', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(self.comparison_dir / f'comparison_sample_{sample_idx:03d}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results):
        """Save comparison results to JSON"""
        output = {
            'test_data': str(self.args.test_data),
            'num_images': len(self.test_loader.dataset),
            'image_size': self.args.image_size,
            'threshold': self.args.threshold,
            'timestamp': datetime.now().isoformat(),
            'models': []
        }
        
        for result in results:
            model_result = {
                'name': result['name'],
                'model_type': result['model_type'],
                'variant': result['variant'],
                'checkpoint': result['checkpoint'],
                'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else int(v) if isinstance(v, np.integer) else v
                           for k, v in result['metrics'].items()}
            }
            output['models'].append(model_result)
        
        output_file = self.output_dir / 'comparison_results.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='Compare multiple trained models')
    
    # Models
    parser.add_argument('--models-config', type=str, required=True,
                       help='Path to models configuration JSON file')
    
    # Data
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size (default: 512)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers (default: 4)')
    parser.add_argument('--compute-loss', action='store_true',
                       help='Compute loss during testing')
    parser.add_argument('--loss', type=str, default='combined',
                       help='Loss function (default: combined)')
    
    # Testing
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold (default: 0.5)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate comparison visualizations (default: True)')
    parser.add_argument('--num-viz-samples', type=int, default=10,
                       help='Number of samples to visualize (default: 10)')
    
    
    # Output
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory (default: comparison_results)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Models config: {args.models_config}")
    print(f"  Test data: {args.test_data}")
    print(f"  Image size: {args.image_size}")
    print(f"  Output: {args.output_dir}")
    
    # Create comparator
    comparator = ModelComparator(args)
    
    # Run comparison
    results = comparator.compare_all()
    
    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
