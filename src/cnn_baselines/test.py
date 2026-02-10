"""
Inference and Testing Script for River Water Segmentation
Supports model evaluation, visualization, and batch prediction
"""

import os
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from models import get_model
from dataset import RiverSegmentationDataset
from utils.metrics import SegmentationMetrics, StreamingMetrics, print_metrics


class Predictor:
    """Model inference and evaluation"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint.get('args', {})
        
        # Initialize model (always RGB - 3 channels)
        model_name = self.config.get('model', 'unet')
        
        self.model = get_model(model_name, n_channels=3, n_classes=1, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded {model_name} model")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Best Val Dice: {checkpoint.get('best_val_dice', 'unknown')}")

    @torch.no_grad()
    def predict_image(self, image_path, return_prob=False):
        """
        Predict segmentation mask for a single image
        
        Args:
            image_path: Path to input image
            return_prob: Return probability map instead of binary mask
        
        Returns:
            Predicted mask (H, W) as numpy array
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Get image size from config
        image_size = self.config.get('image_size', 512)
        image_resized = cv2.resize(image, (image_size, image_size))
        
        # Normalize
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        output = self.model(image_tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize to original size
        prob = cv2.resize(prob, (original_size[1], original_size[0]))
        
        if return_prob:
            return prob
        else:
            return (prob > 0.5).astype(np.uint8)

    @torch.no_grad()
    def predict_batch(self, image_paths, output_dir, visualize=True, save_masks=True):
        """
        Predict masks for multiple images
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save predictions
            visualize: Create visualization images
            save_masks: Save binary masks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if visualize:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        if save_masks:
            mask_dir = os.path.join(output_dir, 'masks')
            os.makedirs(mask_dir, exist_ok=True)
        
        print(f"Processing {len(image_paths)} images...")
        
        for img_path in tqdm(image_paths):
            img_name = Path(img_path).stem
            
            # Predict
            mask = self.predict_image(img_path, return_prob=False)
            prob = self.predict_image(img_path, return_prob=True)
            
            # Save binary mask
            if save_masks:
                mask_path = os.path.join(mask_dir, f"{img_name}.png")
                cv2.imwrite(mask_path, mask * 255)
            
            # Create visualization
            if visualize:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize mask to match image
                if mask.shape != image.shape[:2]:
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    prob_resized = cv2.resize(prob, (image.shape[1], image.shape[0]))
                else:
                    mask_resized = mask
                    prob_resized = prob
                
                # Create visualization
                vis = self._create_visualization(image, mask_resized, prob_resized)
                
                vis_path = os.path.join(vis_dir, f"{img_name}_vis.png")
                cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        print(f"Results saved to {output_dir}")

    def _create_visualization(self, image, mask, prob):
        """Create visualization with original image, mask overlay, and probability map"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Mask overlay
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 255, 255]  # Cyan for water
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        axes[1].imshow(overlay)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Probability map
        im = axes[2].imshow(prob, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Probability Map')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return vis

    @torch.no_grad()
    def evaluate_dataset(self, data_root, batch_size=8, save_results=True, output_dir='./results'):
        """
        Evaluate model on a dataset
        
        Args:
            data_root: Root directory with images/ and masks/
            batch_size: Batch size for evaluation
            save_results: Save detailed results
            output_dir: Output directory for results
        """
        print(f"\nEvaluating on dataset: {data_root}")
        print("=" * 60)
        
        # Setup dataset (always RGB)
        image_size = self.config.get('image_size', 512)
        
        dataset = RiverSegmentationDataset(
            data_root=data_root,
            image_size=(image_size, image_size),
            normalize=True,
            augment=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize metrics
        streaming_metrics = StreamingMetrics(threshold=0.5)
        detailed_metrics = SegmentationMetrics(threshold=0.5)
        
        # Store per-image metrics
        per_image_metrics = []
        
        print(f"Processing {len(dataset)} images...")
        
        for batch in tqdm(dataloader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Predict
            outputs = self.model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Update streaming metrics
            streaming_metrics.update(outputs, masks)
            
            # Compute per-image metrics
            for i in range(images.size(0)):
                img_metrics = detailed_metrics.compute_metrics(
                    preds[i:i+1],
                    masks[i:i+1]
                )
                img_metrics['image_path'] = batch['image_path'][i]
                per_image_metrics.append(img_metrics)
        
        # Get final metrics
        final_metrics = streaming_metrics.get_metrics()
        
        # Print results
        print_metrics(final_metrics, prefix='Overall ')
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save overall metrics
            metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=4)
            
            # Save per-image metrics
            per_image_path = os.path.join(output_dir, 'per_image_metrics.json')
            with open(per_image_path, 'w') as f:
                json.dump(per_image_metrics, f, indent=4)
            
            # Create summary report
            self._create_summary_report(final_metrics, per_image_metrics, output_dir)
            
            print(f"\nResults saved to {output_dir}")
        
        return final_metrics, per_image_metrics

    def _create_summary_report(self, overall_metrics, per_image_metrics, output_dir):
        """Create a summary report"""
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("Model Configuration:\n")
            f.write("-"*60 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("Overall Metrics:\n")
            f.write("-"*60 + "\n")
            f.write(f"Dice Coefficient: {overall_metrics['dice']:.4f}\n")
            f.write(f"IoU Score:        {overall_metrics['iou']:.4f}\n")
            f.write(f"Accuracy:         {overall_metrics['accuracy']:.4f}\n")
            f.write(f"Precision:        {overall_metrics['precision']:.4f}\n")
            f.write(f"Recall:           {overall_metrics['recall']:.4f}\n")
            f.write(f"F1 Score:         {overall_metrics['f1']:.4f}\n")
            f.write(f"Specificity:      {overall_metrics['specificity']:.4f}\n")
            f.write("\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"TP: {overall_metrics['TP']:,} | FP: {overall_metrics['FP']:,}\n")
            f.write(f"FN: {overall_metrics['FN']:,} | TN: {overall_metrics['TN']:,}\n")
            f.write("\n")
            
            # Per-image statistics
            dice_scores = [m['dice'] for m in per_image_metrics]
            iou_scores = [m['iou'] for m in per_image_metrics]
            
            f.write("Per-Image Statistics:\n")
            f.write("-"*60 + "\n")
            f.write(f"Number of images: {len(per_image_metrics)}\n")
            f.write(f"Dice - Mean: {np.mean(dice_scores):.4f}, Std: {np.std(dice_scores):.4f}\n")
            f.write(f"Dice - Min: {np.min(dice_scores):.4f}, Max: {np.max(dice_scores):.4f}\n")
            f.write(f"IoU  - Mean: {np.mean(iou_scores):.4f}, Std: {np.std(iou_scores):.4f}\n")
            f.write(f"IoU  - Min: {np.min(iou_scores):.4f}, Max: {np.max(iou_scores):.4f}\n")
            f.write("\n")
            
            # Best and worst performing images
            sorted_by_dice = sorted(per_image_metrics, key=lambda x: x['dice'], reverse=True)
            
            f.write("Top 5 Best Performing Images (by Dice):\n")
            f.write("-"*60 + "\n")
            for i, m in enumerate(sorted_by_dice[:5], 1):
                f.write(f"{i}. {Path(m['image_path']).name} - Dice: {m['dice']:.4f}, IoU: {m['iou']:.4f}\n")
            f.write("\n")
            
            f.write("Top 5 Worst Performing Images (by Dice):\n")
            f.write("-"*60 + "\n")
            for i, m in enumerate(sorted_by_dice[-5:][::-1], 1):
                f.write(f"{i}. {Path(m['image_path']).name} - Dice: {m['dice']:.4f}, IoU: {m['iou']:.4f}\n")
            f.write("\n")
            
            f.write("="*60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Test River Water Segmentation Models')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['evaluate', 'predict', 'single'],
                       help='Inference mode')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory for evaluation (required for evaluate mode)')
    parser.add_argument('--image-path', type=str, default=None,
                       help='Single image path (for single mode)')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory with images (for predict mode)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--save-masks', action='store_true', default=True,
                       help='Save predicted masks')
    
    # System parameters
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize predictor
    predictor = Predictor(args.checkpoint, device=args.device)
    
    if args.mode == 'evaluate':
        # Evaluate on dataset
        if args.data_root is None:
            raise ValueError("--data-root is required for evaluate mode")
        
        predictor.evaluate_dataset(
            data_root=args.data_root,
            batch_size=args.batch_size,
            save_results=True,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'predict':
        # Predict on directory of images
        if args.image_dir is None:
            raise ValueError("--image-dir is required for predict mode")
        
        # Get all images
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(args.image_dir).glob(ext))
        
        predictor.predict_batch(
            image_paths=[str(p) for p in image_paths],
            output_dir=args.output_dir,
            visualize=args.visualize,
            save_masks=args.save_masks
        )
    
    elif args.mode == 'single':
        # Predict single image
        if args.image_path is None:
            raise ValueError("--image-path is required for single mode")
        
        mask = predictor.predict_image(args.image_path, return_prob=False)
        prob = predictor.predict_image(args.image_path, return_prob=True)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        img_name = Path(args.image_path).stem
        mask_path = os.path.join(args.output_dir, f"{img_name}_mask.png")
        prob_path = os.path.join(args.output_dir, f"{img_name}_prob.png")
        
        cv2.imwrite(mask_path, mask * 255)
        cv2.imwrite(prob_path, (prob * 255).astype(np.uint8))
        
        # Create visualization
        if args.visualize:
            image = cv2.imread(args.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            vis = predictor._create_visualization(image, mask, prob)
            vis_path = os.path.join(args.output_dir, f"{img_name}_vis.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
