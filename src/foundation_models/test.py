"""
Test and Evaluate Transformer Models
RGB Only - Inference and Evaluation
"""

import os
import argparse
import json
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from models import get_model
from dataset import RiverSegmentationDataset
from utils.metrics import SegmentationMetrics, print_metrics


class Evaluator:
    """Model evaluation"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint.get('args', {})
        model_name = self.config.get('model', 'segformer_b2')
        
        # Load model
        self.model = get_model(model_name, num_classes=1)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model: {model_name}")
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Best Dice: {checkpoint.get('best_val_dice', 'unknown')}")

    @torch.no_grad()
    def predict_image(self, image_path, return_prob=False):
        """Predict single image"""
        # Load
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Resize
        image_size = self.config.get('image_size', 512)
        image = cv2.resize(image, (image_size, image_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)
        
        # Predict
        output = self.model(image)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize to original
        prob = cv2.resize(prob, (original_size[1], original_size[0]))
        
        if return_prob:
            return prob
        else:
            return (prob > 0.5).astype(np.uint8)

    @torch.no_grad()
    def evaluate_dataset(self, data_root, batch_size=8, save_results=True, output_dir='./results'):
        """Evaluate on dataset"""
        print(f"\nEvaluating: {data_root}")
        print("=" * 60)
        
        # Setup dataset
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
        
        # Metrics
        metrics_calc = SegmentationMetrics(threshold=0.5)
        all_preds = []
        all_targets = []
        
        print(f"Processing {len(dataset)} images...")
        
        for batch in tqdm(dataloader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            outputs = self.model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = metrics_calc.compute_metrics(all_preds, all_targets)
        
        # Print
        print_metrics(metrics)
        
        # Save
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, 'results.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"\nResults saved to {output_dir}")
        
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Test Transformer Models')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['evaluate', 'single'],
                       help='Mode: evaluate or single')
    
    # Data
    parser.add_argument('--data-root', type=str, default=None,
                       help='Data root for evaluation')
    parser.add_argument('--image-path', type=str, default=None,
                       help='Single image path')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    evaluator = Evaluator(args.checkpoint, device=args.device)
    
    if args.mode == 'evaluate':
        if args.data_root is None:
            raise ValueError("--data-root required for evaluate mode")
        
        evaluator.evaluate_dataset(
            data_root=args.data_root,
            batch_size=args.batch_size,
            save_results=True,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'single':
        if args.image_path is None:
            raise ValueError("--image-path required for single mode")
        
        mask = evaluator.predict_image(args.image_path, return_prob=False)
        prob = evaluator.predict_image(args.image_path, return_prob=True)
        
        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        img_name = Path(args.image_path).stem
        
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_mask.png"), mask * 255)
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_prob.png"), (prob * 255).astype(np.uint8))
        
        print(f"Saved to {args.output_dir}")


if __name__ == '__main__':
    main()
