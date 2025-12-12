"""
Training script for River Water Segmentation Models
Supports all model architectures with comprehensive logging and checkpointing
"""

import os
import time
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import get_model
from losses import get_loss_function
from dataset import get_dataloaders
from metrics import SegmentationMetrics


class Trainer:
    """Training manager for segmentation models"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Setup model (always 3 channels for RGB)
        print(f"Initializing {args.model} model...")
        self.model = get_model(
            args.model,
            n_channels=3,  # Always RGB input
            n_classes=1,
            pretrained=args.pretrained
        ).to(self.device)
        
        # Setup data
        print(f"Loading data from {args.data_root}...")
        self.train_loader, self.val_loader = get_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=(args.image_size, args.image_size),
            train_split=args.train_split,
            seed=args.seed
        )
        
        # Setup loss
        print(f"Using {args.loss} loss function...")
        if args.loss == 'combined':
            self.criterion = get_loss_function(
                args.loss,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                boundary_weight=args.boundary_weight,
                use_boundary=args.use_boundary
            )
        else:
            self.criterion = get_loss_function(args.loss)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Setup scheduler
        if args.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=args.min_lr
            )
        elif args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Setup metrics
        self.metrics = SegmentationMetrics(threshold=0.5)
        
        # Setup logging
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.start_epoch = 0
        self.best_val_dice = 0.0
        self.best_val_iou = 0.0
        
        # Load checkpoint if resuming
        if args.resume:
            self.load_checkpoint(args.resume)
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")


    def setup_directories(self):
        """Create necessary directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{self.args.model}_{self.args.loss}_{timestamp}"
        
        self.exp_dir = os.path.join(self.args.output_dir, exp_name)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=4)
        
        print(f"Experiment directory: {self.exp_dir}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            if self.args.loss == 'combined':
                loss, loss_dict = self.criterion(outputs, masks)
            else:
                loss = self.criterion(outputs, masks)
                loss_dict = {'total': loss.item()}
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            self.optimizer.step()
            
            # Accumulate for metrics
            total_loss += loss.item()
            
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                all_predictions.append(preds.cpu())
                all_targets.append(masks.cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log batch metrics
            if batch_idx % self.args.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/batch_loss', loss.item(), step)
                
                if self.args.loss == 'combined':
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'Train/batch_{k}', v, step)
        
        # Compute epoch metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics.compute_metrics(all_predictions, all_targets)
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc='Validation')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            outputs = self.model(images)
            
            # Compute loss
            if self.args.loss == 'combined':
                loss, _ = self.criterion(outputs, masks)
            else:
                loss = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Accumulate predictions
            preds = torch.sigmoid(outputs) > 0.5
            all_predictions.append(preds.cpu())
            all_targets.append(masks.cpu())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics.compute_metrics(all_predictions, all_targets)
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, metrics

    def save_checkpoint(self, epoch, val_dice, val_iou, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_dice': self.best_val_dice,
            'best_val_iou': self.best_val_iou,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'args': vars(self.args)
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save periodic checkpoints
        if epoch % self.args.save_interval == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        self.best_val_iou = checkpoint.get('best_val_iou', 0.0)
        
        print(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch}/{self.args.epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train - Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
            print(f"Val   - Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/train_dice', train_metrics['dice'], epoch)
            self.writer.add_scalar('Metrics/val_dice', val_metrics['dice'], epoch)
            self.writer.add_scalar('Metrics/train_iou', train_metrics['iou'], epoch)
            self.writer.add_scalar('Metrics/val_iou', val_metrics['iou'], epoch)
            self.writer.add_scalar('Metrics/train_precision', train_metrics['precision'], epoch)
            self.writer.add_scalar('Metrics/val_precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Metrics/train_recall', train_metrics['recall'], epoch)
            self.writer.add_scalar('Metrics/val_recall', val_metrics['recall'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.best_val_iou = val_metrics['iou']
            
            self.save_checkpoint(epoch, val_metrics['dice'], val_metrics['iou'], is_best)
            
            print("-" * 60)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Val Dice: {self.best_val_dice:.4f}")
        print(f"Best Val IoU: {self.best_val_iou:.4f}")
        print("="*60 + "\n")
        
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train River Water Segmentation Models')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights (for DeepLabV3+ variants)')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory containing images/ and masks/')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Input image size (default: 512)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training data split ratio')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                       help='Minimum learning rate for scheduler')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping (0 to disable)')
    
    # Loss parameters
    parser.add_argument('--loss', type=str, default='combined',
                       choices=['bce', 'dice', 'iou', 'focal', 'boundary', 'combined', 'tversky', 'combo'],
                       help='Loss function')
    parser.add_argument('--use-boundary', action='store_true',
                       help='Use boundary loss in combined loss')
    parser.add_argument('--bce-weight', type=float, default=1.0,
                       help='BCE loss weight in combined loss')
    parser.add_argument('--dice-weight', type=float, default=1.0,
                       help='Dice loss weight in combined loss')
    parser.add_argument('--boundary-weight', type=float, default=1.0,
                       help='Boundary loss weight in combined loss')
    
    # Optimizer & Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Logging parameters
    parser.add_argument('--output-dir', type=str, default='./experiments',
                       help='Output directory for experiments')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval (batches)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Checkpoint save interval (epochs)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
