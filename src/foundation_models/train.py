"""
Training Script for Transformer Models
SegFormer-B0, SegFormer-B2, Swin-UNet-Tiny - RGB Only
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
from torch.amp import autocast
from torch.amp import GradScaler

from src.cnn_baselines.models import get_model
from src.utils.losses import get_loss_function
from src.dataset.dataset_loader import get_training_dataloaders
from src.utils.metrics import SegmentationMetrics


class Trainer:
    """Transformer model trainer"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Setup model
        print(f"Initializing {args.model}...")
        self.model = get_model(args.model, varient=args.varient).to(self.device)
        
        # ═══════════════════════════════════════════════════════════
        # Enable gradient checkpointing if requested (saves ~40% memory)
        # ═══════════════════════════════════════════════════════════
        if args.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled (saves ~40% memory)")
            else:
                print("⚠ Warning: Model doesn't support gradient_checkpointing_enable()")
        
        # Setup data
        print(f"Loading data from {args.data_root}...")
        self.train_loader, self.val_loader = get_training_dataloaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Setup loss
        print(f"Using {args.loss} loss...")
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
        
        # ═══════════════════════════════════════════════════════════
        # FIX: Initialize GradScaler for mixed precision
        # ═══════════════════════════════════════════════════════════
        self.scaler = GradScaler('cuda')
        print("✓ Mixed precision enabled with GradScaler")

        if args.optimizer == 'adamw':
                self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999)
            )
        elif args.optimizer == 'adam':
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
        self.metrics = StreamingMetrics(threshold=0.5)  # Uses incremental updates - no memory accumulation!
        
        # Setup logging
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.start_epoch = 0
        self.best_val_dice = 0.0
        self.best_val_iou = 0.0
        
        # Load checkpoint if resuming
        if args.resume:
            self.load_checkpoint()
        
        print(f"\nConfiguration:")
        print(f"  Device: {self.device}")
        print(f"  Model: {args.model}")
        print(f"  Model varient: {args.varient}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Warmup Epochs: {args.warmup_epochs}")
        print(f"  Total Epochs: {args.epochs}")

    def setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        if self.args.warmup_epochs > 0:
            def warmup_cosine(epoch):
                if epoch < self.args.warmup_epochs:
                    return (epoch + 1) / self.args.warmup_epochs
                else:
                    progress = (epoch - self.args.warmup_epochs) / (self.args.epochs - self.args.warmup_epochs)
                    return 0.5 * (1.0 + np.cos(np.pi * progress))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=warmup_cosine
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_lr
            )

    def setup_directories(self):
        """Create necessary directories"""
        exp_name = f"{self.args.model}_{self.args.varient}_{self.args.loss}"
        self.model_dir = os.path.join(self.args.output_dir, self.args.model)
        self.exp_dir = os.path.join(self.model_dir, exp_name)
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
        """Train one epoch"""
        self.model.train()
        
        # Clear cache at epoch start
        torch.cuda.empty_cache()
        
        total_loss = 0
        
        # ═══════════════════════════════════════════════════════════
        # FIX: Use incremental metrics (NO memory accumulation!)
        # ═══════════════════════════════════════════════════════════
        # Reset metrics for this epoch
        self.metrics.reset()
        
        lr = self.optimizer.param_groups[0]['lr']
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} (LR: {lr:.2e})')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(images)
                # Loss
                if self.args.loss == 'combined':
                    loss, loss_dict = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)
                    loss_dict = {'total': loss.item()}
            
            # ═══════════════════════════════════════════════════════════
            # FIX: Use GradScaler for backward pass
            # ═══════════════════════════════════════════════════════════
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.args.clip_grad > 0:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            total_loss += loss.item()
            
            # ═══════════════════════════════════════════════════════════
            # FIX: Update metrics incrementally (NO accumulation!)
            # ═══════════════════════════════════════════════════════════
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                self.metrics.update(preds, masks)  # Incremental update!
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Clear cache periodically to avoid fragmentation
            if batch_idx % 50 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
            
            # Log batch
            if batch_idx % self.args.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/batch_loss', loss.item(), step)

                if self.args.loss == 'combined':
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'Train/batch_{k}', v, step)
        
        # ═══════════════════════════════════════════════════════════
        # FIX: Compute final metrics (NO concatenation!)
        # ═══════════════════════════════════════════════════════════
        metrics = self.metrics.get_metrics()  # StreamingMetrics uses get_metrics()
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, epoch):
        """Validate"""
        self.model.eval()
        
        # Clear cache before validation
        torch.cuda.empty_cache()
        
        total_loss = 0
        
        # ═══════════════════════════════════════════════════════════
        # FIX: Use incremental metrics (NO memory accumulation!)
        # ═══════════════════════════════════════════════════════════
        # Create separate metrics object for validation
        val_metrics = StreamingMetrics(threshold=0.5)
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(images)
            
                if self.args.loss == 'combined':
                    loss, _ = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # ═══════════════════════════════════════════════════════════
            # FIX: Update metrics incrementally (NO accumulation!)
            # ═══════════════════════════════════════════════════════════
            preds = torch.sigmoid(outputs) > 0.5
            val_metrics.update(preds, masks)  # Incremental update!
        
        # ═══════════════════════════════════════════════════════════
        # FIX: Compute final metrics (NO concatenation!)
        # ═══════════════════════════════════════════════════════════
        metrics = val_metrics.get_metrics()  # StreamingMetrics uses get_metrics()
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, metrics

    def save_checkpoint(self, epoch, val_dice, val_iou, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),  # FIX: Add scaler
            'best_val_dice': self.best_val_dice,
            'best_val_iou': self.best_val_iou,
            'args': vars(self.args)
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
            print(f"  ✓ New best - Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Periodic save
        if epoch % self.args.save_interval == 0:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth'))


    def load_checkpoint(self):
        checkpoint_path = f"{self.checkpoint_dir}/latest.pth"
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # FIX: Load scaler state if available
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        self.best_val_iou = checkpoint.get('best_val_iou', 0.0)
        
        print(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("TRAINING START")
        print("="*80 + "\n")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            start = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print
            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch}/{self.args.epochs} - {elapsed:.1f}s - LR: {lr:.6f}")
            print(f"Train - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
            
            # Log
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Dice/train', train_metrics['dice'], epoch)
            self.writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
            self.writer.add_scalar('IoU/train', train_metrics['iou'], epoch)
            self.writer.add_scalar('IoU/val', val_metrics['iou'], epoch)
            self.writer.add_scalar('LR', lr, epoch)
            
            # Save
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.best_val_iou = val_metrics['iou']
            
            self.save_checkpoint(epoch, val_metrics['dice'], val_metrics['iou'], is_best)
            print("-" * 80)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print(f"Best Dice: {self.best_val_dice:.4f}")
        print(f"Best IoU:  {self.best_val_iou:.4f}")
        print("="*80 + "\n")
        
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Transformer Models (RGB Only)')
    
    # Model
    parser.add_argument('--model', type=str, default='sam',
                       choices=['sam', 'dinov2'],
                       help='Model: SAM, DINOv2')
    
    # Data
    parser.add_argument('--data-root', type=str, required=True,
                       help='Data root with images/ and masks/')
    parser.add_argument('--image-size', type=int, default=384,
                       help='Image size (default: 384, saves memory vs 512)')
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
    
    # ═══════════════════════════════════════════════════════════
    # FIX: Set memory allocator config to avoid fragmentation
    # ═══════════════════════════════════════════════════════════
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        # Clear any cached memory
        torch.cuda.empty_cache()
    
    # Train
    trainer = Trainer(args)
    trainer.train()

def train_all_models():
    args = parse_args()
    

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model_names = ['sam', 'dinov2']

    for model_name in model_names:
        args.model = model_name
        print(f"\nTraining {args.model}...")
        varients = get_model_varient(model_name)
        for varient in varients:
            args.varient = varient
            print(f'Model name: {model_name}')
            print(f'Model varient: {varient}')
            # Create trainer and start training
            trainer = Trainer(args)
            trainer.train()

if __name__ == '__main__':
    train_all_models()
