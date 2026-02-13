"""
Unified Training Script for All Model Types
Supports: CNN Baselines, Transformers, Hybrid SOTA, Foundation Models
Configuration via dictionary (no argparse)
"""

import os
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import get_model, get_model_varient
from src.utils.losses import get_loss_function
from src.dataset.dataset_loader import get_training_dataloaders
from src.utils.metrics import SegmentationMetrics


class UnifiedTrainer:
    """Unified trainer for all model architectures"""
    
    def __init__(self, config):
        """
        Initialize trainer with configuration dictionary
        
        Args:
            config (dict): Configuration dictionary with all training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Setup model
        model_name = config['model']['name']
        variant = config['model'].get('variant', None)
        n_channels = config['model'].get('n_channels', 3)
        n_classes = config['model'].get('n_classes', 1)
        
        print(f"\nInitializing {model_name}" + (f" ({variant})" if variant else "") + "...")
        self.model = get_model(
            model_name=model_name,
            variant=variant,
            n_channels=n_channels,
            n_classes=n_classes
        ).to(self.device)
        
        # Setup data
        print(f"Loading data from {config['data']['data_root']}...")
        self.train_loader, self.val_loader = get_training_dataloaders(
            data_dir=config['data']['data_root'],
            batch_size=config['training']['batch_size'],
            num_workers=config['system']['num_workers'],
            augment_train=config['data'].get('augment_train', True),
            image_size=config['data']['image_size']
        )
        
        # Setup loss
        loss_config = config['loss']
        print(f"Using {loss_config['type']} loss...")
        if loss_config['type'] == 'combined':
            self.criterion = get_loss_function(
                loss_config['type'],
                bce_weight=loss_config.get('bce_weight', 1.0),
                dice_weight=loss_config.get('dice_weight', 1.0),
                boundary_weight=loss_config.get('boundary_weight', 1.0),
                use_boundary=loss_config.get('use_boundary', False)
            )
        else:
            self.criterion = get_loss_function(loss_config['type'])
        
        # Setup optimizer
        optimizer_config = config['training']['optimizer']
        if optimizer_config['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay'],
                betas=(0.9, 0.999)
            )
        elif optimizer_config['type'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config['weight_decay']
            )
        
        # Setup scheduler
        scheduler_config = config['training']['scheduler']
        if scheduler_config['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config['type'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_config['type'] == 'warmup_cosine':
            self.scheduler = self.setup_warmup_scheduler()
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
        if config['training'].get('resume', False):
            self.load_checkpoint()
        
        # Print configuration
        self.print_configuration()
    
    def setup_warmup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        warmup_epochs = self.config['training']['scheduler'].get('warmup_epochs', 5)
        total_epochs = self.config['training']['epochs']
        min_lr = self.config['training']['scheduler'].get('min_lr', 1e-6)
        
        def warmup_cosine(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_cosine
        )
    
    def setup_directories(self):
        """Create necessary directories"""
        model_name = self.config['model']['name']
        variant = self.config['model'].get('variant', None)
        loss_type = self.config['loss']['type']
        
        # Create experiment name
        if variant:
            exp_name = f"{model_name}_{variant}_{loss_type}"
        else:
            exp_name = f"{model_name}_{loss_type}"
        
        # Setup directory structure
        output_dir = self.config['system']['output_dir']
        self.model_dir = os.path.join(output_dir, model_name)
        self.exp_dir = os.path.join(self.model_dir, exp_name)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        print(f"Experiment directory: {self.exp_dir}")
    
    def print_configuration(self):
        """Print training configuration"""
        print(f"\n{'='*80}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {self.config['model']['name']}", end="")
        if self.config['model'].get('variant'):
            print(f" ({self.config['model']['variant']})")
        else:
            print()
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch Size: {self.config['training']['batch_size']}")
        print(f"Learning Rate: {self.config['training']['optimizer']['learning_rate']}")
        print(f"Optimizer: {self.config['training']['optimizer']['type']}")
        print(f"Scheduler: {self.config['training']['scheduler']['type']}")
        print(f"Loss: {self.config['loss']['type']}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"{'='*80}\n")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        lr = self.optimizer.param_groups[0]['lr']
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config["training"]["epochs"]} (LR: {lr:.2e})'
        )
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)

            main_out, aux_out = outputs if isinstance(outputs, tuple) else (outputs, None)
            
            # Loss
            if self.config['loss']['type'] == 'combined':
                loss, loss_dict = self.criterion(main_out, masks, aux_out)
            else:
                loss = self.criterion(main_out, masks)
                loss_dict = {'total': loss.item()}
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            clip_grad = self.config['training'].get('clip_grad', 0)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            with torch.no_grad():
                preds = torch.sigmoid(main_out) > 0.5
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log batch
            log_interval = self.config['system'].get('log_interval', 10)
            if batch_idx % log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Batch/Loss', loss.item(), step)
                if self.config['loss']['type'] == 'combined':
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'Batch/{key}', value, step)
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.metrics.compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def validate(self, epoch):
        """Validate"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                
                if self.config['loss']['type'] == 'combined':
                    loss, _ = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.metrics.compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, val_dice, val_iou, is_best):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_dice': self.best_val_dice,
            'best_val_iou': self.best_val_iou,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
            print(f"  ✓ New best - Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Periodic save
        save_interval = self.config['system'].get('save_interval', 10)
        if epoch % save_interval == 0:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth'))
    
    def load_checkpoint(self):
        """Load checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
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
        print("\n" + "="*80)
        print("TRAINING START")
        print("="*80 + "\n")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            start = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Print
            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch}/{self.config['training']['epochs']} - {elapsed:.1f}s - LR: {lr:.6f}")
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


def get_default_config():
    """Get default configuration dictionary"""
    config = {
        'model': {
            'name': 'unet',  # Model name
            'variant': None,  # Model variant (if applicable)
            'n_channels': 3,  # RGB
            'n_classes': 1,  # Binary segmentation
        },
        'data': {
            'data_root': './dataset/processed_1024',  # Path to dataset (with train/val/test folders)
            'image_size': 1024,  # Input image size (512 or 1024)
            'augment_train': True,  # Apply data augmentation to training set
        },
        'training': {
            'batch_size': 4,
            'epochs': 100,
            'clip_grad': 1.0,  # Gradient clipping (0 to disable)
            'resume': False,  # Resume from checkpoint
            'optimizer': {
                'type': 'adam',  # 'adam', 'adamw', 'sgd'
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'momentum': 0.9,  # For SGD
            },
            'scheduler': {
                'type': 'cosine',  # 'cosine', 'step', 'warmup_cosine', 'none'
                'min_lr': 1e-6,
                'step_size': 30,  # For step scheduler
                'gamma': 0.1,  # For step scheduler
                'warmup_epochs': 5,  # For warmup_cosine
            },
        },
        'loss': {
            'type': 'combined',  # 'bce', 'dice', 'iou', 'focal', 'boundary', 'combined'
            'bce_weight': 1.0,
            'dice_weight': 1.0,
            'boundary_weight': 1.0,
            'use_boundary': True,
        },
        'system': {
            'seed': 42,
            'num_workers': 4,
            'output_dir': './experiments',
            'log_interval': 10,
            'save_interval': 10,
        }
    }
    return config


def train_single_model(config):
    """Train a single model with given configuration"""
    # Set seeds
    torch.manual_seed(config['system']['seed'])
    np.random.seed(config['system']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['system']['seed'])
    
    # Create trainer and train
    trainer = UnifiedTrainer(config)
    trainer.train()


def train_all_models(base_config):
    """
    Train all models with their variants
    
    Args:
        base_config: Base configuration dictionary that will be modified for each model
    """
    # Define all models and their variants
    all_models = {
        # CNN Baselines (no variants)
        'unet': [],
        'unetpp': [],
        'resunetpp': [],
        'deeplabv3plus': [],
        'deeplabv3plus_cbam': [],
        
        # Transformers
        'segformer': ['b0', 'b2'],
        'swin_unet': ['tiny'],
        
        # Hybrid SOTA
        'convnext_upernet': ['tiny', 'small', 'base'],
        'hrnet_ocr': ['w18', 'w32', 'w48'],
        
        # Foundation Models
        'sam': ['vit_b', 'vit_l', 'vit_h'],
        'dinov2': ['vit_s', 'vit_b', 'vit_l', 'vit_g'],
    }
    
    # Train each model
    for model_name, variants in all_models.items():
        if not variants:
            # Models without variants
            print(f"\n{'='*80}")
            print(f"Training {model_name}")
            print(f"{'='*80}\n")
            
            config = base_config.copy()
            config['model']['name'] = model_name
            config['model']['variant'] = None
            
            train_single_model(config)
        else:
            # Models with variants
            for variant in variants:
                print(f"\n{'='*80}")
                print(f"Training {model_name} - {variant}")
                print(f"{'='*80}\n")
                
                config = base_config.copy()
                config['model']['name'] = model_name
                config['model']['variant'] = variant
                
                train_single_model(config)


def main():
    """
    Main function - Configure your training here
    """
    # Get base configuration
    config = get_default_config()
    
    # ===== CUSTOMIZE YOUR CONFIGURATION HERE =====
    
    # Data configuration
    config['data']['data_root'] = './dataset/processed_1024'
    config['data']['image_size'] = 1024
    config['data']['train_split'] = 0.9
    
    # Training configuration
    config['training']['batch_size'] = 4
    config['training']['epochs'] = 1
    config['training']['clip_grad'] = 1.0
    
    # Optimizer configuration
    config['training']['optimizer']['type'] = 'adam'
    config['training']['optimizer']['learning_rate'] = 1e-4
    config['training']['optimizer']['weight_decay'] = 0.01
    
    # Scheduler configuration
    config['training']['scheduler']['type'] = 'cosine'
    config['training']['scheduler']['min_lr'] = 1e-6
    
    # Loss configuration
    config['loss']['type'] = 'combined'
    config['loss']['bce_weight'] = 1.0
    config['loss']['dice_weight'] = 1.0
    config['loss']['boundary_weight'] = 1.0
    config['loss']['use_boundary'] = True
    
    # System configuration
    config['system']['seed'] = 42
    config['system']['num_workers'] = 4
    config['system']['output_dir'] = './experiments'
    
    # ===== CHOOSE TRAINING MODE =====
    
    # Option 1: Train a single model
    # config['model']['name'] = 'hrnet_ocr'
    # config['model']['variant'] = 'w18'
    # train_single_model(config)

    # config['model']['name'] = 'sam'
    # config['model']['variant'] = 'vit_b'
    # train_single_model(config)

    # config['model']['name'] = 'dinov2'
    # config['model']['variant'] = 'vit_s'
    # train_single_model(config)
    
    # Option 2: Train specific models
    # models_to_train = [
    #     ('unet', None),
    #     ('segformer', 'b0'),
    #     ('sam', 'vit_b'),
    # ]
    # for model_name, variant in models_to_train:
    #     config['model']['name'] = model_name
    #     config['model']['variant'] = variant
    #     train_single_model(config)
    
    # Option 3: Train all models
    print(f'config : {config}')
    train_all_models(config)


if __name__ == '__main__':
    main()
