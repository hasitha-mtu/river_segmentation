"""
Unified Training Script with Weights & Biases Integration
Supports: CNN Baselines, Transformers, Hybrid SOTA, Foundation Models
Features: Memory optimization, wandb logging, TensorBoard logging
"""

import os
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Weights & Biases
import wandb

from models import get_model, get_model_varient
from src.utils.losses import get_loss_function
from src.dataset.dataset_loader import get_training_dataloaders
from src.utils.metrics import SegmentationMetrics


class UnifiedTrainer:
    """Unified trainer with wandb integration and memory optimization"""
    
    def __init__(self, config):
        """
        Initialize trainer with configuration dictionary
        
        Args:
            config (dict): Configuration dictionary with all training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ===== GPU SETUP & DIAGNOSTICS =====
        if torch.cuda.is_available():
            # Fix cuDNN issues
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            
            print("\n" + "="*80)
            print("GPU MEMORY CHECK")
            print("="*80)
            torch.cuda.empty_cache()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            print("="*80 + "\n")
        
        # Setup directories
        self.setup_directories()
        
        # ===== WANDB INITIALIZATION =====
        self.use_wandb = config['logging'].get('use_wandb', False)
        if self.use_wandb:
            self.init_wandb()
        
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
        
        # ===== WANDB: Watch model =====
        if self.use_wandb and config['logging'].get('watch_model', False):
            wandb.watch(self.model, log='all', log_freq=100)
    
    def init_wandb(self):
        """Initialize Weights & Biases"""
        model_name = self.config['model']['name']
        variant = self.config['model'].get('variant', None)
        
        # Create run name
        run_name = f"{model_name}"
        if variant:
            run_name += f"_{variant}"
        run_name += f"_{self.config['loss']['type']}"
        run_name += f"_bs{self.config['training']['batch_size']}"
        run_name += f"_lr{self.config['training']['optimizer']['learning_rate']}"
        
        # Initialize wandb
        wandb.init(
            project=self.config['logging'].get('wandb_project', 'river-segmentation'),
            name=run_name,
            config=self.config,
            tags=[
                model_name,
                variant if variant else 'no_variant',
                self.config['loss']['type'],
                f"img_{self.config['data']['image_size']}"
            ],
            notes=self.config['logging'].get('wandb_notes', ''),
            dir=self.exp_dir
        )
        
        # Log system info
        if torch.cuda.is_available():
            wandb.config.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version()
            })
        
        print(f"\n✓ Weights & Biases initialized: {wandb.run.url}")
    
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
        if self.use_wandb:
            print(f"Wandb: ENABLED ({wandb.run.name})")
        else:
            print(f"Wandb: DISABLED")
        print(f"{'='*80}\n")
    
    def train_epoch(self, epoch):
        """
        Train one epoch - MEMORY OPTIMIZED with wandb logging
        """
        self.model.train()
        
        total_loss = 0
        # Memory-optimized: Use online metrics instead of accumulating tensors
        running_metrics = {
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
            'intersection': 0, 'union': 0
        }
        
        # For wandb batch-level logging
        batch_losses = []
        
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
            batch_losses.append(loss.item())
            
            # Memory-optimized: Compute metrics online without storing tensors
            with torch.no_grad():
                preds = (torch.sigmoid(main_out) > 0.5).float()
                masks_binary = masks.float()
                
                # Compute confusion matrix components
                tp = (preds * masks_binary).sum().item()
                fp = (preds * (1 - masks_binary)).sum().item()
                fn = ((1 - preds) * masks_binary).sum().item()
                tn = ((1 - preds) * (1 - masks_binary)).sum().item()
                
                # Compute intersection and union for IoU
                intersection = (preds * masks_binary).sum().item()
                union = (preds + masks_binary).clamp(0, 1).sum().item()
                
                # Accumulate
                running_metrics['tp'] += tp
                running_metrics['fp'] += fp
                running_metrics['fn'] += fn
                running_metrics['tn'] += tn
                running_metrics['intersection'] += intersection
                running_metrics['union'] += union
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log batch to TensorBoard and wandb
            log_interval = self.config['system'].get('log_interval', 10)
            if batch_idx % log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                
                # TensorBoard
                self.writer.add_scalar('Batch/Loss', loss.item(), step)
                if self.config['loss']['type'] == 'combined':
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'Batch/{key}', value, step)
                
                # Wandb
                if self.use_wandb:
                    wandb_log = {
                        'train/batch_loss': loss.item(),
                        'train/batch_step': step,
                    }
                    if self.config['loss']['type'] == 'combined':
                        for key, value in loss_dict.items():
                            wandb_log[f'train/batch_{key}'] = value
                    wandb.log(wandb_log)
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Compute final metrics from accumulated values
        avg_loss = total_loss / len(self.train_loader)
        
        # Calculate Dice and IoU from accumulated metrics
        epsilon = 1e-7
        dice = (2 * running_metrics['tp']) / (2 * running_metrics['tp'] + 
                                               running_metrics['fp'] + 
                                               running_metrics['fn'] + epsilon)
        iou = running_metrics['intersection'] / (running_metrics['union'] + epsilon)
        precision = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fp'] + epsilon)
        recall = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fn'] + epsilon)
        
        metrics = {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall
        }
        
        return avg_loss, metrics
    
    def validate(self, epoch):
        """
        Validate - MEMORY OPTIMIZED with wandb logging
        """
        self.model.eval()
        
        total_loss = 0
        running_metrics = {
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
            'intersection': 0, 'union': 0
        }
        
        # For logging sample predictions to wandb
        sample_images = []
        sample_masks = []
        sample_preds = []
        log_samples = self.use_wandb and self.config['logging'].get('log_images', False)
        max_samples = 4  # Number of validation samples to log
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                
                if self.config['loss']['type'] == 'combined':
                    loss, _ = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Compute metrics online
                preds = (torch.sigmoid(outputs) > 0.5).float()
                masks_binary = masks.float()
                
                # Compute confusion matrix components
                tp = (preds * masks_binary).sum().item()
                fp = (preds * (1 - masks_binary)).sum().item()
                fn = ((1 - preds) * masks_binary).sum().item()
                tn = ((1 - preds) * (1 - masks_binary)).sum().item()
                
                # Compute intersection and union
                intersection = (preds * masks_binary).sum().item()
                union = (preds + masks_binary).clamp(0, 1).sum().item()
                
                # Accumulate
                running_metrics['tp'] += tp
                running_metrics['fp'] += fp
                running_metrics['fn'] += fn
                running_metrics['tn'] += tn
                running_metrics['intersection'] += intersection
                running_metrics['union'] += union
                
                # Collect samples for wandb logging
                if log_samples and len(sample_images) < max_samples:
                    sample_images.append(images[0].cpu())
                    sample_masks.append(masks[0].cpu())
                    sample_preds.append(preds[0].cpu())
        
        # Compute final metrics
        avg_loss = total_loss / len(self.val_loader)
        
        epsilon = 1e-7
        dice = (2 * running_metrics['tp']) / (2 * running_metrics['tp'] + 
                                               running_metrics['fp'] + 
                                               running_metrics['fn'] + epsilon)
        iou = running_metrics['intersection'] / (running_metrics['union'] + epsilon)
        precision = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fp'] + epsilon)
        recall = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fn'] + epsilon)
        
        metrics = {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall
        }
        
        # Log sample predictions to wandb
        if log_samples and len(sample_images) > 0:
            self.log_predictions_wandb(sample_images, sample_masks, sample_preds, epoch)
        
        return avg_loss, metrics
    
    def log_predictions_wandb(self, images, masks, preds, epoch):
        """Log sample predictions to wandb"""
        wandb_images = []
        
        for img, mask, pred in zip(images, masks, preds):
            # Convert tensors to numpy
            img_np = img.permute(1, 2, 0).numpy()  # [H, W, C]
            mask_np = mask.squeeze().numpy()  # [H, W]
            pred_np = pred.squeeze().numpy()  # [H, W]
            
            # Normalize image to [0, 1] if needed
            if img_np.min() < 0:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # Create wandb Image with masks overlay
            wandb_images.append(
                wandb.Image(
                    img_np,
                    masks={
                        "ground_truth": {"mask_data": mask_np, "class_labels": {0: "background", 1: "river"}},
                        "predictions": {"mask_data": pred_np, "class_labels": {0: "background", 1: "river"}}
                    },
                    caption=f"Epoch {epoch}"
                )
            )
        
        wandb.log({"val/predictions": wandb_images, "epoch": epoch})
    
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
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best - Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            
            # Log best model to wandb
            if self.use_wandb and self.config['logging'].get('save_model_wandb', False):
                wandb.save(best_path)
        
        # Periodic save
        save_interval = self.config['system'].get('save_interval', 10)
        if epoch % save_interval == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self):
        """Load checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_dice = checkpoint['best_val_dice']
            self.best_val_iou = checkpoint['best_val_iou']
            print(f"Resumed from epoch {self.start_epoch}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    def train(self):
        """Main training loop with wandb logging"""
        print(f"\nStarting training from epoch {self.start_epoch}...")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            print(f"{'='*80}")
            
            # Clear cache at start of epoch
            torch.cuda.empty_cache()
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Clear cache after training
            torch.cuda.empty_cache()
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Clear cache after validation
            torch.cuda.empty_cache()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Print
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Learning Rate: {lr:.2e}")
            print(f"Train - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
            
            # GPU memory
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                print(f"GPU Memory: {gpu_mem:.2f} GB allocated")
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Dice/train', train_metrics['dice'], epoch)
            self.writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
            self.writer.add_scalar('IoU/train', train_metrics['iou'], epoch)
            self.writer.add_scalar('IoU/val', val_metrics['iou'], epoch)
            self.writer.add_scalar('Precision/train', train_metrics['precision'], epoch)
            self.writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
            self.writer.add_scalar('Recall/train', train_metrics['recall'], epoch)
            self.writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
            self.writer.add_scalar('LR', lr, epoch)
            
            # Log to Wandb
            if self.use_wandb:
                wandb_log = {
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/dice': train_metrics['dice'],
                    'train/iou': train_metrics['iou'],
                    'train/precision': train_metrics['precision'],
                    'train/recall': train_metrics['recall'],
                    'val/loss': val_loss,
                    'val/dice': val_metrics['dice'],
                    'val/iou': val_metrics['iou'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'learning_rate': lr,
                }
                
                if torch.cuda.is_available():
                    wandb_log['gpu_memory_gb'] = gpu_mem
                
                wandb.log(wandb_log)
            
            # Save
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.best_val_iou = val_metrics['iou']
                
                # Log best metrics to wandb
                if self.use_wandb:
                    wandb.run.summary['best_val_dice'] = self.best_val_dice
                    wandb.run.summary['best_val_iou'] = self.best_val_iou
                    wandb.run.summary['best_epoch'] = epoch
            
            self.save_checkpoint(epoch, val_metrics['dice'], val_metrics['iou'], is_best)
            print("-" * 80)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print(f"Best Dice: {self.best_val_dice:.4f}")
        print(f"Best IoU:  {self.best_val_iou:.4f}")
        print("="*80 + "\n")
        
        self.writer.close()
        
        if self.use_wandb:
            wandb.finish()


def get_default_config():
    """Get default configuration dictionary"""
    config = {
        'model': {
            'name': 'unet',
            'variant': None,
            'n_channels': 3,
            'n_classes': 1,
        },
        'data': {
            'data_root': './dataset/processed_1024',
            'image_size': 1024,
            'augment_train': True,
        },
        'training': {
            'batch_size': 4,
            'epochs': 100,
            'clip_grad': 1.0,
            'resume': False,
            'optimizer': {
                'type': 'adam',
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'momentum': 0.9,
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6,
                'step_size': 30,
                'gamma': 0.1,
                'warmup_epochs': 5,
            },
        },
        'loss': {
            'type': 'combined',
            'bce_weight': 1.0,
            'dice_weight': 1.0,
            'boundary_weight': 1.0,
            'use_boundary': True,
        },
        'logging': {
            'use_wandb': True,  # Enable/disable wandb
            'wandb_project': 'river-segmentation',  # Wandb project name
            'wandb_notes': 'UAV river segmentation with tree occlusion',
            'watch_model': False,  # Watch model gradients (can be slow)
            'log_images': True,  # Log sample predictions
            'save_model_wandb': True,  # Save best model to wandb
        },
        'system': {
            'seed': 42,
            'num_workers': 0,  # Set to 0 for Windows/WSL2
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
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
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
            
            # Clear GPU between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
                
                # Clear GPU between models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


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
    config['training']['epochs'] = 100  # Full training
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
    
    # Wandb configuration
    config['logging']['use_wandb'] = True
    config['logging']['wandb_project'] = 'river-segmentation'
    config['logging']['wandb_notes'] = 'UAV river segmentation - tree occlusion challenge'
    config['logging']['watch_model'] = False  # Set True to log gradients (slower)
    config['logging']['log_images'] = True
    config['logging']['save_model_wandb'] = True
    
    # System configuration
    config['system']['seed'] = 42
    config['system']['num_workers'] = 0  # 0 for Windows/WSL2
    config['system']['output_dir'] = './experiments'
    
    # ===== CHOOSE TRAINING MODE =====
    
    # Option 1: Train a single model
    config['model']['name'] = 'unet'
    config['model']['variant'] = None
    train_single_model(config)
    
    # Option 2: Train all models
    # train_all_models(config)


if __name__ == '__main__':
    main()
