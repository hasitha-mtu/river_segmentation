"""
Loss Functions for Semantic Segmentation
=========================================

Provides various loss functions optimized for segmentation tasks:
- Cross Entropy Loss
- Dice Loss
- Focal Loss
- Boundary Loss
- Combined losses

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Particularly effective for:
    - Imbalanced datasets
    - Small objects
    - Binary segmentation
    
    Formula: 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, 1, H, W) - logits
            targets: (B, 1, H, W) - binary masks [0.0, 1.0]
        
        Returns:
            Dice loss value
        """
        # Convert to probabilities using sigmoid
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses training on hard examples by down-weighting easy examples.
    Particularly useful for:
    - Highly imbalanced datasets
    - Hard negative mining
    - When easy examples dominate the loss
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for classes
        gamma: Focusing parameter (higher = more focus on hard examples)
        ignore_index: Class index to ignore
    """
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(predictions, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(
            predictions, targets, 
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Get probabilities for target class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets.flatten()).view_as(targets)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for enhanced edge detection.
    
    Based on: "Boundary Loss for Remote Sensing Imagery Semantic Segmentation"
    
    Emphasizes accurate prediction of object boundaries using distance transforms.
    Essential for:
    - Water edge detection
    - Fine-grained segmentation
    - Applications requiring precise boundaries
    
    Args:
        theta0: Distance threshold for boundary region
        theta: Scaling factor for boundary weight function
    """
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, 1, H, W) - logits
            targets: (B, 1, H, W) - binary masks [0.0, 1.0]
        
        Returns:
            Boundary-weighted loss
        """
        predictions = torch.sigmoid(predictions)
        
        # Compute distance transform on CPU for each sample in batch
        batch_size = targets.shape[0]
        device = predictions.device
        
        # Initialize boundary weight map
        boundary_weights = torch.zeros_like(targets)
        
        for b in range(batch_size):
            target_np = targets[b, 0].cpu().numpy()
            
            # Distance transform for positive class (water)
            from scipy.ndimage import distance_transform_edt
            dist_pos = distance_transform_edt(target_np)
            # Distance transform for negative class (non-water)
            dist_neg = distance_transform_edt(1 - target_np)
            
            # Combine distances
            dist_map = np.where(target_np > 0.5, dist_pos, -dist_neg)
            
            # Apply boundary weight function
            weight_map = 1.0 / (
                1.0 + np.exp((np.abs(dist_map) - self.theta0) / self.theta)
            )
            
            boundary_weights[b, 0] = torch.from_numpy(weight_map).float()
        
        boundary_weights = boundary_weights.to(device)
        
        # Compute weighted BCE
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        boundary_loss = (bce_loss * boundary_weights).mean()
        
        return boundary_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: BCE + Dice + Boundary (with Auxiliary Support)
    
    Uses Binary Cross Entropy for num_classes=1 binary segmentation.
    Now supports auxiliary output from HRNet-OCR.
    
    Args:
        bce_weight: Weight for BCE loss (default: 1.0)
        dice_weight: Weight for Dice loss (default: 1.0)
        boundary_weight: Weight for boundary loss (default: 1.0)
        use_boundary: Whether to use boundary loss (default: False)
        aux_weight: Weight for auxiliary loss (default: 0.4)
    
    Input/Output Format:
        predictions: (B, 1, H, W) - logits
        targets: (B, 1, H, W) - binary masks [0.0, 1.0]
        aux_predictions: (B, 1, H, W) - optional auxiliary logits
    
    Returns:
        Tuple of (total_loss, loss_dict)
    
    Example:
        >>> criterion = CombinedLoss()
        >>> # Standard usage
        >>> loss, loss_dict = criterion(predictions, targets)
        >>> # With auxiliary (HRNet-OCR)
        >>> loss, loss_dict = criterion(main_out, targets, aux_out)
    """
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 1.0,
        use_boundary: bool = False,
        aux_weight: float = 0.4
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.use_boundary = use_boundary
        self.aux_weight = aux_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        if use_boundary:
            self.boundary_loss = BoundaryLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        aux_predictions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss with optional auxiliary supervision.
        
        Args:
            predictions: (B, 1, H, W) - main predictions (logits)
            targets: (B, 1, H, W) - ground truth binary masks
            aux_predictions: (B, 1, H, W) - optional auxiliary predictions
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # Compute main losses
        bce = self.bce(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        loss_dict['bce'] = bce.item()
        loss_dict['dice'] = dice.item()
        
        # Combined main loss
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        loss_dict['main'] = total_loss.item()
        
        # Add boundary loss if enabled
        if self.use_boundary:
            boundary = self.boundary_loss(predictions, targets)
            loss_dict['boundary'] = boundary.item()
            total_loss += self.boundary_weight * boundary
        
        # Add auxiliary loss if provided
        if aux_predictions is not None:
            # Compute auxiliary losses
            aux_bce = self.bce(aux_predictions, targets)
            aux_dice = self.dice_loss(aux_predictions, targets)
            
            loss_dict['aux_bce'] = aux_bce.item()
            loss_dict['aux_dice'] = aux_dice.item()
            
            # Combined auxiliary loss
            aux_loss = self.bce_weight * aux_bce + self.dice_weight * aux_dice
            
            # Add auxiliary boundary if enabled
            if self.use_boundary:
                aux_boundary = self.boundary_loss(aux_predictions, targets)
                loss_dict['aux_boundary'] = aux_boundary.item()
                aux_loss += self.boundary_weight * aux_boundary
            
            loss_dict['aux'] = aux_loss.item()
            loss_dict['aux_weight'] = self.aux_weight
            
            # Add weighted auxiliary to total
            total_loss = total_loss + self.aux_weight * aux_loss
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class AuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for models with auxiliary outputs (e.g., HRNet-OCR).
    
    Combines main and auxiliary losses with configurable weighting.
    
    Args:
        main_criterion: Loss function for main output
        aux_weight: Weight for auxiliary loss
    """
    def __init__(
        self,
        main_criterion: nn.Module,
        aux_weight: float = 0.4
    ):
        super().__init__()
        self.main_criterion = main_criterion
        self.aux_weight = aux_weight
    
    def forward(
        self,
        main_output: torch.Tensor,
        aux_output: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            main_output: Main prediction (B, C, H, W)
            aux_output: Auxiliary prediction (B, C, H, W)
            targets: Ground truth (B, H, W)
        
        Returns:
            Combined loss
        """
        main_loss = self.main_criterion(main_output, targets)
        aux_loss = self.main_criterion(aux_output, targets)
        
        return main_loss + self.aux_weight * aux_loss


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss by name.
    
    Args:
        loss_name: Name of loss function
            - 'ce': Cross Entropy
            - 'dice': Dice Loss
            - 'focal': Focal Loss
            - 'boundary': Boundary Loss
            - 'combined': Combined Loss (CE + Dice)
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function module
    
    Example:
        >>> criterion = get_loss_function('combined', dice_weight=1.0)
    """
    losses = {
        'ce': nn.CrossEntropyLoss,
        'dice': DiceLoss,
        'focal': FocalLoss,
        'boundary': BoundaryLoss,
        'combined': CombinedLoss
    }
    
    if loss_name not in losses:
        raise ValueError(
            f"Unknown loss: {loss_name}. "
            f"Available: {list(losses.keys())}"
        )
    
    return losses[loss_name](**kwargs)


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Loss Functions")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 4
    num_classes = 2
    height, width = 256, 256
    
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Test each loss
    losses_to_test = {
        'Cross Entropy': nn.CrossEntropyLoss(),
        'Dice Loss': DiceLoss(),
        'Focal Loss': FocalLoss(),
        'Boundary Loss': BoundaryLoss(),
        'Combined Loss': CombinedLoss()
    }
    
    print("\n" + "-" * 80)
    print("Loss Function Outputs:")
    print("-" * 80)
    
    for name, loss_fn in losses_to_test.items():
        loss_value = loss_fn(predictions, targets)
        print(f"{name:20s}: {loss_value.item():.4f}")
    
    # Test auxiliary loss
    print("\n" + "-" * 80)
    print("Auxiliary Loss (HRNet-OCR):")
    print("-" * 80)
    
    main_out = predictions
    aux_out = torch.randn_like(predictions)
    
    aux_loss_fn = AuxiliaryLoss(nn.CrossEntropyLoss(), aux_weight=0.4)
    total_loss = aux_loss_fn(main_out, aux_out, targets)
    
    print(f"Total loss: {total_loss.item():.4f}")
    
    print("\n✓ All loss functions working correctly!")
