"""
Loss Functions for Semantic Segmentation
Includes: Dice Loss, IoU Loss, Boundary Loss, Combined Losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np
from typing import Optional, Tuple


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class IoULoss(nn.Module):
    """IoU (Jaccard) Loss for binary segmentation"""
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for enhanced edge detection
    Based on: "Boundary Loss for Remote Sensing Imagery Semantic Segmentation"
    """
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, 1, H, W) - logits
            targets: (B, 1, H, W) - binary masks
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
            dist_pos = distance_transform_edt(target_np)
            # Distance transform for negative class (non-water)
            dist_neg = distance_transform_edt(1 - target_np)
            
            # Combine distances
            dist_map = np.where(target_np > 0.5, dist_pos, -dist_neg)
            
            # Apply boundary weight function
            weight_map = 1.0 / (1.0 + np.exp((np.abs(dist_map) - self.theta0) / self.theta))
            
            boundary_weights[b, 0] = torch.from_numpy(weight_map).float()
        
        boundary_weights = boundary_weights.to(device)
        
        # Compute weighted BCE
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        boundary_loss = (bce_loss * boundary_weights).mean()
        
        return boundary_loss


class CombinedLoss(nn.Module):
    """Combined loss: BCE + Dice + Boundary"""
    def __init__(self, bce_weight=1.0, dice_weight=1.0, boundary_weight=1.0, 
                 use_boundary=True, aux_weight: float = 0.4):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.use_boundary = use_boundary
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        if use_boundary:
            self.boundary = BoundaryLoss()
        self.aux_weight = aux_weight

    # def forward(self, predictions, targets):
    #     print(f'CombinedLoss|forward|predictions type: {type(predictions)}')
    #     print(f'CombinedLoss|forward|predictions length: {len(predictions)}')
    #     print(f'CombinedLoss|forward|predictions[0]: {predictions[0]}')
    #     print(f'CombinedLoss|forward|predictions[1]: {predictions[1]}')
    #     print(f'CombinedLoss|forward|targets type: {type(targets)}')
    #     bce_loss = self.bce(predictions, targets)
    #     dice_loss = self.dice(predictions, targets)
        
    #     total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
    #     if self.use_boundary:
    #         boundary_loss = self.boundary(predictions, targets)
    #         total_loss += self.boundary_weight * boundary_loss
    #         return total_loss, {
    #             'bce': bce_loss.item(),
    #             'dice': dice_loss.item(),
    #             'boundary': boundary_loss.item(),
    #             'total': total_loss.item()
    #         }
    #     else:
    #         return total_loss, {
    #             'bce': bce_loss.item(),
    #             'dice': dice_loss.item(),
    #             'total': total_loss.item()
    #         }
    

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
        dice = self.dice(predictions, targets)
        
        loss_dict['bce'] = bce.item()
        loss_dict['dice'] = dice.item()
        
        # Combined main loss
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        loss_dict['main'] = total_loss.item()
        
        # Add boundary loss if enabled
        if self.use_boundary:
            boundary = self.boundary(predictions, targets)
            loss_dict['boundary'] = boundary.item()
            total_loss += self.boundary_weight * boundary
        
        # Add auxiliary loss if provided
        if aux_predictions is not None:
            # Compute auxiliary losses
            aux_bce = self.bce(aux_predictions, targets)
            aux_dice = self.dice(aux_predictions, targets)
            
            loss_dict['aux_bce'] = aux_bce.item()
            loss_dict['aux_dice'] = aux_dice.item()
            
            # Combined auxiliary loss
            aux_loss = self.bce_weight * aux_bce + self.dice_weight * aux_dice
            
            # Add auxiliary boundary if enabled
            if self.use_boundary:
                aux_boundary = self.boundary(aux_predictions, targets)
                loss_dict['aux_boundary'] = aux_boundary.item()
                aux_loss += self.boundary_weight * aux_boundary
            
            loss_dict['aux'] = aux_loss.item()
            loss_dict['aux_weight'] = self.aux_weight
            
            # Add weighted auxiliary to total
            total_loss = total_loss + self.aux_weight * aux_loss
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict



class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with adjustable false positive/negative weights
    Useful for handling class imbalance
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (predictions * targets).sum()
        FP = ((1 - targets) * predictions).sum()
        FN = (targets * (1 - predictions)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class ComboLoss(nn.Module):
    """
    Combo Loss = Dice Loss + Weighted BCE
    Reference: https://arxiv.org/abs/1805.02798
    """
    def __init__(self, alpha=0.5, ce_ratio=0.5, smooth=1.0):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions_sigmoid = torch.sigmoid(predictions)
        
        # Flatten
        predictions_flat = predictions_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        # Dice component
        intersection = (predictions_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (predictions_flat.sum() + targets_flat.sum() + self.smooth)
        
        # Weighted BCE component
        bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction='mean')
        
        # Combo loss
        combo = (self.ce_ratio * bce) - ((1 - self.ce_ratio) * torch.log(dice))
        
        return combo


def get_loss_function(loss_name='combined', **kwargs):
    """
    Factory function to get loss by name
    
    Args:
        loss_name: One of ['bce', 'dice', 'iou', 'focal', 'boundary', 'combined', 
                          'tversky', 'combo']
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function
    """
    losses = {
        'bce': nn.BCEWithLogitsLoss,
        'dice': DiceLoss,
        'iou': IoULoss,
        'focal': FocalLoss,
        'boundary': BoundaryLoss,
        'combined': CombinedLoss,
        'tversky': TverskyLoss,
        'combo': ComboLoss
    }
    
    if loss_name.lower() not in losses:
        raise ValueError(f"Loss {loss_name} not found. Available: {list(losses.keys())}")
    
    return losses[loss_name.lower()](**kwargs)


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    print("-" * 60)
    
    # Create dummy data
    batch_size, height, width = 4, 128, 128
    predictions = torch.randn(batch_size, 1, height, width)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test each loss
    loss_configs = {
        'BCE': ('bce', {}),
        'Dice': ('dice', {}),
        'IoU': ('iou', {}),
        'Focal': ('focal', {'alpha': 0.25, 'gamma': 2.0}),
        'Boundary': ('boundary', {}),
        'Combined': ('combined', {'use_boundary': True}),
        'Tversky': ('tversky', {'alpha': 0.5, 'beta': 0.5}),
        'Combo': ('combo', {})
    }
    
    for name, (loss_type, params) in loss_configs.items():
        loss_fn = get_loss_function(loss_type, **params)
        
        if loss_type == 'combined':
            loss_value, loss_dict = loss_fn(predictions, targets)
            print(f"\n{name} Loss:")
            for k, v in loss_dict.items():
                print(f"  {k}: {v:.4f}")
        else:
            loss_value = loss_fn(predictions, targets)
            print(f"\n{name} Loss: {loss_value.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All loss functions tested successfully!")
