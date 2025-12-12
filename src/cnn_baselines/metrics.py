"""
Evaluation Metrics for Semantic Segmentation
Implements: Dice, IoU, Precision, Recall, F1, Accuracy
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """Compute common segmentation metrics"""
    
    def __init__(self, threshold=0.5, eps=1e-7):
        """
        Args:
            threshold: Threshold for binary classification
            eps: Small epsilon for numerical stability
        """
        self.threshold = threshold
        self.eps = eps

    def dice_coefficient(self, predictions, targets):
        """
        Compute Dice coefficient (F1 score)
        
        Args:
            predictions: Binary predictions (B, 1, H, W)
            targets: Binary targets (B, 1, H, W)
        
        Returns:
            Dice coefficient
        """
        predictions = predictions.float()
        targets = targets.float()
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return dice.item()

    def iou_score(self, predictions, targets):
        """
        Compute Intersection over Union (Jaccard Index)
        
        Args:
            predictions: Binary predictions (B, 1, H, W)
            targets: Binary targets (B, 1, H, W)
        
        Returns:
            IoU score
        """
        predictions = predictions.float()
        targets = targets.float()
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        iou = (intersection + self.eps) / (union + self.eps)
        return iou.item()

    def pixel_accuracy(self, predictions, targets):
        """
        Compute pixel-wise accuracy
        
        Args:
            predictions: Binary predictions (B, 1, H, W)
            targets: Binary targets (B, 1, H, W)
        
        Returns:
            Pixel accuracy
        """
        correct = (predictions == targets).sum()
        total = targets.numel()
        
        accuracy = correct.float() / total
        return accuracy.item()

    def precision_recall(self, predictions, targets):
        """
        Compute precision and recall
        
        Args:
            predictions: Binary predictions (B, 1, H, W)
            targets: Binary targets (B, 1, H, W)
        
        Returns:
            precision, recall
        """
        predictions = predictions.float()
        targets = targets.float()
        
        tp = (predictions * targets).sum()
        fp = (predictions * (1 - targets)).sum()
        fn = ((1 - predictions) * targets).sum()
        
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        
        return precision.item(), recall.item()

    def f1_score(self, precision, recall):
        """
        Compute F1 score from precision and recall
        
        Args:
            precision: Precision value
            recall: Recall value
        
        Returns:
            F1 score
        """
        f1 = (2 * precision * recall + self.eps) / (precision + recall + self.eps)
        return f1

    def specificity(self, predictions, targets):
        """
        Compute specificity (true negative rate)
        
        Args:
            predictions: Binary predictions (B, 1, H, W)
            targets: Binary targets (B, 1, H, W)
        
        Returns:
            Specificity
        """
        predictions = predictions.float()
        targets = targets.float()
        
        tn = ((1 - predictions) * (1 - targets)).sum()
        fp = (predictions * (1 - targets)).sum()
        
        specificity = (tn + self.eps) / (tn + fp + self.eps)
        return specificity.item()

    def confusion_matrix_stats(self, predictions, targets):
        """
        Compute confusion matrix statistics
        
        Args:
            predictions: Binary predictions (B, 1, H, W)
            targets: Binary targets (B, 1, H, W)
        
        Returns:
            Dictionary with TP, TN, FP, FN
        """
        predictions = predictions.float()
        targets = targets.float()
        
        tp = (predictions * targets).sum().item()
        tn = ((1 - predictions) * (1 - targets)).sum().item()
        fp = (predictions * (1 - targets)).sum().item()
        fn = ((1 - predictions) * targets).sum().item()
        
        return {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }

    def compute_metrics(self, predictions, targets):
        """
        Compute all metrics
        
        Args:
            predictions: Binary predictions (B, 1, H, W)
            targets: Binary targets (B, 1, H, W)
        
        Returns:
            Dictionary with all metrics
        """
        dice = self.dice_coefficient(predictions, targets)
        iou = self.iou_score(predictions, targets)
        accuracy = self.pixel_accuracy(predictions, targets)
        precision, recall = self.precision_recall(predictions, targets)
        f1 = self.f1_score(precision, recall)
        specificity = self.specificity(predictions, targets)
        cm_stats = self.confusion_matrix_stats(predictions, targets)
        
        return {
            'dice': dice,
            'iou': iou,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            **cm_stats
        }


class StreamingMetrics:
    """Compute metrics incrementally over batches"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all counters"""
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, predictions, targets):
        """
        Update metrics with new batch
        
        Args:
            predictions: Predictions (B, 1, H, W) - can be logits or probabilities
            targets: Binary targets (B, 1, H, W)
        """
        # Apply sigmoid if predictions are logits
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Binarize predictions
        predictions = (predictions > self.threshold).float()
        targets = targets.float()
        
        # Update confusion matrix
        self.tp += (predictions * targets).sum().item()
        self.tn += ((1 - predictions) * (1 - targets)).sum().item()
        self.fp += (predictions * (1 - targets)).sum().item()
        self.fn += ((1 - predictions) * targets).sum().item()

    def get_metrics(self, eps=1e-7):
        """
        Compute final metrics
        
        Returns:
            Dictionary with all metrics
        """
        # Compute metrics
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        specificity = self.tn / (self.tn + self.fp + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        
        f1 = 2 * precision * recall / (precision + recall + eps)
        dice = f1  # Dice is equivalent to F1 for binary segmentation
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        
        return {
            'dice': dice,
            'iou': iou,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'TP': int(self.tp),
            'TN': int(self.tn),
            'FP': int(self.fp),
            'FN': int(self.fn)
        }


class BoundaryMetrics:
    """Metrics for boundary quality assessment"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def boundary_iou(self, predictions, targets, dilation=5):
        """
        Compute IoU on boundary regions
        
        Args:
            predictions: Binary predictions (B, 1, H, W)
            targets: Binary targets (B, 1, H, W)
            dilation: Dilation size for boundary extraction
        
        Returns:
            Boundary IoU
        """
        import torch.nn.functional as F
        
        # Extract boundaries using morphological operations
        kernel = torch.ones(1, 1, dilation, dilation).to(predictions.device)
        
        # Dilate
        pred_dilated = F.conv2d(predictions.float(), kernel, padding=dilation//2) > 0
        target_dilated = F.conv2d(targets.float(), kernel, padding=dilation//2) > 0
        
        # Erode
        pred_eroded = F.conv2d(predictions.float(), kernel, padding=dilation//2) == (dilation * dilation)
        target_eroded = F.conv2d(targets.float(), kernel, padding=dilation//2) == (dilation * dilation)
        
        # Boundaries = dilated - eroded
        pred_boundary = (pred_dilated.float() - pred_eroded.float()) > 0
        target_boundary = (target_dilated.float() - target_eroded.float()) > 0
        
        # Compute IoU on boundaries
        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum() - intersection
        
        boundary_iou = (intersection + 1e-7) / (union + 1e-7)
        return boundary_iou.item()


def print_metrics(metrics, prefix=''):
    """Pretty print metrics"""
    print(f"\n{prefix}Metrics:")
    print("-" * 60)
    print(f"Dice Coefficient: {metrics['dice']:.4f}")
    print(f"IoU Score:        {metrics['iou']:.4f}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print(f"F1 Score:         {metrics['f1']:.4f}")
    print(f"Specificity:      {metrics['specificity']:.4f}")
    print("\nConfusion Matrix:")
    print(f"TP: {metrics['TP']:,} | FP: {metrics['FP']:,}")
    print(f"FN: {metrics['FN']:,} | TN: {metrics['TN']:,}")
    print("-" * 60)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    height, width = 128, 128
    
    predictions = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test SegmentationMetrics
    print("\n1. Testing SegmentationMetrics:")
    metrics_calc = SegmentationMetrics(threshold=0.5)
    metrics = metrics_calc.compute_metrics(predictions, targets)
    print_metrics(metrics)
    
    # Test StreamingMetrics
    print("\n2. Testing StreamingMetrics:")
    streaming_metrics = StreamingMetrics(threshold=0.5)
    
    # Update with multiple batches
    for i in range(3):
        batch_pred = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        batch_target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        streaming_metrics.update(batch_pred, batch_target)
    
    final_metrics = streaming_metrics.get_metrics()
    print_metrics(final_metrics, prefix='Streaming ')
    
    # Test BoundaryMetrics
    print("\n3. Testing BoundaryMetrics:")
    boundary_calc = BoundaryMetrics(threshold=0.5)
    boundary_iou = boundary_calc.boundary_iou(predictions, targets, dilation=3)
    print(f"Boundary IoU: {boundary_iou:.4f}")
    
    print("\n" + "=" * 60)
    print("All metrics tested successfully!")
