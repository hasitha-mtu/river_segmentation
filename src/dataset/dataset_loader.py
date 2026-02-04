import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for loading image segmentation data.
    Supports data augmentation for training.
    """
    
    def __init__(self, data_dir, split='train', image_transform=None, mask_transform=None, 
                 augment=False):
        """
        Args:
            data_dir: Path to processed dataset directory
            split: One of 'train', 'val', or 'test'
            image_transform: Optional transform to apply to images
            mask_transform: Optional transform to apply to masks
            augment: Whether to apply data augmentation (only for training)
        """
        self.data_dir = Path(data_dir) / split
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        print(f"Loaded {len(self.image_files)} samples from {split} set")
        if self.augment:
            print(f"  Data augmentation: ENABLED")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding mask
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        mask = Image.open(mask_path)
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default: convert to tensor
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask
    
    def _augment(self, image, mask):
        """
        Apply data augmentation to image and mask.
        Important: Same augmentation must be applied to both image and mask.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Random brightness and contrast (only for image)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
        
        return image, mask


def get_dataloaders(data_dir, batch_size=8, num_workers=4, augment_train=True):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to processed dataset directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        augment_train: Whether to apply data augmentation to training set
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Define transforms for images
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Define transforms for masks (just convert to tensor)
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = SegmentationDataset(
        data_dir, 
        split='train',
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=augment_train
    )
    
    val_dataset = SegmentationDataset(
        data_dir,
        split='val',
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=False  # No augmentation for validation
    )
    
    test_dataset = SegmentationDataset(
        data_dir,
        split='test',
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=False  # No augmentation for test
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Example: Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="dataset/processed",
        batch_size=8,
        num_workers=4,
        augment_train=True
    )
    
    # Test loading a batch
    print("\nTesting data loading...")
    for images, masks in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask unique values: {torch.unique(masks)}")
        break
