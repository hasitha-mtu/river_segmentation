import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


def mask_to_tensor(mask, image_size=None):
    """
    Convert PIL mask to tensor with shape (1, H, W).
    Module-level function to support multiprocessing on Windows.
    
    Args:
        mask: PIL Image in grayscale mode
        image_size: Optional tuple (H, W) to resize mask
        
    Returns:
        torch.Tensor: Float tensor with shape (1, H, W) and values in [0, 1]
    """
    # Resize if needed
    if image_size is not None:
        mask = mask.resize((image_size, image_size), Image.NEAREST)
    
    mask_np = np.array(mask, dtype=np.float32)
    # Normalize to [0, 1] if needed
    if mask_np.max() > 1.0:
        mask_np = mask_np / 255.0
    # Convert to tensor and add channel dimension
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (H, W) -> (1, H, W)
    return mask_tensor


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for loading image segmentation data.
    Supports data augmentation for training.
    """
    
    def __init__(self, data_dir, split='train', image_transform=None, mask_transform=None, 
                 augment=False, image_size=None):
        """
        Args:
            data_dir: Path to processed dataset directory
            split: One of 'train', 'val', or 'test'
            image_transform: Optional transform to apply to images
            mask_transform: Optional transform to apply to masks
            augment: Whether to apply data augmentation (only for training)
            image_size: Size to resize images (single int for square images)
        """
        self.data_dir = Path(data_dir) / split
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.split = split
        self.augment = augment and (split == 'train')
        self.image_size = image_size
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        print(f"Loaded {len(self.image_files)} samples from {split} set")
        if self.image_size:
            print(f"  Image size: {self.image_size}x{self.image_size}")
        if self.augment:
            print(f"  Data augmentation: ENABLED")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding mask (convert to grayscale to ensure single channel)
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        mask = Image.open(mask_path).convert('L')
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            # Pass image_size to mask_transform if it's a callable
            if self.image_size is not None and callable(self.mask_transform):
                try:
                    mask = self.mask_transform(mask, self.image_size)
                except TypeError:
                    mask = self.mask_transform(mask)
            else:
                mask = self.mask_transform(mask)
        else:
            # Default: convert to tensor
            if self.image_size is not None:
                mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()
        
        # Ensure mask has the correct shape: (1, H, W) for binary segmentation
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension (H, W) -> (1, H, W)
        elif mask.dim() == 3 and mask.shape[0] == 3:
            # If mask somehow has 3 channels, take only the first channel
            mask = mask[0:1, :, :]
        
        # Ensure mask values are in [0, 1] range
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        return {'image': image, 'mask': mask}
    
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


def get_training_dataloaders(data_dir, batch_size=8, num_workers=4, augment_train=True, image_size=1024):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to processed dataset directory (containing train/val/test folders)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        augment_train: Whether to apply data augmentation to training set
        image_size: Size to resize images (single int for square images)
        
    Returns:
        train_loader, val_loader
    """
    return (
        get_dataloaders(data_dir, batch_size=batch_size, num_workers=num_workers, 
                       augment_train=augment_train, data_loader_type='train', image_size=image_size), 
        get_dataloaders(data_dir, batch_size=batch_size, num_workers=num_workers, 
                       augment_train=augment_train, data_loader_type='val', image_size=image_size)
    )


def get_dataloaders(data_dir, batch_size=8, num_workers=4, augment_train=True, data_loader_type='train', image_size=1024):
    """
    Create train, validation, or test dataloader.
    
    Args:
        data_dir: Path to processed dataset directory (containing train/val/test folders)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        augment_train: Whether to apply data augmentation to training set
        data_loader_type: Type of dataloader ('train', 'val', or 'test')
        image_size: Size to resize images (single int for square images)
        
    Returns:
        DataLoader for specified split
    """
    
    # Define transforms for images
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to specified size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Define transforms for masks (use module-level mask_to_tensor for Windows multiprocessing compatibility)
    mask_transform = mask_to_tensor

    if data_loader_type == 'train':
        train_dataset = SegmentationDataset(
            data_dir, 
            split='train',
            image_transform=image_transform,
            mask_transform=mask_transform,
            augment=augment_train,
            image_size=image_size
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        return train_loader

    elif data_loader_type == 'val':
        val_dataset = SegmentationDataset(
            data_dir,
            split='val',
            image_transform=image_transform,
            mask_transform=mask_transform,
            augment=False,  # No augmentation for validation
            image_size=image_size
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        return val_loader
    
    elif data_loader_type == 'test':
        test_dataset = SegmentationDataset(
            data_dir,
            split='test',
            image_transform=image_transform,
            mask_transform=mask_transform,
            augment=False,  # No augmentation for test
            image_size=image_size
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        return test_loader
    
    else:
        return None


# Example usage
if __name__ == "__main__":
    # Example 1: Default 1024x1024 images
    print("="*80)
    print("Testing with 1024x1024 images")
    print("="*80)
    train_loader_1024, val_loader_1024 = get_training_dataloaders(
        data_dir="dataset/processed_1024",
        batch_size=4,
        num_workers=4,
        augment_train=True,
        image_size=1024
    )
    
    # Test loading a batch
    print("\nTesting data loading...")
    for batch in train_loader_1024:
        images = batch['image']
        masks = batch['mask']
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask unique values: {torch.unique(masks)}")
        break
    
    # Example 2: Resized 512x512 images
    print("\n" + "="*80)
    print("Testing with 512x512 images")
    print("="*80)
    train_loader_512, val_loader_512 = get_training_dataloaders(
        data_dir="dataset/processed_1024",
        batch_size=8,
        num_workers=4,
        augment_train=True,
        image_size=512
    )
    
    # Test loading a batch
    print("\nTesting data loading...")
    for batch in train_loader_512:
        images = batch['image']
        masks = batch['mask']
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask unique values: {torch.unique(masks)}")
        break
