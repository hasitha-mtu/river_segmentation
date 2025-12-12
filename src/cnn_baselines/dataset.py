"""
Dataset loader for River Water Segmentation
Standard RGB image to binary mask segmentation
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob


class RiverSegmentationDataset(Dataset):
    """
    Dataset for river water segmentation (RGB -> Binary Mask)
    
    Expected directory structure:
        data_root/
            images/
                img1.png
                img2.png
                ...
            masks/
                img1.png
                img2.png
                ...
    
    Masks should be binary: 0 (background) and 255 (water)
    """
    
    def __init__(self, 
                 data_root=None,
                 image_paths=None,
                 mask_paths=None,
                 image_dir='images',
                 mask_dir='masks',
                 transform=None,
                 image_size=(512, 512),
                 normalize=True,
                 augment=False):
        """
        Args:
            data_root: Root directory containing images/ and masks/ folders
            image_paths: List of image paths (alternative to data_root)
            mask_paths: List of mask paths (alternative to data_root)
            image_dir: Name of images subdirectory (default: 'images')
            mask_dir: Name of masks subdirectory (default: 'masks')
            transform: Custom Albumentations transform pipeline (optional)
            image_size: Target image size (H, W) - default 512x512
            normalize: Apply ImageNet normalization
            augment: Apply data augmentation (for training)
        """
        
        # Load paths
        if image_paths is not None and mask_paths is not None:
            self.image_paths = sorted(image_paths)
            self.mask_paths = sorted(mask_paths)
        elif data_root is not None:
            img_dir = os.path.join(data_root, image_dir)
            mask_dir_path = os.path.join(data_root, mask_dir)
            
            # Support multiple image formats
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
            self.image_paths = []
            for ext in image_extensions:
                self.image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
            self.image_paths = sorted(self.image_paths)
            
            # Get corresponding mask paths
            self.mask_paths = []
            for img_path in self.image_paths:
                img_name = os.path.basename(img_path)
                # Try same extension first, then .png
                mask_path = os.path.join(mask_dir_path, img_name)
                if not os.path.exists(mask_path):
                    base_name = os.path.splitext(img_name)[0]
                    mask_path = os.path.join(mask_dir_path, base_name + '.png')
                self.mask_paths.append(mask_path)
        else:
            raise ValueError("Either provide data_root or both image_paths and mask_paths")
        
        # Validate paths
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Number of images ({len(self.image_paths)}) != masks ({len(self.mask_paths)})"
        
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        self.image_size = image_size
        self.normalize = normalize
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_transforms(augment)
        
        print(f"Loaded {len(self.image_paths)} image-mask pairs")

    def _get_transforms(self, augment=False):
        """Get default transforms"""
        
        if augment:
            # Training augmentations
            transform_list = [
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                                  rotate_limit=15, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, 
                                              contrast_limit=0.2, p=1),
                    A.HueSaturationValue(hue_shift_limit=20, 
                                        sat_shift_limit=30, 
                                        val_shift_limit=20, p=1),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                ], p=0.3),
            ]
        else:
            # Validation/test transforms - only resize
            transform_list = [
                A.Resize(self.image_size[0], self.image_size[1]),
            ]
        
        # Add normalization if needed
        if self.normalize:
            transform_list.append(
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            )
        
        transform_list.append(ToTensorV2())
        
        return A.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load RGB image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load binary mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Ensure mask has channel dimension and is float
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': image_path,
            'mask_path': mask_path
        }


def get_dataloaders(data_root,
                   batch_size=8,
                   num_workers=4,
                   image_size=(512, 512),
                   train_split=0.8,
                   seed=42):
    """
    Create train and validation dataloaders
    
    Args:
        data_root: Root directory with images/ and masks/ folders
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        image_size: Target image size (H, W) - default 512x512
        train_split: Fraction of data for training (default 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader
    """
    
    # Get all image paths
    img_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'masks')
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    image_paths = sorted(image_paths)
    
    mask_paths = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, img_name)
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(mask_dir, base_name + '.png')
        mask_paths.append(mask_path)
    
    # Split into train/val
    np.random.seed(seed)
    indices = np.random.permutation(len(image_paths))
    split_idx = int(len(indices) * train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_image_paths = [image_paths[i] for i in train_indices]
    train_mask_paths = [mask_paths[i] for i in train_indices]
    val_image_paths = [image_paths[i] for i in val_indices]
    val_mask_paths = [mask_paths[i] for i in val_indices]
    
    # Create datasets
    train_dataset = RiverSegmentationDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        image_size=image_size,
        normalize=True,
        augment=True
    )
    
    val_dataset = RiverSegmentationDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        image_size=image_size,
        normalize=True,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Dataset module for River Water Segmentation")
    print("="*60)
    print("\nExpected directory structure:")
    print("""
    data_root/
    ├── images/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── masks/
        ├── img1.png  (binary: 0=background, 255=water)
        ├── img2.png
        └── ...
    """)
    print("\nUsage:")
    print("""
    from dataset import get_dataloaders
    
    train_loader, val_loader = get_dataloaders(
        data_root='./your_data',
        batch_size=8,
        image_size=(512, 512),
        train_split=0.8
    )
    
    for batch in train_loader:
        images = batch['image']  # (B, 3, 512, 512) RGB images
        masks = batch['mask']    # (B, 1, 512, 512) binary masks
        break
    """)
