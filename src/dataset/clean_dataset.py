import os
from pathlib import Path

def clean_dataset(dataset_path="dataset/raw"):
    """
    Remove images that don't have corresponding masks.
    
    Args:
        dataset_path: Path to the raw dataset folder containing 'images' and 'masks' subfolders
    """
    images_dir = Path(dataset_path) / "images"
    masks_dir = Path(dataset_path) / "masks"
    
    # Check if directories exist
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return
    
    if not masks_dir.exists():
        print(f"Error: Masks directory not found at {masks_dir}")
        return
    
    # Get all image files (.jpg)
    image_files = list(images_dir.glob("*.jpg"))
    
    print(f"Found {len(image_files)} images in total")
    
    removed_count = 0
    kept_count = 0
    
    # Check each image for corresponding mask
    for image_path in image_files:
        # Get the base name without extension
        base_name = image_path.stem
        
        # Check if corresponding mask exists (.png)
        mask_path = masks_dir / f"{base_name}.png"
        
        if not mask_path.exists():
            print(f"Removing: {image_path.name} (no corresponding mask found)")
            image_path.unlink()  # Delete the image file
            removed_count += 1
        else:
            kept_count += 1
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Images kept: {kept_count}")
    print(f"  Images removed: {removed_count}")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Run the cleanup
    clean_dataset()
