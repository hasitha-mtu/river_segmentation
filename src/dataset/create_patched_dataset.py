import os
import re
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
from datetime import datetime
import json
import shutil


def parse_filename_date(filename):
    """
    Extract date from DJI filename format: DJI_YYYYMMDDHHMMSS_####_V.jpg
    Returns datetime object or None
    """
    match = re.search(r'DJI_(\d{4})(\d{2})(\d{2})\d{6}_\d{4}', filename)
    if match:
        year, month, day = match.groups()
        return datetime(int(year), int(month), int(day))
    return None


def get_gps_data(image_path):
    """
    Extract GPS coordinates from image EXIF data.
    Returns (latitude, longitude) or None
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if not exif_data:
            return None
        
        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                for gps_tag in value:
                    gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_tag_name] = value[gps_tag]
        
        if not gps_info:
            return None
        
        # Convert GPS coordinates to decimal degrees
        def convert_to_degrees(value):
            d, m, s = value
            return float(d) + float(m) / 60.0 + float(s) / 3600.0
        
        lat = convert_to_degrees(gps_info.get('GPSLatitude', [0, 0, 0]))
        lon = convert_to_degrees(gps_info.get('GPSLongitude', [0, 0, 0]))
        
        # Handle N/S and E/W
        if gps_info.get('GPSLatitudeRef') == 'S':
            lat = -lat
        if gps_info.get('GPSLongitudeRef') == 'W':
            lon = -lon
        
        return (lat, lon)
    
    except Exception as e:
        print(f"Error reading GPS from {image_path.name}: {e}")
        return None


def calculate_distance(coord1, coord2):
    """
    Calculate approximate distance between two GPS coordinates in meters.
    Using simple euclidean distance (good enough for small areas).
    """
    if coord1 is None or coord2 is None:
        return float('inf')
    
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Approximate conversion (1 degree ≈ 111km at equator)
    lat_diff = (lat2 - lat1) * 111000
    lon_diff = (lon2 - lon1) * 111000 * np.cos(np.radians(lat1))
    
    return np.sqrt(lat_diff**2 + lon_diff**2)


def analyze_dataset(raw_path="dataset/raw"):
    """
    Analyze the dataset: parse dates, extract GPS, calculate overlaps.
    Returns structured metadata about all images.
    """
    print("="*70)
    print("ANALYZING DATASET")
    print("="*70)
    
    images_dir = Path(raw_path) / "images"
    masks_dir = Path(raw_path) / "masks"
    
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return None
    
    # Collect all valid image-mask pairs with metadata
    image_metadata = []
    
    print("\nScanning images and extracting metadata...")
    image_files = sorted(list(images_dir.glob("*.jpg")))
    
    for i, img_path in enumerate(image_files):
        mask_path = masks_dir / f"{img_path.stem}.png"
        
        if not mask_path.exists():
            continue
        
        # Parse date from filename
        date = parse_filename_date(img_path.name)
        
        # Extract GPS
        gps = get_gps_data(img_path)
        
        # Extract sequence number
        seq_match = re.search(r'_(\d{4})_V\.jpg', img_path.name)
        sequence = int(seq_match.group(1)) if seq_match else None
        
        image_metadata.append({
            'filename': img_path.name,
            'path': str(img_path),
            'mask_path': str(mask_path),
            'date': date,
            'gps': gps,
            'sequence': sequence
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images...")
    
    print(f"\nFound {len(image_metadata)} valid image-mask pairs")
    
    # Group by date
    date_groups = {}
    for item in image_metadata:
        date_str = item['date'].strftime('%Y/%m/%d') if item['date'] else 'Unknown'
        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(item)
    
    print(f"\n{'='*70}")
    print("DATE DISTRIBUTION:")
    print(f"{'='*70}")
    for date_str, items in sorted(date_groups.items()):
        print(f"  {date_str}: {len(items)} images")
    
    # Calculate spatial statistics
    print(f"\n{'='*70}")
    print("SPATIAL ANALYSIS:")
    print(f"{'='*70}")
    
    for date_str, items in sorted(date_groups.items()):
        # Sort by sequence number or GPS
        items_sorted = sorted(items, key=lambda x: x['sequence'] if x['sequence'] else 0)
        
        # Calculate distances between consecutive images
        distances = []
        for i in range(len(items_sorted) - 1):
            if items_sorted[i]['gps'] and items_sorted[i+1]['gps']:
                dist = calculate_distance(items_sorted[i]['gps'], items_sorted[i+1]['gps'])
                distances.append(dist)
        
        if distances:
            print(f"\n  {date_str}:")
            print(f"    Average distance between consecutive images: {np.mean(distances):.2f}m")
            print(f"    Min distance: {np.min(distances):.2f}m")
            print(f"    Max distance: {np.max(distances):.2f}m")
            print(f"    Total flight path: {np.sum(distances):.2f}m ({np.sum(distances)/1000:.2f}km)")
    
    return image_metadata, date_groups


def create_splits_with_buffers(date_groups, buffer_size=10, 
                               train_ratio=0.6, val_ratio=0.18, test_ratio=0.22):
    """
    Create train/val/test splits with buffer zones to prevent data leakage.
    
    CRITICAL STRATEGY for seasonal variation:
    - Split EACH date separately with buffer zones
    - Combine splits across dates so each split has both seasons
    - This prevents both spatial leakage AND seasonal domain shift
    
    Example with 2 dates (winter/summer):
      Date 1: [train] [buffer] [val] [buffer] [test]
      Date 2: [train] [buffer] [val] [buffer] [test]
      Final: train=D1_train+D2_train (both seasons in training!)
    
    Args:
        date_groups: Dictionary of {date_str: [image_metadata]}
        buffer_size: Number of images to skip as buffer (default: 10)
        train_ratio: Target proportion for training (default: 0.6)
        val_ratio: Target proportion for validation (default: 0.18)
        test_ratio: Target proportion for testing (default: 0.22)
        Note: Actual ratios will be lower due to buffer zones
    
    Returns:
        splits: Dictionary with train, val, test lists
    """
    print(f"\n{'='*70}")
    print("CREATING SPLITS WITH BUFFER ZONES")
    print(f"STRATEGY: Mixed seasonal split to prevent domain shift")
    print(f"{'='*70}")
    
    dates = sorted(date_groups.keys())
    
    # Initialize combined splits
    combined_splits = {
        'train': [],
        'val': [],
        'test': [],
        'buffers': []
    }
    
    # Process each date separately
    for date_str in dates:
        images = sorted(date_groups[date_str], key=lambda x: x['sequence'] if x['sequence'] else 0)
        n = len(images)
        
        print(f"\nProcessing {date_str}: {n} images")
        
        # Calculate split points with buffers
        # train | buffer | val | buffer | test
        train_end = int(n * train_ratio)
        buffer1_end = min(train_end + buffer_size, n)
        val_end = min(buffer1_end + int(n * val_ratio), n)
        buffer2_end = min(val_end + buffer_size, n)
        
        # Create splits for this date
        date_train = images[:train_end]
        date_buffer1 = images[train_end:buffer1_end]
        date_val = images[buffer1_end:val_end]
        date_buffer2 = images[val_end:buffer2_end]
        date_test = images[buffer2_end:]
        
        # Add to combined splits
        combined_splits['train'].extend(date_train)
        combined_splits['val'].extend(date_val)
        combined_splits['test'].extend(date_test)
        combined_splits['buffers'].extend(date_buffer1)
        combined_splits['buffers'].extend(date_buffer2)
        
        print(f"  Train:      {len(date_train):3d} images")
        print(f"  Buffer 1:   {len(date_buffer1):3d} images (DISCARDED)")
        print(f"  Val:        {len(date_val):3d} images")
        print(f"  Buffer 2:   {len(date_buffer2):3d} images (DISCARDED)")
        print(f"  Test:       {len(date_test):3d} images")
    
    # Print combined statistics
    print(f"\n{'='*70}")
    print(f"COMBINED SPLITS (Mixed Seasonal):")
    print(f"{'='*70}")
    
    total_used = len(combined_splits['train']) + len(combined_splits['val']) + len(combined_splits['test'])
    total_buffer = len(combined_splits['buffers'])
    total = total_used + total_buffer
    
    print(f"  Train:      {len(combined_splits['train']):4d} images ({len(combined_splits['train'])/total_used*100:5.1f}%)")
    print(f"  Validation: {len(combined_splits['val']):4d} images ({len(combined_splits['val'])/total_used*100:5.1f}%)")
    print(f"  Test:       {len(combined_splits['test']):4d} images ({len(combined_splits['test'])/total_used*100:5.1f}%)")
    print(f"  Buffers:    {total_buffer:4d} images (DISCARDED)")
    print(f"\n  Total images used: {total_used}/{total} ({total_used/total*100:.1f}%)")
    
    # Print seasonal distribution
    print(f"\n{'='*70}")
    print(f"SEASONAL DISTRIBUTION (prevents domain shift):")
    print(f"{'='*70}")
    
    for split_name in ['train', 'val', 'test']:
        split_images = combined_splits[split_name]
        date_counts = {}
        for img in split_images:
            date_str = img['date'].strftime('%Y/%m/%d') if img['date'] else 'Unknown'
            date_counts[date_str] = date_counts.get(date_str, 0) + 1
        
        print(f"\n  {split_name.capitalize()}:")
        for date_str in sorted(date_counts.keys()):
            print(f"    {date_str}: {date_counts[date_str]:3d} images")
    
    return {
        'train': combined_splits['train'],
        'val': combined_splits['val'],
        'test': combined_splits['test']
    }


def extract_patches(image_path, mask_path, patch_size=512, stride=512):
    """
    Extract patches from image and mask.
    
    Args:
        image_path: Path to image
        mask_path: Path to mask
        patch_size: Size of patches (default: 512)
        stride: Stride for patch extraction (default: 512, no overlap)
    
    Returns:
        List of (image_patch, mask_patch, position) tuples
    """
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    h, w = img_array.shape[:2]
    patches = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = img_array[y:y+patch_size, x:x+patch_size]
            mask_patch = mask_array[y:y+patch_size, x:x+patch_size]
            
            # Check if patch has sufficient content (not just edges/empty)
            # You can adjust this threshold based on your data
            if mask_patch.sum() > 0:  # At least some mask content
                patches.append({
                    'image': Image.fromarray(img_patch),
                    'mask': Image.fromarray(mask_patch),
                    'position': (x, y)
                })
    
    return patches


def create_patched_dataset(
    raw_path="dataset/raw",
    output_path="dataset/processed",
    patch_size=512,
    stride=512,
    buffer_size=10
):
    """
    Main function to create patched dataset with proper splits.
    """
    print(f"\n{'='*70}")
    print("DATASET PREPARATION PIPELINE")
    print(f"{'='*70}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Stride: {stride}")
    print(f"Buffer size: {buffer_size} images")
    
    # Step 1: Analyze dataset
    result = analyze_dataset(raw_path)
    if result is None:
        return
    
    image_metadata, date_groups = result
    
    # Step 2: Create splits
    splits = create_splits_with_buffers(date_groups, buffer_size=buffer_size)
    
    # Step 3: Create output directories
    output_path = Path(output_path)
    for split_name in ['train', 'val', 'test']:
        (output_path / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split_name / "masks").mkdir(parents=True, exist_ok=True)
    
    # Step 4: Extract patches for each split
    print(f"\n{'='*70}")
    print("EXTRACTING PATCHES")
    print(f"{'='*70}")
    
    split_stats = {}
    
    for split_name in ['train', 'val', 'test']:
        print(f"\nProcessing {split_name} set...")
        images = splits[split_name]
        
        total_patches = 0
        patch_counter = 0
        
        for i, img_meta in enumerate(images):
            # Extract patches
            patches = extract_patches(
                img_meta['path'],
                img_meta['mask_path'],
                patch_size=patch_size,
                stride=stride
            )
            
            # Save patches
            for patch in patches:
                patch_filename = f"{Path(img_meta['filename']).stem}_patch_{patch_counter:04d}"
                
                # Save image patch
                img_out_path = output_path / split_name / "images" / f"{patch_filename}.jpg"
                patch['image'].save(img_out_path, "JPEG", quality=95)
                
                # Save mask patch
                mask_out_path = output_path / split_name / "masks" / f"{patch_filename}.png"
                patch['mask'].save(mask_out_path, "PNG")
                
                patch_counter += 1
            
            total_patches += len(patches)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(images)} images, {total_patches} patches so far...")
        
        split_stats[split_name] = {
            'images': len(images),
            'patches': total_patches
        }
        
        print(f"  Completed {split_name}: {len(images)} images → {total_patches} patches")
    
    # Step 5: Save metadata
    metadata = {
        'patch_size': patch_size,
        'stride': stride,
        'buffer_size': buffer_size,
        'splits': split_stats,
        'date_distribution': {date: len(items) for date, items in date_groups.items()}
    }
    
    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print("DATASET CREATION COMPLETED!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    print(f"\nFinal Statistics:")
    for split_name in ['train', 'val', 'test']:
        stats = split_stats[split_name]
        print(f"  {split_name.capitalize():12s}: {stats['images']:3d} images → {stats['patches']:4d} patches")
    
    total_patches = sum(s['patches'] for s in split_stats.values())
    print(f"  {'Total':12s}: {total_patches:4d} patches")
    
    return metadata


if __name__ == "__main__":
    """
    Configuration presets for different patch sizes:
    
    512x512 patches:
      - stride=512 (no overlap): ~70 patches/image, ~6,300 total patches
      - stride=256 (50% overlap): ~280 patches/image, ~25,000 total patches
    
    1024x1024 patches (RECOMMENDED for high-res details):
      - stride=1024 (no overlap): ~15 patches/image, ~6,300 total patches
      - stride=768 (25% overlap): ~24 patches/image, ~10,000 total patches ⭐
      - stride=640 (37.5% overlap): ~35 patches/image, ~14,700 total patches
      - stride=512 (50% overlap): ~54 patches/image, ~22,700 total patches
    
    For river segmentation under tree canopy, RECOMMENDED:
      patch_size=1024, stride=768 (25% overlap)
    
    Rationale:
      - Preserves fine details better than 512x512
      - 25% overlap ensures boundary features are well-captured
      - Balanced dataset size (~10k patches)
      - Reasonable training time
    """
    
    # OPTION 1: 512x512 patches (original)
    create_patched_dataset(
        raw_path="dataset/raw",
        output_path="dataset/processed_512",
        patch_size=512,
        stride=512,
        buffer_size=10
    )
    
    # OPTION 2: 1024x1024 patches with 25% overlap (RECOMMENDED)
    # create_patched_dataset(
    #     raw_path="dataset/raw",
    #     output_path="dataset/processed_1024",
    #     patch_size=1024,
    #     stride=768,  # 25% overlap - good balance
    #     buffer_size=10
    # )
    
    # OPTION 3: 1024x1024 patches with no overlap (faster training)
    # create_patched_dataset(
    #     raw_path="dataset/raw",
    #     output_path="dataset/processed_1024_no_overlap",
    #     patch_size=1024,
    #     stride=1024,
    #     buffer_size=10
    # )
    
    # OPTION 4: 1024x1024 patches with 50% overlap (maximum data)
    # create_patched_dataset(
    #     raw_path="dataset/raw",
    #     output_path="dataset/processed_1024_50overlap",
    #     patch_size=1024,
    #     stride=512,
    #     buffer_size=10
    # )
