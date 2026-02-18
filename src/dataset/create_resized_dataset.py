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

        def convert_to_degrees(value):
            d, m, s = value
            return float(d) + float(m) / 60.0 + float(s) / 3600.0

        lat = convert_to_degrees(gps_info.get('GPSLatitude', [0, 0, 0]))
        lon = convert_to_degrees(gps_info.get('GPSLongitude', [0, 0, 0]))

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

    lat_diff = (lat2 - lat1) * 111000
    lon_diff = (lon2 - lon1) * 111000 * np.cos(np.radians(lat1))

    return np.sqrt(lat_diff**2 + lon_diff**2)


def analyze_dataset(raw_path="dataset/raw"):
    """
    Analyze the dataset: parse dates, extract GPS, calculate overlaps.
    Returns structured metadata about all images.
    """
    print("=" * 70)
    print("ANALYZING DATASET")
    print("=" * 70)

    images_dir = Path(raw_path) / "images"
    masks_dir = Path(raw_path) / "masks"

    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return None

    image_metadata = []

    print("\nScanning images and extracting metadata...")
    image_files = sorted(list(images_dir.glob("*.jpg")))

    for i, img_path in enumerate(image_files):
        mask_path = masks_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            continue

        date = parse_filename_date(img_path.name)
        gps = get_gps_data(img_path)

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

    print(f"\n{'='*70}")
    print("SPATIAL ANALYSIS:")
    print(f"{'='*70}")

    for date_str, items in sorted(date_groups.items()):
        items_sorted = sorted(items, key=lambda x: x['sequence'] if x['sequence'] else 0)

        distances = []
        for i in range(len(items_sorted) - 1):
            if items_sorted[i]['gps'] and items_sorted[i + 1]['gps']:
                dist = calculate_distance(items_sorted[i]['gps'], items_sorted[i + 1]['gps'])
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
    """
    print(f"\n{'='*70}")
    print("CREATING SPLITS WITH BUFFER ZONES")
    print(f"STRATEGY: Mixed seasonal split to prevent domain shift")
    print(f"{'='*70}")

    dates = sorted(date_groups.keys())

    combined_splits = {
        'train': [],
        'val': [],
        'test': [],
        'buffers': []
    }

    for date_str in dates:
        images = sorted(date_groups[date_str], key=lambda x: x['sequence'] if x['sequence'] else 0)
        n = len(images)

        print(f"\nProcessing {date_str}: {n} images")

        train_end = int(n * train_ratio)
        buffer1_end = min(train_end + buffer_size, n)
        val_end = min(buffer1_end + int(n * val_ratio), n)
        buffer2_end = min(val_end + buffer_size, n)

        date_train = images[:train_end]
        date_buffer1 = images[train_end:buffer1_end]
        date_val = images[buffer1_end:val_end]
        date_buffer2 = images[val_end:buffer2_end]
        date_test = images[buffer2_end:]

        combined_splits['train'].extend(date_train)
        combined_splits['val'].extend(date_val)
        combined_splits['test'].extend(date_test)
        combined_splits['buffers'].extend(date_buffer1 + date_buffer2)

        print(f"  Train: {len(date_train)}, Buffer1: {len(date_buffer1)}, "
              f"Val: {len(date_val)}, Buffer2: {len(date_buffer2)}, Test: {len(date_test)}")

    total = sum(len(v) for v in combined_splits.values())
    total_used = len(combined_splits['train']) + len(combined_splits['val']) + len(combined_splits['test'])
    total_buffer = len(combined_splits['buffers'])

    print(f"\n{'='*70}")
    print("SPLIT SUMMARY:")
    print(f"{'='*70}")
    print(f"  Train:      {len(combined_splits['train']):4d} images ({len(combined_splits['train'])/total*100:.1f}%)")
    print(f"  Val:        {len(combined_splits['val']):4d} images ({len(combined_splits['val'])/total*100:.1f}%)")
    print(f"  Test:       {len(combined_splits['test']):4d} images ({len(combined_splits['test'])/total*100:.1f}%)")
    print(f"  Buffers:    {total_buffer:4d} images (DISCARDED)")
    print(f"\n  Total images used: {total_used}/{total} ({total_used/total*100:.1f}%)")

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


def resize_pair(image_path, mask_path, target_size=512):
    """
    Resize an image and its corresponding mask to target_size x target_size.

    Uses:
      - LANCZOS resampling for the RGB image (best quality downscaling)
      - NEAREST resampling for the mask (preserves label values exactly)

    Args:
        image_path: Path to the source image
        mask_path:  Path to the source mask
        target_size: Output size in pixels (default: 512)

    Returns:
        Tuple of (resized_image, resized_mask) as PIL Images
    """
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)

    resized_image = image.resize((target_size, target_size), Image.LANCZOS)

    # NEAREST resampling is critical for masks — avoids interpolating class labels
    resized_mask = mask.resize((target_size, target_size), Image.NEAREST)

    return resized_image, resized_mask


def create_resized_dataset(
    raw_path="dataset/raw",
    output_path="dataset/processed_512_resized",
    target_size=512,
    buffer_size=10
):
    """
    Main function to create a resized dataset with proper train/val/test splits.

    Each image-mask pair is resized to target_size x target_size (1-to-1 mapping),
    preserving the full scene context without any cropping or patching.
    """
    print(f"\n{'='*70}")
    print("DATASET PREPARATION PIPELINE  (resize mode)")
    print(f"{'='*70}")
    print(f"Target size:  {target_size}x{target_size}")
    print(f"Buffer size:  {buffer_size} images")

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

    # Step 4: Resize and save each image-mask pair
    print(f"\n{'='*70}")
    print(f"RESIZING IMAGES TO {target_size}x{target_size}")
    print(f"{'='*70}")

    split_stats = {}

    for split_name in ['train', 'val', 'test']:
        print(f"\nProcessing {split_name} set...")
        images = splits[split_name]
        saved = 0
        skipped = 0

        for i, img_meta in enumerate(images):
            try:
                resized_img, resized_mask = resize_pair(
                    img_meta['path'],
                    img_meta['mask_path'],
                    target_size=target_size
                )

                stem = Path(img_meta['filename']).stem

                # Save resized image
                img_out = output_path / split_name / "images" / f"{stem}.jpg"
                resized_img.save(img_out, "JPEG", quality=95)

                # Save resized mask
                mask_out = output_path / split_name / "masks" / f"{stem}.png"
                resized_mask.save(mask_out, "PNG")

                saved += 1

            except Exception as e:
                print(f"  [WARNING] Failed to process {img_meta['filename']}: {e}")
                skipped += 1

            if (i + 1) % 20 == 0:
                print(f"  Saved {saved}/{len(images)} images...")

        split_stats[split_name] = {'images': saved, 'skipped': skipped}
        print(f"  Completed {split_name}: {saved} saved, {skipped} skipped")

    # Step 5: Save metadata
    metadata = {
        'mode': 'resize',
        'target_size': target_size,
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
    total = 0
    for split_name in ['train', 'val', 'test']:
        n = split_stats[split_name]['images']
        total += n
        print(f"  {split_name.capitalize():12s}: {n:4d} images")
    print(f"  {'Total':12s}: {total:4d} images")

    return metadata


if __name__ == "__main__":
    create_resized_dataset(
        raw_path="dataset/raw",
        output_path="dataset/processed_512_resized",
        target_size=512,
        buffer_size=10
    )
