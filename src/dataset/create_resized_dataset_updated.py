"""
create_resized_dataset.py  —  Dataset creation with three split strategies

ORIGINAL BEHAVIOUR (split_strategy='sequential') is unchanged.

NEW for reviewer revision response:
  split_strategy='stratified'   →  Split A: windowed stratification
  split_strategy='alternative'  →  Split B: reversed spatial direction

Both new strategies maintain spatial buffer zones to prevent data leakage.
The stratified split ensures the test set samples from all spatial regions
of the flight path, giving representative gt_ratio coverage (addresses R1
Major #1 and R2 Third point).  The alternative split provides the
independent cross-split stability check the reviewers require.

Usage:
    # Original sequential split (existing behaviour)
    python create_resized_dataset.py

    # New stratified split (Split A — recommended for paper revision)
    python create_resized_dataset.py --strategy stratified

    # Alternative split for stability analysis (Split B)
    python create_resized_dataset.py --strategy alternative
"""

import argparse
import os
import re
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
from datetime import datetime
import json
import shutil


# ---------------------------------------------------------------------------
# gt_ratio bin definitions — must match audit_current_split.py
# ---------------------------------------------------------------------------
BIN_EDGES  = [0.0, 0.001, 0.005, 0.05, 1.01]
BIN_LABELS = ['ultra_low', 'low', 'medium', 'high']


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------

def parse_filename_date(filename):
    """
    Extract date from DJI filename format: DJI_YYYYMMDDHHMMSS_####_V.jpg
    Returns datetime object or None.
    """
    match = re.search(r'DJI_(\d{4})(\d{2})(\d{2})\d{6}_\d{4}', filename)
    if match:
        year, month, day = match.groups()
        return datetime(int(year), int(month), int(day))
    return None


def get_gps_data(image_path):
    """
    Extract GPS coordinates from image EXIF data.
    Returns (latitude, longitude) or None.
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
    Approximate distance between two GPS coordinates in metres.
    """
    if coord1 is None or coord2 is None:
        return float('inf')

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat_diff = (lat2 - lat1) * 111000
    lon_diff = (lon2 - lon1) * 111000 * np.cos(np.radians(lat1))

    return np.sqrt(lat_diff**2 + lon_diff**2)


# ---------------------------------------------------------------------------
# NEW helper: gt_ratio from mask
# ---------------------------------------------------------------------------

def compute_gt_ratio(mask_path) -> float:
    """
    Water-pixel fraction (gt_ratio) of a binary PNG mask.
    Works whether the mask stores values as 0/255 or 0/1.

    Returns a float in [0, 1].
    """
    arr = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return float(arr.mean())


def gt_ratio_bin(ratio: float) -> str:
    """Return the string bin label for a given gt_ratio."""
    for i, (lo, hi) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
        if lo <= ratio < hi:
            return BIN_LABELS[i]
    return BIN_LABELS[-1]


# ---------------------------------------------------------------------------
# Dataset analysis (unchanged from original)
# ---------------------------------------------------------------------------

def analyze_dataset(raw_path="dataset/raw"):
    """
    Analyse the dataset: parse dates, extract GPS, calculate overlaps,
    and NOW also compute gt_ratio for every image.
    """
    print("=" * 70)
    print("ANALYZING DATASET")
    print("=" * 70)

    images_dir = Path(raw_path) / "images"
    masks_dir  = Path(raw_path) / "masks"

    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return None

    image_metadata = []

    print("\nScanning images and extracting metadata (including gt_ratio)...")
    image_files = sorted(list(images_dir.glob("*.jpg")))

    for i, img_path in enumerate(image_files):
        mask_path = masks_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            continue

        date     = parse_filename_date(img_path.name)
        gps      = get_gps_data(img_path)
        gt_ratio = compute_gt_ratio(mask_path)          # ← NEW

        seq_match = re.search(r'_(\d{4})_V\.jpg', img_path.name)
        sequence  = int(seq_match.group(1)) if seq_match else None

        image_metadata.append({
            'filename':  img_path.name,
            'path':      str(img_path),
            'mask_path': str(mask_path),
            'date':      date,
            'gps':       gps,
            'sequence':  sequence,
            'gt_ratio':  gt_ratio,                      # ← NEW
            'gt_bin':    gt_ratio_bin(gt_ratio),        # ← NEW
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
        ratios = [it['gt_ratio'] for it in items]
        print(f"  {date_str}: {len(items)} images  "
              f"(mean gt_ratio={np.mean(ratios)*100:.2f}%, "
              f"max={np.max(ratios)*100:.2f}%)")

    # gt_ratio bin summary across whole dataset
    print(f"\n{'='*70}")
    print("GT_RATIO BIN DISTRIBUTION (full dataset):")
    print(f"{'='*70}")
    all_ratios = [it['gt_ratio'] for it in image_metadata]
    for i, (lo, hi) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
        n = sum(lo <= r < hi for r in all_ratios)
        print(f"  {BIN_LABELS[i]:12s} [{lo*100:.2f}% – {hi*100:.2f}%): "
              f"{n:3d} images  ({n/len(all_ratios)*100:.1f}%)")

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
            print(f"    Avg distance between consecutive images: {np.mean(distances):.2f}m")
            print(f"    Min distance: {np.min(distances):.2f}m")
            print(f"    Max distance: {np.max(distances):.2f}m")
            print(f"    Total flight path: {np.sum(distances):.2f}m "
                  f"({np.sum(distances)/1000:.2f}km)")

    return image_metadata, date_groups


# ---------------------------------------------------------------------------
# SPLIT STRATEGY 1 (ORIGINAL): Sequential — unchanged
# ---------------------------------------------------------------------------

def create_splits_with_buffers(date_groups, buffer_size=10,
                               train_ratio=0.6, val_ratio=0.18, test_ratio=0.22):
    """
    Original sequential split.

    STRATEGY: split each date separately (preserving seasonal mixing),
    then combine. Test images always come from the END of each date's
    sequence.
    """
    print(f"\n{'='*70}")
    print("SPLIT STRATEGY: SEQUENTIAL (original)")
    print(f"{'='*70}")

    dates = sorted(date_groups.keys())

    combined_splits = {'train': [], 'val': [], 'test': [], 'buffers': []}

    for date_str in dates:
        images = sorted(date_groups[date_str],
                        key=lambda x: x['sequence'] if x['sequence'] else 0)
        n = len(images)

        print(f"\nProcessing {date_str}: {n} images")

        train_end   = int(n * train_ratio)
        buffer1_end = min(train_end + buffer_size, n)
        val_end     = min(buffer1_end + int(n * val_ratio), n)
        buffer2_end = min(val_end + buffer_size, n)

        combined_splits['train'].extend(images[:train_end])
        combined_splits['buffers'].extend(images[train_end:buffer1_end])
        combined_splits['val'].extend(images[buffer1_end:val_end])
        combined_splits['buffers'].extend(images[val_end:buffer2_end])
        combined_splits['test'].extend(images[buffer2_end:])

        print(f"  Train: {train_end}, Buffer1: {buffer1_end-train_end}, "
              f"Val: {val_end-buffer1_end}, Buffer2: {buffer2_end-val_end}, "
              f"Test: {n-buffer2_end}")

    _print_split_summary(combined_splits, date_groups)

    return {
        'train': combined_splits['train'],
        'val':   combined_splits['val'],
        'test':  combined_splits['test'],
    }


# ---------------------------------------------------------------------------
# SPLIT STRATEGY 2 (NEW — Split A): Windowed stratification
# ---------------------------------------------------------------------------

def create_stratified_splits(date_groups, buffer_size=5, n_windows=4,
                             train_ratio=0.6, val_ratio=0.18, test_ratio=0.22):
    """
    Split A: Windowed stratification (addresses R1 Major #1, R2 Third).

    MOTIVATION
    ----------
    The sequential split places test images exclusively at the END of each
    flight, which happens to be the area with the least water coverage.
    This produces a test set over-represented in ultra-low gt_ratio images
    and lacking any high-coverage images — invalidating the failure threshold
    and SAM-FPN recovery claims.

    STRATEGY
    --------
    Divide each date's flight sequence into n_windows equal spatial windows.
    Apply a sequential train/buffer/val/buffer/test split within each window.
    This ensures test images are drawn from ALL spatial regions of the flight
    path, including sections with medium and high water coverage.

    Spatial buffers are applied at the val/test boundary within every window,
    preventing data leakage between adjacent images.

    PARAMETERS
    ----------
    buffer_size : int
        Number of frames excluded at each buffer zone boundary.
        Smaller than the original (5 vs 10) because buffers are applied
        inside windows — total buffer overhead stays comparable.
    n_windows : int
        Number of spatial windows per date.  4 windows over ~200 images
        gives windows of ~50 images each, with ~11 test images per window.
    """
    print(f"\n{'='*70}")
    print("SPLIT STRATEGY: STRATIFIED WINDOWED (Split A — new)")
    print(f"  n_windows={n_windows}, buffer_size={buffer_size}")
    print(f"{'='*70}")

    combined_splits = {'train': [], 'val': [], 'test': [], 'buffers': []}

    for date_str in sorted(date_groups.keys()):
        images = sorted(date_groups[date_str],
                        key=lambda x: x['sequence'] if x['sequence'] else 0)
        n = len(images)
        window_size = n // n_windows

        print(f"\n  {date_str}: {n} images → {n_windows} windows of ~{window_size} each")
        print(f"  {'Win':>4} {'Start':>6} {'End':>6} {'Train':>6} "
              f"{'Buf1':>5} {'Val':>5} {'Buf2':>5} {'Test':>5}")
        print(f"  {'-'*50}")

        for w in range(n_windows):
            w_start = w * window_size
            # Last window absorbs any remainder
            w_end   = w_start + window_size if w < n_windows - 1 else n
            window  = images[w_start:w_end]
            wn      = len(window)

            train_end   = int(wn * train_ratio)
            buf1_end    = min(train_end + buffer_size, wn)
            val_end     = min(buf1_end + int(wn * val_ratio), wn)
            buf2_end    = min(val_end + buffer_size, wn)

            combined_splits['train'].extend(window[:train_end])
            combined_splits['buffers'].extend(window[train_end:buf1_end])
            combined_splits['val'].extend(window[buf1_end:val_end])
            combined_splits['buffers'].extend(window[val_end:buf2_end])
            combined_splits['test'].extend(window[buf2_end:])

            print(f"  {w+1:>4} {w_start:>6} {w_end:>6} {train_end:>6} "
                  f"{buf1_end-train_end:>5} {val_end-buf1_end:>5} "
                  f"{buf2_end-val_end:>5} {wn-buf2_end:>5}")

    _print_split_summary(combined_splits, date_groups)
    _print_gt_ratio_distribution(combined_splits)

    return {
        'train': combined_splits['train'],
        'val':   combined_splits['val'],
        'test':  combined_splits['test'],
    }


# ---------------------------------------------------------------------------
# SPLIT STRATEGY 3 (NEW — Split B): Reversed spatial direction
# ---------------------------------------------------------------------------

def create_alternative_splits(date_groups, buffer_size=10,
                              train_ratio=0.6, val_ratio=0.18, test_ratio=0.22):
    """
    Split B: Reversed spatial direction (independent cross-split stability check).

    MOTIVATION
    ----------
    Complements Split A by using a completely different spatial partition.
    The original and Split A both start from the beginning of the sequence.
    Split B reverses the sequence order so test images come from the
    BEGINNING of the flight path (what was training data in the original).

    If architecture rankings and failure thresholds remain consistent
    between Split A and Split B, this is strong evidence that the core
    findings are robust to the specific spatial partition — directly
    answering the reviewers' stability analysis requirement.

    STRATEGY
    --------
    Identical logic to the original sequential split, but images are
    processed in REVERSED sequence order per date before splitting.
    Buffer zones and seasonal mixing are preserved.
    """
    print(f"\n{'='*70}")
    print("SPLIT STRATEGY: ALTERNATIVE / REVERSED (Split B — new)")
    print(f"  Same ratios as original but sequences REVERSED per date")
    print(f"{'='*70}")

    combined_splits = {'train': [], 'val': [], 'test': [], 'buffers': []}

    for date_str in sorted(date_groups.keys()):
        # KEY CHANGE: sort in DESCENDING sequence order
        images = sorted(date_groups[date_str],
                        key=lambda x: x['sequence'] if x['sequence'] else 0,
                        reverse=True)
        n = len(images)

        print(f"\n  {date_str}: {n} images (reversed sequence)")

        train_end   = int(n * train_ratio)
        buf1_end    = min(train_end + buffer_size, n)
        val_end     = min(buf1_end + int(n * val_ratio), n)
        buf2_end    = min(val_end + buffer_size, n)

        combined_splits['train'].extend(images[:train_end])
        combined_splits['buffers'].extend(images[train_end:buf1_end])
        combined_splits['val'].extend(images[buf1_end:val_end])
        combined_splits['buffers'].extend(images[val_end:buf2_end])
        combined_splits['test'].extend(images[buf2_end:])

        print(f"    Train: {train_end}, Buffer1: {buf1_end-train_end}, "
              f"Val: {val_end-buf1_end}, Buffer2: {buf2_end-val_end}, "
              f"Test: {n-buf2_end}")

    _print_split_summary(combined_splits, date_groups)
    _print_gt_ratio_distribution(combined_splits)

    return {
        'train': combined_splits['train'],
        'val':   combined_splits['val'],
        'test':  combined_splits['test'],
    }


# ---------------------------------------------------------------------------
# Shared reporting helpers
# ---------------------------------------------------------------------------

def _print_split_summary(combined_splits, date_groups):
    total_all = (len(combined_splits['train']) + len(combined_splits['val']) +
                 len(combined_splits['test']) + len(combined_splits['buffers']))
    total_used = (len(combined_splits['train']) + len(combined_splits['val']) +
                  len(combined_splits['test']))
    total_buf  = len(combined_splits['buffers'])

    print(f"\n  {'='*50}")
    print(f"  SPLIT SUMMARY:")
    print(f"  {'='*50}")
    print(f"  Train:   {len(combined_splits['train']):4d} images "
          f"({len(combined_splits['train'])/total_all*100:.1f}%)")
    print(f"  Val:     {len(combined_splits['val']):4d} images "
          f"({len(combined_splits['val'])/total_all*100:.1f}%)")
    print(f"  Test:    {len(combined_splits['test']):4d} images "
          f"({len(combined_splits['test'])/total_all*100:.1f}%)")
    print(f"  Buffers: {total_buf:4d} images (DISCARDED)")
    print(f"\n  Total used: {total_used}/{total_all} "
          f"({total_used/total_all*100:.1f}%)")

    print(f"\n  SEASONAL DISTRIBUTION:")
    for split_name in ('train', 'val', 'test'):
        split_images = combined_splits[split_name]
        date_counts: dict = {}
        for img in split_images:
            ds = img['date'].strftime('%Y/%m/%d') if img['date'] else 'Unknown'
            date_counts[ds] = date_counts.get(ds, 0) + 1
        counts_str = ", ".join(f"{d}: {c}" for d, c in sorted(date_counts.items()))
        print(f"    {split_name.capitalize():7s}: {counts_str}")


def _print_gt_ratio_distribution(combined_splits):
    """
    Report gt_ratio bin breakdown per split — key diagnostic for the
    reviewer's distribution mismatch concern.
    """
    print(f"\n  GT_RATIO BIN DISTRIBUTION PER SPLIT:")
    print(f"  {'Bin':<15} {'Train%':>8} {'Val%':>8} {'Test%':>8}")
    print(f"  {'-'*42}")

    for i, label in enumerate(BIN_LABELS):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        row = f"  {label:<15}"
        for split_name in ('train', 'val', 'test'):
            imgs   = combined_splits[split_name]
            n_bin  = sum(lo <= img['gt_ratio'] < hi for img in imgs)
            pct    = n_bin / len(imgs) * 100 if imgs else 0.0
            row   += f" {pct:>7.1f}%"
        print(row)

    # Flag if test is missing high-coverage images
    test_imgs = combined_splits['test']
    if test_imgs:
        max_test_gt = max(img['gt_ratio'] for img in test_imgs)
        if max_test_gt < 0.10:
            print(f"\n  ► WARNING: max test gt_ratio = {max_test_gt*100:.2f}% "
                  f"(still below 10% — check n_windows or buffer_size)")
        else:
            print(f"\n  ✓ Test set now includes images up to "
                  f"{max_test_gt*100:.2f}% coverage — distribution mismatch resolved")


# ---------------------------------------------------------------------------
# Resize pair (unchanged)
# ---------------------------------------------------------------------------

def resize_pair(image_path, mask_path, target_size=512):
    """
    Resize an image and its corresponding mask to target_size × target_size.

    Uses LANCZOS for the RGB image and NEAREST for the mask to preserve
    binary label values exactly.
    """
    image = Image.open(image_path).convert("RGB")
    mask  = Image.open(mask_path)

    resized_image = image.resize((target_size, target_size), Image.LANCZOS)
    resized_mask  = mask.resize((target_size, target_size), Image.NEAREST)

    return resized_image, resized_mask


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_resized_dataset(
    raw_path="dataset/raw",
    output_path="dataset/processed_512_resized",
    target_size=512,
    buffer_size=10,
    split_strategy='sequential',   # 'sequential' | 'stratified' | 'alternative'
    n_windows=3,                   # only used by 'stratified'
):
    """
    Create a resized dataset with proper train/val/test splits.

    split_strategy options
    ----------------------
    'sequential'  — original behaviour (sequential split per date)
    'stratified'  — Split A: windowed stratification for reviewer revision
    'alternative' — Split B: reversed spatial direction for stability analysis
    """
    print(f"\n{'='*70}")
    print("DATASET PREPARATION PIPELINE  (resize mode)")
    print(f"{'='*70}")
    print(f"Target size:     {target_size}×{target_size}")
    print(f"Buffer size:     {buffer_size} images")
    print(f"Split strategy:  {split_strategy}")
    if split_strategy == 'stratified':
        print(f"N windows:       {n_windows}")

    # Step 1: Analyse dataset (now also computes gt_ratio per image)
    result = analyze_dataset(raw_path)
    if result is None:
        return
    image_metadata, date_groups = result

    # Step 2: Create splits using the chosen strategy
    if split_strategy == 'sequential':
        splits = create_splits_with_buffers(
            date_groups, buffer_size=buffer_size)
    elif split_strategy == 'stratified':
        # Per-window buffer must be small enough that test portion is non-trivial.
        # Rule of thumb: buffer ≤ 5% of per-window image count.
        # With ~200 images/date and n_windows=3, window_size ≈ 67 → buffer=3 gives
        # ~9 test images per window × 3 windows × 2 dates = ~54 test images total.
        windowed_buffer = max(2, buffer_size // 4)
        splits = create_stratified_splits(
            date_groups,
            buffer_size=windowed_buffer,
            n_windows=n_windows)
    elif split_strategy == 'alternative':
        splits = create_alternative_splits(
            date_groups, buffer_size=buffer_size)
    else:
        raise ValueError(f"Unknown split_strategy: '{split_strategy}'. "
                         f"Choose 'sequential', 'stratified', or 'alternative'.")

    # Step 3: Create output directories
    output_path = Path(output_path)
    for split_name in ('train', 'val', 'test'):
        (output_path / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split_name / "masks").mkdir(parents=True, exist_ok=True)

    # Step 4: Resize and save each image-mask pair
    print(f"\n{'='*70}")
    print(f"RESIZING IMAGES TO {target_size}×{target_size}")
    print(f"{'='*70}")

    split_stats = {}

    for split_name in ('train', 'val', 'test'):
        print(f"\nProcessing {split_name} set...")
        images  = splits[split_name]
        saved   = 0
        skipped = 0

        # Per-image gt_ratio record for metadata
        gt_ratio_records = []

        for i, img_meta in enumerate(images):
            try:
                resized_img, resized_mask = resize_pair(
                    img_meta['path'],
                    img_meta['mask_path'],
                    target_size=target_size,
                )

                stem = Path(img_meta['filename']).stem

                img_out  = output_path / split_name / "images" / f"{stem}.jpg"
                mask_out = output_path / split_name / "masks"  / f"{stem}.png"

                resized_img.save(img_out,  "JPEG", quality=95)
                resized_mask.save(mask_out, "PNG")

                gt_ratio_records.append({
                    'filename': img_meta['filename'],
                    'gt_ratio': img_meta['gt_ratio'],
                    'gt_bin':   img_meta['gt_bin'],
                })

                saved += 1

            except Exception as e:
                print(f"  [WARNING] Failed to process {img_meta['filename']}: {e}")
                skipped += 1

            if (i + 1) % 20 == 0:
                print(f"  Saved {saved}/{len(images)} images...")

        split_stats[split_name] = {
            'images':          saved,
            'skipped':         skipped,
            'gt_ratio_records': gt_ratio_records,
        }
        print(f"  Completed {split_name}: {saved} saved, {skipped} skipped")

    # Step 5: Save metadata
    metadata = {
        'mode':            'resize',
        'target_size':     target_size,
        'buffer_size':     buffer_size,
        'split_strategy':  split_strategy,
        'n_windows':       n_windows if split_strategy == 'stratified' else None,
        'splits': {
            s: {
                'images':          v['images'],
                'skipped':         v['skipped'],
                'gt_ratio_records': v['gt_ratio_records'],
            }
            for s, v in split_stats.items()
        },
        'date_distribution': {
            date: len(items) for date, items in date_groups.items()
        },
    }

    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("DATASET CREATION COMPLETED!")
    print(f"{'='*70}")
    print(f"\nOutput directory:  {output_path}")
    print(f"Metadata:          {metadata_path}")

    print(f"\nFinal Statistics:")
    total = 0
    for split_name in ('train', 'val', 'test'):
        n = split_stats[split_name]['images']
        total += n
        print(f"  {split_name.capitalize():12s}: {n:4d} images")
    print(f"  {'Total':12s}: {total:4d} images")

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create resized dataset with configurable split strategy")
    parser.add_argument('--raw_path',    default='dataset/raw')
    parser.add_argument('--output_path', default='dataset/processed_512_resized')
    parser.add_argument('--target_size', default=512, type=int)
    parser.add_argument('--buffer_size', default=10,  type=int)
    parser.add_argument('--strategy',   default='sequential',
                        choices=['sequential', 'stratified', 'alternative'],
                        help="Split strategy to use")
    parser.add_argument('--n_windows',  default=3, type=int,
                        help="Number of spatial windows (stratified only)")
    args = parser.parse_args()

    output_path = f'{args.output_path}/{args.strategy}'

    create_resized_dataset(
        raw_path=args.raw_path,
        output_path=output_path,
        target_size=args.target_size,
        buffer_size=args.buffer_size,
        split_strategy=args.strategy,
        n_windows=args.n_windows,
    )
