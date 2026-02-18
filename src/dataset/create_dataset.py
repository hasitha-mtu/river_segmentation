"""
create_dataset.py
=================
Creates a train / val / test dataset from raw UAV image-mask pairs.

Supports two processing modes:
  - 'resize'  : each image → one 512×512 resized output  (1-to-1)
  - 'patch'   : each image → N overlapping/non-overlapping patches

CRITICAL GUARANTEE
------------------
ALL outputs derived from the same source image are assigned to exactly ONE
split (train, val, or test).  No source image can ever appear in two splits.

This is enforced at three points:
  1. verify_no_leakage()   – checks split lists before any file is written
  2. Output filenames      – always contain the source stem, so contamination
                             would be immediately visible in the file names
  3. audit_output_dirs()   – after writing, scans disk and confirms zero
                             cross-split contamination
"""

import re
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_filename_date(filename):
    """Extract date from DJI filename: DJI_YYYYMMDDHHMMSS_####_V.jpg"""
    match = re.search(r'DJI_(\d{4})(\d{2})(\d{2})\d{6}_\d{4}', filename)
    if match:
        year, month, day = match.groups()
        return datetime(int(year), int(month), int(day))
    return None


def get_gps_data(image_path):
    """Extract (lat, lon) from image EXIF, or return None."""
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
                    gps_info[GPSTAGS.get(gps_tag, gps_tag)] = value[gps_tag]

        if not gps_info:
            return None

        def to_degrees(v):
            d, m, s = v
            return float(d) + float(m) / 60.0 + float(s) / 3600.0

        lat = to_degrees(gps_info.get('GPSLatitude', [0, 0, 0]))
        lon = to_degrees(gps_info.get('GPSLongitude', [0, 0, 0]))
        if gps_info.get('GPSLatitudeRef') == 'S':
            lat = -lat
        if gps_info.get('GPSLongitudeRef') == 'W':
            lon = -lon
        return (lat, lon)

    except Exception as e:
        print(f"  [WARN] GPS read failed for {Path(image_path).name}: {e}")
        return None


def calculate_distance(coord1, coord2):
    """Approximate distance (metres) between two (lat, lon) pairs."""
    if coord1 is None or coord2 is None:
        return float('inf')
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat_diff = (lat2 - lat1) * 111_000
    lon_diff = (lon2 - lon1) * 111_000 * np.cos(np.radians(lat1))
    return np.sqrt(lat_diff**2 + lon_diff**2)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dataset(raw_path="dataset/raw"):
    """
    Scan raw directory, pair images with masks, extract metadata.
    Returns (image_metadata list, date_groups dict) or None on error.
    """
    print("=" * 70)
    print("ANALYZING DATASET")
    print("=" * 70)

    images_dir = Path(raw_path) / "images"
    masks_dir  = Path(raw_path) / "masks"

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return None

    image_metadata = []
    image_files = sorted(images_dir.glob("*.jpg"))
    print(f"\nScanning {len(image_files)} images…")

    for i, img_path in enumerate(image_files):
        mask_path = masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue

        date = parse_filename_date(img_path.name)
        gps  = get_gps_data(img_path)
        seq_match = re.search(r'_(\d{4})_V\.jpg', img_path.name)

        image_metadata.append({
            'filename': img_path.name,
            'stem':     img_path.stem,          # used as unique source ID
            'path':     str(img_path),
            'mask_path': str(mask_path),
            'date':     date,
            'gps':      gps,
            'sequence': int(seq_match.group(1)) if seq_match else None,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(image_files)} processed…")

    print(f"\nFound {len(image_metadata)} valid image-mask pairs")

    # Group by date
    date_groups: dict[str, list] = {}
    for item in image_metadata:
        key = item['date'].strftime('%Y/%m/%d') if item['date'] else 'Unknown'
        date_groups.setdefault(key, []).append(item)

    print(f"\n{'='*70}\nDATE DISTRIBUTION:\n{'='*70}")
    for date_str, items in sorted(date_groups.items()):
        print(f"  {date_str}: {len(items)} images")

    print(f"\n{'='*70}\nSPATIAL ANALYSIS:\n{'='*70}")
    for date_str, items in sorted(date_groups.items()):
        ordered = sorted(items, key=lambda x: x['sequence'] or 0)
        dists = [
            calculate_distance(ordered[i]['gps'], ordered[i + 1]['gps'])
            for i in range(len(ordered) - 1)
            if ordered[i]['gps'] and ordered[i + 1]['gps']
        ]
        if dists:
            print(f"\n  {date_str}:")
            print(f"    Avg consecutive distance : {np.mean(dists):.2f} m")
            print(f"    Min / Max                : {np.min(dists):.2f} m / {np.max(dists):.2f} m")
            print(f"    Total flight path        : {np.sum(dists)/1000:.2f} km")

    return image_metadata, date_groups


# ─────────────────────────────────────────────────────────────────────────────
# Split creation
# ─────────────────────────────────────────────────────────────────────────────

def create_splits_with_buffers(
    date_groups,
    buffer_size=10,
    train_ratio=0.60,
    val_ratio=0.18,
    test_ratio=0.22,
):
    """
    Build train/val/test splits with spatial buffer zones.

    Splitting is done PER DATE so that every split contains images from
    every flight session (prevents seasonal domain shift).

    Layout per date:
        [train] [buffer] [val] [buffer] [test]

    Buffer images are discarded — they act as a spatial gap between splits
    so that highly-overlapping consecutive images don't leak across splits.
    """
    print(f"\n{'='*70}")
    print("CREATING SPLITS WITH BUFFER ZONES")
    print(f"  Strategy : mixed-seasonal split (each split sees all seasons)")
    print(f"  Buffers  : {buffer_size} images discarded between each zone")
    print(f"{'='*70}")

    combined = {'train': [], 'val': [], 'test': [], 'buffers': []}

    for date_str in sorted(date_groups):
        images = sorted(date_groups[date_str], key=lambda x: x['sequence'] or 0)
        n = len(images)

        train_end   = int(n * train_ratio)
        buf1_end    = min(train_end + buffer_size, n)
        val_end     = min(buf1_end  + int(n * val_ratio), n)
        buf2_end    = min(val_end   + buffer_size, n)

        d_train  = images[:train_end]
        d_buf1   = images[train_end:buf1_end]
        d_val    = images[buf1_end:val_end]
        d_buf2   = images[val_end:buf2_end]
        d_test   = images[buf2_end:]

        combined['train'].extend(d_train)
        combined['val'].extend(d_val)
        combined['test'].extend(d_test)
        combined['buffers'].extend(d_buf1 + d_buf2)

        print(f"\n  {date_str}  ({n} images)")
        print(f"    Train   : {len(d_train):3d}   Buffer1 : {len(d_buf1):2d} (discarded)")
        print(f"    Val     : {len(d_val):3d}   Buffer2 : {len(d_buf2):2d} (discarded)")
        print(f"    Test    : {len(d_test):3d}")

    used  = len(combined['train']) + len(combined['val']) + len(combined['test'])
    total = used + len(combined['buffers'])

    print(f"\n{'='*70}\nCOMBINED SPLITS (mixed seasonal):\n{'='*70}")
    for s in ['train', 'val', 'test']:
        n = len(combined[s])
        print(f"  {s.capitalize():12s}: {n:4d} images  ({n/used*100:5.1f}% of used)")
    print(f"  {'Buffers':12s}: {len(combined['buffers']):4d} images  (discarded)")
    print(f"  Total used  : {used}/{total}  ({used/total*100:.1f}%)")

    print(f"\n{'='*70}\nSEASONAL DISTRIBUTION:\n{'='*70}")
    for s in ['train', 'val', 'test']:
        counts: dict[str, int] = {}
        for img in combined[s]:
            key = img['date'].strftime('%Y/%m/%d') if img['date'] else 'Unknown'
            counts[key] = counts.get(key, 0) + 1
        print(f"\n  {s.capitalize()}:")
        for k in sorted(counts):
            print(f"    {k}: {counts[k]:3d} images")

    return {'train': combined['train'], 'val': combined['val'], 'test': combined['test']}


# ─────────────────────────────────────────────────────────────────────────────
# No-leakage verification  (runs BEFORE any file is written)
# ─────────────────────────────────────────────────────────────────────────────

def verify_no_leakage(splits: dict) -> bool:
    """
    Confirm that no source image stem appears in more than one split.

    This is the primary safeguard: if the split logic ever has a bug,
    this catches it before a single output file is written.

    Returns True if clean, raises RuntimeError on violation.
    """
    print(f"\n{'='*70}")
    print("LEAKAGE VERIFICATION  (pre-write check)")
    print(f"{'='*70}")

    stem_to_split: dict[str, str] = {}
    violations: list[str] = []

    for split_name, images in splits.items():
        for img in images:
            stem = img['stem']
            if stem in stem_to_split:
                msg = (f"  ✗  '{stem}'  appears in both "
                       f"'{stem_to_split[stem]}' and '{split_name}'")
                violations.append(msg)
                print(msg)
            else:
                stem_to_split[stem] = split_name

    if violations:
        raise RuntimeError(
            f"\n[FATAL] {len(violations)} source image(s) appear in multiple splits.\n"
            "Dataset creation aborted to prevent data leakage."
        )

    total = sum(len(v) for v in splits.values())
    print(f"  ✓  All {total} source images are assigned to exactly one split.")
    print(f"  ✓  Zero cross-split contamination detected.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Post-write audit  (runs AFTER all files are written)
# ─────────────────────────────────────────────────────────────────────────────

def audit_output_dirs(output_path: Path) -> bool:
    """
    Scan the output directory structure and verify that no source image stem
    (the part of the filename before the first '_patch_' or the full stem for
    resized images) appears in more than one split folder.

    This catches any accidental file copies that might have been made outside
    the normal pipeline.

    Returns True if clean, prints a warning table on violation.
    """
    print(f"\n{'='*70}")
    print("POST-WRITE AUDIT  (on-disk verification)")
    print(f"{'='*70}")

    stem_to_splits: dict[str, list[str]] = {}

    for split_name in ['train', 'val', 'test']:
        img_dir = output_path / split_name / "images"
        if not img_dir.exists():
            continue
        for f in img_dir.iterdir():
            # Derive source stem: strip '_patch_NNNN' suffix if present
            raw_stem = f.stem
            source_stem = re.sub(r'_patch_\d+$', '', raw_stem)
            stem_to_splits.setdefault(source_stem, []).append(split_name)

    violations = {
        stem: splits
        for stem, splits in stem_to_splits.items()
        if len(set(splits)) > 1
    }

    if violations:
        print(f"\n  [WARNING] {len(violations)} source image(s) found in multiple splits:")
        for stem, splits in violations.items():
            unique = sorted(set(splits))
            print(f"    '{stem}'  →  {unique}")
        print("\n  Dataset may be contaminated — review the output directory.")
        return False

    total_sources = len(stem_to_splits)
    print(f"  ✓  {total_sources} unique source images across all splits.")
    print(f"  ✓  No source image found in more than one split.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Image processing: resize
# ─────────────────────────────────────────────────────────────────────────────

def resize_pair(image_path, mask_path, target_size=512):
    """
    Resize an image-mask pair to target_size × target_size.

    - LANCZOS  for the RGB image  (best-quality downscaling)
    - NEAREST  for the mask       (preserves integer class labels exactly)
    """
    image = Image.open(image_path).convert("RGB")
    mask  = Image.open(mask_path)

    return (
        image.resize((target_size, target_size), Image.LANCZOS),
        mask.resize( (target_size, target_size), Image.NEAREST),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image processing: patch
# ─────────────────────────────────────────────────────────────────────────────

def extract_patches(image_path, mask_path, patch_size=512, stride=512):
    """
    Slice image and mask into fixed-size patches.

    Only patches that contain at least one positive mask pixel are kept,
    so background-only tiles don't bloat the dataset.

    Returns a list of dicts: {image, mask, position (x, y)}.
    """
    img_arr  = np.array(Image.open(image_path).convert("RGB"))
    mask_arr = np.array(Image.open(mask_path))

    h, w = img_arr.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch  = img_arr[ y:y+patch_size, x:x+patch_size]
            mask_patch = mask_arr[y:y+patch_size, x:x+patch_size]

            if mask_patch.sum() > 0:           # skip all-background patches
                patches.append({
                    'image':    Image.fromarray(img_patch),
                    'mask':     Image.fromarray(mask_patch),
                    'position': (x, y),
                })

    return patches


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def create_dataset(
    raw_path="dataset/raw",
    output_path="dataset/processed_512_resized",
    mode="resize",          # "resize" or "patch"
    target_size=512,        # used by resize mode
    patch_size=512,         # used by patch mode
    stride=512,             # used by patch mode
    buffer_size=10,
):
    """
    End-to-end dataset creation pipeline.

    mode='resize' : every source image → one resized 512×512 pair  (1-to-1)
    mode='patch'  : every source image → N patches of patch_size    (1-to-N)

    In BOTH modes, all outputs from a given source image are placed in
    exactly one split.  This is verified before and after writing.
    """
    assert mode in ("resize", "patch"), f"mode must be 'resize' or 'patch', got '{mode}'"

    print(f"\n{'='*70}")
    print(f"DATASET PREPARATION PIPELINE  [mode={mode}]")
    print(f"{'='*70}")
    if mode == "resize":
        print(f"  Target size : {target_size}×{target_size}")
    else:
        print(f"  Patch size  : {patch_size}×{patch_size}  |  stride={stride}")
    print(f"  Buffer size : {buffer_size} images")

    # ── Step 1: analyse ──────────────────────────────────────────────────────
    result = analyze_dataset(raw_path)
    if result is None:
        return None
    image_metadata, date_groups = result

    # ── Step 2: split ────────────────────────────────────────────────────────
    splits = create_splits_with_buffers(date_groups, buffer_size=buffer_size)

    # ── Step 3: verify BEFORE writing ────────────────────────────────────────
    #   Raises RuntimeError and aborts if any source stem is in two splits.
    verify_no_leakage(splits)

    # ── Step 4: create output directories ────────────────────────────────────
    out = Path(output_path)
    for s in ['train', 'val', 'test']:
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "masks").mkdir(parents=True, exist_ok=True)

    # ── Step 5: process and save ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"WRITING OUTPUT  [{mode} mode]")
    print(f"{'='*70}")

    split_stats: dict[str, dict] = {}

    for split_name in ['train', 'val', 'test']:
        print(f"\n  Processing {split_name} split…")
        images = splits[split_name]
        saved, skipped, total_outputs = 0, 0, 0

        for i, img_meta in enumerate(images):
            stem = img_meta['stem']   # e.g. DJI_20230615120000_0042_V

            try:
                if mode == "resize":
                    # ── resize: one output per source image ─────────────────
                    res_img, res_mask = resize_pair(
                        img_meta['path'], img_meta['mask_path'], target_size
                    )
                    # Filename = original stem — source is immediately obvious
                    (out / split_name / "images" / f"{stem}.jpg").parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    res_img.save( out / split_name / "images" / f"{stem}.jpg",  "JPEG", quality=95)
                    res_mask.save(out / split_name / "masks"  / f"{stem}.png",  "PNG")
                    total_outputs += 1

                else:
                    # ── patch: N outputs per source image ────────────────────
                    patches = extract_patches(
                        img_meta['path'], img_meta['mask_path'],
                        patch_size=patch_size, stride=stride
                    )
                    for p_idx, patch in enumerate(patches):
                        # Filename encodes source stem + patch index
                        # e.g. DJI_20230615120000_0042_V_patch_0007
                        base = f"{stem}_patch_{p_idx:04d}"
                        patch['image'].save(
                            out / split_name / "images" / f"{base}.jpg", "JPEG", quality=95
                        )
                        patch['mask'].save(
                            out / split_name / "masks" / f"{base}.png", "PNG"
                        )
                        total_outputs += 1

                saved += 1

            except Exception as e:
                print(f"    [WARN] Skipping {img_meta['filename']}: {e}")
                skipped += 1

            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(images)} source images done  "
                      f"({total_outputs} outputs written so far)…")

        split_stats[split_name] = {
            'source_images': saved,
            'outputs':       total_outputs,
            'skipped':       skipped,
        }
        print(f"    ✓ {split_name}: {saved} source images → {total_outputs} output files")

    # ── Step 6: audit on disk AFTER writing ──────────────────────────────────
    audit_output_dirs(out)

    # ── Step 7: save metadata ─────────────────────────────────────────────────
    metadata = {
        'mode':              mode,
        'target_size':       target_size  if mode == 'resize' else None,
        'patch_size':        patch_size   if mode == 'patch'  else None,
        'stride':            stride       if mode == 'patch'  else None,
        'buffer_size':       buffer_size,
        'leakage_verified':  True,
        'splits':            split_stats,
        'date_distribution': {d: len(v) for d, v in date_groups.items()},
    }
    meta_path = out / "dataset_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DATASET CREATION COMPLETED")
    print(f"{'='*70}")
    print(f"  Output     : {out}")
    print(f"  Metadata   : {meta_path}")
    print(f"\n  {'Split':12s}  {'Source images':>14s}  {'Output files':>13s}")
    print(f"  {'-'*44}")
    total_src = total_out = 0
    for s in ['train', 'val', 'test']:
        n_src = split_stats[s]['source_images']
        n_out = split_stats[s]['outputs']
        total_src += n_src
        total_out += n_out
        print(f"  {s.capitalize():12s}  {n_src:>14d}  {n_out:>13d}")
    print(f"  {'-'*44}")
    print(f"  {'Total':12s}  {total_src:>14d}  {total_out:>13d}")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Option A: resize to 512×512  (1 source → 1 output) ──────────────────
    # create_dataset(
    #     raw_path    = "dataset/raw",
    #     output_path = "dataset/processed_512_resized",
    #     mode        = "resize",
    #     target_size = 512,
    #     buffer_size = 10,
    # )

    # ── Option B: 512×512 patches, no overlap  (1 source → N outputs) ───────
    create_dataset(
        raw_path    = "dataset/raw",
        output_path = "dataset/processed_512_patch",
        mode        = "patch",
        patch_size  = 512,
        stride      = 512,
        buffer_size = 10,
    )

    # ── Option C: 1024×1024 patches, 25 % overlap  (RECOMMENDED for rivers) ─
    # create_dataset(
    #     raw_path    = "dataset/raw",
    #     output_path = "dataset/processed_1024_patch",
    #     mode        = "patch",
    #     patch_size  = 1024,
    #     stride      = 768,
    #     buffer_size = 10,
    # )
