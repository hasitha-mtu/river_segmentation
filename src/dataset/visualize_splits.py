import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re


def visualize_splits(metadata_path="dataset/processed/dataset_metadata.json", 
                     raw_path="dataset/raw"):
    """
    Visualize the spatial distribution of train/val/test splits along the flight path.
    This helps verify there's no spatial overlap between splits.
    """
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("="*70)
    print("SPLIT VISUALIZATION")
    print("="*70)
    
    # We'll create a simple visualization showing which images went to which split
    # Since we don't have the full GPS data in metadata, we'll need to re-scan
    
    from create_patched_dataset import get_gps_data, parse_filename_date
    
    images_dir = Path(raw_path) / "images"
    masks_dir = Path(raw_path) / "masks"
    
    # Collect image info
    image_data = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        mask_path = masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue
        
        date = parse_filename_date(img_path.name)
        gps = get_gps_data(img_path)
        seq_match = re.search(r'_(\d{4})_V\.jpg', img_path.name)
        sequence = int(seq_match.group(1)) if seq_match else None
        
        if date and gps:
            image_data.append({
                'filename': img_path.name,
                'date': date.strftime('%Y/%m/%d'),
                'gps': gps,
                'sequence': sequence
            })
    
    # Group by date
    date_groups = {}
    for item in image_data:
        date_str = item['date']
        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(item)
    
    # Create figure with subplots for each date
    fig, axes = plt.subplots(len(date_groups), 1, figsize=(15, 5*len(date_groups)))
    if len(date_groups) == 1:
        axes = [axes]
    
    for idx, (date_str, items) in enumerate(sorted(date_groups.items())):
        ax = axes[idx]
        
        # Sort by sequence
        items_sorted = sorted(items, key=lambda x: x['sequence'])
        
        # Extract GPS coordinates
        lats = [item['gps'][0] for item in items_sorted]
        lons = [item['gps'][1] for item in items_sorted]
        sequences = [item['sequence'] for item in items_sorted]
        
        # Plot the flight path
        ax.plot(lons, lats, 'k-', alpha=0.3, linewidth=1, label='Flight path')
        
        # Determine which images went to which split for THIS date
        # Based on the mixed seasonal strategy
        n = len(items_sorted)
        
        # Using the ratios from create_splits_with_buffers
        train_ratio = 0.6
        val_ratio = 0.18
        buffer_size = 10
        
        train_end = int(n * train_ratio)
        buffer1_end = min(train_end + buffer_size, n)
        val_end = min(buffer1_end + int(n * val_ratio), n)
        buffer2_end = min(val_end + buffer_size, n)
        
        # Train
        if train_end > 0:
            ax.scatter(lons[:train_end], lats[:train_end], 
                      c='blue', s=50, alpha=0.6, label=f'Train ({train_end})', zorder=3)
        
        # Buffer 1
        if buffer1_end > train_end:
            ax.scatter(lons[train_end:buffer1_end], lats[train_end:buffer1_end],
                      c='gray', s=50, alpha=0.6, marker='x', 
                      label=f'Buffer ({buffer1_end-train_end})', zorder=3)
        
        # Val
        if val_end > buffer1_end:
            ax.scatter(lons[buffer1_end:val_end], lats[buffer1_end:val_end],
                      c='green', s=50, alpha=0.6, label=f'Val ({val_end-buffer1_end})', zorder=3)
        
        # Buffer 2
        if buffer2_end > val_end:
            ax.scatter(lons[val_end:buffer2_end], lats[val_end:buffer2_end],
                      c='gray', s=50, alpha=0.6, marker='x', 
                      label=f'Buffer ({buffer2_end-val_end})', zorder=3)
        
        # Test
        if buffer2_end < n:
            ax.scatter(lons[buffer2_end:], lats[buffer2_end:],
                      c='red', s=50, alpha=0.6, label=f'Test ({n-buffer2_end})', zorder=3)
        
        # Add season annotation
        season = "Winter (March)" if "03/24" in date_str else "Summer (July)" if "07/28" in date_str else ""
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Flight Path - {date_str} ({season})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add sequence numbers to a few points for reference
        step = max(1, n // 10)
        for i in range(0, n, step):
            ax.annotate(f'{sequences[i]}', (lons[i], lats[i]), 
                       fontsize=8, alpha=0.5, xytext=(5, 5), 
                       textcoords='offset points')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(metadata_path).parent / "split_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Print statistics
    print(f"\n{'='*70}")
    print("STATISTICS:")
    print(f"{'='*70}")
    for split, stats in metadata['splits'].items():
        print(f"{split.capitalize():12s}: {stats['images']:3d} images → {stats['outputs']:4d} outputs")
    
    plt.show()


def print_split_details(metadata_path="dataset/processed/dataset_metadata.json"):
    """
    Print detailed information about the splits.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("\n" + "="*70)
    print("DATASET METADATA")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Patch size: {metadata['patch_size']}x{metadata['patch_size']}")
    print(f"  Stride: {metadata['stride']}")
    print(f"  Buffer size: {metadata['buffer_size']} images")
    
    print(f"\nDate Distribution (original images):")
    for date, count in metadata['date_distribution'].items():
        print(f"  {date}: {count} images")
    
    print(f"\nSplit Statistics:")
    total_images = sum(s['images'] for s in metadata['splits'].values())
    total_outputs = sum(s['outputs'] for s in metadata['splits'].values())
    
    for split, stats in metadata['splits'].items():
        img_pct = stats['images'] / total_images * 100
        patch_pct = stats['outputs'] / total_outputs * 100
        outputs_per_img = stats['outputs'] / stats['images'] if stats['images'] > 0 else 0
        
        print(f"\n  {split.capitalize()}:")
        print(f"    Images:  {stats['images']:3d} ({img_pct:5.1f}%)")
        print(f"    Patches: {stats['outputs']:4d} ({patch_pct:5.1f}%)")
        print(f"    Patches per image: {outputs_per_img:.1f}")
    
    print(f"\n  Total:")
    print(f"    Images:  {total_images}")
    print(f"    Patches: {total_outputs}")


if __name__ == "__main__":
    # # Print metadata
    # print_split_details()
    
    # # Create visualization
    # print("\nGenerating visualization...")
    # visualize_splits()

    
    print_split_details("dataset/processed_512_resized/dataset_metadata.json")
    visualize_splits("dataset/processed_512_resized/dataset_metadata.json", 
                     "dataset/raw")
