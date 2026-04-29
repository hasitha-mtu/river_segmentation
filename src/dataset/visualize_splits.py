import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import re

plt.rcParams['font.family'] = 'serif'

BIN_EDGES  = [0.0, 0.001, 0.005, 0.05, 1.01]
BIN_LABELS = ['Ultra-low\n(<0.1%)', 'Low\n(0.1–0.5%)', 'Medium\n(0.5–5%)', 'High\n(>5%)']
SPLIT_COLORS = {'train': '#4C72B0', 'val': '#55A868', 'test': '#C44E52'}

def visualize_splits(metadata_path="dataset/processed_512_resized/dataset_metadata.json", 
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
        n = len(items_sorted)
        
        # Extract GPS coordinates
        lats = [item['gps'][0] for item in items_sorted]
        lons = [item['gps'][1] for item in items_sorted]
        sequences = [item['sequence'] for item in items_sorted]
        
        # Plot the flight path
        ax.plot(lons, lats, 'k-', alpha=0.3, linewidth=1, label='Flight path')
        
        # Build lookup: filename → split assignment
        # This reads directly from the saved gt_ratio_records in metadata,
        # so it is always accurate regardless of what split strategy was used.
        filename_to_split = {}
        for split_name in ['train', 'val', 'test']:
            for record in metadata.get('splits', {}).get(split_name, {}).get('gt_ratio_records', []):
                # Records store filename with extension; strip it for matching
                fname = Path(record['filename']).stem
                filename_to_split[fname] = split_name

        # Draw each image coloured by its actual split assignment.
        # Images not found in metadata (discarded buffers) are coloured gray.
        label_style = {
            'train':  ('#4C72B0', 'o', 0.45, 40),
            'val':    ('#55A868', 'o', 0.55, 40),
            'test':   ('#C44E52', 'o', 0.85, 60),
            'buffer': ('gray',   'x', 0.30, 25),
        }
        counts = {k: 0 for k in label_style}

        for item in items_sorted:
            fname  = Path(item['filename']).stem
            label  = filename_to_split.get(fname, 'buffer')
            counts[label] += 1
            col, marker, alpha, size = label_style[label]
            ax.scatter(item['gps'][1], item['gps'][0],
                       c=col, s=size, alpha=alpha, marker=marker,
                       label='_nolegend_', zorder=3)

        # Add one legend entry per split showing final counts
        for label, (col, marker, alpha, size) in label_style.items():
            if counts[label] > 0:
                ax.scatter([], [], c=col, s=size, alpha=min(alpha+0.2, 1.0),
                           marker=marker,
                           label=f"{label.capitalize()} ({counts[label]})")
        
        # Add season annotation
        season = "Winter (March)" if "03/24" in date_str else "Summer (July)" if "07/28" in date_str else ""
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        # ax.set_title(f'Flight Path - {date_str} ({season})', fontsize=14, fontweight='bold')
        ax.set_title(f'Flight Path - {date_str} ({season})', y=-0.3, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add sequence numbers to a few points for reference
        step = max(1, n // 10)
        for i in range(0, n, step):
            ax.annotate(f'{sequences[i]}', (lons[i], lats[i]), 
                       fontsize=8, alpha=0.5, xytext=(5, 5), 
                       textcoords='offset points')
    
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    
    # Save figure
    output_path = Path(metadata_path).parent / "split_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Print statistics
    print(f"\n{'='*70}")
    print("STATISTICS:")
    print(f"{'='*70}")
    for split, stats in metadata['splits'].items():
        print(f"{split.capitalize():12s}: {stats['images']:3d} images")
    
    plt.show()


def print_split_details(metadata_path="dataset/processed_512_resized/dataset_metadata.json"):
    """
    Print detailed information about the resized dataset splits.
    Works with the metadata format produced by create_resized_dataset.py.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print("\n" + "="*70)
    print("DATASET METADATA  (resized dataset)")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Mode:          {metadata.get('mode', 'resize')}")
    print(f"  Target size:   {metadata.get('target_size', '?')}×{metadata.get('target_size', '?')}")
    print(f"  Buffer size:   {metadata.get('buffer_size', '?')} images per window boundary")
    print(f"  Split strategy: {metadata.get('split_strategy', 'sequential')}")
    if metadata.get('n_windows'):
        print(f"  N windows:     {metadata['n_windows']}")

    print(f"\nDate Distribution (original images):")
    for date, count in metadata['date_distribution'].items():
        print(f"  {date}: {count} images")

    print(f"\nSplit Statistics:")
    total_images = sum(s['images'] for s in metadata['splits'].values())

    for split, stats in metadata['splits'].items():
        img_pct = stats['images'] / total_images * 100 if total_images else 0

        # gt_ratio summary if available
        records = stats.get('gt_ratio_records', [])
        gt_info = ""
        if records:
            import numpy as np
            gts = [r['gt_ratio'] for r in records]
            gt_info = (f"  mean gt={np.mean(gts)*100:.2f}%  "
                       f"max gt={np.max(gts)*100:.2f}%")

        print(f"\n  {split.capitalize()}:")
        print(f"    Images:  {stats['images']:3d} ({img_pct:5.1f}%)")
        if stats.get('skipped', 0):
            print(f"    Skipped: {stats['skipped']}")
        if gt_info:
            print(f"    gt_ratio:{gt_info}")

    print(f"\n  Total images: {total_images}")


def visualize_gt_ratio_distribution(metadata_path="dataset/processed_512_resized/dataset_metadata.json"):
    """
    Plot gt_ratio distributions across train/val/test splits.

    Reads gt_ratio_records saved by create_resized_dataset.py and produces:
      - Histogram per split (log-scale x-axis to show ultra-low tail)
      - Grouped bar chart of bin percentages (the key reviewer diagnostic)
      - Empirical CDF overlay

    This figure should be included in the revision response as evidence that
    the new stratified split resolves the distribution mismatch (R1 Major #1).
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    splits = ['train', 'val', 'test']
    ratios = {}
    for split in splits:
        records = metadata.get('splits', {}).get(split, {}).get('gt_ratio_records', [])
        if not records:
            print(f"  WARNING: no gt_ratio_records found for {split}. "
                  f"Re-run create_resized_dataset.py to regenerate metadata.")
            return
        ratios[split] = np.array([r['gt_ratio'] for r in records])

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                            top=0.90, bottom=0.08, left=0.07, right=0.97)

    log_bins = np.logspace(np.log10(5e-4), np.log10(0.25), 30)

    # Row 1: histograms per split
    for col, split in enumerate(splits):
        ax = fig.add_subplot(gs[0, col])
        r  = ratios[split]
        ax.hist(r, bins=log_bins, color=SPLIT_COLORS[split],
                edgecolor='white', linewidth=0.4, alpha=0.88)
        ax.set_xscale('log')
        ax.set_title(f'{split.capitalize()}  (n={len(r)})', fontweight='bold')
        ax.set_xlabel('gt_ratio (log scale)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.2, which='both')
        # reviewer threshold lines
        for thr, lbl, col_name in [(0.001,'0.1%','gray'),(0.005,'0.5%','darkorange'),(0.10,'10%','firebrick')]:
            ax.axvline(thr, color=col_name, linestyle='--', linewidth=1, alpha=0.75)
            ax.text(thr*1.1, ax.get_ylim()[1]*0.88, lbl, fontsize=7,
                    color=col_name, rotation=90, va='top')
        ax.axvline(r.max(), color=SPLIT_COLORS[split], linestyle=':', linewidth=1.3)

    # Row 2 left: grouped bin bar chart
    ax_bar = fig.add_subplot(gs[1, 0])
    x, w   = np.arange(len(BIN_LABELS)), 0.25
    for i, split in enumerate(splits):
        r    = ratios[split]
        pcts = [100 * np.mean((r >= BIN_EDGES[j]) & (r < BIN_EDGES[j+1]))
                for j in range(len(BIN_LABELS))]
        ax_bar.bar(x + (i-1)*w, pcts, w, color=SPLIT_COLORS[split],
                   label=split.capitalize(), edgecolor='white', linewidth=0.4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(BIN_LABELS, fontsize=8.5)
    ax_bar.set_ylabel('Percentage of split (%)')
    ax_bar.set_title('gt_ratio Bin Breakdown (Reviewer Diagnostic)', fontweight='bold')
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.2, axis='y')

    # Row 2 mid: CDF
    ax_cdf = fig.add_subplot(gs[1, 1])
    for split in splits:
        r   = np.sort(ratios[split])
        cdf = np.arange(1, len(r)+1) / len(r)
        ax_cdf.plot(r, cdf, color=SPLIT_COLORS[split], linewidth=2.2,
                    label=f'{split.capitalize()} (n={len(r)})')
    for thr, lbl in [(0.001,'0.1%'),(0.005,'0.5%'),(0.10,'10%')]:
        ax_cdf.axvline(thr, color='gray', linestyle=':', linewidth=0.9)
        ax_cdf.text(thr*1.1, 0.03, lbl, fontsize=7.5, color='gray',
                    rotation=90, va='bottom')
    ax_cdf.set_xscale('log')
    ax_cdf.set_xlabel('gt_ratio (log scale)')
    ax_cdf.set_ylabel('Cumulative fraction')
    ax_cdf.set_title('Empirical CDF', fontweight='bold')
    ax_cdf.legend(fontsize=9)
    ax_cdf.grid(True, alpha=0.2, which='both')

    # Row 2 right: stats table
    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis('off')
    row_labels = ['n', 'Mean%', 'Median%', 'Min%', 'Max%',
                  'Ultra-low', 'Low', 'Medium', 'High', '>10% cov.']
    table_data = [
        [str(len(ratios[s])) for s in splits],
        [f'{ratios[s].mean()*100:.2f}%'       for s in splits],
        [f'{np.median(ratios[s])*100:.2f}%'   for s in splits],
        [f'{ratios[s].min()*100:.4f}%'         for s in splits],
        [f'{ratios[s].max()*100:.2f}%'         for s in splits],
        [f'{np.mean(ratios[s]<0.001)*100:.0f}%' for s in splits],
        [f'{np.mean((ratios[s]>=0.001)&(ratios[s]<0.005))*100:.0f}%' for s in splits],
        [f'{np.mean((ratios[s]>=0.005)&(ratios[s]<0.05))*100:.0f}%'  for s in splits],
        [f'{np.mean(ratios[s]>=0.05)*100:.0f}%'  for s in splits],
        [f'{np.sum(ratios[s]>=0.10)}'             for s in splits],
    ]
    tbl = ax_tbl.table(cellText=table_data, rowLabels=row_labels,
                       colLabels=['Train','Val','Test'],
                       cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)
    for col, split in enumerate(splits):
        tbl[0, col].set_facecolor(SPLIT_COLORS[split])
        tbl[0, col].set_text_props(color='white', fontweight='bold')
    # Highlight test column for rows 6-10 (bin rows)
    for row in range(6, 11):
        tbl[row, 2].set_facecolor('#fde0dc')
    ax_tbl.set_title('Summary Statistics', fontweight='bold', pad=12)

    strategy = metadata.get('split_strategy', 'sequential')
    fig.suptitle(
        f'gt_ratio Distribution — {strategy.capitalize()} Split\n'
        f'(n_windows={metadata.get("n_windows","N/A")}, buffer={metadata.get("buffer_size","N/A")})',
        fontsize=12, fontweight='bold', y=0.97
    )

    out_path = Path(metadata_path).parent / "gt_ratio_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  gt_ratio distribution saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    choices  = ['sequential', 'stratified', 'alternative']
    RAW      = "dataset/raw"
    for choice in choices:
        METADATA = f"dataset/processed_512_resized/{choice}/dataset_metadata.json"

        print_split_details(METADATA)

        print("\nGenerating spatial split visualization...")
        visualize_splits(METADATA, RAW)

        print("\nGenerating gt_ratio distribution (reviewer diagnostic)...")
        visualize_gt_ratio_distribution(METADATA)
