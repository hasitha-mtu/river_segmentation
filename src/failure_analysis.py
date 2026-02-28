"""
failure_analysis.py
===================
Failure case analysis and pixel class distribution for UAV river segmentation.

What this script does
---------------------
1. PIXEL CLASS DISTRIBUTION
   Scans all masks in every split (train / val / test) and computes:
     - Per-image water pixel ratio
     - Per-split and overall class balance
     - Histogram of coverage ratios
     - Images ranked by coverage (most / least water)

2. COVERAGE–PERFORMANCE GRADIENT
   Bins test images by ground-truth water coverage ratio (from
   test_results_per_image.csv) and computes mean Dice per bin per model,
   revealing how performance degrades as visible water area shrinks.

3. IMAGE-LEVEL FAILURE IDENTIFICATION
   For each test image, measures how many models fail (Dice < threshold)
   and computes a consensus difficulty score (mean Dice across all models).
   Identifies the hardest images and what they share in terms of coverage.

4. COMPUTABLE IMAGE STATISTICS
   For every test image, computes directly measurable properties from the
   raw pixels — no manual labels, no assumptions about scene content:

   (a) Luminance (mean, std, min, max of greyscale pixel values 0–255)
       — low mean = dark scene overall; high std = high contrast
   (b) Green excess index  ExG = 2G − R − B  (per Woebbecke et al., 1995)
       — positive values indicate green-dominant (vegetation) pixels
       — mean ExG over the image is a proxy for canopy density
   (c) HSV saturation (mean, std)
       — high saturation = vivid colour (water glint, strong reflections)
       — low saturation = grey/washed-out areas (shadow, overcast)
   (d) HSV value / brightness (mean of V channel)
       — distinct from luminance: low V means dark pixels regardless of hue
   (e) Laplacian variance (edge density / texture complexity)
       — high value = busy texture (e.g., rippled water, cluttered canopy)
       — low value = smooth uniform regions
   (f) Prediction confidence (mean sigmoid probability, from CSV)
       — model's own estimate of how much of the image is water

   Pearson correlations between each image statistic and per-image Dice
   are computed across all models and reported.

5. CROSS-MODEL CONSENSUS
   For each pair of failure thresholds (Dice < 0.5, < 0.1, == 0.0), counts
   how many models fail per image and lists images sorted by failure count.

Outputs (written to --output_dir):
   class_distribution.csv           — per-image coverage stats for all splits
   class_distribution_summary.txt   — human-readable class balance table
   coverage_performance.csv         — mean Dice per coverage bin per model
   image_stats.csv                  — computable image statistics per test image
   image_stats_correlations.csv     — Pearson r between image stats and Dice
   failure_summary.csv              — per-image failure counts across all models
   failure_hardest.txt              — hardest images ranked by consensus Dice
   *.png figures (if matplotlib installed)

Usage
-----
    python failure_analysis.py \\
        --data_root       ./dataset/processed_512_resized \\
        --per_image_csv   ./results/test_results_per_image.csv \\
        --output_dir      ./results/failure_analysis

    # Adjust coverage bins explicitly:
    python failure_analysis.py \\
        --coverage_bins 0 0.002 0.01 0.03 0.10 1.0

Notes
-----
* Requires only: numpy, Pillow, scipy, matplotlib (optional)
* Does NOT require model checkpoints or GPU — works entirely on saved CSVs
  and the raw image/mask files.
* All statistics are computed directly from pixels; no scene labelling is
  performed or assumed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# ── Optional deps ────────────────────────────────────────────────────────────
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy not found — Pearson correlations will use numpy fallback.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not found — figures will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pearson_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Pearson correlation coefficient and two-tailed p-value.
    Returns (r, p). Uses scipy if available, otherwise numpy.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float('nan'), float('nan')
    if SCIPY_AVAILABLE:
        r, p = scipy_stats.pearsonr(x, y)
        return float(r), float(p)
    # numpy fallback (no p-value)
    r = float(np.corrcoef(x, y)[0, 1])
    return r, float('nan')


def load_mask_as_binary(path: Path) -> np.ndarray:
    """Load a mask PNG/JPG as a binary 0/1 uint8 array."""
    img = Image.open(path).convert('L')
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as an (H, W, 3) uint8 RGB array."""
    return np.array(Image.open(path).convert('RGB'))


def compute_image_statistics(rgb: np.ndarray) -> dict:
    """
    Compute directly measurable pixel statistics from an RGB image.
    All values are computed from raw pixel data — no scene assumptions.

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3), dtype uint8, values 0-255

    Returns
    -------
    dict with keys described in module docstring.
    """
    r_ch = rgb[:, :, 0].astype(np.float32)
    g_ch = rgb[:, :, 1].astype(np.float32)
    b_ch = rgb[:, :, 2].astype(np.float32)

    # ── (a) Luminance (greyscale approximation, ITU-R BT.601) ────────────────
    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    lum_mean = float(lum.mean())
    lum_std  = float(lum.std())
    lum_min  = float(lum.min())
    lum_max  = float(lum.max())

    # ── (b) Excess green index ExG = 2G − R − B  (Woebbecke et al. 1995) ────
    #   Normalise channels to [0, 1] first so ExG is scale-independent.
    r_n = r_ch / 255.0
    g_n = g_ch / 255.0
    b_n = b_ch / 255.0
    exg = 2.0 * g_n - r_n - b_n          # range: [-2, 2]
    exg_mean = float(exg.mean())          # >0 indicates green-dominant scene
    exg_positive_frac = float((exg > 0).mean())  # fraction of pixels with positive ExG

    # ── (c,d) HSV saturation and value ───────────────────────────────────────
    # Convert RGB [0,255] to HSV [0,1] manually to avoid cv2 dependency.
    r_n2 = r_ch / 255.0
    g_n2 = g_ch / 255.0
    b_n2 = b_ch / 255.0

    cmax = np.maximum(np.maximum(r_n2, g_n2), b_n2)
    cmin = np.minimum(np.minimum(r_n2, g_n2), b_n2)
    delta = cmax - cmin

    # V channel (brightness)
    v_channel = cmax
    v_mean = float(v_channel.mean())
    v_std  = float(v_channel.std())

    # S channel (saturation): 0 when cmax == 0 (black pixel)
    s_channel = np.where(cmax > 0, delta / (cmax + 1e-9), 0.0)
    s_mean = float(s_channel.mean())
    s_std  = float(s_channel.std())

    # High-saturation fraction: proxy for specularly vivid regions
    high_sat_frac = float((s_channel > 0.5).mean())

    # Low-value fraction: proxy for dark (shadow / heavily occluded) pixels
    low_val_frac = float((v_channel < 0.25).mean())

    # ── (e) Laplacian variance (edge density / texture complexity) ───────────
    # Use a simple 3×3 discrete Laplacian on the luminance channel.
    # Pads with reflected border to avoid edge artefacts.
    lum_norm = lum / 255.0
    # Manual 3x3 Laplacian convolution without scipy/cv2
    kernel = np.array([[0,  1, 0],
                        [1, -4, 1],
                        [0,  1, 0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(lum_norm, 1, mode='reflect')
    windows = sliding_window_view(padded, (3, 3))  # shape: (H, W, 3, 3)
    laplacian = (windows * kernel).sum(axis=(-2, -1))
    laplacian_var = float(laplacian.var())

    return {
        'lum_mean'        : round(lum_mean, 4),
        'lum_std'         : round(lum_std, 4),
        'lum_min'         : round(lum_min, 4),
        'lum_max'         : round(lum_max, 4),
        'exg_mean'        : round(exg_mean, 4),
        'exg_positive_frac': round(exg_positive_frac, 4),
        'hsv_s_mean'      : round(s_mean, 4),
        'hsv_s_std'       : round(s_std, 4),
        'hsv_v_mean'      : round(v_mean, 4),
        'hsv_v_std'       : round(v_std, 4),
        'high_sat_frac'   : round(high_sat_frac, 4),
        'low_val_frac'    : round(low_val_frac, 4),
        'laplacian_var'   : round(laplacian_var, 6),
    }


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PIXEL CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def analyse_class_distribution(
    data_root: Path,
    splits: list[str],
    output_dir: Path,
) -> list[dict]:
    """
    Scan all mask files across splits and compute per-image and per-split
    water pixel ratios. Saves CSV and summary text. Returns per-image records.
    """
    print("\n" + "="*70)
    print("1. PIXEL CLASS DISTRIBUTION")
    print("="*70)

    all_records: list[dict] = []
    split_stats: dict[str, dict] = {}

    for split in splits:
        mask_dir = data_root / split / 'masks'
        if not mask_dir.exists():
            print(f"  [SKIP] {split}/masks not found: {mask_dir}")
            continue

        mask_paths = sorted(mask_dir.glob('*.png')) + sorted(mask_dir.glob('*.jpg'))
        if not mask_paths:
            print(f"  [SKIP] No mask files in {mask_dir}")
            continue

        print(f"\n  Split: {split}  ({len(mask_paths)} images)")

        water_pixels = 0
        total_pixels = 0
        ratios = []

        for mp in mask_paths:
            mask = load_mask_as_binary(mp)
            n_water = int(mask.sum())
            n_total = mask.size
            ratio = n_water / n_total

            water_pixels += n_water
            total_pixels += n_total
            ratios.append(ratio)

            all_records.append({
                'split'       : split,
                'image'       : mp.stem,
                'n_pixels'    : n_total,
                'water_pixels': n_water,
                'bg_pixels'   : n_total - n_water,
                'water_ratio' : round(ratio, 6),
                'bg_ratio'    : round(1.0 - ratio, 6),
            })

        ratios = np.array(ratios)
        split_stats[split] = {
            'n_images'       : len(mask_paths),
            'total_pixels'   : total_pixels,
            'water_pixels'   : water_pixels,
            'bg_pixels'      : total_pixels - water_pixels,
            'water_pct'      : 100.0 * water_pixels / total_pixels,
            'bg_pct'         : 100.0 * (total_pixels - water_pixels) / total_pixels,
            'ratio_mean'     : float(ratios.mean()),
            'ratio_std'      : float(ratios.std()),
            'ratio_median'   : float(np.median(ratios)),
            'ratio_min'      : float(ratios.min()),
            'ratio_max'      : float(ratios.max()),
            'n_zero_water'   : int((ratios == 0.0).sum()),
            'n_lt_1pct'      : int((ratios < 0.01).sum()),
            'n_lt_2pct'      : int((ratios < 0.02).sum()),
            'n_gt_10pct'     : int((ratios > 0.10).sum()),
        }

    if not all_records:
        print("  [ERROR] No mask files found in any split.")
        return all_records

    # ── Compute overall across all splits ────────────────────────────────────
    total_w = sum(s['water_pixels'] for s in split_stats.values())
    total_b = sum(s['bg_pixels'] for s in split_stats.values())
    total_p = total_w + total_b
    all_ratios = np.array([r['water_ratio'] for r in all_records])

    split_stats['ALL'] = {
        'n_images'     : len(all_records),
        'total_pixels' : total_p,
        'water_pixels' : total_w,
        'bg_pixels'    : total_b,
        'water_pct'    : 100.0 * total_w / total_p,
        'bg_pct'       : 100.0 * total_b / total_p,
        'ratio_mean'   : float(all_ratios.mean()),
        'ratio_std'    : float(all_ratios.std()),
        'ratio_median' : float(np.median(all_ratios)),
        'ratio_min'    : float(all_ratios.min()),
        'ratio_max'    : float(all_ratios.max()),
        'n_zero_water' : int((all_ratios == 0.0).sum()),
        'n_lt_1pct'    : int((all_ratios < 0.01).sum()),
        'n_lt_2pct'    : int((all_ratios < 0.02).sum()),
        'n_gt_10pct'   : int((all_ratios > 0.10).sum()),
    }

    # ── Summary text ─────────────────────────────────────────────────────────
    lines = [
        "PIXEL CLASS DISTRIBUTION SUMMARY",
        "="*70,
        f"{'Split':<8} {'N':>5} {'Water%':>8} {'Bg%':>8} "
        f"{'Mean_r':>8} {'Std_r':>8} {'Median_r':>9} "
        f"{'Min_r':>8} {'Max_r':>8} {'n<1%':>6} {'n<2%':>6} {'n>10%':>7}",
        "-"*70,
    ]
    for sp, st in split_stats.items():
        lines.append(
            f"{sp:<8} {st['n_images']:>5} {st['water_pct']:>7.3f}% "
            f"{st['bg_pct']:>7.3f}% "
            f"{st['ratio_mean']:>8.4f} {st['ratio_std']:>8.4f} "
            f"{st['ratio_median']:>9.4f} "
            f"{st['ratio_min']:>8.4f} {st['ratio_max']:>8.4f} "
            f"{st['n_lt_1pct']:>6} {st['n_lt_2pct']:>6} {st['n_gt_10pct']:>7}"
        )
    lines += [
        "-"*70,
        "",
        "Coverage distribution across ALL images:",
    ]
    # Percentile table
    pcts = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    pct_vals = np.percentile(all_ratios, pcts)
    for p, v in zip(pcts, pct_vals):
        lines.append(f"  p{p:>3d}: {v:.6f}  ({v*100:.3f}% water)")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    out_txt = output_dir / 'class_distribution_summary.txt'
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(summary_text, encoding='utf-8')
    print(f"\n  Saved: {out_txt}")

    save_csv(all_records, output_dir / 'class_distribution.csv')

    # ── Figure: coverage histogram per split ─────────────────────────────────
    if MATPLOTLIB_AVAILABLE:
        _plot_coverage_histogram(all_records, splits, output_dir)

    return all_records


def _plot_coverage_histogram(
    records: list[dict],
    splits: list[str],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(splits) + 1,
                              figsize=(4 * (len(splits) + 1), 4),
                              sharey=False)
    colours = {'train': '#2196F3', 'val': '#FF9800', 'test': '#4CAF50'}
    edges = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0])
    labels = ['0–0.5%', '0.5–1%', '1–2%', '2–5%', '5–10%', '10–20%', '>20%']

    all_ratios_by_split: dict[str, list] = defaultdict(list)
    for r in records:
        all_ratios_by_split[r['split']].append(r['water_ratio'])

    for ax_i, split in enumerate(splits):
        if split not in all_ratios_by_split:
            continue
        ratios = np.array(all_ratios_by_split[split])
        counts, _ = np.histogram(ratios, bins=edges)
        ax = axes[ax_i]
        ax.bar(range(len(labels)), counts,
               color=colours.get(split, '#9E9E9E'), edgecolor='white', linewidth=0.5)
        ax.set_title(f'{split.capitalize()} (n={len(ratios)})', fontsize=11)
        ax.set_xlabel('Water coverage bin', fontsize=9)
        ax.set_ylabel('Number of images', fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        for bar, cnt in zip(ax.patches, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(cnt), ha='center', va='bottom', fontsize=8)

    # All-split overlay
    ax_all = axes[-1]
    all_ratios = np.array([r['water_ratio'] for r in records])
    for sp in splits:
        if sp not in all_ratios_by_split:
            continue
        r = np.array(all_ratios_by_split[sp])
        ax_all.hist(r, bins=edges,
                    label=sp, color=colours.get(sp, '#9E9E9E'),
                    alpha=0.65, edgecolor='white', linewidth=0.5)
    ax_all.set_title('All splits combined', fontsize=11)
    ax_all.set_xlabel('Water coverage ratio', fontsize=9)
    ax_all.set_ylabel('Number of images', fontsize=9)
    ax_all.set_xticks(range(len(labels)))
    ax_all.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax_all.legend(fontsize=8)

    fig.suptitle('Pixel Class Distribution — Water Coverage per Image', fontsize=13, y=1.02)
    fig.tight_layout()
    out = output_dir / 'fig_class_distribution.png'
    fig.savefig(out, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. COVERAGE–PERFORMANCE GRADIENT
# ─────────────────────────────────────────────────────────────────────────────

def analyse_coverage_performance(
    per_image_df: list[dict],
    coverage_bins: list[float],
    output_dir: Path,
) -> None:
    """
    Bin test images by ground-truth water coverage ratio and compute
    mean / median / std Dice per bin per model.
    """
    print("\n" + "="*70)
    print("2. COVERAGE–PERFORMANCE GRADIENT")
    print("="*70)

    # Group per-image records by model
    models: list[str] = sorted(set(r['model'] for r in per_image_df))
    bins = np.array(coverage_bins)
    bin_labels = [
        f"{bins[i]*100:.2g}–{bins[i+1]*100:.2g}%"
        for i in range(len(bins) - 1)
    ]

    # Build bin index for each record
    for rec in per_image_df:
        gt = float(rec['gt_ratio'])
        idx = int(np.searchsorted(bins[1:], gt, side='right'))
        idx = min(idx, len(bin_labels) - 1)
        rec['_bin_idx'] = idx
        rec['_bin_label'] = bin_labels[idx]

    # Count images per bin (model-independent — use one model to avoid duplication)
    bin_image_counts: dict[int, int] = defaultdict(int)
    first_model = models[0]
    for rec in per_image_df:
        if rec['model'] == first_model:
            bin_image_counts[rec['_bin_idx']] += 1

    output_rows: list[dict] = []
    print(f"\n  {'Bin':<14} {'N':>4}  ", end='')
    for m in models:
        short = m[:14]
        print(f"  {short:<14}", end='')
    print()
    print("  " + "-" * (20 + 16 * len(models)))

    for bin_idx, bin_label in enumerate(bin_labels):
        n_imgs = bin_image_counts.get(bin_idx, 0)
        row = {'bin': bin_label, 'n_images': n_imgs}
        print(f"  {bin_label:<14} {n_imgs:>4}  ", end='')
        for model in models:
            dices = [float(r['dice'])
                     for r in per_image_df
                     if r['model'] == model and r['_bin_idx'] == bin_idx]
            if dices:
                mean_d  = np.mean(dices)
                std_d   = np.std(dices)
                med_d   = np.median(dices)
                n_fail  = sum(1 for d in dices if d < 0.5)
            else:
                mean_d = std_d = med_d = float('nan')
                n_fail = 0
            row[f'{model}_mean'] = round(mean_d, 4) if np.isfinite(mean_d) else ''
            row[f'{model}_std']  = round(std_d,  4) if np.isfinite(std_d)  else ''
            row[f'{model}_n_fail'] = n_fail
            print(f"  {mean_d:>6.4f}±{std_d:>5.4f}", end='') if np.isfinite(mean_d) \
                else print(f"  {'—':^14}", end='')
        print()
        output_rows.append(row)

    save_csv(output_rows, output_dir / 'coverage_performance.csv')

    if MATPLOTLIB_AVAILABLE:
        _plot_coverage_performance(per_image_df, models, bin_labels, bin_image_counts, output_dir)


def _plot_coverage_performance(
    per_image_df, models, bin_labels, bin_image_counts, output_dir
):
    # Family colours matching the paper's convention
    family_map = {
        'deeplabv3plus': 'CNN Baseline',
        'deeplabv3plus_cbam': 'CNN Baseline',
        'unetpp': 'CNN Baseline',
        'resunetpp': 'CNN Baseline',
        'unet': 'CNN Baseline',
        'hrnet_ocr_w18': 'Hybrid SOTA',
        'hrnet_ocr_w32': 'Hybrid SOTA',
        'hrnet_ocr_w48': 'Hybrid SOTA',
        'convnext_upernet_base': 'Hybrid SOTA',
        'convnext_upernet_small': 'Hybrid SOTA',
        'convnext_upernet_tiny': 'Hybrid SOTA',
        'segformer_b2': 'Transformer',
        'segformer_b0': 'Transformer',
        'swin_unet_tiny': 'Transformer',
        'sam_vit_b': 'Foundation',
        'sam_vit_l': 'Foundation',
        'sam_vit_h': 'Foundation',
        'dinov2_vit_b': 'Foundation',
        'dinov2_vit_s': 'Foundation',
        'dinov2_vit_l': 'Foundation',
    }
    family_colours = {
        'CNN Baseline': '#2196F3',
        'Hybrid SOTA' : '#FF9800',
        'Transformer' : '#9C27B0',
        'Foundation'  : '#F44336',
    }

    x = np.arange(len(bin_labels))
    fig, ax = plt.subplots(figsize=(12, 5))

    plotted_families = set()
    for model in models:
        means = []
        for bin_idx in range(len(bin_labels)):
            dices = [float(r['dice'])
                     for r in per_image_df
                     if r['model'] == model and r['_bin_idx'] == bin_idx]
            means.append(np.mean(dices) if dices else float('nan'))

        family = family_map.get(model, 'Other')
        colour = family_colours.get(family, '#9E9E9E')
        label  = family if family not in plotted_families else None
        plotted_families.add(family)
        ax.plot(x, means, marker='o', markersize=4, linewidth=1.2,
                color=colour, alpha=0.7, label=label)

    # Secondary axis: image count per bin
    ax2 = ax.twinx()
    counts = [bin_image_counts.get(i, 0) for i in range(len(bin_labels))]
    ax2.bar(x, counts, alpha=0.15, color='grey', label='n images')
    ax2.set_ylabel('Number of test images per bin', fontsize=9, color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')

    ax.set_xlabel('Ground-truth water coverage bin', fontsize=11)
    ax.set_ylabel('Mean Dice (per bin)', fontsize=11)
    ax.set_title('Coverage–Performance Gradient across Architecture Families', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='Dice = 0.5 threshold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    fig.tight_layout()
    out = output_dir / 'fig_coverage_performance.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CROSS-MODEL FAILURE IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def analyse_failures(
    per_image_df: list[dict],
    thresholds: list[float],
    output_dir: Path,
) -> None:
    """
    For each test image, count how many models fail at each Dice threshold.
    Rank images by consensus difficulty (mean Dice across all models).
    """
    print("\n" + "="*70)
    print("3. CROSS-MODEL FAILURE IDENTIFICATION")
    print("="*70)

    models = sorted(set(r['model'] for r in per_image_df))
    images = sorted(set(r['image'] for r in per_image_df))
    n_models = len(models)

    # Build image × model dice matrix
    dice_matrix: dict[str, dict[str, float]] = {
        img: {} for img in images
    }
    gt_ratios: dict[str, float] = {}
    for rec in per_image_df:
        dice_matrix[rec['image']][rec['model']] = float(rec['dice'])
        if rec['image'] not in gt_ratios:
            gt_ratios[rec['image']] = float(rec['gt_ratio'])

    rows = []
    for img in images:
        model_dices = [dice_matrix[img].get(m, float('nan')) for m in models]
        valid_dices = [d for d in model_dices if np.isfinite(d)]
        consensus_dice = float(np.mean(valid_dices)) if valid_dices else float('nan')
        consensus_std  = float(np.std(valid_dices))  if len(valid_dices) > 1 else float('nan')
        row = {
            'image'          : img,
            'gt_ratio'       : round(gt_ratios.get(img, float('nan')), 6),
            'consensus_dice' : round(consensus_dice, 4),
            'consensus_std'  : round(consensus_std, 4),
            'n_models_evaluated': len(valid_dices),
        }
        for thr in thresholds:
            n_fail = sum(1 for d in valid_dices if d < thr)
            row[f'n_fail_lt{int(thr*100):03d}'] = n_fail
            row[f'frac_fail_lt{int(thr*100):03d}'] = round(n_fail / n_models, 4) if n_models > 0 else float('nan')
        rows.append(row)

    rows.sort(key=lambda r: r['consensus_dice'])
    save_csv(rows, output_dir / 'failure_summary.csv')

    # ── Text report ───────────────────────────────────────────────────────────
    lines = [
        "FAILURE SUMMARY — Images ranked by consensus Dice (hardest first)",
        f"Total test images: {len(images)} | Models: {n_models}",
        "="*80,
    ]
    header = (f"{'Image':<42} {'gt_ratio':>9} {'Mean_Dice':>9} "
              f"{'Std':>6}")
    for thr in thresholds:
        header += f"  {'<'+str(int(thr*100))+'%_fail':>10}"
    lines.append(header)
    lines.append("-"*80)

    # Print top-20 hardest
    for row in rows[:20]:
        line = (f"{row['image']:<42} "
                f"{row['gt_ratio']:>9.6f} "
                f"{row['consensus_dice']:>9.4f} "
                f"{row['consensus_std']:>6.4f}")
        for thr in thresholds:
            key = f'n_fail_lt{int(thr*100):03d}'
            line += f"  {row[key]:>4}/{n_models:<4}"
        lines.append(line)

    lines += [
        "",
        "FAILURE THRESHOLD SUMMARY (number of images where ≥N models fail):",
        "-"*60,
    ]
    for thr in thresholds:
        key = f'n_fail_lt{int(thr*100):03d}'
        for min_models in [1, 5, 10, 15, n_models]:
            count = sum(1 for r in rows if r[key] >= min_models)
            lines.append(
                f"  Dice < {thr:.2f}: {count:>3} images have "
                f">= {min_models:>2}/{n_models} models failing"
            )
        lines.append("")

    # Coverage breakdown for hardest quartile
    n_hard = max(1, len(rows) // 4)
    hard_ratios = [r['gt_ratio'] for r in rows[:n_hard] if np.isfinite(r['gt_ratio'])]
    easy_ratios = [r['gt_ratio'] for r in rows[-n_hard:] if np.isfinite(r['gt_ratio'])]
    if hard_ratios and easy_ratios:
        lines += [
            f"COVERAGE COMPARISON: Hardest vs Easiest quartile ({n_hard} images each)",
            "-"*60,
            f"  Hardest quartile — mean gt_ratio: {np.mean(hard_ratios):.5f}  "
            f"median: {np.median(hard_ratios):.5f}",
            f"  Easiest quartile — mean gt_ratio: {np.mean(easy_ratios):.5f}  "
            f"median: {np.median(easy_ratios):.5f}",
        ]

    report = "\n".join(lines)
    print("\n" + report)

    out_txt = output_dir / 'failure_hardest.txt'
    out_txt.write_text(report, encoding='utf-8')
    print(f"\n  Saved: {out_txt}")

    if MATPLOTLIB_AVAILABLE:
        _plot_failure_heatmap(rows, models, dice_matrix, output_dir)


def _plot_failure_heatmap(rows, models, dice_matrix, output_dir):
    """Heatmap of per-image × per-model Dice scores (hardest images first)."""
    images_sorted = [r['image'] for r in rows]  # already sorted hardest first
    # Limit to 64 images for readability
    imgs_display = images_sorted[:64]

    matrix = np.zeros((len(imgs_display), len(models)))
    for i, img in enumerate(imgs_display):
        for j, m in enumerate(models):
            matrix[i, j] = dice_matrix[img].get(m, float('nan'))

    fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.7), max(8, len(imgs_display) * 0.15)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0.0, vmax=1.0, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Dice')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(len(imgs_display)))
    ax.set_yticklabels([img[:30] for img in imgs_display], fontsize=6)
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('Test image (hardest → easiest, top → bottom)', fontsize=10)
    ax.set_title('Per-image × Per-model Dice Heatmap', fontsize=12)

    fig.tight_layout()
    out = output_dir / 'fig_failure_heatmap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPUTABLE IMAGE STATISTICS + CORRELATION WITH DICE
# ─────────────────────────────────────────────────────────────────────────────

def analyse_image_statistics(
    data_root: Path,
    per_image_df: list[dict],
    output_dir: Path,
) -> None:
    """
    Compute directly measurable pixel statistics for every test image.
    Correlate each statistic with per-image Dice across all models.
    """
    print("\n" + "="*70)
    print("4. COMPUTABLE IMAGE STATISTICS")
    print("="*70)

    img_dir = data_root / 'test' / 'images'
    if not img_dir.exists():
        print(f"  [SKIP] Test image directory not found: {img_dir}")
        return

    img_paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    if not img_paths:
        print(f"  [SKIP] No images in {img_dir}")
        return

    print(f"\n  Computing statistics for {len(img_paths)} test images…")

    # Build lookup: image filename stem → list of (model, dice, pred_prob, gt_ratio)
    per_image_lookup: dict[str, list[dict]] = defaultdict(list)
    for rec in per_image_df:
        stem = Path(rec['image']).stem
        per_image_lookup[stem].append(rec)

    stat_rows: list[dict] = []

    for img_path in img_paths:
        rgb = load_image_rgb(img_path)
        stats = compute_image_statistics(rgb)

        # Attach per-model dice values and gt_ratio
        records = per_image_lookup.get(img_path.stem, [])
        gt_ratio = float(records[0]['gt_ratio']) if records else float('nan')

        row = {
            'image'   : img_path.name,
            'gt_ratio': round(gt_ratio, 6),
            **stats,
        }
        for rec in records:
            row[f'dice_{rec["model"]}']      = float(rec['dice'])
            row[f'pred_prob_{rec["model"]}'] = float(rec['pred_prob'])

        stat_rows.append(row)

    save_csv(stat_rows, output_dir / 'image_stats.csv')

    # ── Correlations ─────────────────────────────────────────────────────────
    print("\n  Computing correlations between image statistics and Dice…")

    stat_keys = [
        'gt_ratio', 'lum_mean', 'lum_std', 'exg_mean', 'exg_positive_frac',
        'hsv_s_mean', 'hsv_s_std', 'hsv_v_mean', 'hsv_v_std',
        'high_sat_frac', 'low_val_frac', 'laplacian_var',
    ]
    models = sorted(set(r['model'] for r in per_image_df))

    corr_rows: list[dict] = []
    for stat in stat_keys:
        x_vals = []
        for row in stat_rows:
            x_vals.append(row.get(stat, float('nan')))
        x_arr = np.array(x_vals, dtype=float)

        corr_row: dict = {'statistic': stat}
        all_r: list[float] = []

        for model in models:
            y_arr = np.array([row.get(f'dice_{model}', float('nan'))
                               for row in stat_rows], dtype=float)
            r, p = pearson_r(x_arr, y_arr)
            corr_row[f'r_{model}'] = round(r, 4) if np.isfinite(r) else ''
            corr_row[f'p_{model}'] = round(p, 4) if np.isfinite(p) else ''
            if np.isfinite(r):
                all_r.append(r)

        corr_row['mean_r_across_models'] = round(float(np.mean(all_r)), 4) if all_r else ''
        corr_row['std_r_across_models']  = round(float(np.std(all_r)),  4) if all_r else ''
        corr_rows.append(corr_row)

    save_csv(corr_rows, output_dir / 'image_stats_correlations.csv')

    # Print summary table
    lines = [
        "\n  CORRELATION SUMMARY (Pearson r with per-image Dice, averaged across models)",
        f"  {'Statistic':<26} {'Mean r':>8}  {'Std r':>7}  {'Interpretation'}",
        "  " + "-"*75,
    ]
    stat_descriptions = {
        'gt_ratio'         : 'Ground-truth water fraction (primary driver)',
        'lum_mean'         : 'Mean luminance (image brightness)',
        'lum_std'          : 'Luminance std dev (contrast)',
        'exg_mean'         : 'Mean ExG index (vegetation density proxy)',
        'exg_positive_frac': 'Fraction of pixels with ExG > 0',
        'hsv_s_mean'       : 'Mean HSV saturation',
        'hsv_s_std'        : 'HSV saturation std dev',
        'hsv_v_mean'       : 'Mean HSV value (brightness)',
        'hsv_v_std'        : 'HSV value std dev',
        'high_sat_frac'    : 'Fraction of high-saturation pixels (S > 0.5)',
        'low_val_frac'     : 'Fraction of dark pixels (V < 0.25)',
        'laplacian_var'    : 'Laplacian variance (texture/edge complexity)',
    }
    for row in sorted(corr_rows,
                       key=lambda r: -abs(float(r['mean_r_across_models']) or 0)):
        mean_r = row['mean_r_across_models']
        std_r  = row['std_r_across_models']
        desc   = stat_descriptions.get(row['statistic'], '')
        lines.append(f"  {row['statistic']:<26} {mean_r:>8}  {std_r:>7}  {desc}")
    print("\n".join(lines))

    if MATPLOTLIB_AVAILABLE:
        _plot_stat_correlations(corr_rows, stat_descriptions, output_dir)
        _plot_coverage_scatter(stat_rows, models[:5], output_dir)


def _plot_stat_correlations(corr_rows, stat_descriptions, output_dir):
    """Bar chart of mean Pearson r per image statistic."""
    stats_sorted = sorted(
        corr_rows,
        key=lambda r: float(r['mean_r_across_models'] or 0)
    )
    labels = [r['statistic'] for r in stats_sorted]
    mean_rs = [float(r['mean_r_across_models'] or 0) for r in stats_sorted]
    std_rs  = [float(r['std_r_across_models']  or 0) for r in stats_sorted]

    colours = ['#F44336' if v < 0 else '#2196F3' for v in mean_rs]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(labels)), mean_rs, xerr=std_rs,
                   color=colours, alpha=0.8, error_kw={'linewidth': 1.0, 'capsize': 3})
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Mean Pearson r with per-image Dice (across all models)", fontsize=10)
    ax.set_title("Image Statistics vs. Segmentation Dice — Correlation Summary", fontsize=12)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    fig.tight_layout()
    out = output_dir / 'fig_stat_correlations.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {out}")


def _plot_coverage_scatter(stat_rows, models, output_dir):
    """Scatter: gt_ratio vs Dice for selected models."""
    n = min(len(models), 5)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    colours = ['#2196F3', '#FF9800', '#9C27B0', '#F44336', '#4CAF50']
    for ax, model, colour in zip(axes, models[:n], colours):
        xs = [row['gt_ratio'] for row in stat_rows]
        ys = [row.get(f'dice_{model}', float('nan')) for row in stat_rows]
        xs_arr = np.array(xs, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        valid = np.isfinite(xs_arr) & np.isfinite(ys_arr)
        ax.scatter(xs_arr[valid], ys_arr[valid],
                   color=colour, alpha=0.6, s=30, edgecolors='white', linewidths=0.3)
        # Trend line
        if valid.sum() > 2:
            m_, b_ = np.polyfit(xs_arr[valid], ys_arr[valid], 1)
            xfit = np.linspace(xs_arr[valid].min(), xs_arr[valid].max(), 50)
            ax.plot(xfit, m_ * xfit + b_, color='black', linewidth=1.2, linestyle='--')
            r, _ = pearson_r(xs_arr[valid], ys_arr[valid])
            ax.text(0.05, 0.05, f'r = {r:.3f}', transform=ax.transAxes,
                    fontsize=9, color='black')
        ax.set_title(model, fontsize=9)
        ax.set_xlabel('gt_ratio', fontsize=8)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('Dice', fontsize=10)
    fig.suptitle('Ground-truth Coverage vs. Dice (test images)', fontsize=12)
    fig.tight_layout()
    out = output_dir / 'fig_coverage_scatter.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Failure case analysis and pixel class distribution
            for UAV river segmentation results.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--data_root',
        default='./dataset/processed_512_resized',
        help='Root directory containing train/val/test splits with images/ and masks/ subdirs.',
    )
    p.add_argument(
        '--per_image_csv',
        default='./24GB_results/results/test_results_per_image.csv',
        help='Path to test_results_per_image.csv from evaluate_models.py.',
    )
    p.add_argument(
        '--output_dir',
        default='./24GB_results/results/failure_analysis',
        help='Directory for output CSVs, text reports, and figures.',
    )
    p.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to include in class distribution analysis.',
    )
    p.add_argument(
        '--coverage_bins',
        nargs='+',
        type=float,
        default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0],
        help=(
            'Bin edges (as fractions 0–1) for coverage–performance gradient. '
            'Example: 0 0.005 0.01 0.02 0.05 0.10 0.20 1.0'
        ),
    )
    p.add_argument(
        '--failure_thresholds',
        nargs='+',
        type=float,
        default=[0.5, 0.1, 0.0],
        help='Dice thresholds used to define "failure" in cross-model analysis.',
    )
    p.add_argument(
        '--skip_image_stats',
        action='store_true',
        help='Skip per-image pixel statistic computation (faster, no figures).',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*70)
    print('FAILURE CASE ANALYSIS & PIXEL CLASS DISTRIBUTION')
    print('='*70)
    print(f'Data root       : {args.data_root}')
    print(f'Per-image CSV   : {args.per_image_csv}')
    print(f'Output dir      : {output_dir}')

    # ── Load per-image CSV ────────────────────────────────────────────────────
    per_image_csv = Path(args.per_image_csv)
    if not per_image_csv.exists():
        print(f'\n[ERROR] Per-image CSV not found: {per_image_csv}')
        print('Run evaluate_models.py first to generate this file.')
        sys.exit(1)

    per_image_df: list[dict] = []
    with open(per_image_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            per_image_df.append(row)

    n_imgs   = len(set(r['image'] for r in per_image_df))
    n_models = len(set(r['model'] for r in per_image_df))
    print(f'\nLoaded per-image CSV: {len(per_image_df)} rows '
          f'({n_imgs} images × {n_models} models)')

    data_root = Path(args.data_root)

    # ── Run analyses ──────────────────────────────────────────────────────────
    analyse_class_distribution(
        data_root  = data_root,
        splits     = args.splits,
        output_dir = output_dir,
    )

    analyse_coverage_performance(
        per_image_df   = per_image_df,
        coverage_bins  = sorted(args.coverage_bins),
        output_dir     = output_dir,
    )

    analyse_failures(
        per_image_df = per_image_df,
        thresholds   = args.failure_thresholds,
        output_dir   = output_dir,
    )

    if not args.skip_image_stats:
        analyse_image_statistics(
            data_root    = data_root,
            per_image_df = per_image_df,
            output_dir   = output_dir,
        )

    print('\n' + '='*70)
    print('Analysis complete.')
    print(f'All outputs saved to: {output_dir}')
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
