import pandas as pd
import seaborn as sns

def create_test_results_per_image_plots(file_path, output_dir):
    # 1. Load the per-image results
    df = pd.read_csv(file_path)

    # 2. Define Model-to-Family Mapping for high-level grouping
    family_map = {
        'deeplabv3plus': 'CNN Baseline', 'deeplabv3plus_cbam': 'CNN Baseline', 
        'unet': 'CNN Baseline', 'unetpp': 'CNN Baseline', 'resunetpp': 'CNN Baseline',
        'hrnet_ocr_w18': 'Hybrid SOTA', 'hrnet_ocr_w32': 'Hybrid SOTA', 'hrnet_ocr_w48': 'Hybrid SOTA',
        'convnext_upernet_base': 'Hybrid SOTA', 'convnext_upernet_small': 'Hybrid SOTA', 'convnext_upernet_tiny': 'Hybrid SOTA',
        'segformer_b0': 'Transformer', 'segformer_b2': 'Transformer', 'swin_unet_tiny': 'Transformer',
        'dinov2_vit_b': 'Foundation', 'dinov2_vit_l': 'Foundation', 'dinov2_vit_s': 'Foundation',
        'sam_vit_b': 'Foundation', 'sam_vit_h': 'Foundation', 'sam_vit_l': 'Foundation',
        'sam_fpn_vit_b': 'Foundation', 'sam_fpn_vit_h': 'Foundation', 'sam_fpn_vit_l': 'Foundation'
    }
    df['Family'] = df['model'].map(family_map)

    # 3. Set the style for a journal-ready visualization
    sns.set_theme(style="whitegrid", font_scale=1.2, rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
    })

    # --- FIGURE A: Box Plot by Model Family ---
    plt.figure(figsize=(10, 6))
    # Sort families by median dice to show a clear performance hierarchy
    family_order = df.groupby('Family')['dice'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Family', y='dice', order=family_order, palette='viridis', width=0.6)

    # plt.title('Statistical Robustness: Dice Score Distribution by Model Family', pad=20)
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Model Family')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplot_family_dice.png', dpi=1500)

    # --- FIGURE B: Detailed Box Plot (All 23 Models Ranked) ---
    plt.figure(figsize=(16, 8))
    # Sort individual models by median dice for better comparison
    model_order = df.groupby('model')['dice'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='model', y='dice', order=model_order, palette='magma')

    plt.xticks(rotation=45, ha='right')
    # plt.title('Performance Consistency across All Benchmarked Models', pad=20)
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Model Architecture')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplot_all_models_dice.png', dpi=1000)

    print("Images saved as 'boxplot_family_dice.png' and 'boxplot_all_models_dice.png'.")


if __name__ == '__main__':
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        MATPLOTLIB_AVAILABLE = True
        plt.rcParams['font.family'] = 'serif'

        create_test_results_per_image_plots('C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results/test_results_per_image.csv', 
                                        'C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results')

    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        print("[WARN] matplotlib not found — figures will be skipped.")
    