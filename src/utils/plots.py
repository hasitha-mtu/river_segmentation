import pandas as pd
import seaborn as sns
import os
import diptest


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

# # --- PUBLICATION STYLE SETUP ---
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# def generate_failure_figure(csv_path, test_root, output_path):
#     # Load the wide-format evaluation data
#     df = pd.read_csv(csv_path)
    
#     # 1. Define the images to showcase
#     # Row 1: The 'Easy' Case (High visibility/coverage)
#     # Row 2: The 'Hard' Case (March failure cluster - Frame 0135)
#     selected_images = [
#         {'id': 'DJI_20250324094701_0122_V.jpg', 'label': 'Easy Case', 'ratio': '8.05%'},
#         {'id': 'DJI_20250324094710_0135_V.jpg', 'label': 'Hard Case', 'ratio': '0.07%'}
#     ]
    
#     # 2. Define Columns: Display Name -> Subfolder path
#     # 'generated_masks' is assumed to contain subfolders for each model
#     cols = [
#         ('Original', 'images'),
#         ('Ground Truth', 'masks'),
#         ('DeepLabV3+', 'generated_masks/deeplabv3plus'),
#         ('HRNet-W48', 'generated_masks/hrnet_ocr_w48'),
#         ('SAM-ViT-H', 'generated_masks/sam_vit_h'),
#         ('SAM-FPN-H', 'generated_masks/sam_fpn_vit_h')
#     ]

#     fig, axes = plt.subplots(len(selected_images), len(cols), figsize=(18, 7))

#     for r_idx, img_info in enumerate(selected_images):
#         jpg_name = img_info['id']
#         png_name = jpg_name.replace('.jpg', '.png')
        
#         for c_idx, (display_name, subfolder) in enumerate(cols):
#             ax = axes[r_idx, c_idx]
            
#             # Construct path: test/images/*.jpg OR test/masks/*.png OR test/generated_masks/model/*.png
#             filename = jpg_name if display_name == 'Original' else png_name
#             full_path = os.path.join(test_root, subfolder, filename)
            
#             try:
#                 # Load Image
#                 img = mpimg.imread(full_path)
#                 ax.imshow(img)
                
#                 # Add Dice Score Annotation for Model Predictions
#                 if 'generated_masks' in subfolder:
#                     # Extract folder name to match CSV column (e.g., 'deeplabv3plus')
#                     model_key = subfolder.split('/')[-1]
#                     dice_col = f"dice_{model_key}"
#                     dice_val = df.loc[df['image'] == jpg_name, dice_col].values[0]
                    
#                     color = 'lime' if dice_val > 0.5 else 'red'
#                     ax.text(0.95, 0.05, f"Dice: {dice_val:.2f}", transform=ax.transAxes,
#                             ha='right', va='bottom', color=color, fontsize=12,
#                             fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=2))
                            
#             except FileNotFoundError:
#                 ax.text(0.5, 0.5, "File Not Found", ha='center', va='center', color='gray', fontsize=10)
            
#             # Column Titles (Top row only)
#             if r_idx == 0:
#                 ax.set_title(display_name, fontsize=14, pad=12, fontweight='bold')
            
#             # Row Labels (First column only)
#             if c_idx == 0:
#                 ax.set_ylabel(f"{img_info['label']}\n($\hat{{r}}={img_info['ratio']}$)", 
#                               fontsize=13, labelpad=15, fontweight='bold')

#             # Clean up axes
#             ax.set_xticks([])
#             ax.set_yticks([])

#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0.02, hspace=0.05)
#     plt.savefig(output_path, dpi=600, bbox_inches='tight')
#     print(f"Publication-quality figure saved to: {output_path}")

def generate_failure_figure(csv_path, test_root, output_path):
    # 1. Setup Publication Style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    # Load the long-format evaluation data
    df = pd.read_csv(csv_path)
    
    # 2. Define the images to showcase
    selected_images = [
        {'id': 'DJI_20250324094701_0122_V.jpg', 'label': 'Easy Case', 'ratio': '8.05%'},
        {'id': 'DJI_20250324094710_0135_V.jpg', 'label': 'Hard Case', 'ratio': '0.07%'}
    ]
    
    # 3. Define Columns: Display Name -> Subfolder path
    # The subfolder name must match the 'model' name in your CSV
    cols = [
        ('Original', 'images'),
        ('Ground Truth', 'masks'),
        ('DeepLabV3+', 'generated_masks/deeplabv3plus'),
        ('HRNet-W48', 'generated_masks/hrnet_ocr_w48'),
        ('SAM-ViT-H', 'generated_masks/sam_vit_h'),
        ('SAM-FPN-H', 'generated_masks/sam_fpn_vit_h')
    ]

    fig, axes = plt.subplots(len(selected_images), len(cols), figsize=(18, 7))

    for r_idx, img_info in enumerate(selected_images):
        jpg_name = img_info['id']
        png_name = jpg_name.replace('.jpg', '.png')
        
        for c_idx, (display_name, subfolder) in enumerate(cols):
            ax = axes[r_idx, c_idx]
            
            # Construct file path
            filename = jpg_name if display_name == 'Original' else png_name
            full_path = os.path.join(test_root, subfolder, filename)
            
            try:
                # Load Image
                if os.path.exists(full_path):
                    img = mpimg.imread(full_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "File Missing", ha='center', va='center', color='gray')
                
                # --- FIXED: Handle Long-Format CSV lookup ---
                if 'generated_masks' in subfolder:
                    # Extract folder name to match 'model' column (e.g., 'deeplabv3plus')
                    model_key = subfolder.split('/')[-1]
                    
                    # Filter for specific image AND specific model
                    mask = (df['image'] == jpg_name) & (df['model'] == model_key)
                    row = df[mask]
                    
                    if not row.empty:
                        dice_val = row['dice'].values[0]
                        color = 'lime' if dice_val > 0.5 else 'red'
                        ax.text(0.95, 0.05, f"Dice: {dice_val:.2f}", transform=ax.transAxes,
                                ha='right', va='bottom', color=color, fontsize=12,
                                fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=2))
                            
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                ax.text(0.5, 0.5, "Error", ha='center', va='center', color='red')
            
            # Column Titles (Top row only)
            if r_idx == 0:
                ax.set_title(display_name, fontsize=14, pad=12, fontweight='bold')
            
            # Row Labels (First column only)
            # FIXED: Added 'r' prefix for raw string to handle LaTeX \hat
            if c_idx == 0:
                # We separate the Label from the LaTeX math to avoid parsing errors
                row_label = f"{img_info['label']}\n" + rf"($\hat{{r}}={img_info['ratio'].replace('%', r'\%')}$)"
                
                ax.set_ylabel(row_label, 
                              fontsize=13, 
                              labelpad=15, 
                              fontweight='bold',
                              multialignment='center')

            # Clean up axes
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    plt.savefig(output_path, dpi=2000, bbox_inches='tight')
    plt.close() # Close figure to free memory
    print(f"Publication-quality figure saved to: {output_path}")

# --- USER SETTINGS ---
# Set these paths to your local folders before running dataset\processed_512_resized\test\generated_masks
IMAGE_DIRECTORY = 'C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/dataset/processed_512_resized/test/images'
MASK_BASE_DIRECTORY = 'C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/dataset/processed_512_resized/test/generated_masks' # This folder should contain subfolders for each model
CSV_FILE = 'C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/dataset/processed_512_resized/test/test_results_per_image.csv'


def perform_segmentation_analysis(file_path):
    plt.rcParams['font.family'] = 'serif'
    # 1. Load your data
    df = pd.read_csv(file_path)
    
    # 2. Statistical Analysis: Hartigan's Dip Test per Model
    results = []
    unique_models = df['model'].unique()
    
    print(f"{'Model':<30} | {'Dip Stat':<10} | {'p-value':<10} | {'Result'}")
    print("-" * 75)
    
    for model_name in unique_models:
        # Filter scores for this specific model
        model_scores = df[df['model'] == model_name]['dice'].dropna().values
        
        # We need at least 3 points for a dip test
        if len(model_scores) > 5:
            dip_stat, p_val = diptest.diptest(model_scores)
            is_bimodal = "Bimodal (p<0.05)" if p_val < 0.05 else "Unimodal"
            
            results.append({
                'model': model_name,
                'dip_stat': dip_stat,
                'p_val': p_val,
                'classification': is_bimodal
            })
            print(f"{model_name:<30} | {dip_stat:<10.4f} | {p_val:<10.4f} | {is_bimodal}")

    # 3. Visualization: Faceted Histogram/KDE Plot
    # We'll pick 4 representative models to keep the plot clean
    # Or you can plot all by removing the 'representative_models' filter
    representative_models = ['convnext_upernet_base', 'deeplabv3_plus', 'unet_plus_plus', 'sam_fpn_vit_h']
    plot_df = df[df['model'].isin(representative_models)]
    
    sns.set_style("whitegrid")
    g = sns.FacetGrid(plot_df, col="model", col_wrap=2, height=4, aspect=1.5, sharey=False)
    
    # Map the histogram and KDE
    g.map_dataframe(sns.histplot, x="dice", bins=20, kde=True, color='royalblue', alpha=0.6)
    
    # Add failure threshold and labels
    for ax in g.axes.flat:
        ax.axvspan(0, 0.5, color='red', alpha=0.1, label='Failure Zone')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Dice Score")
        ax.set_ylabel("Image Frequency")
        
        # Annotate the modes
        ax.text(0.1, ax.get_ylim()[1]*0.7, "Failure Mode", color='red', fontsize=10, weight='bold')
        ax.text(0.75, ax.get_ylim()[1]*0.7, "Success Mode", color='green', fontsize=10, weight='bold')

    plt.subplots_adjust(top=0.9)
    # g.fig.suptitle("Dice Score Distribution: Bimodality & Performance Cliff Analysis", fontsize=16)
    
    plt.savefig('C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results/bimodal_distribution.png', 
                dpi=1500, 
                bbox_inches='tight')
    pd.DataFrame(results).to_csv('C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results/dip_test_results.csv', 
                                 index=False)


def plot_best_model_distributions(file_path):
    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # 2. Define the 'Best' models (User Selection)
    # Mapping CSV names to Display Names for the plot
    best_models = {
        'deeplabv3plus': 'DeepLabV3+ (CNN)',
        'segformer_b2': 'SegFormer (Transformer)',
        'convnext_upernet_base': 'ConvNeXt-Base (Hybrid)',
        'sam_fpn_vit_h': 'SAM-FPN (Foundation-Tuned)'
    }
    
    # Optional: Add a 'Vanilla' model for the "Argument" contrast
    contrast_model = {'sam_vit_b': 'Vanilla SAM (Bimodal)'}
    
    # 3. Setup the Figure
    sns.set_context("paper", font_scale=1.2)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    
    # Data provided by user for p-values
    p_values = {
        'deeplabv3plus': 0.3323,
        'segformer_b2': 0.9677,
        'convnext_upernet_base': 0.7390,
        'sam_fpn_vit_h': 0.6962
    }

    for i, (model_id, display_name) in enumerate(best_models.items()):
        data = df[df['model'] == model_id]['dice']
        
        # Plot Histogram + KDE
        # sns.histplot(data, bins=15, kde=True, ax=axes[i], color='#34495e', alpha=0.6)
        sns.histplot(data, bins=15, kde=True, ax=axes[i], color='royalblue', alpha=0.6)
        
        # Annotations
        axes[i].set_title(f"{display_name}\n$p_{{dip}} = {p_values[model_id]:.3f}$", fontsize=14)
        axes[i].set_xlim(0, 1)
        axes[i].set_xlabel("Dice Score")
        
        # Mark the "Performance Cliff"
        axes[i].axvspan(0, 0.5, color='#e74c3c', alpha=0.1)
        if i == 0:
            axes[i].set_ylabel("Frequency (Images)")
            axes[i].text(0.1, 1, "Failure Zone", color='#c0392b', fontsize=10, rotation=90)

    plt.tight_layout()
    plt.savefig('C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results/best_models_distribution.png', dpi=1500, bbox_inches='tight')
    print("Plot generated: best_models_distribution.png")
    

if __name__ == '__main__':
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import matplotlib.ticker as mticker
        MATPLOTLIB_AVAILABLE = True
        plt.rcParams['font.family'] = 'serif'

        # create_test_results_per_image_plots('C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results/test_results_per_image.csv', 
        #                                 'C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results')

        # generate_journal_qualitative_plot(CSV_FILE, 
        #                                   IMAGE_DIRECTORY, 
        #                                   MASK_BASE_DIRECTORY, 
        #                                   'C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results/figure_qualitative_results.png')

        # # Ensure 'test_results_per_image.csv' and the 'test/' directory are in your script folder
        # generate_failure_figure('C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results/test_results_per_image.csv', 
        #                         'C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/dataset/processed_512_resized/test', 
        #                         'C:/Users/AdikariAdikari/PycharmProjects/river_segmentation/24GB_results/results/figure_qualitative_results.png')
        

        perform_segmentation_analysis(CSV_FILE)
        plot_best_model_distributions(CSV_FILE)


    except ImportError: 
        MATPLOTLIB_AVAILABLE = False
        print("[WARN] matplotlib not found — figures will be skipped.")
    