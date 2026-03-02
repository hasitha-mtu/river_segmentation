import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("./24GB_results/results/test_results.csv")

# # # Sort by Dice (descending)
# df = df.sort_values("dice", ascending=False)

# families = sorted(df["family"].unique())

# # Different markers per family (no explicit color setting)
# marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "*"]
# family_marker = {fam: marker_cycle[i % len(marker_cycle)] for i, fam in enumerate(families)}

# # Annotate only top 2 Dice models per family (reduce clutter)
# top_models = (
#     df.groupby("family")
#       .apply(lambda x: x.nlargest(2, "dice"))
#       .reset_index(drop=True)
# )

# def scatter_by_family(x, y, xlabel, ylabel, filename):
#     plt.figure(figsize=(7, 5))
    
#     for fam in families:
#         sub = df[df["family"] == fam]
#         plt.scatter(
#             sub[x], sub[y],
#             marker=family_marker[fam],
#             label=fam,
#             alpha=0.9
#         )

#     # Annotate selected models only
#     for _, r in top_models.iterrows():
#         plt.annotate(
#             r["model"],
#             (r[x], r[y]),
#             textcoords="offset points",
#             xytext=(4, 4),
#             fontsize=8
#         )

#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True, alpha=0.3)
#     plt.legend(title="Family", fontsize=9)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=600)
#     plt.close()


# # 1. Dice vs Model Parameters
# scatter_by_family(
#     x="params_M",
#     y="dice",
#     xlabel="Model Parameters (M)",
#     ylabel="Dice",
#     filename="./24GB_results/results/scatter_dice_vs_params.png"
# )

# # 2. Dice vs Inference Time
# scatter_by_family(
#     x="inference_ms",
#     y="dice",
#     xlabel="Inference Time (ms)",
#     ylabel="Dice",
#     filename="./24GB_results/results/scatter_dice_vs_inference.png"
# )

# # 3. Boundary F1 vs Dice
# scatter_by_family(
#     x="dice",
#     y="boundary_f1",
#     xlabel="Dice",
#     ylabel="Boundary F1",
#     filename="./24GB_results/results/scatter_boundary_vs_dice.png"
# )

# # 4. Precision vs Recall
# scatter_by_family(
#     x="recall",
#     y="precision",
#     xlabel="Recall",
#     ylabel="Precision",
#     filename="./24GB_results/results/scatter_precision_vs_recall.png"
# )

# # Load the dataset
# df = pd.read_csv('test_results.csv')

# Set the visual style
sns.set_theme(style="whitegrid")

plt.rcParams['font.family'] = 'serif'

# Create a figure with 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Dice Score vs Model Parameters (Model Complexity)
# We use a log scale for parameters because SAM models are significantly larger than others
sns.scatterplot(ax=axes[0, 0], data=df, x='params_M', y='dice', hue='family', s=120, alpha=0.8)
axes[0, 0].set_title('Model Complexity vs. Segmentation Performance', y=-0.25, fontsize=14, pad=20)
axes[0, 0].set_xlabel('Parameters (Millions) - Log Scale', fontsize=12)
axes[0, 0].set_ylabel('Dice Score (F1)', fontsize=12)
axes[0, 0].set_xscale('log')

# Plot 2: Dice Score vs Inference Time (Latency)
sns.scatterplot(ax=axes[0, 1], data=df, x='inference_ms', y='dice', hue='family', s=120, alpha=0.8)
axes[0, 1].set_title('Inference Latency vs. Performance', y=-0.25, fontsize=14, pad=20)
axes[0, 1].set_xlabel('Wall-clock Inference Time (ms)', fontsize=12)
axes[0, 1].set_ylabel('Dice Score (F1)', fontsize=12)

# Plot 3: Boundary F1 vs Dice Score (Contour Accuracy vs Region Accuracy)
sns.scatterplot(ax=axes[1, 0], data=df, x='dice', y='boundary_f1', hue='family', s=120, alpha=0.8)
axes[1, 0].set_title('Boundary-F1 vs. Region Dice Score', y=-0.25, fontsize=14, pad=20)
axes[1, 0].set_xlabel('Dice Score (Region Overlap)', fontsize=12)
axes[1, 0].set_ylabel('Boundary F1 (Contour Alignment)', fontsize=12)

# Plot 4: Precision vs Recall (Model Bias)
sns.scatterplot(ax=axes[1, 1], data=df, x='recall', y='precision', hue='family', s=120, alpha=0.8)
axes[1, 1].set_title('Precision-Recall Tradeoff', y=-0.25, fontsize=14, pad=20)
axes[1, 1].set_xlabel('Recall (Sensitivity)', fontsize=12)
axes[1, 1].set_ylabel('Precision (Positive Predictive Value)', fontsize=12)

# Global layout adjustments
# plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.2)

# Save the plot
plt.savefig('./24GB_results/results/segmentation_analysis1.png', dpi=1000)

print("Plot successfully generated and saved as 'segmentation_analysis.png'")




# Load the dataset
df = pd.read_csv("./24GB_results/results/test_results.csv")

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

# Create a figure with increased height to accommodate bottom titles
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Function to add subplot with title below
def plot_data(ax, x_col, y_col, x_label, y_label, title_text, use_log=False):
    sns.scatterplot(ax=ax, data=df, x=x_col, y=y_col, hue='family', s=120, alpha=0.8)
    if use_log:
        ax.set_xscale('log')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    # y=-0.25 positions the title below the x-axis labels
    ax.set_title(title_text, y=-0.3, fontsize=14,  pad=20)

# Plot 1: Complexity
plot_data(axes[0, 0], 'params_M', 'dice', 'Parameters (Millions)', 'Dice Score', 
          '(a) Model Complexity vs. Performance', use_log=True)

# Plot 2: Latency
plot_data(axes[0, 1], 'inference_ms', 'dice', 'Wall-clock Inference Time (ms)', 'Dice Score', 
          '(b) Inference Latency vs. Performance')

# Plot 3: Boundary-F1
plot_data(axes[1, 0], 'dice', 'boundary_f1', 'Dice Score', 'Boundary F1', 
          '(c) Boundary-F1 vs. Dice Score')

# Plot 4: Precision-Recall
plot_data(axes[1, 1], 'recall', 'precision', 'Recall', 'Precision', 
          '(d) Precision-Recall Balance')

# Adjust vertical spacing (hspace) so the top row's titles don't overlap the bottom plots
plt.subplots_adjust(hspace=0.5, wspace=0.2)

# Save the finalized image
plt.savefig('./24GB_results/results/segmentation_analysis.png', dpi=1000, bbox_inches='tight')