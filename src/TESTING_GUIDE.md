# Testing and Visualization Guide
## For CNN, Transformer, and Foundation Models

Complete guide for testing your trained models on test datasets and generating beautiful visualizations.

---

## 📁 **File Overview**

| File | Purpose |
|------|---------|
| `test_models.py` | Test single model, compute metrics, visualize predictions |
| `compare_models.py` | Compare multiple models side-by-side |
| `models_config_example.json` | Configuration file for model comparison |

---

## 🚀 **Quick Start**

### **1. Test Single Model**

Test one model and visualize results:

```bash
python test_models.py \
    --model sam \
    --variant vit_b \
    --checkpoint experiments/sam/best.pth \
    --test-data data/test \
    --image-size 384 \
    --output-dir results/sam_test
```

**Output:**
- `results/sam_test/test_results.json` - Metrics in JSON
- `results/sam_test/visualizations/` - Individual predictions
- `results/sam_test/visualizations/summary.png` - Grid overview

---

### **2. Compare Multiple Models**

Create `models_config.json`:
```json
{
  "models": [
    {
      "name": "U-Net",
      "model_type": "unet",
      "variant": null,
      "checkpoint": "experiments/unet/best.pth"
    },
    {
      "name": "SAM ViT-B",
      "model_type": "sam",
      "variant": "vit_b",
      "checkpoint": "experiments/sam/best.pth"
    },
    {
      "name": "DINOv2",
      "model_type": "dinov2",
      "variant": "vit_b",
      "checkpoint": "experiments/dinov2/best.pth"
    }
  ]
}
```

Run comparison:
```bash
python compare_models.py \
    --models-config models_config.json \
    --test-data data/test \
    --image-size 384 \
    --output-dir results/comparison
```

**Output:**
- `comparison_results.json` - All results
- `comparison_table.csv` - Metrics table
- `comparison_bars.png` - Bar chart
- `comparison_radar.png` - Radar chart
- `comparison_heatmap.png` - Heatmap
- `comparison_sample_*.png` - Side-by-side predictions

---

## 📊 **What You Get**

### **Single Model Testing**

#### **1. Metrics JSON** (`test_results.json`)
```json
{
  "model": "sam",
  "metrics": {
    "dice": 0.9234,
    "iou": 0.8567,
    "accuracy": 0.9812,
    "precision": 0.9345,
    "recall": 0.9123,
    "f1": 0.9234,
    "specificity": 0.9889,
    "TP": 1234567,
    "TN": 8765432,
    "FP": 12345,
    "FN": 23456
  }
}
```

#### **2. Individual Visualizations**

Each image gets a 2×2 grid:
- **Top Left:** Original image
- **Top Right:** Ground truth overlay (blue)
- **Bottom Left:** Prediction overlay (red)
- **Bottom Right:** Probability heatmap

#### **3. Summary Grid**

16-image overview with:
- Blue = Ground truth
- Red = Prediction
- Purple = Correct (overlap)

---

### **Multi-Model Comparison**

#### **1. Comparison Table** (CSV)

| Model | Type | Dice | IoU | Accuracy | Precision | Recall | F1 |
|-------|------|------|-----|----------|-----------|--------|-----|
| U-Net | unet | 0.8912 | 0.8034 | 0.9678 | 0.9123 | 0.8712 | 0.8912 |
| SAM ViT-B | sam | 0.9234 | 0.8567 | 0.9812 | 0.9345 | 0.9123 | 0.9234 |
| DINOv2 | dinov2 | 0.9156 | 0.8445 | 0.9756 | 0.9267 | 0.9045 | 0.9156 |

#### **2. Bar Chart**

Grouped bars for each metric across all models.

#### **3. Radar Chart**

Multi-dimensional view of model performance.

#### **4. Heatmap**

Color-coded metric matrix.

#### **5. Side-by-Side Visualizations**

For each sample image:
- Original + Ground Truth
- Each model's prediction
- Error maps (TP=green, FP=red, FN=blue)

---

## 🎨 **Visualization Examples**

### **Color Scheme:**

| Color | Meaning |
|-------|---------|
| Blue | Ground Truth |
| Red | Prediction |
| Purple | Correct (Overlap) |
| Green | True Positive |
| Red (error map) | False Positive |
| Blue (error map) | False Negative |
| Black (error map) | True Negative |

---

## 💻 **Detailed Usage**

### **Test Single Model (`test_models.py`)**

#### **Required Arguments:**
```bash
--model            # Model type: unet, sam, dinov2, swin-unet
--checkpoint       # Path to best.pth file
--test-data        # Path to test data directory
```

#### **Optional Arguments:**
```bash
--varient          # Model variant: vit_b, vit_l, vit_h (for transformers)
--image-size       # Image size (default: 512)
--batch-size       # Batch size (default: 4)
--num-workers      # Data loading workers (default: 4)
--threshold        # Prediction threshold (default: 0.5)
--compute-loss     # Also compute loss
--loss             # Loss function (default: combined)
--visualize        # Generate visualizations (default: True)
--max-viz          # Max images to visualize (default: 50)
--output-dir       # Output directory (default: test_results)
```

#### **Full Example:**
```bash
python test_models.py \
    --model sam \
    --varient vit_b \
    --checkpoint experiments/sam_vit_b/best.pth \
    --test-data data/test \
    --image-size 384 \
    --batch-size 8 \
    --num-workers 4 \
    --threshold 0.5 \
    --compute-loss \
    --loss combined \
    --visualize \
    --max-viz 100 \
    --output-dir results/sam_vit_b_test
```

---

### **Compare Models (`compare_models.py`)**

#### **Required Arguments:**
```bash
--models-config    # Path to models configuration JSON
--test-data        # Path to test data directory
```

#### **Optional Arguments:**
```bash
--image-size       # Image size (default: 512)
--batch-size       # Batch size (default: 4)
--num-workers      # Data loading workers (default: 4)
--threshold        # Prediction threshold (default: 0.5)
--visualize        # Generate visualizations (default: True)
--num-viz-samples  # Number of samples to compare (default: 10)
--output-dir       # Output directory (default: comparison_results)
```

#### **Full Example:**
```bash
python compare_models.py \
    --models-config my_models.json \
    --test-data data/test \
    --image-size 384 \
    --batch-size 8 \
    --num-workers 4 \
    --threshold 0.5 \
    --visualize \
    --num-viz-samples 20 \
    --output-dir results/full_comparison
```

---

## 📝 **Creating Models Configuration File**

### **Template:**
```json
{
  "models": [
    {
      "name": "Display Name",
      "model_type": "model_identifier",
      "variant": "variant_name_or_null",
      "checkpoint": "path/to/best.pth",
      "description": "Optional description"
    }
  ]
}
```

### **Model Types:**

| Model Type | Variants | Example |
|------------|----------|---------|
| `unet` | None | `"variant": null` |
| `sam` | `vit_b`, `vit_l`, `vit_h` | `"variant": "vit_b"` |
| `dinov2` | `vit_s`, `vit_b`, `vit_l`, `vit_g` | `"variant": "vit_b"` |
| `swin-unet` | `tiny`, `small`, `base` | `"variant": "small"` |

### **Complete Example:**
```json
{
  "models": [
    {
      "name": "Baseline U-Net",
      "model_type": "unet",
      "variant": null,
      "checkpoint": "experiments/unet/best.pth",
      "description": "Standard U-Net with ResNet34 encoder"
    },
    {
      "name": "Swin-UNet (Tiny)",
      "model_type": "swin-unet",
      "variant": "tiny",
      "checkpoint": "experiments/swin_unet_tiny/best.pth",
      "description": "Swin Transformer U-Net - Tiny variant"
    },
    {
      "name": "SAM Base",
      "model_type": "sam",
      "variant": "vit_b",
      "checkpoint": "experiments/sam_base/best.pth",
      "description": "SAM with ViT-B backbone"
    },
    {
      "name": "SAM Large",
      "model_type": "sam",
      "variant": "vit_l",
      "checkpoint": "experiments/sam_large/best.pth",
      "description": "SAM with ViT-L backbone - Best performer"
    },
    {
      "name": "DINOv2 Base",
      "model_type": "dinov2",
      "variant": "vit_b",
      "checkpoint": "experiments/dinov2_base/best.pth",
      "description": "DINOv2 self-supervised ViT-B"
    }
  ]
}
```

---

## 🪟 **Windows Quick Commands**

### **Test Single Model:**
```cmd
python test_models.py --model sam --varient vit_b --checkpoint experiments\sam\best.pth --test-data data\test --output-dir results\sam_test
```

### **Compare Models:**
```cmd
python compare_models.py --models-config models_config.json --test-data data\test --output-dir results\comparison
```

### **View Results:**
```cmd
REM Open results folder
explorer results\comparison

REM View metrics
type results\comparison\comparison_results.json
```

---

## 🐧 **Linux/Mac Quick Commands**

### **Test Single Model:**
```bash
python test_models.py \
    --model sam \
    --varient vit_b \
    --checkpoint experiments/sam/best.pth \
    --test-data data/test \
    --output-dir results/sam_test
```

### **Compare Models:**
```bash
python compare_models.py \
    --models-config models_config.json \
    --test-data data/test \
    --output-dir results/comparison
```

### **View Results:**
```bash
# Open results folder
xdg-open results/comparison  # Linux
open results/comparison      # Mac

# View metrics
cat results/comparison/comparison_results.json
```

---

## 📈 **Metrics Explained**

| Metric | Formula | Range | Best | Description |
|--------|---------|-------|------|-------------|
| **Dice** | 2TP/(2TP+FP+FN) | 0-1 | 1.0 | Overlap similarity |
| **IoU** | TP/(TP+FP+FN) | 0-1 | 1.0 | Jaccard index |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 0-1 | 1.0 | Pixel accuracy |
| **Precision** | TP/(TP+FP) | 0-1 | 1.0 | Positive predictive value |
| **Recall** | TP/(TP+FN) | 0-1 | 1.0 | Sensitivity |
| **F1** | 2×P×R/(P+R) | 0-1 | 1.0 | Harmonic mean of P&R |
| **Specificity** | TN/(TN+FP) | 0-1 | 1.0 | True negative rate |

**Legend:**
- TP = True Positives (correct water pixels)
- TN = True Negatives (correct non-water pixels)
- FP = False Positives (predicted water, actually non-water)
- FN = False Negatives (predicted non-water, actually water)

---

## 💡 **Tips & Best Practices**

### **1. Test Set Preparation**
```
data/test/
├── images/
│   ├── test_001.jpg
│   ├── test_002.jpg
│   └── ...
└── masks/
    ├── test_001.png
    ├── test_002.png
    └── ...
```

### **2. Image Size Consistency**
- Use same `--image-size` as training
- Model expects specific size (check your training config)
- Common sizes: 256, 384, 512

### **3. Threshold Tuning**
- Default: 0.5
- Higher (0.6-0.7): More precision, less recall
- Lower (0.3-0.4): More recall, less precision
- Test multiple: `for t in 0.3 0.4 0.5 0.6 0.7; do python test_models.py ... --threshold $t; done`

### **4. Memory Optimization**
- Reduce `--batch-size` if OOM
- Reduce `--image-size` if needed
- Reduce `--max-viz` for less disk usage

### **5. Batch Processing**
Test all models at once:
```bash
# Bash
for model in unet sam dinov2; do
    python test_models.py \
        --model $model \
        --checkpoint experiments/$model/best.pth \
        --test-data data/test \
        --output-dir results/${model}_test
done
```

```cmd
REM Windows
for %%m in (unet sam dinov2) do (
    python test_models.py --model %%m --checkpoint experiments\%%m\best.pth --test-data data\test --output-dir results\%%m_test
)
```

---

## 🔧 **Troubleshooting**

### **Issue: Model fails to load**
```
Error: Unexpected key in state_dict
```

**Solution:** Check model type and variant match checkpoint:
```bash
# List checkpoint contents
python -c "import torch; print(torch.load('best.pth').keys())"
```

---

### **Issue: CUDA out of memory**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Reduce batch size
--batch-size 1

# Reduce image size
--image-size 256

# Both
--batch-size 1 --image-size 256
```

---

### **Issue: No visualizations generated**
```
Warning: No visualizations found
```

**Solution:**
```bash
# Enable visualization
--visualize

# Check max viz isn't 0
--max-viz 50
```

---

### **Issue: Test data not found**
```
FileNotFoundError: data/test not found
```

**Solution:** Check directory structure:
```bash
# Verify path
ls data/test/images
ls data/test/masks

# Use absolute path
--test-data /full/path/to/data/test
```

---

## 📋 **Complete Workflow Example**

### **Step 1: Train Models**
```bash
# Train multiple models
python train.py --model unet --data-root data/train
python train.py --model sam --varient vit_b --data-root data/train
python train.py --model dinov2 --varient vit_b --data-root data/train
```

### **Step 2: Create Config**
Create `models_to_compare.json`:
```json
{
  "models": [
    {
      "name": "U-Net",
      "model_type": "unet",
      "variant": null,
      "checkpoint": "experiments/unet/best.pth"
    },
    {
      "name": "SAM",
      "model_type": "sam",
      "variant": "vit_b",
      "checkpoint": "experiments/sam/best.pth"
    },
    {
      "name": "DINOv2",
      "model_type": "dinov2",
      "variant": "vit_b",
      "checkpoint": "experiments/dinov2/best.pth"
    }
  ]
}
```

### **Step 3: Compare**
```bash
python compare_models.py \
    --models-config models_to_compare.json \
    --test-data data/test \
    --image-size 384 \
    --output-dir results/final_comparison \
    --num-viz-samples 20
```

### **Step 4: Analyze**
```bash
# View metrics
cat results/final_comparison/comparison_results.json

# Open visualizations
xdg-open results/final_comparison/comparisons/

# Check table
cat results/final_comparison/comparison_table.csv
```

---

## ✅ **Summary**

**For Single Model:**
```bash
python test_models.py \
    --model MODEL_TYPE \
    --checkpoint path/to/best.pth \
    --test-data data/test
```

**For Multiple Models:**
```bash
python compare_models.py \
    --models-config config.json \
    --test-data data/test
```

**You Get:**
- ✅ Quantitative metrics (Dice, IoU, etc.)
- ✅ Beautiful visualizations
- ✅ Comparison charts
- ✅ Side-by-side predictions
- ✅ Error analysis
- ✅ Publication-ready figures

**All saved to organized output directories!** 🎉
