# River Water Segmentation Models - PyTorch Implementation

A comprehensive PyTorch implementation of 5 state-of-the-art segmentation architectures for river water detection from UAV imagery. **Input: 512×512 RGB images → Output: Binary water masks**

## 🏗️ Implemented Models

- **U-Net**: Classic encoder-decoder with skip connections
- **U-Net++**: Nested U-Net with dense skip pathways  
- **ResUNet++**: Residual U-Net with ASPP and Squeeze-Excitation
- **DeepLabV3+**: State-of-the-art with ResNet50 backbone
- **DeepLabV3+ + CBAM**: Enhanced with attention and boundary loss

## 📋 Key Features

✅ **5 Model Architectures** - From lightweight U-Net to DeepLabV3+ with CBAM  
✅ **8 Loss Functions** - BCE, Dice, IoU, Focal, Boundary, Combined, Tversky, Combo  
✅ **CBAM Attention** - With residual connections for stability  
✅ **Boundary Loss** - Distance-weighted for precise water boundaries  
✅ **Comprehensive Metrics** - Dice, IoU, Precision, Recall, F1, Accuracy  
✅ **Data Augmentation** - Extensive pipeline for robust training  
✅ **TensorBoard Logging** - Real-time visualization  
✅ **Production Ready** - Checkpointing, resume, inference tools  

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Data Preparation
```
your_data/
├── images/
│   ├── img1.png        # RGB images (any size)
│   ├── img2.png
│   └── ...
└── masks/
    ├── img1.png        # Binary masks (0=background, 255=water)
    ├── img2.png
    └── ...
```

### Training

**Basic Training (U-Net)**
```bash
python train.py \
    --model unet \
    --data-root ./your_data \
    --batch-size 8 \
    --epochs 100 \
    --learning-rate 1e-4
```

**Best Configuration (DeepLabV3+ + CBAM + Boundary Loss)**
```bash
python train.py \
    --model deeplabv3plus_cbam \
    --data-root ./your_data \
    --loss combined \
    --use-boundary \
    --bce-weight 1.0 \
    --dice-weight 1.0 \
    --boundary-weight 1.0 \
    --pretrained \
    --batch-size 8 \
    --epochs 100 \
    --learning-rate 1e-4
```

**Resume Training**
```bash
python train.py \
    --model unet \
    --data-root ./your_data \
    --resume experiments/unet_combined_20241210/checkpoints/latest.pth
```

### Monitoring
```bash
tensorboard --logdir experiments/
```
Open http://localhost:6006 to view training progress.

### Testing & Evaluation

**Evaluate on Test Dataset**
```bash
python test.py \
    --checkpoint experiments/.../checkpoints/best.pth \
    --mode evaluate \
    --data-root ./test_data \
    --batch-size 8 \
    --output-dir ./evaluation_results
```

**Predict on New Images**
```bash
python test.py \
    --checkpoint experiments/.../checkpoints/best.pth \
    --mode predict \
    --image-dir ./new_images \
    --output-dir ./predictions \
    --visualize \
    --save-masks
```

**Single Image Prediction**
```bash
python test.py \
    --checkpoint experiments/.../checkpoints/best.pth \
    --mode single \
    --image-path ./image.png \
    --output-dir ./output \
    --visualize
```

## 📊 Model Comparison

Run comparison on your hardware:
```bash
python utils.py --mode compare
```

### Model Characteristics

| Model | Parameters | Speed | Memory | Best For |
|-------|-----------|-------|---------|----------|
| **U-Net** | ~31M | ★★★★★ | ★★★★★ | Fast inference, general use |
| **U-Net++** | ~36M | ★★★★☆ | ★★★★☆ | Better boundaries |
| **ResUNet++** | ~38M | ★★★☆☆ | ★★★★☆ | Complex scenes, fine details |
| **DeepLabV3+** | ~42M | ★★☆☆☆ | ★★★☆☆ | State-of-the-art accuracy |
| **DeepLabV3+ + CBAM** | ~45M | ★★☆☆☆ | ★★☆☆☆ | Forest canopy, attention |

## 🎯 Training Arguments

### Essential Arguments
```
--model              Model: unet, unetpp, resunetpp, deeplabv3plus, deeplabv3plus_cbam
--data-root          Path to data directory (with images/ and masks/ folders)
--batch-size         Batch size (default: 4)
--epochs             Number of epochs (default: 100)
--learning-rate      Learning rate (default: 1e-4)
```

### Loss Function Arguments
```
--loss               Loss: bce, dice, iou, focal, boundary, combined (default: combined)
--use-boundary       Enable boundary loss in combined loss
--bce-weight         BCE weight in combined loss (default: 1.0)
--dice-weight        Dice weight in combined loss (default: 1.0)
--boundary-weight    Boundary weight in combined loss (default: 1.0)
```

### Training Configuration
```
--pretrained         Use ImageNet pretrained weights (for DeepLabV3+ variants)
--image-size         Input size (default: 512)
--train-split        Train/val split ratio (default: 0.8)
--weight-decay       Weight decay (default: 1e-5)
--scheduler          LR scheduler: cosine, step, none (default: cosine)
--clip-grad          Gradient clipping (default: 1.0)
```

### System & Output
```
--num-workers        Data loading workers (default: 4)
--seed               Random seed (default: 42)
--output-dir         Output directory (default: ./experiments)
--resume             Resume from checkpoint
```

## 📈 Loss Functions Guide

| Loss | Best For | When to Use |
|------|----------|-------------|
| **Combined** | Best overall | Start here (BCE + Dice + Boundary) |
| **Dice** | Imbalanced data | Water is minority class |
| **BCE** | Balanced data | Roughly equal water/background |
| **Focal** | Severe imbalance | Very small water regions |
| **Boundary** | Edge precision | Need accurate boundaries |
| **IoU** | Tight segmentation | Minimize false positives |
| **Tversky** | Tune precision/recall | Adjust FP/FN trade-off |

**Recommendation**: Start with Combined loss (`--loss combined --use-boundary`)

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Option 1: Reduce batch size
--batch-size 4

# Option 2: Reduce image size
--image-size 256

# Option 3: Use U-Net instead of DeepLabV3+
--model unet
```

### Poor Performance
```bash
# 1. Verify data
# - Ensure masks are binary (0 and 255)
# - Check image-mask correspondence
# - Verify correct directory structure

# 2. Try different loss
--loss combined --use-boundary

# 3. Use pretrained weights (for DeepLabV3+ variants)
--pretrained

# 4. Adjust learning rate
--learning-rate 1e-5  # Lower for fine-tuning
--learning-rate 1e-3  # Higher for faster start
```

### NaN Loss
```bash
# Enable gradient clipping
--clip-grad 1.0

# Reduce learning rate
--learning-rate 1e-5

# Check for corrupted images/masks
```

## 📁 Project Structure

```
.
├── models.py          # All 5 model architectures
├── losses.py          # 8 loss functions including Boundary Loss
├── dataset.py         # Data loading and augmentation
├── metrics.py         # Evaluation metrics
├── train.py           # Training pipeline with TensorBoard
├── test.py            # Inference and evaluation
├── utils.py           # Model comparison and testing
├── quick_start.py     # Example code and tutorials
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 🎓 Usage Examples

### Example 1: Train All Models
```bash
#!/bin/bash
for model in unet unetpp resunetpp deeplabv3plus deeplabv3plus_cbam; do
    python train.py \
        --model $model \
        --data-root ./data \
        --loss combined \
        --use-boundary \
        --batch-size 8 \
        --epochs 100 \
        --output-dir ./experiments/$model
done
```

### Example 2: Compare Loss Functions
```bash
for loss in bce dice iou combined; do
    python train.py \
        --model unet \
        --data-root ./data \
        --loss $loss \
        --output-dir ./loss_comparison/$loss
done
```

### Example 3: Hyperparameter Tuning
```bash
for lr in 1e-3 1e-4 1e-5; do
    python train.py \
        --model deeplabv3plus_cbam \
        --data-root ./data \
        --learning-rate $lr \
        --output-dir ./tuning/lr_$lr
done
```

## 🧪 Testing

### Test Model Architecture
```bash
python utils.py --mode single --model unet
```

### Compare All Models
```bash
python utils.py --mode compare
```

### Test Loss Functions
```bash
python utils.py --mode losses
```

### Test Metrics
```bash
python utils.py --mode metrics
```

### Benchmark Training Speed
```bash
python utils.py --mode benchmark --model deeplabv3plus_cbam --batch-size 8
```

## 📊 Expected Results

### Training Time (on NVIDIA RTX 3090)
- **U-Net**: ~2 hours for 100 epochs
- **U-Net++**: ~3 hours for 100 epochs
- **ResUNet++**: ~3.5 hours for 100 epochs
- **DeepLabV3+**: ~5 hours for 100 epochs
- **DeepLabV3+ + CBAM**: ~6 hours for 100 epochs

### Typical Performance (River Bride Dataset)
- **Dice Score**: 0.85-0.92
- **IoU Score**: 0.75-0.85
- **Precision**: 0.88-0.94
- **Recall**: 0.82-0.90

## 🔬 Research Context

This implementation was developed for flash flood forecasting in Irish rivers with forest canopy occlusion:

- **Study Area**: River Bride catchment, County Cork, Ireland
- **Challenge**: 50-80% forest canopy occlusion over narrow rivers (3-5m width)
- **Key Finding**: CBAM attention provides 10.84% improvement over baseline
- **Application**: Near-real-time flash flood forecasting for emergency management

## 📝 Code Quality

- ✅ Production-ready implementation
- ✅ Comprehensive error handling
- ✅ Detailed documentation
- ✅ Modular and extensible design
- ✅ Type hints and docstrings
- ✅ Best practices for PyTorch

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## 📄 License

MIT License

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: Hasitha Wijesuriya, PhD Researcher, MTU Cork

## 🙏 Acknowledgments

- River Bride catchment data collection
- MTU Cork research support  
- Ireland's Office of Public Works
- PyTorch and Albumentations communities

---

**Ready to start?** 
1. Prepare your data (images/ and masks/ folders)
2. Run `python train.py --model unet --data-root ./your_data`
3. Monitor with `tensorboard --logdir experiments/`
4. Evaluate with `python test.py --checkpoint path/to/best.pth --mode evaluate`

For detailed examples: `python quick_start.py`


# Transformer Models for River Water Segmentation

Self-contained implementation of transformer-based segmentation models for river water detection from UAV imagery. **RGB input only.**

## 📋 Models

- **SegFormer-B0**: 3.7M params - Ultra-fast (60 FPS), real-time forecasting
- **SegFormer-B2**: 28M params - **RECOMMENDED** - Best accuracy/speed balance
- **Swin-UNet-Tiny**: 27M params - Superior boundary detection

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```
your_data/
├── images/      # RGB images (.png, .jpg, .tif)
│   ├── img1.png
│   └── img2.png
└── masks/       # Binary masks (0=background, 255=water)
    ├── img1.png
    └── img2.png
```

### 3. Train SegFormer-B2
```bash
python train.py \
    --model segformer_b2 \
    --data-root /path/to/your_data \
    --batch-size 10 \
    --epochs 100 \
    --learning-rate 6e-5 \
    --warmup-epochs 5
```

### 4. Evaluate
```bash
python test.py \
    --checkpoint experiments/segformer_b2_.../checkpoints/best.pth \
    --mode evaluate \
    --data-root /path/to/test_data
```

## 📊 Training Examples

### SegFormer-B0 (Fast)
```bash
python train.py \
    --model segformer_b0 \
    --data-root ./data \
    --batch-size 16 \
    --epochs 100
```
**Best for:** Real-time deployment, limited GPU

### SegFormer-B2 (Recommended)
```bash
python train.py \
    --model segformer_b2 \
    --data-root ./data \
    --batch-size 10 \
    --epochs 100 \
    --loss dice
```
**Best for:** Research, best overall performance

### Swin-UNet-Tiny (Boundaries)
```bash
python train.py \
    --model swin_unet_tiny \
    --data-root ./data \
    --batch-size 8 \
    --epochs 100 \
    --loss combined \
    --use-boundary
```
**Best for:** Fine boundary detection

## 🎯 Loss Functions

Available losses: `bce`, `dice`, `iou`, `focal`, `boundary`, `combined`

**Recommended:**
- Start with: `--loss dice`
- Best accuracy: `--loss combined --use-boundary`

Example:
```bash
python train.py \
    --model segformer_b2 \
    --data-root ./data \
    --loss combined \
    --use-boundary \
    --bce-weight 1.0 \
    --dice-weight 1.0 \
    --boundary-weight 1.0
```

## 📈 Hyperparameters

### Recommended Settings

**SegFormer-B0:**
- Batch size: 16
- Learning rate: 6e-5
- Warmup: 3-5 epochs
- Weight decay: 0.01

**SegFormer-B2:**
- Batch size: 8-10
- Learning rate: 6e-5
- Warmup: 5-10 epochs
- Weight decay: 0.01

**Swin-UNet-Tiny:**
- Batch size: 8
- Learning rate: 5e-5
- Warmup: 5-10 epochs
- Weight decay: 0.01

### Important Parameters

```bash
--learning-rate 6e-5     # Lower than CNNs (typically 1e-4)
--warmup-epochs 5        # Critical for transformers
--weight-decay 0.01      # Higher than CNNs (typically 1e-5)
--clip-grad 1.0          # Gradient clipping
```

## 🔬 For Your Research

### River Bride Catchment Study

Your data: 415 RGB images from River Bride at Crookstown (March-July 2025)

**Training:**
```bash
python train.py \
    --model segformer_b2 \
    --data-root ./river_bride_data \
    --image-size 512 \
    --batch-size 10 \
    --epochs 100 \
    --loss dice \
    --warmup-epochs 5
```

### CERI 2026 Conference

**Experiments to run:**

1. **Baseline Transformer:**
```bash
python train.py --model segformer_b2 --data-root ./data --loss dice
```

2. **With Boundary Loss:**
```bash
python train.py --model segformer_b2 --data-root ./data --loss combined --use-boundary
```

3. **Fast Model (Real-time):**
```bash
python train.py --model segformer_b0 --data-root ./data --batch-size 16
```

### Expected Results

| Model | Dice (Expected) | IoU | FPS | Memory |
|-------|----------------|-----|-----|---------|
| SegFormer-B0 | 0.85-0.89 | 0.75-0.82 | 60 | Low |
| SegFormer-B2 | 0.88-0.92 | 0.78-0.85 | 28 | Medium |
| Swin-UNet | 0.87-0.91 | 0.77-0.84 | 25 | Medium |

## 📂 File Structure

```
transformer_segmentation/
├── models.py          # SegFormer & Swin-UNet implementations
├── dataset.py         # RGB data loader
├── losses.py          # Loss functions
├── metrics.py         # Evaluation metrics
├── train.py           # Training script
├── test.py            # Evaluation script
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## 🎓 Model Details

### SegFormer Architecture
- Hierarchical transformer encoder (4 stages)
- Multi-scale features (1/4, 1/8, 1/16, 1/32)
- All-MLP decoder
- Efficient self-attention with spatial reduction

### Swin-UNet Architecture
- Swin Transformer encoder
- U-Net decoder with skip connections
- Shifted window attention
- Patch-based processing

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir experiments/
```

**What to monitor:**
- First 5 epochs: LR increases (warmup)
- Loss: Should decrease smoothly
- Dice/IoU: Should improve steadily
- LR: Cosine decay after warmup

### Good Training Signs
✅ Smooth loss decrease  
✅ Steady metric improvement  
✅ No NaN losses  
✅ Validation tracks training  

### Problem Signs
🔴 NaN loss → Lower LR, increase warmup  
🔴 No improvement → Try different loss  
🔴 Overfitting → Reduce epochs, increase weight decay  

## 🚨 Troubleshooting

### Out of Memory
```bash
--batch-size 4      # Reduce batch
--image-size 384    # Smaller images
--model segformer_b0  # Lighter model
```

### Poor Performance
```bash
--loss combined --use-boundary  # Better loss
--warmup-epochs 10              # More warmup
--epochs 150                    # More training
```

### Slow Convergence
```bash
--learning-rate 8e-5  # Slightly higher
--warmup-epochs 10    # More warmup
```

## 🎯 Testing

### Single Image
```bash
python test.py \
    --checkpoint path/to/best.pth \
    --mode single \
    --image-path path/to/image.png \
    --output-dir ./predictions
```

### Full Dataset
```bash
python test.py \
    --checkpoint path/to/best.pth \
    --mode evaluate \
    --data-root path/to/test_data \
    --batch-size 8 \
    --output-dir ./evaluation
```

## 📝 Command Reference

### Full Training Command
```bash
python train.py \
    --model segformer_b2 \
    --data-root /path/to/data \
    --image-size 512 \
    --batch-size 10 \
    --epochs 100 \
    --learning-rate 6e-5 \
    --weight-decay 0.01 \
    --warmup-epochs 5 \
    --loss dice \
    --num-workers 4 \
    --output-dir ./experiments
```

### Resume Training
```bash
python train.py \
    --model segformer_b2 \
    --data-root /path/to/data \
    --resume experiments/segformer_b2_.../checkpoints/latest.pth
```

## 💡 Tips

1. **Start small:** Test with SegFormer-B0 first
2. **Use warmup:** Always enable for transformers
3. **Monitor memory:** Adjust batch size accordingly
4. **Try different losses:** Dice is good baseline, Combined is best
5. **Be patient:** Transformers may be slower to converge initially

## 📚 References

- SegFormer: [NeurIPS 2021](https://arxiv.org/abs/2105.15203)
- Swin Transformer: [ICCV 2021](https://arxiv.org/abs/2103.14030)
- Swin-UNet: [ECCV 2022](https://arxiv.org/abs/2105.05537)

## 📧 Support

For issues:
1. Check that images are RGB (3 channels)
2. Verify masks are binary (0 and 255)
3. Ensure data structure matches expected format
4. Try reducing batch size if OOM

## ✅ Checklist

Before training:
- [ ] Installed requirements
- [ ] Data in correct format (images/ and masks/)
- [ ] Masks are binary (0 and 255)
- [ ] Images are RGB
- [ ] Checked GPU memory

Ready to train transformers! 🚀


### After Training Evalute Models

```bash
python src/evaluate_models.py --experiments_dir ./experiments --data_root ./dataset/processed_512_resized --save_predictions --num_pred_samples 8

```

