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
--batch-size         Batch size (default: 8)
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
