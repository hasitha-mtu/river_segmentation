# Quick Setup Guide - River Water Segmentation

## 📦 What You Have

Complete PyTorch implementation for river water segmentation:
- **Input**: 512×512 RGB images
- **Output**: Binary water masks (0=background, 1=water)
- **Models**: U-Net, U-Net++, ResUNet++, DeepLabV3+, DeepLabV3+ + CBAM
- **Losses**: BCE, Dice, IoU, Focal, Boundary, Combined, and more

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
```
your_data/
├── images/          # RGB images (.png, .jpg, etc.)
│   ├── img1.png
│   └── img2.png
└── masks/           # Binary masks (0=background, 255=water)
    ├── img1.png
    └── img2.png
```

**Important**: 
- Masks must be **binary** (0 and 255 values)
- Image and mask filenames must **match**
- Images will be automatically resized to 512×512

### 3. Test Installation
```bash
python utils.py --mode compare
```

### 4. Start Training
```bash
python train.py \
    --model unet \
    --data-root ./your_data \
    --batch-size 8 \
    --epochs 100
```

Done! 🎉

## 🎯 Recommended Workflows

### Workflow 1: Quick Start (Fastest)
```bash
# Train baseline U-Net
python train.py \
    --model unet \
    --data-root ./data \
    --loss dice \
    --batch-size 16 \
    --epochs 100
```
**Time**: ~2 hours | **Use for**: Fast results, prototyping

### Workflow 2: Best Accuracy (Recommended)
```bash
# Train DeepLabV3+ with CBAM and Boundary Loss
python train.py \
    --model deeplabv3plus_cbam \
    --data-root ./data \
    --loss combined \
    --use-boundary \
    --pretrained \
    --batch-size 8 \
    --epochs 100
```
**Time**: ~6 hours | **Use for**: Best performance, production

### Workflow 3: Balanced (Good Speed + Accuracy)
```bash
# Train ResUNet++
python train.py \
    --model resunetpp \
    --data-root ./data \
    --loss combined \
    --batch-size 12 \
    --epochs 100
```
**Time**: ~3.5 hours | **Use for**: Balance between speed and accuracy

## 📊 Monitor Training

### TensorBoard (Real-time Graphs)
```bash
tensorboard --logdir experiments/
```
Then open: http://localhost:6006

**What to watch**:
- Loss should decrease steadily
- Dice/IoU should increase
- Validation metrics should track training

### Console Output
Training will show:
```
Epoch 10/100 - 42.3s
Train Loss: 0.234 | Val Loss: 0.289
Train - Dice: 0.892, IoU: 0.805
Val   - Dice: 0.867, IoU: 0.765
```

## 🧪 Evaluate Your Model

### Full Evaluation on Test Set
```bash
python test.py \
    --checkpoint experiments/.../best.pth \
    --mode evaluate \
    --data-root ./test_data \
    --batch-size 8 \
    --output-dir ./results
```

**Outputs**:
- `evaluation_metrics.json` - Overall metrics
- `per_image_metrics.json` - Per-image results
- `evaluation_report.txt` - Human-readable summary

### Predict on New Images
```bash
python test.py \
    --checkpoint experiments/.../best.pth \
    --mode predict \
    --image-dir ./new_images \
    --output-dir ./predictions \
    --visualize
```

**Outputs**:
- `masks/` - Binary prediction masks
- `visualizations/` - Side-by-side comparisons

### Single Image Test
```bash
python test.py \
    --checkpoint experiments/.../best.pth \
    --mode single \
    --image-path ./my_image.png \
    --output-dir ./output \
    --visualize
```

## 🔧 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size
--batch-size 4

# Or reduce image size
--image-size 256
```

### Issue 2: Poor Segmentation Results
**Check your data**:
```python
# Verify mask is binary
import cv2
mask = cv2.imread('masks/img1.png', 0)
print(f"Unique values: {np.unique(mask)}")  # Should be [0, 255]

# Check image-mask correspondence
import matplotlib.pyplot as plt
img = cv2.imread('images/img1.png')
mask = cv2.imread('masks/img1.png', 0)
plt.subplot(121); plt.imshow(img)
plt.subplot(122); plt.imshow(mask)
plt.show()
```

**Try different loss**:
```bash
# For imbalanced data (small water regions)
--loss focal

# For precise boundaries
--loss combined --use-boundary
```

**Use pretrained weights**:
```bash
# Only for DeepLabV3+ variants
--model deeplabv3plus --pretrained
```

### Issue 3: NaN Loss
```bash
# Enable gradient clipping
--clip-grad 1.0

# Reduce learning rate
--learning-rate 1e-5
```

### Issue 4: Training Too Slow
```bash
# Use smaller model
--model unet

# Reduce batch size if GPU underutilized
--batch-size 16

# Reduce workers if I/O bound
--num-workers 2
```

## 📈 Model Selection Guide

| If you need... | Use this model | Command |
|----------------|----------------|---------|
| **Fastest inference** | U-Net | `--model unet` |
| **Best accuracy** | DeepLabV3+ + CBAM | `--model deeplabv3plus_cbam --pretrained` |
| **Good balance** | ResUNet++ | `--model resunetpp` |
| **Better boundaries** | U-Net++ | `--model unetpp` |
| **Forest canopy** | DeepLabV3+ + CBAM | `--model deeplabv3plus_cbam --loss combined --use-boundary` |

## 💡 Pro Tips

### Tip 1: Always Start Simple
```bash
# First, train baseline U-Net to verify your data is correct
python train.py --model unet --data-root ./data --epochs 50

# Then move to complex models
python train.py --model deeplabv3plus_cbam --pretrained ...
```

### Tip 2: Use Combined Loss
```bash
# Best results with BCE + Dice + Boundary
--loss combined --use-boundary
```

### Tip 3: Monitor Overfitting
If validation loss increases while training loss decreases:
```bash
# Add more augmentation (already default in code)
# Or reduce model complexity
--model unet  # Instead of deeplabv3plus_cbam
```

### Tip 4: Resume Failed Training
```bash
# Training interrupted? Resume from last checkpoint
python train.py \
    --model unet \
    --data-root ./data \
    --resume experiments/.../checkpoints/latest.pth
```

### Tip 5: Compare Models
```bash
# Test all models on your hardware
python utils.py --mode compare

# Benchmark specific model
python utils.py --mode benchmark --model deeplabv3plus_cbam
```

## 📋 Checklist Before Training

- [ ] Data in correct format (`images/` and `masks/` folders)
- [ ] Masks are binary (0 and 255 values)
- [ ] Image-mask filenames match
- [ ] Installed all dependencies (`pip install -r requirements.txt`)
- [ ] Tested installation (`python utils.py --mode compare`)
- [ ] GPU available (optional but recommended)

## 🎓 Example Training Session

Here's what a complete training session looks like:

```bash
# 1. Verify setup
python utils.py --mode compare

# 2. Start training
python train.py \
    --model deeplabv3plus_cbam \
    --data-root ./river_data \
    --loss combined \
    --use-boundary \
    --pretrained \
    --batch-size 8 \
    --epochs 100 \
    --learning-rate 1e-4

# Output will show:
# Initializing deeplabv3plus_cbam model...
# Loaded deeplabv3plus_cbam model
# Loading data from ./river_data...
# Loaded 332 image-mask pairs
# Train samples: 265, Val samples: 67
# ...
# Epoch 1/100 - 45.2s
# Train Loss: 0.523 | Val Loss: 0.487
# Train - Dice: 0.721, IoU: 0.564
# Val   - Dice: 0.745, IoU: 0.594
# ...

# 3. Monitor with TensorBoard (in another terminal)
tensorboard --logdir experiments/

# 4. After training completes
# Best model saved at: experiments/.../checkpoints/best.pth

# 5. Evaluate
python test.py \
    --checkpoint experiments/.../checkpoints/best.pth \
    --mode evaluate \
    --data-root ./test_data \
    --output-dir ./final_results
```

## 🚀 Next Steps

After successful training:

1. **Evaluate thoroughly** on held-out test set
2. **Visualize predictions** to understand failure cases
3. **Tune hyperparameters** if needed (learning rate, loss weights)
4. **Compare models** to find best for your use case
5. **Deploy** for inference on new images

## 📚 Additional Resources

- **Full documentation**: See `README.md`
- **Code examples**: Run `python quick_start.py`
- **Model comparison**: `python utils.py --mode compare`
- **Test losses**: `python utils.py --mode losses`

## 🆘 Getting Help

1. Check this guide for common issues
2. Review `README.md` for detailed documentation
3. Test with `python utils.py --mode [test_type]`
4. Check TensorBoard for training curves
5. Open an issue on GitHub

---

**Ready to train?** Just run:
```bash
python train.py --model unet --data-root ./your_data
```

Good luck! 🎉
