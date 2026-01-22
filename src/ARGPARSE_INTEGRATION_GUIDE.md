# Improved Argparse Integration Guide

## 🎯 **What's Improved**

The new argparse system provides:

1. ✅ **Smart validation** - Automatically validates variants based on model type
2. ✅ **Clear error messages** - Tells you exactly what's wrong
3. ✅ **Default variants** - Uses sensible defaults when variant not specified
4. ✅ **Helpful documentation** - Shows valid variants for each model
5. ✅ **Type safety** - Prevents invalid model-variant combinations

---

## 📋 **Model Configuration**

### **Models Without Variants:**
- `unet` - Standard U-Net
- `resunet` - Residual U-Net

### **Models With Variants:**

| Model | Variants | Default |
|-------|----------|---------|
| `sam` | vit_b, vit_l, vit_h |
| `dinov2` | vit_s, vit_b, vit_l, vit_g | 
| `swin-unet` | tiny |
| `segformer` | b0, b2 |
| `convnext-unet` | tiny, small, base |
| `hrnet-ocr` | w18, w32, w48 |

---

## ✅ **Valid Command Examples**

### **Model Without Variant:**
```bash
# Correct - no variant needed
python test_models.py --model unet --checkpoint best.pth --test-data data/test

# Wrong - unet doesn't have variants!
python test_models.py --model unet --variant vit_b --checkpoint best.pth --test-data data/test
# Error: Model 'unet' does not have variants. Remove --variant argument.
```

### **Model With Variant:**
```bash
# Correct - explicit variant
python test_models.py --model sam --variant vit_b --checkpoint best.pth --test-data data/test

# Correct - uses default (vit_b)
python test_models.py --model sam --checkpoint best.pth --test-data data/test
# Output: No variant specified. Using default: vit_b

# Wrong - invalid variant
python test_models.py --model sam --variant tiny --checkpoint best.pth --test-data data/test
# Error: Invalid variant 'tiny' for model 'sam'. Valid variants: vit_b, vit_l, vit_h
```

---

## 🔧 **How to Integrate**

### **Step 1: Add to Your Project**

Copy `improved_argparse.py` to your project directory:
```
your_project/
├── improved_argparse.py    # ← New file
├── test_models.py
├── compare_models.py
├── train.py
└── ...
```

---

### **Step 2: Update test_models.py**

Replace the argparse section:

```python
# OLD (before):
def parse_args():
    parser = argparse.ArgumentParser(description='Test trained segmentation model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model type (e.g., sam, dinov2, unet)')
    parser.add_argument('--varient', type=str, default=None,
                       help='Model variant (e.g., vit_b, vit_l)')
    # ... other arguments ...
    
    return parser.parse_args()


# NEW (after):
from improved_argparse import add_model_arguments, validate_args_post_parse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test trained segmentation model',
        formatter_class=argparse.RawDescriptionHelpFormatter  # For nice help text
    )
    
    # Add model arguments with smart validation
    parser = add_model_arguments(parser)
    
    # ... other arguments (checkpoint, test-data, etc.) ...
    
    args = parser.parse_args()
    
    # Validate model-variant combination
    args = validate_args_post_parse(args)
    
    return args
```

---

### **Step 3: Update compare_models.py**

Same pattern:

```python
from improved_argparse import add_model_arguments, validate_args_post_parse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare multiple trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--models-config', type=str, required=True,
                       help='Path to models configuration JSON')
    # ... other arguments ...
    
    args = parser.parse_args()
    
    # No need for post-validation here (config file handles model-variant)
    return args
```

---

### **Step 4: Update train.py**

```python
from improved_argparse import add_model_arguments, validate_args_post_parse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train segmentation model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add model arguments
    parser = add_model_arguments(parser)
    
    # Data arguments
    parser.add_argument('--data-root', type=str, required=True)
    # ... other arguments ...
    
    args = parser.parse_args()
    args = validate_args_post_parse(args)
    
    return args
```

---

## 🎨 **What Users See**

### **Help Text:**

```bash
python test_models.py --help
```

**Output:**
```
Model architecture type.

Available models:
  • unet                 - U-Net architecture (no variants)
  • resunet              - Residual U-Net (no variants)
  • sam                  - Segment Anything Model (SAM) (variants: vit_b, vit_l, vit_h)
  • dinov2               - DINOv2 foundation model (variants: vit_s, vit_b, vit_l, vit_g)
  • swin-unet            - Swin Transformer U-Net (variants: tiny, small, base, large)
  ...

Model variant (required for some models).

Variants by model type:
  • sam                 : vit_b, vit_l, vit_h (default: vit_b)
  • dinov2              : vit_s, vit_b, vit_l, vit_g (default: vit_b)
  • swin-unet           : tiny, small, base, large (default: tiny)
  ...

Models without variants: unet, resunet
```

---

### **Error Messages:**

```bash
# Wrong model name
python test_models.py --model unet2 ...
# Error: argument --model: invalid choice: 'unet2' 
#        (choose from 'unet', 'resunet', 'sam', 'dinov2', ...)

# Variant for model without variants
python test_models.py --model unet --variant vit_b ...
# Error: Model 'unet' does not have variants. Remove --variant argument.

# Invalid variant
python test_models.py --model sam --variant tiny ...
# Error: Invalid variant 'tiny' for model 'sam'. 
#        Valid variants: vit_b, vit_l, vit_h

# Missing required variant (if no default)
python test_models.py --model custom-model ...
# Error: Model 'custom-model' requires a variant. 
#        Valid variants: variant1, variant2
```

---

## 🆕 **Adding New Models**

### **Model Without Variants:**

```python
# In improved_argparse.py:

MODEL_VARIANTS = {
    # ... existing models ...
    
    'my-new-model': {
        'variants': None,  # No variants
        'description': 'My New Model Architecture'
    },
}
```

**Usage:**
```bash
python test_models.py --model my-new-model --checkpoint best.pth ...
```

---

### **Model With Variants:**

```python
MODEL_VARIANTS = {
    # ... existing models ...
    
    'my-transformer': {
        'variants': ['small', 'medium', 'large'],
        'default': 'small',  # Optional: if not provided, variant is required
        'description': 'My Transformer Model'
    },
}
```

**Usage:**
```bash
# Explicit variant
python test_models.py --model my-transformer --variant large ...

# Uses default (small)
python test_models.py --model my-transformer ...
```

---

## 📝 **Complete Integration Example**

### **test_models.py (updated)**

```python
"""
Test Script with Improved Argparse
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Import improved argparse functions
from improved_argparse import add_model_arguments, validate_args_post_parse

from models import get_model
from dataset import RiverSegmentationDataset
from metrics import StreamingMetrics


def parse_args():
    """Parse command line arguments with smart model-variant validation"""
    parser = argparse.ArgumentParser(
        description='Test trained segmentation model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add model arguments (with automatic validation)
    parser = add_model_arguments(parser)
    
    # Testing arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (best.pth)')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size (default: 512)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold (default: 0.5)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations (default: True)')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory (default: test_results)')
    
    # Parse and validate
    args = parser.parse_args()
    args = validate_args_post_parse(args)
    
    return args


def main():
    args = parse_args()
    
    print(f"Testing {args.model}" + 
          (f" ({args.variant})" if args.variant else ""))
    
    # Load model (variant is automatically None or valid string)
    model = get_model(args.model, varient=args.variant)
    
    # ... rest of testing code ...


if __name__ == "__main__":
    main()
```

---

## 🧪 **Testing the Integration**

### **Test Valid Commands:**

```bash
# Model without variant
python test_models.py --model unet --checkpoint best.pth --test-data data/test
# ✓ Works: variant=None

# Model with explicit variant
python test_models.py --model sam --variant vit_b --checkpoint best.pth --test-data data/test
# ✓ Works: variant=vit_b

# Model with default variant
python test_models.py --model sam --checkpoint best.pth --test-data data/test
# ✓ Works: variant=vit_b (default)
# Output: No variant specified. Using default: vit_b

# All DINOv2 variants
python test_models.py --model dinov2 --variant vit_s --checkpoint best.pth --test-data data/test
python test_models.py --model dinov2 --variant vit_b --checkpoint best.pth --test-data data/test
python test_models.py --model dinov2 --variant vit_l --checkpoint best.pth --test-data data/test
python test_models.py --model dinov2 --variant vit_g --checkpoint best.pth --test-data data/test
# ✓ All work
```

### **Test Invalid Commands:**

```bash
# Wrong: variant for model without variants
python test_models.py --model unet --variant vit_b --checkpoint best.pth --test-data data/test
# ✗ Error: Model 'unet' does not have variants. Remove --variant argument.

# Wrong: invalid variant
python test_models.py --model sam --variant small --checkpoint best.pth --test-data data/test
# ✗ Error: Invalid variant 'small' for model 'sam'. Valid variants: vit_b, vit_l, vit_h

# Wrong: unknown model
python test_models.py --model transformer --checkpoint best.pth --test-data data/test
# ✗ Error: argument --model: invalid choice: 'transformer'
```

---

## 💡 **Benefits**

### **Before (Old System):**
```bash
# No validation - silently wrong!
python test_models.py --model sam --variant tiny ...
# Crashes later with cryptic error in model loading

# No help for variants
python test_models.py --help
# Just says: --variant STR (no info about valid values)
```

### **After (New System):**
```bash
# Immediate validation with clear error
python test_models.py --model sam --variant tiny ...
# Error: Invalid variant 'tiny' for model 'sam'. Valid variants: vit_b, vit_l, vit_h

# Detailed help
python test_models.py --help
# Shows all models, their variants, and defaults!
```

---

## ✅ **Checklist**

Integration steps:
- [ ] Copy `improved_argparse.py` to project
- [ ] Update `test_models.py` imports
- [ ] Update `compare_models.py` imports
- [ ] Update `train.py` imports
- [ ] Test with valid commands
- [ ] Test with invalid commands (should give clear errors)
- [ ] Update documentation/README

Testing:
- [ ] Model without variant works
- [ ] Model with explicit variant works
- [ ] Model with default variant works
- [ ] Invalid variant gives clear error
- [ ] Variant for no-variant model gives clear error
- [ ] Help text shows all models and variants

---

## 🎓 **Advanced: Custom Validation**

You can add custom validation logic:

```python
from improved_argparse import (
    add_model_arguments, 
    validate_args_post_parse,
    MODEL_VARIANTS
)

def parse_args():
    parser = argparse.ArgumentParser(...)
    parser = add_model_arguments(parser)
    
    # ... other arguments ...
    
    args = parser.parse_args()
    args = validate_args_post_parse(args)
    
    # Custom validation
    if args.model == 'sam' and args.image_size < 384:
        parser.error("SAM models require image_size >= 384")
    
    if args.model in ['dinov2', 'sam'] and args.batch_size > 4:
        print("Warning: Large models may OOM with batch_size > 4")
    
    return args
```

---

## 📚 **Summary**

**Old System:**
```python
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--variant', type=str, default=None)
# No validation, no help, errors discovered at runtime
```

**New System:**
```python
from improved_argparse import add_model_arguments, validate_args_post_parse

parser = add_model_arguments(parser)
args = validate_args_post_parse(args)
# ✓ Automatic validation
# ✓ Clear error messages
# ✓ Helpful documentation
# ✓ Default variants
# ✓ Type safety
```

**Result:** Better UX, fewer errors, clearer documentation! 🎉
