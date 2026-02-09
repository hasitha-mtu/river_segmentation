"""
Image Size Comparison Script
Compare model performance with different input image sizes (512 vs 1024)
"""

from train_unified import train_single_model, get_default_config
import copy


def train_with_different_sizes():
    """
    Train models with both 512x512 and 1024x1024 input sizes
    to compare performance and training efficiency
    """
    
    # Base configuration
    base_config = get_default_config()
    base_config['data']['data_root'] = './dataset/processed_1024'
    base_config['training']['epochs'] = 100
    
    # Models to test
    models_to_test = [
        ('unet', None),
        ('segformer', 'b0'),
        ('resunetpp', None),
    ]
    
    # Image sizes to test
    image_sizes = [512, 1024]
    
    for model_name, variant in models_to_test:
        for image_size in image_sizes:
            print("\n" + "="*80)
            print(f"Training {model_name}" + (f"-{variant}" if variant else ""))
            print(f"Image Size: {image_size}x{image_size}")
            print("="*80 + "\n")
            
            # Create config for this experiment
            config = copy.deepcopy(base_config)
            config['model']['name'] = model_name
            config['model']['variant'] = variant
            config['data']['image_size'] = image_size
            
            # Adjust batch size based on image size and model
            if image_size == 1024:
                config['training']['batch_size'] = 4
            else:  # 512
                config['training']['batch_size'] = 8
            
            # Train
            train_single_model(config)
            
            print(f"\n✓ Completed: {model_name}" + (f"-{variant}" if variant else "") + f" @ {image_size}x{image_size}")


def train_512_experiments():
    """
    Train multiple models with 512x512 images
    Faster training for initial experiments
    """
    print("\n" + "="*80)
    print("FAST EXPERIMENTS - 512x512 Images")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['data']['image_size'] = 512
    config['training']['batch_size'] = 8
    config['training']['epochs'] = 100
    
    models = [
        ('unet', None),
        ('unetpp', None),
        ('resunetpp', None),
        ('deeplabv3plus', None),
        ('segformer', 'b0'),
    ]
    
    for model_name, variant in models:
        print(f"\n{'='*80}")
        print(f"Training {model_name}" + (f"-{variant}" if variant else ""))
        print(f"{'='*80}\n")
        
        config['model']['name'] = model_name
        config['model']['variant'] = variant
        train_single_model(config)


def train_1024_best_models():
    """
    Train best performing models with full 1024x1024 resolution
    For final paper results
    """
    print("\n" + "="*80)
    print("FINAL RESULTS - 1024x1024 Images")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['data']['image_size'] = 1024
    config['training']['batch_size'] = 4
    config['training']['epochs'] = 150
    
    # Best models based on 512 experiments
    best_models = [
        ('unet', None),
        ('resunetpp', None),
        ('deeplabv3plus_cbam', None),
        ('segformer', 'b0'),
        ('segformer', 'b2'),
    ]
    
    for model_name, variant in best_models:
        print(f"\n{'='*80}")
        print(f"Training {model_name}" + (f"-{variant}" if variant else ""))
        print(f"{'='*80}\n")
        
        config['model']['name'] = model_name
        config['model']['variant'] = variant
        
        # Adjust batch size for larger models
        if model_name == 'segformer' and variant == 'b2':
            config['training']['batch_size'] = 2
        
        train_single_model(config)


def compare_resolution_impact():
    """
    Systematic comparison of resolution impact on a single model
    Train U-Net at different resolutions
    """
    print("\n" + "="*80)
    print("RESOLUTION IMPACT STUDY - U-Net")
    print("="*80)
    
    base_config = get_default_config()
    base_config['data']['data_root'] = './dataset/processed_1024'
    base_config['model']['name'] = 'unet'
    base_config['model']['variant'] = None
    base_config['training']['epochs'] = 100
    
    # Test different resolutions
    resolutions = [256, 512, 768, 1024]
    batch_sizes = [16, 8, 6, 4]  # Adjust for GPU memory
    
    for resolution, batch_size in zip(resolutions, batch_sizes):
        print(f"\n{'='*80}")
        print(f"Training U-Net @ {resolution}x{resolution} (batch_size={batch_size})")
        print(f"{'='*80}\n")
        
        config = copy.deepcopy(base_config)
        config['data']['image_size'] = resolution
        config['training']['batch_size'] = batch_size
        
        train_single_model(config)


def quick_test_both_sizes():
    """
    Quick test with both sizes to verify everything works
    """
    print("\n" + "="*80)
    print("QUICK TEST - Both Image Sizes")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['model']['name'] = 'unet'
    config['training']['epochs'] = 2  # Just 2 epochs for testing
    config['system']['output_dir'] = './test_experiments'
    
    for image_size in [512, 1024]:
        print(f"\n{'='*80}")
        print(f"Testing with {image_size}x{image_size} images")
        print(f"{'='*80}\n")
        
        config['data']['image_size'] = image_size
        config['training']['batch_size'] = 8 if image_size == 512 else 4
        
        train_single_model(config)
        
        print(f"\n✓ Test completed for {image_size}x{image_size}")


# ============================================================================
# Recommended Workflow
# ============================================================================

def recommended_workflow():
    """
    Recommended workflow for your research:
    
    1. Quick test to verify everything works
    2. Fast 512 experiments to identify best models
    3. Full 1024 training for final results
    """
    
    print("\n" + "="*80)
    print("RECOMMENDED WORKFLOW")
    print("="*80)
    print("""
    Step 1: Quick Test (uncomment to run)
    - Verify data loading and training pipeline
    - Takes ~5 minutes
    """)
    # quick_test_both_sizes()
    
    print("""
    Step 2: Fast 512 Experiments (uncomment to run)
    - Train multiple models quickly with 512x512 images
    - Identify best performing architectures
    - Takes ~1-2 days for 5-10 models
    """)
    # train_512_experiments()
    
    print("""
    Step 3: Final 1024 Results (uncomment to run)
    - Train best models with full 1024x1024 resolution
    - Generate final paper results
    - Takes ~3-5 days for 5 models
    """)
    # train_1024_best_models()
    
    print("""
    Optional: Resolution Study (uncomment to run)
    - Systematic study of resolution impact
    - Useful for ablation analysis
    """)
    # compare_resolution_impact()


# ============================================================================
# Configuration Tips
# ============================================================================

CONFIGURATION_TIPS = """
Configuration Tips for Different Image Sizes:
==============================================

512x512 Images:
--------------
✓ Faster training (2-3x speedup)
✓ Larger batch sizes possible
✓ Good for initial experiments
✓ May miss fine details in narrow rivers

Recommended settings:
- batch_size: 8-16 (depending on model)
- epochs: 100
- Use for: Initial model comparison, ablation studies

1024x1024 Images:
-----------------
✓ Better detail preservation
✓ More accurate boundary detection
✓ Better for narrow water features
✓ Essential for final results

Recommended settings:
- batch_size: 4-8 (depending on model)
- epochs: 100-150
- Use for: Final model evaluation, paper results

Batch Size Guidelines by Model and Resolution:
-----------------------------------------------
512x512:
  • U-Net, DeepLabV3+: 16
  • ResU-Net++: 12
  • SegFormer-B0: 16
  • SegFormer-B2: 8
  • SAM-ViT-B: 4

1024x1024:
  • U-Net, DeepLabV3+: 8
  • ResU-Net++: 6
  • SegFormer-B0: 8
  • SegFormer-B2: 4
  • SAM-ViT-B: 2

Memory Requirements (Approximate):
-----------------------------------
512x512 @ batch_size=8:  ~8-12 GB GPU
1024x1024 @ batch_size=4: ~12-16 GB GPU
1024x1024 @ batch_size=8: ~20-24 GB GPU (large models may need more)
"""


# ============================================================================
# Main
# ============================================================================

def main():
    """
    Choose which experiment to run
    """
    print(CONFIGURATION_TIPS)
    
    # Uncomment the experiment you want to run:
    
    # Option 1: Quick test (recommended first step)
    quick_test_both_sizes()
    
    # Option 2: Fast 512 experiments
    # train_512_experiments()
    
    # Option 3: Full 1024 results
    # train_1024_best_models()
    
    # Option 4: Compare both sizes for same models
    # train_with_different_sizes()
    
    # Option 5: Resolution impact study
    # compare_resolution_impact()
    
    # Option 6: Follow recommended workflow
    # recommended_workflow()


if __name__ == '__main__':
    main()
