"""
Practical Training Examples
Copy the example you need and modify as required
"""

from train_unified import train_single_model, train_all_models, get_default_config
from config_examples import get_cnn_config, get_transformer_config, get_foundation_model_config
import copy


# ============================================================================
# EXAMPLE 1: Train Your First Model (Quick Start)
# ============================================================================
def example_1_quick_start():
    """Train a single U-Net model with default settings"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Quick Start - Train U-Net")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['model']['name'] = 'unet'
    
    train_single_model(config)


# ============================================================================
# EXAMPLE 2: Train All CNN Baselines
# ============================================================================
def example_2_train_all_cnns():
    """Train all CNN baseline models"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Train All CNN Baselines")
    print("="*80)
    
    config = get_cnn_config()
    config['data']['data_root'] = './dataset/processed_1024'
    
    cnn_models = ['unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam']
    
    for model_name in cnn_models:
        print(f"\n{'='*80}")
        print(f"Training {model_name}")
        print(f"{'='*80}\n")
        
        config['model']['name'] = model_name
        train_single_model(config)


# ============================================================================
# EXAMPLE 3: Train All Transformer Variants
# ============================================================================
def example_3_train_all_transformers():
    """Train all transformer models with their variants"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Train All Transformers")
    print("="*80)
    
    config = get_transformer_config()
    config['data']['data_root'] = './dataset/processed_1024'
    
    transformer_models = {
        'segformer': ['b0', 'b2'],
        'swin_unet': ['tiny'],
    }
    
    for model_name, variants in transformer_models.items():
        for variant in variants:
            print(f"\n{'='*80}")
            print(f"Training {model_name}-{variant}")
            print(f"{'='*80}\n")
            
            config['model']['name'] = model_name
            config['model']['variant'] = variant
            train_single_model(config)


# ============================================================================
# EXAMPLE 4: Train Foundation Models (SAM & DINOv2)
# ============================================================================
def example_4_train_foundation_models():
    """Train foundation models - requires more GPU memory"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Train Foundation Models")
    print("="*80)
    
    config = get_foundation_model_config()
    config['data']['data_root'] = './dataset/processed_1024'
    
    # Train SAM variants
    sam_variants = ['vit_b']  # Start with smallest, add 'vit_l', 'vit_h' if you have GPU memory
    for variant in sam_variants:
        print(f"\n{'='*80}")
        print(f"Training SAM-{variant}")
        print(f"{'='*80}\n")
        
        config['model']['name'] = 'sam'
        config['model']['variant'] = variant
        train_single_model(config)
    
    # Train DINOv2 variants
    dino_variants = ['vit_s']  # Start with smallest
    for variant in dino_variants:
        print(f"\n{'='*80}")
        print(f"Training DINOv2-{variant}")
        print(f"{'='*80}\n")
        
        config['model']['name'] = 'dinov2'
        config['model']['variant'] = variant
        train_single_model(config)


# ============================================================================
# EXAMPLE 5: Ablation Study - Different Loss Functions
# ============================================================================
def example_5_loss_ablation():
    """Compare different loss functions on U-Net"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Loss Function Ablation Study")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['model']['name'] = 'unet'
    
    loss_functions = ['bce', 'dice', 'focal', 'combined']
    
    for loss_type in loss_functions:
        print(f"\n{'='*80}")
        print(f"Training with {loss_type} loss")
        print(f"{'='*80}\n")
        
        config['loss']['type'] = loss_type
        train_single_model(config)


# ============================================================================
# EXAMPLE 6: Hyperparameter Search - Learning Rates
# ============================================================================
def example_6_lr_search():
    """Test different learning rates"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Learning Rate Search")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['model']['name'] = 'unet'
    config['training']['epochs'] = 50  # Shorter for quick comparison
    
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    
    for lr in learning_rates:
        print(f"\n{'='*80}")
        print(f"Training with LR={lr}")
        print(f"{'='*80}\n")
        
        config['training']['optimizer']['learning_rate'] = lr
        train_single_model(config)


# ============================================================================
# EXAMPLE 7: Compare Optimizers
# ============================================================================
def example_7_optimizer_comparison():
    """Compare different optimizers"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Optimizer Comparison")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['model']['name'] = 'unet'
    
    optimizers = [
        {'type': 'adam', 'learning_rate': 1e-4, 'weight_decay': 1e-4},
        {'type': 'adamw', 'learning_rate': 1e-4, 'weight_decay': 0.01},
        {'type': 'sgd', 'learning_rate': 1e-3, 'weight_decay': 1e-4, 'momentum': 0.9},
    ]
    
    for opt_config in optimizers:
        print(f"\n{'='*80}")
        print(f"Training with {opt_config['type']} optimizer")
        print(f"{'='*80}\n")
        
        config['training']['optimizer'] = opt_config
        train_single_model(config)


# ============================================================================
# EXAMPLE 8: Compare Schedulers
# ============================================================================
def example_8_scheduler_comparison():
    """Compare different learning rate schedulers"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Scheduler Comparison")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['model']['name'] = 'unet'
    
    schedulers = [
        {'type': 'cosine', 'min_lr': 1e-6},
        {'type': 'step', 'step_size': 30, 'gamma': 0.1},
        {'type': 'warmup_cosine', 'min_lr': 1e-6, 'warmup_epochs': 5},
    ]
    
    for sched_config in schedulers:
        print(f"\n{'='*80}")
        print(f"Training with {sched_config['type']} scheduler")
        print(f"{'='*80}\n")
        
        config['training']['scheduler'] = sched_config
        train_single_model(config)


# ============================================================================
# EXAMPLE 9: Multi-Seed Training for Robustness
# ============================================================================
def example_9_multi_seed():
    """Train with multiple seeds to assess model robustness"""
    print("\n" + "="*80)
    print("EXAMPLE 9: Multi-Seed Training")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['model']['name'] = 'unet'
    
    seeds = [42, 123, 456, 789, 999]
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Training with seed={seed}")
        print(f"{'='*80}\n")
        
        config['system']['seed'] = seed
        train_single_model(config)


# ============================================================================
# EXAMPLE 10: Paper Results - Train All Models for Publication
# ============================================================================
def example_10_paper_results():
    """
    Train all models for paper results
    This is comprehensive and will take a long time!
    """
    print("\n" + "="*80)
    print("EXAMPLE 10: Train All Models for Paper")
    print("="*80)
    print("\nWARNING: This will train ALL models and may take several days!")
    print("Make sure you have sufficient GPU resources and time.")
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    
    # Use the train_all_models function
    train_all_models(config)


# ============================================================================
# EXAMPLE 11: Resume Training After Interruption
# ============================================================================
def example_11_resume_training():
    """Resume training from checkpoint"""
    print("\n" + "="*80)
    print("EXAMPLE 11: Resume Training")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['model']['name'] = 'segformer'
    config['model']['variant'] = 'b0'
    config['training']['resume'] = True  # Key setting for resuming
    config['training']['epochs'] = 150  # Can extend training
    
    train_single_model(config)


# ============================================================================
# EXAMPLE 12: Quick Test Run (For Debugging)
# ============================================================================
def example_12_quick_test():
    """Quick test run with small settings for debugging"""
    print("\n" + "="*80)
    print("EXAMPLE 12: Quick Test Run")
    print("="*80)
    
    config = get_default_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['data']['image_size'] = 512  # Smaller for speed
    config['model']['name'] = 'unet'
    config['training']['batch_size'] = 2
    config['training']['epochs'] = 2  # Just 2 epochs
    config['system']['output_dir'] = './test_experiments'
    
    train_single_model(config)


# ============================================================================
# EXAMPLE 13: Compare Model Sizes (Variants)
# ============================================================================
def example_13_model_size_comparison():
    """Compare different model sizes (SegFormer variants)"""
    print("\n" + "="*80)
    print("EXAMPLE 13: Model Size Comparison")
    print("="*80)
    
    config = get_transformer_config()
    config['data']['data_root'] = './dataset/processed_1024'
    config['model']['name'] = 'segformer'
    
    variants = ['b0', 'b2']  # From smallest to largest
    
    for variant in variants:
        print(f"\n{'='*80}")
        print(f"Training SegFormer-{variant}")
        print(f"{'='*80}\n")
        
        config['model']['variant'] = variant
        train_single_model(config)


# ============================================================================
# EXAMPLE 14: Best Models Only (For Final Evaluation)
# ============================================================================
def example_14_best_models_only():
    """Train only the best performing models for final evaluation"""
    print("\n" + "="*80)
    print("EXAMPLE 14: Best Models Only")
    print("="*80)
    
    # Based on preliminary results, train only top performers
    best_models = [
        ('unet', None, 'cnn'),
        ('resunetpp', None, 'cnn'),
        ('segformer', 'b0', 'transformer'),
        ('sam', 'vit_b', 'foundation'),
    ]
    
    for model_name, variant, model_type in best_models:
        print(f"\n{'='*80}")
        print(f"Training {model_name}" + (f"-{variant}" if variant else ""))
        print(f"{'='*80}\n")
        
        # Use appropriate config for model type
        if model_type == 'cnn':
            config = get_cnn_config()
        elif model_type == 'transformer':
            config = get_transformer_config()
        else:  # foundation
            config = get_foundation_model_config()
        
        config['data']['data_root'] = './dataset/processed_1024'
        config['model']['name'] = model_name
        config['model']['variant'] = variant
        
        train_single_model(config)


# ============================================================================
# Main Menu
# ============================================================================
def main():
    """
    Choose which example to run
    """
    print("\n" + "="*80)
    print("UNIFIED TRAINING SYSTEM - EXAMPLES")
    print("="*80)
    print("\nAvailable Examples:")
    print("  1. Quick Start - Train U-Net")
    print("  2. Train All CNN Baselines")
    print("  3. Train All Transformers")
    print("  4. Train Foundation Models")
    print("  5. Loss Function Ablation")
    print("  6. Learning Rate Search")
    print("  7. Optimizer Comparison")
    print("  8. Scheduler Comparison")
    print("  9. Multi-Seed Training")
    print(" 10. Train All Models for Paper (LONG!)")
    print(" 11. Resume Training")
    print(" 12. Quick Test (Debugging)")
    print(" 13. Model Size Comparison")
    print(" 14. Best Models Only")
    
    # Uncomment the example you want to run:
    
    # example_1_quick_start()
    # example_2_train_all_cnns()
    # example_3_train_all_transformers()
    # example_4_train_foundation_models()
    # example_5_loss_ablation()
    # example_6_lr_search()
    # example_7_optimizer_comparison()
    # example_8_scheduler_comparison()
    # example_9_multi_seed()
    # example_10_paper_results()
    # example_11_resume_training()
    # example_12_quick_test()
    # example_13_model_size_comparison()
    # example_14_best_models_only()
    
    # Or run a specific example directly:
    example_1_quick_start()


if __name__ == '__main__':
    main()
