"""
Model Reference Guide
Quick reference for all supported models with recommended configurations
"""

# ============================================================================
# MODEL CHARACTERISTICS
# ============================================================================

MODEL_CHARACTERISTICS = {
    # CNN Baselines
    'unet': {
        'type': 'CNN',
        'variants': None,
        'params_approx': '31M',
        'memory_gb': '~4-6',
        'recommended_batch_size': 8,
        'recommended_lr': 1e-4,
        'recommended_optimizer': 'adam',
        'recommended_scheduler': 'cosine',
        'notes': 'Baseline encoder-decoder. Fast and reliable.'
    },
    'unetpp': {
        'type': 'CNN',
        'variants': None,
        'params_approx': '36M',
        'memory_gb': '~5-7',
        'recommended_batch_size': 6,
        'recommended_lr': 1e-4,
        'recommended_optimizer': 'adam',
        'recommended_scheduler': 'cosine',
        'notes': 'Dense skip connections. Better feature fusion.'
    },
    'resunetpp': {
        'type': 'CNN',
        'variants': None,
        'params_approx': '42M',
        'memory_gb': '~6-8',
        'recommended_batch_size': 6,
        'recommended_lr': 1e-4,
        'recommended_optimizer': 'adam',
        'recommended_scheduler': 'cosine',
        'notes': 'Residual blocks + attention. Good for fine details.'
    },
    'deeplabv3plus': {
        'type': 'CNN',
        'variants': None,
        'params_approx': '41M',
        'memory_gb': '~6-8',
        'recommended_batch_size': 6,
        'recommended_lr': 1e-4,
        'recommended_optimizer': 'adam',
        'recommended_scheduler': 'cosine',
        'notes': 'ASPP + encoder-decoder. Multi-scale features.'
    },
    'deeplabv3plus_cbam': {
        'type': 'CNN',
        'variants': None,
        'params_approx': '43M',
        'memory_gb': '~6-8',
        'recommended_batch_size': 6,
        'recommended_lr': 1e-4,
        'recommended_optimizer': 'adam',
        'recommended_scheduler': 'cosine',
        'notes': 'DeepLabV3+ with CBAM attention. Better feature weighting.'
    },
    
    # Transformers
    'segformer': {
        'type': 'Transformer',
        'variants': ['b0', 'b2'],
        'params_approx': {
            'b0': '3.8M',
            'b2': '27M',
        },
        'memory_gb': {
            'b0': '~4-6',
            'b2': '~8-12',
        },
        'recommended_batch_size': {
            'b0': 8,
            'b2': 4,
        },
        'recommended_lr': 6e-5,
        'recommended_optimizer': 'adamw',
        'recommended_scheduler': 'warmup_cosine',
        'warmup_epochs': 5,
        'notes': 'Hierarchical transformer. Efficient and accurate.'
    },
    'swin_unet': {
        'type': 'Transformer',
        'variants': ['tiny'],
        'params_approx': {
            'tiny': '27M',
        },
        'memory_gb': {
            'tiny': '~8-12',
        },
        'recommended_batch_size': {
            'tiny': 4,
        },
        'recommended_lr': 6e-5,
        'recommended_optimizer': 'adamw',
        'recommended_scheduler': 'warmup_cosine',
        'warmup_epochs': 5,
        'notes': 'Swin Transformer with U-Net decoder. Good for local details.'
    },
    
    # Hybrid SOTA
    'convnext_upernet': {
        'type': 'Hybrid',
        'variants': ['tiny', 'small', 'base'],
        'params_approx': {
            'tiny': '28M',
            'small': '50M',
            'base': '89M',
        },
        'memory_gb': {
            'tiny': '~8-12',
            'small': '~12-16',
            'base': '~16-20',
        },
        'recommended_batch_size': {
            'tiny': 4,
            'small': 2,
            'base': 2,
        },
        'recommended_lr': 6e-5,
        'recommended_optimizer': 'adamw',
        'recommended_scheduler': 'warmup_cosine',
        'warmup_epochs': 5,
        'notes': 'Modern ConvNet + UPerNet head. SOTA performance.'
    },
    'hrnet_ocr': {
        'type': 'Hybrid',
        'variants': ['w18', 'w32', 'w48'],
        'params_approx': {
            'w18': '15M',
            'w32': '41M',
            'w48': '65M',
        },
        'memory_gb': {
            'w18': '~6-8',
            'w32': '~10-14',
            'w48': '~14-18',
        },
        'recommended_batch_size': {
            'w18': 6,
            'w32': 4,
            'w48': 2,
        },
        'recommended_lr': 1e-4,
        'recommended_optimizer': 'adam',
        'recommended_scheduler': 'cosine',
        'notes': 'Multi-resolution + OCR module. Excellent for boundaries.'
    },
    
    # Foundation Models
    'sam': {
        'type': 'Foundation',
        'variants': ['vit_b', 'vit_l', 'vit_h'],
        'params_approx': {
            'vit_b': '89M',
            'vit_l': '308M',
            'vit_h': '636M',
        },
        'memory_gb': {
            'vit_b': '~16-20',
            'vit_l': '~20-28',
            'vit_h': '~28-40',
        },
        'recommended_batch_size': {
            'vit_b': 2,
            'vit_l': 1,
            'vit_h': 1,
        },
        'recommended_lr': 1e-5,
        'recommended_optimizer': 'adamw',
        'recommended_scheduler': 'warmup_cosine',
        'warmup_epochs': 3,
        'notes': 'Segment Anything Model. Needs fine-tuning. Very powerful.'
    },
    'dinov2': {
        'type': 'Foundation',
        'variants': ['vit_s', 'vit_b', 'vit_l', 'vit_g'],
        'params_approx': {
            'vit_s': '22M',
            'vit_b': '86M',
            'vit_l': '304M',
            'vit_g': '1.1B',
        },
        'memory_gb': {
            'vit_s': '~6-10',
            'vit_b': '~16-20',
            'vit_l': '~20-28',
            'vit_g': '~40+',
        },
        'recommended_batch_size': {
            'vit_s': 4,
            'vit_b': 2,
            'vit_l': 1,
            'vit_g': 1,
        },
        'recommended_lr': 1e-5,
        'recommended_optimizer': 'adamw',
        'recommended_scheduler': 'warmup_cosine',
        'warmup_epochs': 3,
        'notes': 'Self-supervised vision features. Excellent representations.'
    },
}


# ============================================================================
# TRAINING RECOMMENDATIONS BY USE CASE
# ============================================================================

TRAINING_RECOMMENDATIONS = {
    'baseline': {
        'description': 'Quick baseline for comparison',
        'recommended_models': ['unet'],
        'epochs': 100,
        'batch_size': 4,
        'notes': 'Fast training, reliable results'
    },
    
    'best_performance': {
        'description': 'Best overall performance (accuracy + speed)',
        'recommended_models': ['resunetpp', 'deeplabv3plus_cbam', 'segformer_b0'],
        'epochs': 100,
        'batch_size': 4,
        'notes': 'Good balance of accuracy and efficiency'
    },
    
    'sota_performance': {
        'description': 'State-of-the-art performance (accuracy focused)',
        'recommended_models': ['convnext_upernet_base', 'sam_vit_b', 'segformer_b2'],
        'epochs': 100,
        'batch_size': 4,
        'notes': 'Highest accuracy, longer training time'
    },
    
    'fast_inference': {
        'description': 'Fast inference for real-time applications',
        'recommended_models': ['unet', 'segformer_b0', 'hrnet_ocr_w18'],
        'epochs': 100,
        'batch_size': 4,
        'notes': 'Optimized for speed'
    },
    
    'limited_memory': {
        'description': 'Training with limited GPU memory (<8GB)',
        'recommended_models': ['unet', 'segformer_b0', 'dinov2_vit_s'],
        'epochs': 100,
        'batch_size': 4,
        'notes': 'Small models, reduced batch size'
    },
    
    'paper_results': {
        'description': 'Comprehensive comparison for publication',
        'recommended_models': [
            'unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam',
            'segformer_b0', 'segformer_b2', 'swin_unet_tiny',
            'convnext_upernet_tiny', 'hrnet_ocr_w32',
            'sam_vit_b', 'dinov2_vit_b'
        ],
        'epochs': 100,
        'batch_size': 4,
        'notes': 'Train all major architectures'
    },
}


# ============================================================================
# HYPERPARAMETER RECOMMENDATIONS
# ============================================================================

HYPERPARAMETER_RANGES = {
    'batch_size': {
        'small_models': [4, 8, 16],  # U-Net, SegFormer-B0
        'medium_models': [2, 4, 8],  # ResUNet++, DeepLabV3+
        'large_models': [1, 2, 4],   # SAM, ConvNeXt-Base
    },
    
    'learning_rate': {
        'cnn': [1e-3, 5e-4, 1e-4, 5e-5],
        'transformer': [1e-4, 6e-5, 1e-5],
        'foundation': [5e-5, 1e-5, 5e-6],
    },
    
    'optimizer': {
        'cnn': 'adam',
        'transformer': 'adamw',
        'foundation': 'adamw',
    },
    
    'scheduler': {
        'cnn': 'cosine',
        'transformer': 'warmup_cosine',
        'foundation': 'warmup_cosine',
    },
    
    'epochs': {
        'quick_test': 10,
        'normal': 100,
        'thorough': 150,
        'foundation': 50,  # Pretrained models need fewer epochs
    },
}


# ============================================================================
# LOSS FUNCTION RECOMMENDATIONS
# ============================================================================

LOSS_RECOMMENDATIONS = {
    'balanced_dataset': {
        'recommended': 'combined',
        'config': {
            'bce_weight': 1.0,
            'dice_weight': 1.0,
            'boundary_weight': 1.0,
            'use_boundary': True,
        },
        'notes': 'Works well for most cases'
    },
    
    'imbalanced_dataset': {
        'recommended': 'focal',
        'notes': 'Better for datasets with class imbalance'
    },
    
    'fine_details': {
        'recommended': 'combined',
        'config': {
            'bce_weight': 0.5,
            'dice_weight': 1.0,
            'boundary_weight': 2.0,  # Higher weight for boundaries
            'use_boundary': True,
        },
        'notes': 'Emphasizes boundary accuracy'
    },
    
    'shape_preservation': {
        'recommended': 'dice',
        'notes': 'Focus on shape and region accuracy'
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_info(model_name, variant=None):
    """Get information about a specific model"""
    if model_name not in MODEL_CHARACTERISTICS:
        return f"Model '{model_name}' not found"
    
    info = MODEL_CHARACTERISTICS[model_name]
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}" + (f" ({variant})" if variant else ""))
    print(f"{'='*60}")
    print(f"Type: {info['type']}")
    
    if variant and isinstance(info['params_approx'], dict):
        print(f"Parameters: {info['params_approx'][variant]}")
        print(f"Memory: {info['memory_gb'][variant]}")
        print(f"Recommended Batch Size: {info['recommended_batch_size'][variant]}")
    else:
        print(f"Parameters: {info['params_approx']}")
        print(f"Memory: {info['memory_gb']}")
        print(f"Recommended Batch Size: {info['recommended_batch_size']}")
    
    print(f"Recommended LR: {info['recommended_lr']}")
    print(f"Recommended Optimizer: {info['recommended_optimizer']}")
    print(f"Recommended Scheduler: {info['recommended_scheduler']}")
    if 'warmup_epochs' in info:
        print(f"Warmup Epochs: {info['warmup_epochs']}")
    print(f"\nNotes: {info['notes']}")
    print(f"{'='*60}\n")


def print_all_models():
    """Print summary of all models"""
    print("\n" + "="*80)
    print("ALL SUPPORTED MODELS")
    print("="*80 + "\n")
    
    for category in ['CNN', 'Transformer', 'Hybrid', 'Foundation']:
        print(f"\n{category} Models:")
        print("-" * 60)
        
        for model_name, info in MODEL_CHARACTERISTICS.items():
            if info['type'] == category:
                variants_str = ""
                if info['variants']:
                    variants_str = f" (variants: {', '.join(info['variants'])})"
                print(f"  • {model_name}{variants_str}")
                print(f"    {info['notes']}")


def get_recommended_config(model_name, variant=None):
    """Get recommended configuration for a model"""
    if model_name not in MODEL_CHARACTERISTICS:
        return None
    
    info = MODEL_CHARACTERISTICS[model_name]
    
    config = {
        'model': {
            'name': model_name,
            'variant': variant,
            'n_channels': 3,
            'n_classes': 1,
        },
        'training': {
            'optimizer': {
                'type': info['recommended_optimizer'],
                'learning_rate': info['recommended_lr'],
                'weight_decay': 0.01 if info['recommended_optimizer'] == 'adamw' else 1e-4,
            },
            'scheduler': {
                'type': info['recommended_scheduler'],
                'min_lr': 1e-6,
            },
        },
    }
    
    if variant and isinstance(info['recommended_batch_size'], dict):
        config['training']['batch_size'] = info['recommended_batch_size'][variant]
    else:
        config['training']['batch_size'] = info['recommended_batch_size']
    
    if 'warmup_epochs' in info:
        config['training']['scheduler']['warmup_epochs'] = info['warmup_epochs']
    
    return config


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Example usage
    print_all_models()
    
    print("\n" + "="*80)
    print("MODEL DETAILS EXAMPLES")
    print("="*80)
    
    # Get info for specific models
    get_model_info('unet')
    get_model_info('segformer', 'b0')
    get_model_info('sam', 'vit_b')
    
    # Show recommendations
    print("\n" + "="*80)
    print("TRAINING RECOMMENDATIONS")
    print("="*80)
    
    for use_case, rec in TRAINING_RECOMMENDATIONS.items():
        print(f"\n{use_case.upper().replace('_', ' ')}:")
        print(f"rec: {rec}")
        print(f"Description: {rec['description']}")
        print(f"Models: {', '.join(rec['recommended_models'])}")
        print(f"Epochs: {rec['epochs']}")
        print(f"Batch Size: {rec['batch_size']}")
        print(f"Notes: {rec['notes']}")
