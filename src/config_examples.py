"""
Example Configurations for Different Training Scenarios
Copy and modify these in your main() function
"""

# ===== Example 1: CNN Baseline Configuration =====
def get_cnn_config():
    """Configuration optimized for CNN models (U-Net, DeepLabV3+, etc.)"""
    config = {
        'model': {
            'name': 'unet',
            'variant': None,
            'n_channels': 3,
            'n_classes': 1,
        },
        'data': {
            'data_root': './dataset/processed_1024',
            'image_size': 1024,
            'train_split': 0.9,
        },
        'training': {
            'batch_size': 8,  # CNNs can handle larger batch sizes
            'epochs': 150,
            'clip_grad': 1.0,
            'resume': False,
            'optimizer': {
                'type': 'adam',
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6,
            },
        },
        'loss': {
            'type': 'combined',
            'bce_weight': 1.0,
            'dice_weight': 1.0,
            'boundary_weight': 1.0,
            'use_boundary': True,
        },
        'system': {
            'seed': 42,
            'num_workers': 8,
            'output_dir': './experiments',
            'log_interval': 10,
            'save_interval': 10,
        }
    }
    return config


# ===== Example 2: Transformer Configuration =====
def get_transformer_config():
    """Configuration optimized for Transformer models (SegFormer, Swin-UNet)"""
    config = {
        'model': {
            'name': 'segformer',
            'variant': 'b0',
            'n_channels': 3,
            'n_classes': 1,
        },
        'data': {
            'data_root': './dataset/processed_1024',
            'image_size': 1024,
            'train_split': 0.9,
        },
        'training': {
            'batch_size': 4,  # Transformers need smaller batch sizes
            'epochs': 100,
            'clip_grad': 1.0,
            'resume': False,
            'optimizer': {
                'type': 'adamw',  # AdamW is better for transformers
                'learning_rate': 6e-5,  # Lower LR for transformers
                'weight_decay': 0.01,
            },
            'scheduler': {
                'type': 'warmup_cosine',  # Warmup helps transformers
                'min_lr': 1e-6,
                'warmup_epochs': 5,
            },
        },
        'loss': {
            'type': 'combined',
            'bce_weight': 1.0,
            'dice_weight': 1.0,
            'boundary_weight': 1.0,
            'use_boundary': True,
        },
        'system': {
            'seed': 42,
            'num_workers': 4,
            'output_dir': './experiments',
            'log_interval': 10,
            'save_interval': 10,
        }
    }
    return config


# ===== Example 3: Foundation Model Configuration =====
def get_foundation_model_config():
    """Configuration for foundation models (SAM, DINOv2)"""
    config = {
        'model': {
            'name': 'sam',
            'variant': 'vit_b',
            'n_channels': 3,
            'n_classes': 1,
        },
        'data': {
            'data_root': './dataset/processed_1024',
            'image_size': 1024,
            'train_split': 0.9,
        },
        'training': {
            'batch_size': 2,  # Foundation models are very large
            'epochs': 50,  # Often need fewer epochs with pretrained models
            'clip_grad': 1.0,
            'resume': False,
            'optimizer': {
                'type': 'adamw',
                'learning_rate': 1e-5,  # Very small LR for fine-tuning
                'weight_decay': 0.01,
            },
            'scheduler': {
                'type': 'warmup_cosine',
                'min_lr': 1e-7,
                'warmup_epochs': 3,
            },
        },
        'loss': {
            'type': 'combined',
            'bce_weight': 1.0,
            'dice_weight': 1.0,
            'boundary_weight': 1.0,
            'use_boundary': True,
        },
        'system': {
            'seed': 42,
            'num_workers': 4,
            'output_dir': './experiments',
            'log_interval': 5,
            'save_interval': 5,
        }
    }
    return config


# ===== Example 4: Quick Testing Configuration =====
def get_quick_test_config():
    """Configuration for quick testing/debugging"""
    config = {
        'model': {
            'name': 'unet',
            'variant': None,
            'n_channels': 3,
            'n_classes': 1,
        },
        'data': {
            'data_root': './dataset/processed_1024',
            'image_size': 512,  # Smaller for faster testing
            'train_split': 0.9,
        },
        'training': {
            'batch_size': 2,
            'epochs': 5,  # Just a few epochs for testing
            'clip_grad': 1.0,
            'resume': False,
            'optimizer': {
                'type': 'adam',
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6,
            },
        },
        'loss': {
            'type': 'dice',  # Simple loss for testing
            'bce_weight': 1.0,
            'dice_weight': 1.0,
        },
        'system': {
            'seed': 42,
            'num_workers': 2,
            'output_dir': './test_experiments',
            'log_interval': 1,
            'save_interval': 1,
        }
    }
    return config


# ===== Example 5: Resume Training Configuration =====
def get_resume_config():
    """Configuration for resuming training from checkpoint"""
    config = {
        'model': {
            'name': 'segformer',
            'variant': 'b0',
            'n_channels': 3,
            'n_classes': 1,
        },
        'data': {
            'data_root': './dataset/processed_1024',
            'image_size': 1024,
            'train_split': 0.9,
        },
        'training': {
            'batch_size': 4,
            'epochs': 150,  # Can extend training
            'clip_grad': 1.0,
            'resume': True,  # Resume from latest checkpoint
            'optimizer': {
                'type': 'adamw',
                'learning_rate': 6e-5,
                'weight_decay': 0.01,
            },
            'scheduler': {
                'type': 'warmup_cosine',
                'min_lr': 1e-6,
                'warmup_epochs': 5,
            },
        },
        'loss': {
            'type': 'combined',
            'bce_weight': 1.0,
            'dice_weight': 1.0,
            'boundary_weight': 1.0,
            'use_boundary': True,
        },
        'system': {
            'seed': 42,
            'num_workers': 4,
            'output_dir': './experiments',
            'log_interval': 10,
            'save_interval': 10,
        }
    }
    return config


# ===== Example 6: Ablation Study Configuration =====
def get_ablation_configs():
    """Generate multiple configs for ablation studies"""
    base_config = {
        'model': {
            'name': 'unet',
            'variant': None,
            'n_channels': 3,
            'n_classes': 1,
        },
        'data': {
            'data_root': './dataset/processed_1024',
            'image_size': 1024,
            'train_split': 0.9,
        },
        'training': {
            'batch_size': 4,
            'epochs': 100,
            'clip_grad': 1.0,
            'resume': False,
            'optimizer': {
                'type': 'adam',
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6,
            },
        },
        'loss': {
            'type': 'combined',
            'bce_weight': 1.0,
            'dice_weight': 1.0,
            'boundary_weight': 1.0,
            'use_boundary': True,
        },
        'system': {
            'seed': 42,
            'num_workers': 4,
            'output_dir': './experiments',
            'log_interval': 10,
            'save_interval': 10,
        }
    }
    
    # Different loss functions to test
    configs = []
    for loss_type in ['bce', 'dice', 'focal', 'combined']:
        config = base_config.copy()
        config['loss']['type'] = loss_type
        configs.append(config)
    
    return configs


# ===== How to Use These Configurations =====
"""
In your train_unified.py main() function, use like this:

# Option 1: Use a predefined config
from config_examples import get_cnn_config
config = get_cnn_config()
train_single_model(config)

# Option 2: Modify a predefined config
config = get_transformer_config()
config['training']['batch_size'] = 8
config['training']['epochs'] = 150
train_single_model(config)

# Option 3: Use for ablation studies
configs = get_ablation_configs()
for config in configs:
    train_single_model(config)

# Option 4: Train specific models with base config
config = get_cnn_config()
models_to_train = [
    ('unet', None),
    ('resunetpp', None),
    ('deeplabv3plus', None),
]
for model_name, variant in models_to_train:
    config['model']['name'] = model_name
    config['model']['variant'] = variant
    train_single_model(config)
"""
