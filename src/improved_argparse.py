"""
Improved Argument Parser with Model-Variant Validation
=======================================================

Smart argparse that validates variants based on model type.

Author: Hasitha
Date: December 2025
"""

import argparse
import sys


# Define valid variants for each model type
MODEL_VARIANTS = {
    'unet': {
        'variants': None,  # No variants
        'description': 'U-Net architecture'
    },
    'unetpp': {
        'variants': None,  # No variants
        'description': 'U-Net++ architecture'
    },
    'resunetpp': {
        'variants': None,
        'description': 'Residual U-Net ++'
    },
    'deeplabv3plus': {
        'variants': None,
        'description': 'DeepLabV3+'
    },
    'deeplabv3plus_cbam': {
        'variants': None,
        'description': 'DeepLabV3+ CBAM'
    },
    'sam': {
        'variants': ['vit_b', 'vit_l', 'vit_h'],
        'default': 'vit_b',
        'description': 'Segment Anything Model (SAM)'
    },
    'dinov2': {
        'variants': ['vit_s', 'vit_b', 'vit_l', 'vit_g'],
        'default': 'vit_b',
        'description': 'DINOv2 foundation model'
    },
    'swin_unet': {
        'variants': ['tiny'],
        'default': 'tiny',
        'description': 'Swin Transformer U-Net'
    },
    'segformer': {
        'variants': ['b0', 'b2'],
        'default': 'b0',
        'description': 'SegFormer'
    },
    'convnext_upernet': {
        'variants': ['tiny', 'small', 'base'],
        'default': 'tiny',
        'description': 'ConvNeXt U-Net'
    },
    'hrnet_ocr': {
        'variants': ['w18', 'w32', 'w48'],
        'default': 'w18',
        'description': 'HRNet with OCR'
    }
}


def validate_model_variant(model, variant):
    """
    Validate that variant is valid for the given model type.
    
    Args:
        model: Model type string
        variant: Variant string (can be None)
    
    Returns:
        Validated variant (or None if model has no variants)
    
    Raises:
        argparse.ArgumentTypeError: If validation fails
    """
    if model not in MODEL_VARIANTS:
        valid_models = ', '.join(MODEL_VARIANTS.keys())
        raise argparse.ArgumentTypeError(
            f"Unknown model type '{model}'. "
            f"Valid models: {valid_models}"
        )
    
    model_info = MODEL_VARIANTS[model]
    valid_variants = model_info['variants']
    
    # Model has no variants
    if valid_variants is None:
        if variant is not None:
            raise argparse.ArgumentTypeError(
                f"Model '{model}' does not have variants. "
                f"Remove --variant argument."
            )
        return None
    
    # Model has variants
    if variant is None:
        # Use default variant
        default_variant = model_info.get('default')
        if default_variant:
            print(f"No variant specified. Using default: {default_variant}")
            return default_variant
        else:
            raise argparse.ArgumentTypeError(
                f"Model '{model}' requires a variant. "
                f"Valid variants: {', '.join(valid_variants)}"
            )
    
    # Validate provided variant
    if variant not in valid_variants:
        raise argparse.ArgumentTypeError(
            f"Invalid variant '{variant}' for model '{model}'. "
            f"Valid variants: {', '.join(valid_variants)}"
        )
    
    return variant


class ModelVariantAction(argparse.Action):
    """Custom action to validate model and variant together"""
    
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        
        # If this is the variant argument, validate against model
        if self.dest == 'variant':
            model = getattr(namespace, 'model', None)
            if model:
                try:
                    validated = validate_model_variant(model, values)
                    setattr(namespace, self.dest, validated)
                except argparse.ArgumentTypeError as e:
                    parser.error(str(e))


def create_model_help():
    """Create detailed help text for model argument"""
    help_lines = ["Model architecture type.\n\nAvailable models:"]
    
    for model, info in MODEL_VARIANTS.items():
        variants = info['variants']
        desc = info['description']
        
        if variants:
            variants_str = f" (variants: {', '.join(variants)})"
        else:
            variants_str = " (no variants)"
        
        help_lines.append(f"  • {model:20s} - {desc}{variants_str}")
    
    return '\n'.join(help_lines)


def create_variant_help():
    """Create detailed help text for variant argument"""
    help_lines = [
        "Model variant (required for some models).\n\n"
        "Variants by model type:"
    ]
    
    for model, info in MODEL_VARIANTS.items():
        variants = info['variants']
        if variants:
            default = info.get('default', variants[0])
            help_lines.append(
                f"  • {model:20s}: {', '.join(variants)} (default: {default})"
            )
    
    help_lines.append("\nModels without variants: " + 
                     ", ".join([m for m, i in MODEL_VARIANTS.items() 
                               if i['variants'] is None]))
    
    return '\n'.join(help_lines)


def add_model_arguments(parser):
    """
    Add model and variant arguments to parser with validation.
    
    Args:
        parser: ArgumentParser instance
    
    Returns:
        parser: Modified parser
    """
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=MODEL_VARIANTS.keys(),
        help=create_model_help()
    )
    
    parser.add_argument(
        '--variant',
        type=str,
        default=None,
        action=ModelVariantAction,
        help=create_variant_help()
    )
    
    return parser


def validate_args_post_parse(args):
    """
    Validate model-variant combination after parsing all arguments.
    
    Args:
        args: Parsed arguments namespace
    
    Returns:
        args: Validated arguments
    
    Raises:
        SystemExit: If validation fails
    """
    try:
        validated_variant = validate_model_variant(args.model, args.variant)
        args.variant = validated_variant
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    return args


# ============================================================================
# Example Usage in Training/Testing Scripts
# ============================================================================

def create_train_parser():
    """Example: Create parser for training script"""
    parser = argparse.ArgumentParser(
        description='Train segmentation model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add model arguments with validation
    parser = add_model_arguments(parser)
    
    # Data arguments
    parser.add_argument('--data-root', type=str, required=True,
                       help='Data root directory')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size (default: 512)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    
    return parser


def create_test_parser():
    """Example: Create parser for testing script"""
    parser = argparse.ArgumentParser(
        description='Test trained segmentation model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add model arguments with validation
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
    
    return parser


# ============================================================================
# Testing & Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Model-Variant Argument Parser - Examples")
    print("=" * 80)
    
    # Example 1: Valid model without variant
    print("\n1. Valid model without variant:")
    print("   Command: --model unet")
    try:
        result = validate_model_variant('unet', None)
        print(f"   ✓ Result: model=unet, variant={result}")
    except argparse.ArgumentTypeError as e:
        print(f"   ✗ Error: {e}")
    
    # Example 2: Invalid - providing variant for model without variants
    print("\n2. Invalid - variant for model without variants:")
    print("   Command: --model unet --variant vit_b")
    try:
        result = validate_model_variant('unet', 'vit_b')
        print(f"   ✓ Result: model=unet, variant={result}")
    except argparse.ArgumentTypeError as e:
        print(f"   ✗ Error: {e}")
    
    # Example 3: Valid model with variant
    print("\n3. Valid model with variant:")
    print("   Command: --model sam --variant vit_b")
    try:
        result = validate_model_variant('sam', 'vit_b')
        print(f"   ✓ Result: model=sam, variant={result}")
    except argparse.ArgumentTypeError as e:
        print(f"   ✗ Error: {e}")
    
    # Example 4: Model with variants, no variant provided (uses default)
    print("\n4. Model with variants, no variant provided (uses default):")
    print("   Command: --model sam")
    try:
        result = validate_model_variant('sam', None)
        print(f"   ✓ Result: model=sam, variant={result}")
    except argparse.ArgumentTypeError as e:
        print(f"   ✗ Error: {e}")
    
    # Example 5: Invalid variant for model
    print("\n5. Invalid variant for model:")
    print("   Command: --model sam --variant tiny")
    try:
        result = validate_model_variant('sam', 'tiny')
        print(f"   ✓ Result: model=sam, variant={result}")
    except argparse.ArgumentTypeError as e:
        print(f"   ✗ Error: {e}")
    
    # Example 6: Unknown model
    print("\n6. Unknown model:")
    print("   Command: --model unknown")
    try:
        result = validate_model_variant('unknown', None)
        print(f"   ✓ Result: model=unknown, variant={result}")
    except argparse.ArgumentTypeError as e:
        print(f"   ✗ Error: {e}")
    
    # Example 7: Show help text
    print("\n" + "=" * 80)
    print("Model Help Text:")
    print("=" * 80)
    print(create_model_help())
    
    print("\n" + "=" * 80)
    print("Variant Help Text:")
    print("=" * 80)
    print(create_variant_help())
    
    # Example 8: Full parser test
    print("\n" + "=" * 80)
    print("Full Parser Example:")
    print("=" * 80)
    
    parser = create_test_parser()
    
    # Test valid arguments
    test_cases = [
        ['--model', 'unet', '--checkpoint', 'model.pth', '--test-data', 'data/test'],
        ['--model', 'sam', '--variant', 'vit_b', '--checkpoint', 'model.pth', '--test-data', 'data/test'],
        ['--model', 'sam', '--checkpoint', 'model.pth', '--test-data', 'data/test'],  # Uses default
    ]
    
    for i, test_args in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {' '.join(test_args)}")
        try:
            args = parser.parse_args(test_args)
            args = validate_args_post_parse(args)
            print(f"  ✓ Parsed: model={args.model}, variant={args.variant}")
        except SystemExit:
            print(f"  ✗ Failed to parse")
