"""
Quick Fix for 11GB Workstation OOM
===================================

Works on 8GB laptop but fails on 11GB workstation?
This is an environment issue - here's the fix!

Author: Hasitha
Date: December 2025
"""

# CRITICAL FIX #1: Set memory allocator BEFORE importing torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution

import torch
import torch.nn as nn


def apply_workstation_fixes():
    """Apply all fixes for workstation OOM issues"""
    
    print("=" * 80)
    print("Applying Workstation OOM Fixes")
    print("=" * 80)
    
    # Fix 1: Force garbage collection
    import gc
    gc.collect()
    print("✓ Forced garbage collection")
    
    # Fix 2: Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("✓ Cleared CUDA cache")
    
    # Fix 3: Set cuDNN settings
    torch.backends.cudnn.benchmark = True  # Can help with memory
    torch.backends.cudnn.enabled = True
    print("✓ Configured cuDNN")
    
    # Fix 4: Disable TF32 (can cause issues)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("✓ Disabled TF32")
    
    # Fix 5: Set memory fraction (prevent fragmentation)
    if torch.cuda.is_available():
        # Reserve 90% of GPU memory (prevents fragmentation)
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        print("✓ Set memory fraction to 0.9")
    
    print("=" * 80)
    print("All fixes applied!")
    print("=" * 80)


# Apply fixes when this module is imported
apply_workstation_fixes()


# ============================================================================
# USAGE IN YOUR TRAINING SCRIPT
# ============================================================================

"""
Add this at the VERY TOP of your train.py, BEFORE any other imports:

    import workstation_fix  # Apply fixes first
    import torch
    # ... rest of your code ...

Or copy the environment variable setting:

    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    import torch
    # ... rest of your code ...
"""


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WORKSTATION FIX - Test Results")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print(f"\n✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Test allocation
        print("\nTesting memory allocation...")
        try:
            x = torch.randn(2, 3, 512, 512, device='cuda')
            print(f"  ✓ Allocated 2×3×512×512 tensor")
            print(f"  Memory used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            del x
            torch.cuda.empty_cache()
            print(f"  ✓ Freed memory")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    else:
        print("✗ CUDA not available")
    
    print("\n" + "=" * 80)
    print("To use this fix:")
    print("=" * 80)
    print("""
    METHOD 1: Import at top of train.py
    ------------------------------------
    import workstation_fix  # Add this line at the very top
    import torch
    # ... rest of your code
    
    METHOD 2: Set environment variable
    ----------------------------------
    # Windows CMD:
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
    python train.py ...
    
    # Windows PowerShell:
    $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512,expandable_segments:True"
    python train.py ...
    
    # Linux:
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
    python train.py ...
    
    METHOD 3: Add to train.py before imports
    -----------------------------------------
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    import torch  # Must be AFTER setting env var
    """)
