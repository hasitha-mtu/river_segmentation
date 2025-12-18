"""
GPU Environment Diagnostic - 8GB vs 11GB Comparison
===================================================

If models work on 8GB laptop but fail on 11GB workstation,
it's an environment issue, not memory capacity!

Author: Hasitha
Date: December 2025
"""

import torch
import sys
import os


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_pytorch_environment():
    """Check PyTorch and CUDA environment"""
    print_section("PyTorch Environment")
    
    print(f"Python version:     {sys.version}")
    print(f"PyTorch version:    {torch.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version:       {torch.version.cuda}")
        print(f"cuDNN version:      {torch.backends.cudnn.version()}")
        print(f"Number of GPUs:     {torch.cuda.device_count()}")
        print(f"Current GPU:        {torch.cuda.current_device()}")
        print(f"GPU Name:           {torch.cuda.get_device_name(0)}")
        
        # GPU properties
        props = torch.cuda.get_device_properties(0)
        print(f"Total memory:       {props.total_memory / 1024**3:.2f} GB")
        print(f"Multi-processor:    {props.multi_processor_count}")
        print(f"CUDA capability:    {props.major}.{props.minor}")


def check_memory_allocator():
    """Check CUDA memory allocator settings"""
    print_section("CUDA Memory Allocator Settings")
    
    # Check environment variables
    env_vars = [
        'PYTORCH_CUDA_ALLOC_CONF',
        'CUDA_LAUNCH_BLOCKING',
        'PYTORCH_NO_CUDA_MEMORY_CACHING',
        'CUDA_VISIBLE_DEVICES'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var:35s}: {value}")
    
    # Check PyTorch memory allocator
    if torch.cuda.is_available():
        print(f"\nMemory allocator:")
        try:
            # PyTorch 2.0+
            backend = torch.cuda.memory._get_allocator_backend()
            print(f"  Backend: {backend}")
        except:
            print(f"  Backend: native (default)")


def check_current_memory():
    """Check current GPU memory usage"""
    print_section("Current GPU Memory Usage")
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    # Get memory stats
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(0) / 1024**3
    
    print(f"Currently allocated:  {allocated:.2f} GB")
    print(f"Currently reserved:   {reserved:.2f} GB")
    print(f"Max allocated:        {max_allocated:.2f} GB")
    print(f"Max reserved:         {max_reserved:.2f} GB")
    
    # Available memory
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    available = total - reserved
    print(f"\nTotal GPU memory:     {total:.2f} GB")
    print(f"Available:            {available:.2f} GB")


def check_background_processes():
    """Check for background GPU processes"""
    print_section("Background GPU Processes")
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.stdout.strip():
            print("GPU processes found:")
            print(result.stdout)
        else:
            print("No background GPU processes detected")
    except Exception as e:
        print(f"Could not check: {e}")
        print("Run manually: nvidia-smi")


def test_memory_allocation():
    """Test memory allocation patterns"""
    print_section("Memory Allocation Test")
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    print("Testing allocation patterns...")
    
    sizes = [
        (1, 3, 256, 256),
        (2, 3, 256, 256),
        (1, 3, 384, 384),
        (2, 3, 384, 384),
        (1, 3, 512, 512),
    ]
    
    for size in sizes:
        try:
            # Allocate
            x = torch.randn(*size, device='cuda')
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            
            print(f"  Shape {size}: {allocated:.2f} GB - ✓")
            
            # Free
            del x
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"  Shape {size}: FAILED - {str(e)[:50]}")


def check_cudnn_settings():
    """Check cuDNN settings"""
    print_section("cuDNN Settings")
    
    if torch.cuda.is_available():
        print(f"cuDNN enabled:          {torch.backends.cudnn.enabled}")
        print(f"cuDNN benchmark:        {torch.backends.cudnn.benchmark}")
        print(f"cuDNN deterministic:    {torch.backends.cudnn.deterministic}")
        print(f"cuDNN allow_tf32:       {torch.backends.cudnn.allow_tf32}")
        
        if hasattr(torch.backends.cuda, 'matmul'):
            print(f"CUDA matmul allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")


def compare_configurations():
    """Provide comparison guidance"""
    print_section("Configuration Comparison Guide")
    
    print("""
LIKELY CAUSES (8GB works, 11GB fails):

1. **Memory Fragmentation** (MOST LIKELY)
   - 11GB GPU has fragmented memory from previous runs
   - Solution: Reboot workstation or reset GPU
   
2. **Different PyTorch Versions**
   - 8GB laptop: Older PyTorch (better memory management)
   - 11GB workstation: Newer PyTorch (different allocator)
   - Check versions above
   
3. **Different CUDA Versions**
   - Memory allocator behavior changed between CUDA versions
   - Check CUDA versions above
   
4. **cuDNN Settings**
   - benchmark=True can use more memory
   - Check settings above
   
5. **Background Processes**
   - Something else using GPU on workstation
   - Check nvidia-smi output
   
6. **Driver Differences**
   - Newer/older drivers handle memory differently
   
7. **Memory Allocator Backend**
   - PyTorch 2.0+ introduced new allocators
   - Can cause different behavior

RUN THIS SCRIPT ON BOTH MACHINES:
1. On 8GB laptop (working)
2. On 11GB workstation (failing)
3. Compare outputs
4. Identify differences
""")


def provide_solutions():
    """Provide solutions"""
    print_section("Solutions to Try")
    
    print("""
SOLUTION 1: Clear GPU Memory (TRY FIRST!)
==========================================
Windows:
    # Kill all Python
    taskkill /F /IM python.exe /T
    
    # Restart CUDA (may need admin)
    # Or just reboot

Linux:
    # Kill all Python
    pkill -9 python
    
    # Reset GPU (requires root)
    sudo nvidia-smi --gpu-reset
    
    # Or reboot
    sudo reboot


SOLUTION 2: Set Memory Allocator (IMPORTANT!)
==============================================
Add to your script BEFORE any CUDA operations:

Windows (PowerShell):
    $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512,expandable_segments:True"
    python train.py ...

Windows (CMD):
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
    python train.py ...

Linux:
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
    python train.py ...

Or in Python at the very start:
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    import torch  # Must be AFTER setting env var


SOLUTION 3: Match Laptop Settings
==================================
If laptop uses different versions, try matching:

PyTorch version:
    pip install torch==X.X.X  # Use same version as laptop
    
cuDNN settings (in train.py before training):
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


SOLUTION 4: Disable Memory Caching
===================================
In train.py, add after each epoch:
    torch.cuda.empty_cache()
    
Or more aggressively after each batch:
    if batch_idx % 10 == 0:
        torch.cuda.empty_cache()


SOLUTION 5: Use Native Allocator
=================================
Force PyTorch to use native allocator:

    export PYTORCH_CUDA_ALLOC_CONF=backend:native
    python train.py ...


SOLUTION 6: Smaller Max Split Size
===================================
Limit maximum split size:

    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    python train.py ...
""")


def main():
    """Run all diagnostics"""
    print("=" * 80)
    print("GPU ENVIRONMENT DIAGNOSTIC")
    print("8GB Works But 11GB Fails - Why?")
    print("=" * 80)
    
    check_pytorch_environment()
    check_memory_allocator()
    check_current_memory()
    check_background_processes()
    check_cudnn_settings()
    test_memory_allocation()
    compare_configurations()
    provide_solutions()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("""
1. Run this script on BOTH machines
2. Compare the outputs
3. Note any differences in:
   - PyTorch version
   - CUDA version
   - Memory allocator settings
   - cuDNN settings
4. Try Solution 1 (clear GPU) first
5. Try Solution 2 (set PYTORCH_CUDA_ALLOC_CONF) second
6. If still failing, match versions to laptop
""")


if __name__ == "__main__":
    main()
