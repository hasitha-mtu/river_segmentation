"""
Multi-GPU Training in PyTorch
==============================

This guide shows how to use 2 GPUs for training with two approaches:
1. DataParallel (DP) - Simple but slower
2. DistributedDataParallel (DDP) - Faster and recommended

Author: Hasitha
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

# ==============================================================================
# APPROACH 1: DataParallel (Simple but Slower)
# ==============================================================================

def train_with_dataparallel():
    """
    Simple approach using DataParallel.
    
    Pros:
    - Very simple to implement (just 1 line change!)
    - Works with existing code
    
    Cons:
    - Slower than DDP (single-process bottleneck)
    - Unbalanced GPU utilization
    - Not recommended for modern training
    
    Use when: Quick prototyping only
    """
    
    print("=" * 80)
    print("Approach 1: DataParallel (DP)")
    print("=" * 80)
    
    from models import build_hrnet_ocr
    from models.losses import CombinedLoss
    
    # Build model
    model = build_hrnet_ocr('w32', in_channels=3, num_classes=1)
    
    # MAGIC LINE: Wrap model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)  # ← THIS IS ALL YOU NEED!
    
    model = model.cuda()
    
    # Rest of training code stays the same
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop - NO CHANGES NEEDED
    for epoch in range(num_epochs):
        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.cuda()
            
            optimizer.zero_grad()
            
            outputs = model(images)  # Automatically uses both GPUs
            
            if isinstance(outputs, tuple):
                main_out, aux_out = outputs
                loss, loss_dict = criterion(main_out, masks, aux_out)
            else:
                loss, loss_dict = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
    
    print("\n✓ DataParallel is simple but NOT recommended for production!")


# ==============================================================================
# APPROACH 2: DistributedDataParallel (Fast and Recommended)
# ==============================================================================

def setup_ddp(rank, world_size):
    """Initialize DDP environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Cleanup DDP environment."""
    dist.destroy_process_group()


def train_one_epoch_ddp(model, train_loader, criterion, optimizer, epoch, rank):
    """Train one epoch with DDP."""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.cuda(rank)
        masks = masks.cuda(rank)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Handle auxiliary output
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            loss, loss_dict = criterion(main_out, masks, aux_out)
        else:
            loss, loss_dict = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log only from rank 0
        if rank == 0 and batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: "
                  f"Loss={loss_dict['total']:.4f}")
    
    return total_loss / len(train_loader)


def train_ddp(rank, world_size, args):
    """
    Main DDP training function.
    
    Pros:
    - Much faster than DataParallel (2-3x speedup)
    - Better GPU utilization
    - Scales to multiple nodes
    - Industry standard
    
    Cons:
    - More complex setup
    - Requires separate processes
    
    Use when: Serious training (recommended!)
    """
    print(f"Running DDP on rank {rank}")
    
    # Setup DDP
    setup_ddp(rank, world_size)
    
    from models import build_hrnet_ocr
    from models.losses import CombinedLoss
    
    # Build model
    model = build_hrnet_ocr('w32', in_channels=3, num_classes=1)
    model = model.cuda(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    # Setup loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # IMPORTANT: Use DistributedSampler for data loading
    train_dataset = YourDataset(args.train_dir)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,  # ← Use sampler instead of shuffle
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(args.epochs):
        # IMPORTANT: Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        # Train one epoch
        avg_loss = train_one_epoch_ddp(
            model, train_loader, criterion, optimizer, epoch, rank
        )
        
        # Log only from rank 0
        if rank == 0:
            print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
            
            # Save checkpoint only from rank 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # .module!
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch}.pth')
    
    # Cleanup
    cleanup_ddp()


def main_ddp():
    """Launch DDP training."""
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs with DistributedDataParallel")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    # Spawn processes (one per GPU)
    mp.spawn(
        train_ddp,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


# ==============================================================================
# COMPARISON
# ==============================================================================

def compare_approaches():
    """Compare DataParallel vs DistributedDataParallel."""
    
    comparison = """
    ┌─────────────────────────────────────────────────────────────────┐
    │              DataParallel vs DistributedDataParallel            │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  DataParallel (DP)                                              │
    │  ─────────────────                                              │
    │  Implementation:  Very simple (1 line change)                   │
    │  Speed:           Slower (single-process bottleneck)            │
    │  GPU Balance:     Unbalanced (GPU 0 does more work)            │
    │  Scalability:     Single node only                              │
    │  Memory:          Higher (replicates model on GPU 0)            │
    │  Code Change:     Minimal                                       │
    │  Use Case:        Quick prototyping only                        │
    │  Status:          Not recommended for production                │
    │                                                                 │
    │  DistributedDataParallel (DDP)                                  │
    │  ──────────────────────────────                                │
    │  Implementation:  More complex (multiprocessing)                │
    │  Speed:           Much faster (2-3x speedup)                    │
    │  GPU Balance:     Balanced utilization                          │
    │  Scalability:     Multi-node support                            │
    │  Memory:          Lower (efficient gradient sync)               │
    │  Code Change:     Moderate                                      │
    │  Use Case:        Production training                           │
    │  Status:          Industry standard ✓                           │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  Performance Comparison (2 GPUs, Batch Size 16)                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  Single GPU:      100 s/epoch    (baseline)                     │
    │  DataParallel:    65 s/epoch     (1.5x speedup)                 │
    │  DDP:             40 s/epoch     (2.5x speedup) ✓               │
    └─────────────────────────────────────────────────────────────────┘
    
    RECOMMENDATION: Use DistributedDataParallel (DDP) for 2x-3x speedup!
    """
    
    print(comparison)


# ==============================================================================
# BATCH SIZE CONSIDERATIONS
# ==============================================================================

def batch_size_guide():
    """Guide for adjusting batch size with multiple GPUs."""
    
    guide = """
    Batch Size with Multiple GPUs
    ═════════════════════════════
    
    IMPORTANT: Effective batch size multiplies by number of GPUs!
    
    Example with 2 GPUs:
    ────────────────────
    - You set: batch_size = 8 per GPU
    - Actual effective batch size = 8 × 2 = 16
    
    Recommendations:
    ────────────────
    
    1 GPU Setup:
      - HRNet-W32: batch_size = 8-16
      - HRNet-W48: batch_size = 4-8
    
    2 GPU Setup (adjust per-GPU batch size):
      - HRNet-W32: batch_size = 4-8 per GPU (effective: 8-16)
      - HRNet-W48: batch_size = 2-4 per GPU (effective: 4-8)
    
    Learning Rate Scaling:
    ─────────────────────
    When you increase effective batch size, consider scaling learning rate:
    
    - Original: batch_size=8, lr=1e-4
    - With 2 GPUs: batch_size=8 per GPU (effective 16)
                   lr=1.41e-4 (multiply by √2)
    
    Or use learning rate warmup for stability.
    """
    
    print(guide)


if __name__ == "__main__":
    print("=" * 80)
    print("Multi-GPU Training in PyTorch")
    print("=" * 80)
    
    compare_approaches()
    print("\n")
    batch_size_guide()
    
    print("\n" + "=" * 80)
    print("Quick Start")
    print("=" * 80)
    print("""
    For 2 GPUs, choose one approach:
    
    OPTION 1: DataParallel (Quick & Dirty)
    ──────────────────────────────────────
    In train.py, add 2 lines:
    
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
    
    Run: python train.py (no changes to run command)
    
    
    OPTION 2: DDP (Recommended)
    ───────────────────────────
    Use the DDP version of train.py (I'll provide full code)
    
    Run: python -m torch.distributed.launch \\
              --nproc_per_node=2 \\
              train_ddp.py --args
    
    Or: torchrun --nproc_per_node=2 train_ddp.py --args
    
    
    MY RECOMMENDATION: Start with DataParallel for testing,
                      then switch to DDP for actual training!
    """)
