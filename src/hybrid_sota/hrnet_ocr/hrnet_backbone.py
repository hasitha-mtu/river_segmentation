"""
HRNet Backbone Module
=====================

Implements High-Resolution Network (HRNet) for feature extraction.
HRNet maintains high-resolution representations throughout the network
through parallel multi-resolution branches with repeated fusion.

Key innovations:
- Parallel branches at multiple resolutions
- Repeated multi-scale fusion
- No aggressive downsampling → preserves spatial details
- High-resolution features maintained throughout

Reference:
    Wang et al. "Deep High-Resolution Representation Learning for Visual Recognition" TPAMI 2020

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BasicBlock(nn.Module):
    """
    Basic residual block for HRNet.
    
    Simple 3x3 → 3x3 conv structure with residual connection.
    Used in higher-resolution branches for efficiency.
    
    Architecture:
        Input → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+) → ReLU → Output
                                                         ↑
                                                    Residual/Downsample
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Convolution stride
        downsample: Optional downsampling layer for residual
    """
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for HRNet.
    
    1x1 → 3x3 → 1x1 conv structure (reduce → process → expand).
    Used in lower-resolution branches and early stages.
    More efficient for processing low-resolution features.
    
    Architecture:
        Input → Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU
              → Conv1x1 → BN → (+) → ReLU → Output
                            ↑
                       Residual/Downsample
    
    Args:
        in_channels: Input channels
        out_channels: Bottleneck channels (output will be out_channels * expansion)
        stride: Convolution stride
        downsample: Optional downsampling layer for residual
    """
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class HighResolutionModule(nn.Module):
    """
    High-Resolution Module: Core building block of HRNet.
    
    Maintains multiple parallel branches at different resolutions
    and performs multi-scale feature fusion via exchange units.
    
    Key Innovation: Instead of recovering high-resolution from low-resolution
    (like U-Net), HRNet keeps high-resolution features throughout!
    
    Architecture (4 branches example):
        ┌─ HR Branch (1/4)  ────────────────────────┐
        ├─ MR Branch (1/8)  ────────────────────────┤
        ├─ LR Branch (1/16) ────────────────────────┤  Multi-scale
        └─ VLR Branch (1/32)────────────────────────┘  Exchange
                                ↓
        ┌─ Fused HR  ────────────────────────────────┐
        ├─ Fused MR  ────────────────────────────────┤
        ├─ Fused LR  ────────────────────────────────┤
        └─ Fused VLR ────────────────────────────────┘
    
    Args:
        num_branches: Number of parallel branches
        block: Type of residual block (BasicBlock or Bottleneck)
        num_blocks: Number of blocks in each branch
        num_channels: Channel count for each branch
        fuse_method: Fusion method ('sum' or 'cat')
    """
    def __init__(
        self,
        num_branches: int,
        block: nn.Module,
        num_blocks: List[int],
        num_channels: List[int],
        fuse_method: str = 'sum'
    ):
        super().__init__()
        
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        
        # Build parallel branches
        self.branches = nn.ModuleList([
            self._make_branch(
                branch_index=i,
                block=block,
                num_blocks=num_blocks[i],
                num_channels=num_channels[i]
            )
            for i in range(num_branches)
        ])
        
        # Fusion layers for multi-scale exchange
        self.fuse_layers = self._make_fuse_layers(num_channels)
        self.relu = nn.ReLU(inplace=False)
    
    def _make_branch(
        self,
        branch_index: int,
        block: nn.Module,
        num_blocks: int,
        num_channels: int
    ) -> nn.Sequential:
        """Build a single branch with multiple blocks."""
        layers = []
        
        for i in range(num_blocks):
            layers.append(
                block(
                    in_channels=num_channels,
                    out_channels=num_channels // block.expansion
                )
            )
        
        return nn.Sequential(*layers)
    
    def _make_fuse_layers(self, num_channels: List[int]) -> nn.ModuleList:
        """
        Create fusion layers for multi-scale feature exchange.
        
        Branch indexing in HRNet:
        - Branch 0: 1/4 resolution (HIGHEST resolution)
        - Branch 1: 1/8 resolution
        - Branch 2: 1/16 resolution  
        - Branch 3: 1/32 resolution (LOWEST resolution)
        
        For each output branch i, fuse information from all input branches j:
        - j == i (same scale): identity
        - j > i  (LOWER res → HIGHER res): upsample
        - j < i  (HIGHER res → LOWER res): downsample
        
        Returns:
            ModuleList of fusion layers
        """
        if self.num_branches == 1:
            return None
        
        fuse_layers = nn.ModuleList()
        
        for i in range(self.num_branches):  # For each output branch
            fuse_layer = nn.ModuleList()
            
            for j in range(self.num_branches):  # From each input branch
                if j > i:  # LOWER resolution → HIGHER resolution (upsample)
                    # Use 1x1 conv + bilinear upsampling
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels[j], num_channels[i],
                                kernel_size=1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels[i]),
                            nn.Upsample(
                                scale_factor=2**(j-i),  # j > i, so this gives upsampling
                                mode='bilinear', 
                                align_corners=False
                            )
                        )
                    )
                    
                elif j == i:  # Same resolution (identity)
                    fuse_layer.append(None)
                    
                else:  # HIGHER resolution → LOWER resolution (downsample)
                    # Chain of strided convs to downsample
                    downsample_layers = []
                    for k in range(i - j):  # Number of 2x downsamples needed
                        if k == i - j - 1:
                            # Last downsampling layer
                            downsample_layers.extend([
                                nn.Conv2d(
                                    num_channels[j], num_channels[i],
                                    kernel_size=3, stride=2, padding=1, bias=False
                                ),
                                nn.BatchNorm2d(num_channels[i])
                            ])
                        else:
                            # Intermediate downsampling layers
                            downsample_layers.extend([
                                nn.Conv2d(
                                    num_channels[j], num_channels[j],
                                    kernel_size=3, stride=2, padding=1, bias=False
                                ),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True)
                            ])
                    fuse_layer.append(nn.Sequential(*downsample_layers))
            
            fuse_layers.append(fuse_layer)
        
        return fuse_layers
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through parallel branches with multi-scale fusion.
        
        Args:
            x: List of feature maps, one per branch
        
        Returns:
            List of fused feature maps
        """
        # Single branch case
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        
        # Process each branch independently
        branch_outputs = []
        for i in range(self.num_branches):
            branch_outputs.append(self.branches[i](x[i]))
        
        # Fuse features across branches
        fused_outputs = []
        for i in range(self.num_branches):
            # Start with same-scale features
            fusion_sum = branch_outputs[i]
            
            # Add features from other branches
            for j in range(self.num_branches):
                if i != j and self.fuse_layers[i][j] is not None:
                    fusion_sum = fusion_sum + self.fuse_layers[i][j](branch_outputs[j])
            
            fused_outputs.append(self.relu(fusion_sum))
        
        return fused_outputs


class HRNetBackbone(nn.Module):
    """
    HRNet Backbone for semantic segmentation.
    
    Architecture progressively increases number of parallel branches:
    - Stem: Initial downsampling to 1/4 resolution
    - Stage 1: Single high-resolution branch
    - Stage 2: 2 parallel branches (1/4, 1/8)
    - Stage 3: 3 parallel branches (1/4, 1/8, 1/16) with 4 repeated modules
    - Stage 4: 4 parallel branches (1/4, 1/8, 1/16, 1/32) with 3 repeated modules
    
    Each stage performs repeated multi-scale feature fusion.
    
    Args:
        in_channels: Input image channels
        base_channels: Base channel count (18, 32, or 48 for W18, W32, W48)
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 48):
        super().__init__()
        
        self.base_channels = base_channels
        
        # Stem: Initial downsampling (1/4 resolution)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: Single high-resolution branch
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        
        # Transition to Stage 2 (add second branch)
        self.transition1 = self._make_transition_layer(
            [256],  # Output of stage 1
            [base_channels, base_channels * 2]
        )
        
        # Stage 2: 2 branches
        self.stage2 = nn.Sequential(
            HighResolutionModule(
                num_branches=2,
                block=BasicBlock,
                num_blocks=[4, 4],
                num_channels=[base_channels, base_channels * 2]
            )
        )
        
        # Transition to Stage 3 (add third branch)
        self.transition2 = self._make_transition_layer(
            [base_channels, base_channels * 2],
            [base_channels, base_channels * 2, base_channels * 4]
        )
        
        # Stage 3: 3 branches, repeated 4 times
        self.stage3 = nn.Sequential(*[
            HighResolutionModule(
                num_branches=3,
                block=BasicBlock,
                num_blocks=[4, 4, 4],
                num_channels=[base_channels, base_channels * 2, base_channels * 4]
            )
            for _ in range(4)
        ])
        
        # Transition to Stage 4 (add fourth branch)
        self.transition3 = self._make_transition_layer(
            [base_channels, base_channels * 2, base_channels * 4],
            [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        )
        
        # Stage 4: 4 branches, repeated 3 times
        self.stage4 = nn.Sequential(*[
            HighResolutionModule(
                num_branches=4,
                block=BasicBlock,
                num_blocks=[4, 4, 4, 4],
                num_channels=[base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
            )
            for _ in range(3)
        ])
        
        # Store output channels for decoder
        self.out_channels = [
            base_channels, 
            base_channels * 2, 
            base_channels * 4, 
            base_channels * 8
        ]
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize backbone weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(
        self,
        block: nn.Module,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Build a layer of residual blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = [block(in_channels, out_channels, stride, downsample)]
        in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_transition_layer(
        self,
        num_channels_pre: List[int],
        num_channels_cur: List[int]
    ) -> nn.ModuleList:
        """
        Create transition layers between stages.
        
        Adds new lower-resolution branches when moving to next stage.
        """
        num_branches_cur = len(num_channels_cur)
        num_branches_pre = len(num_channels_pre)
        
        transition_layers = nn.ModuleList()
        
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                # Existing branch - adjust channels if needed
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre[i], num_channels_cur[i],
                                kernel_size=3, padding=1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                # New branch - downsample from last existing branch
                downsample_layers = []
                for j in range(i - num_branches_pre + 1):
                    in_channels = num_channels_pre[-1]
                    out_channels = num_channels_cur[i] if j == i - num_branches_pre else in_channels
                    downsample_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels, out_channels,
                                kernel_size=3, stride=2, padding=1, bias=False
                            ),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*downsample_layers))
        
        return transition_layers
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through HRNet backbone.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            List of 4 feature maps at different resolutions
            [f1 (1/4), f2 (1/8), f3 (1/16), f4 (1/32)]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Transition 1 → Stage 2
        x_list = []
        for i, trans in enumerate(self.transition1):
            if trans is not None:
                x_list.append(trans(x))
            else:
                x_list.append(x)
        x_list = self.stage2(x_list)
        
        # Transition 2 → Stage 3
        x_list_new = []
        for i, trans in enumerate(self.transition2):
            if trans is not None:
                if i < len(x_list):
                    x_list_new.append(trans(x_list[i]))
                else:
                    x_list_new.append(trans(x_list[-1]))
            else:
                x_list_new.append(x_list[i])
        x_list = self.stage3(x_list_new)
        
        # Transition 3 → Stage 4
        x_list_new = []
        for i, trans in enumerate(self.transition3):
            if trans is not None:
                if i < len(x_list):
                    x_list_new.append(trans(x_list[i]))
                else:
                    x_list_new.append(trans(x_list[-1]))
            else:
                x_list_new.append(x_list[i])
        x_list = self.stage4(x_list_new)
        
        return x_list


# Predefined HRNet configurations
HRNET_CONFIGS = {
    'w18': {'base_channels': 18},
    'w32': {'base_channels': 32},
    'w48': {'base_channels': 48}
}


def build_hrnet_backbone(variant: str = 'w48', in_channels: int = 3) -> HRNetBackbone:
    """
    Factory function to build HRNet backbone variants.
    
    Args:
        variant: Model size ('w18', 'w32', 'w48')
        in_channels: Number of input channels
    
    Returns:
        HRNetBackbone instance
    
    Example:
        >>> backbone = build_hrnet_backbone('w48', in_channels=1)
        >>> features = backbone(torch.randn(1, 1, 512, 512))
        >>> print([f.shape for f in features])
    """
    if variant not in HRNET_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(HRNET_CONFIGS.keys())}")
    
    config = HRNET_CONFIGS[variant]
    return HRNetBackbone(in_channels=in_channels, **config)


if __name__ == "__main__":
    # Test backbone
    print("Testing HRNet Backbone...")
    
    backbone = build_hrnet_backbone('w48', in_channels=3)
    x = torch.randn(2, 3, 512, 512)
    
    features = backbone(x)
    
    print(f"Input shape: {x.shape}")
    for i, feat in enumerate(features):
        print(f"Branch {i+1} (1/{4*2**i} res): {feat.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in backbone.parameters())
    print(f"\nTotal parameters: {params:,}")
