"""
ConvNeXt Encoder Module
=======================

Implements the ConvNeXt backbone for feature extraction.
ConvNeXt brings transformer design principles to CNNs while maintaining efficiency.

Key components:
- Layer Normalization for 2D feature maps
- ConvNeXt blocks with depthwise convolutions
- Stochastic depth for regularization
- Multi-stage hierarchical feature extraction

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import List
import math


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2D feature maps (channels-first format).
    
    ConvNeXt uses LayerNorm instead of BatchNorm for better stability.
    
    Args:
        num_channels: Number of channels
        eps: Small value to avoid division by zero
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) for regularization.
    
    Randomly drops entire samples during training to prevent overfitting
    and improve generalization.
    
    Args:
        drop_prob: Probability of dropping a path
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block: Modernized ResNet bottleneck with transformer principles.
    
    Design choices inspired by transformers:
    - Depthwise conv (7x7) for spatial mixing (similar to self-attention)
    - Inverted bottleneck (expand → compress) like transformer FFN
    - LayerNorm instead of BatchNorm
    - GELU activation (used in transformers)
    - Larger kernel (7x7) for increased receptive field
    
    Architecture:
        Input → DWConv7x7 → LayerNorm → Linear(4x) → GELU → Linear → Scale → DropPath → Output
                ↓_________________________________________________________________↑
                                        Residual Connection
    
    Args:
        dim: Number of input/output channels
        drop_path: Stochastic depth rate
        layer_scale_init_value: Initial value for layer scale parameter
    """
    def __init__(
        self, 
        dim: int, 
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        
        # Depthwise convolution (7x7) - spatial mixing
        self.dwconv = nn.Conv2d(
            dim, dim, 
            kernel_size=7, 
            padding=3, 
            groups=dim  # Depthwise: each channel processed separately
        )
        
        # Layer normalization
        self.norm = LayerNorm2d(dim)
        
        # Pointwise/channel mixing (inverted bottleneck)
        # Expand to 4x channels, then compress back
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 1x1 conv via Linear
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer scale for better optimization
        # Learnable per-channel scaling factor
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        
        # Stochastic depth for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        # Depthwise conv + norm
        x = self.dwconv(x)
        x = self.norm(x)
        
        # Channel-wise operations (need to permute for Linear layers)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Apply layer scale
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Residual connection with stochastic depth
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtStage(nn.Module):
    """
    ConvNeXt Stage: Downsampling layer + sequence of ConvNeXt blocks.
    
    Each stage processes features at a specific resolution and consists of:
    1. Downsampling layer (2x2 conv with stride 2) - except first stage
    2. Sequence of ConvNeXt blocks
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        depth: Number of ConvNeXt blocks in this stage
        drop_path_rates: List of drop path rates for each block
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path_rates: List[float]
    ):
        super().__init__()
        
        # Downsampling layer (2x2 conv with stride 2)
        # Only downsample if channels change (not for first block in stage)
        self.downsample = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        ) if in_channels != out_channels else nn.Identity()
        
        # Stack of ConvNeXt blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=out_channels,
                drop_path=drop_path_rates[i]
            )
            for i in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt Backbone Encoder for feature extraction.
    
    Extracts hierarchical multi-scale features at 4 different resolutions:
    - C1: 1/4 resolution   (after stem)
    - C2: 1/8 resolution   (stage 1)
    - C3: 1/16 resolution  (stage 2)
    - C4: 1/32 resolution  (stage 3)
    
    The aggressive stem (4x4 conv, stride 4) immediately reduces resolution
    by 4x, which is more efficient than gradual downsampling.
    
    Architecture:
        Input (H×W) → Stem → Stage1 → Stage2 → Stage3 → Stage4
                       ↓      ↓        ↓        ↓        ↓
        Resolution:   H/4    H/4      H/8      H/16     H/32
        Channels:     C0     C1       C2       C3       C4
    
    Args:
        in_channels: Number of input channels (3 for RGB, 1 for luminance)
        depths: Number of blocks in each stage [3, 3, 9, 3] for Base
        dims: Channel dimensions for each stage [128, 256, 512, 1024] for Base
        drop_path_rate: Maximum drop path rate (linearly increases across stages)
    """
    def __init__(
        self,
        in_channels: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [128, 256, 512, 1024],
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        
        # Stem: Aggressive downsampling (4x4 conv with stride 4)
        # Reduces spatial dimensions by 4x immediately
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        
        # Calculate drop path rates (linearly increasing across all blocks)
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Build 4 stages
        self.stages = nn.ModuleList()
        cur_dp_idx = 0
        
        for i in range(4):
            stage = ConvNeXtStage(
                in_channels=dims[i-1] if i > 0 else dims[0],
                out_channels=dims[i],
                depth=depths[i],
                drop_path_rates=dp_rates[cur_dp_idx:cur_dp_idx + depths[i]]
            )
            self.stages.append(stage)
            cur_dp_idx += depths[i]
        
        # Output normalization for each stage
        self.norms = nn.ModuleList([LayerNorm2d(dims[i]) for i in range(4)])
        
        # Store output channel dimensions
        self.out_channels = dims
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using proper initialization schemes."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            List of feature maps [C1, C2, C3, C4] at different resolutions
        """
        features = []
        
        x = self.stem(x)  # 1/4 resolution
        
        for i, (stage, norm) in enumerate(zip(self.stages, self.norms)):
            x = stage(x)
            x = norm(x)
            features.append(x)
        
        return features


# Predefined ConvNeXt configurations
CONVNEXT_CONFIGS = {
    'tiny': {
        'depths': [3, 3, 9, 3],
        'dims': [96, 192, 384, 768],
        'drop_path_rate': 0.1
    },
    'small': {
        'depths': [3, 3, 27, 3],
        'dims': [96, 192, 384, 768],
        'drop_path_rate': 0.2
    },
    'base': {
        'depths': [3, 3, 27, 3],
        'dims': [128, 256, 512, 1024],
        'drop_path_rate': 0.3
    },
    'large': {
        'depths': [3, 3, 27, 3],
        'dims': [192, 384, 768, 1536],
        'drop_path_rate': 0.4
    },
    'xlarge': {
        'depths': [3, 3, 27, 3],
        'dims': [256, 512, 1024, 2048],
        'drop_path_rate': 0.5
    }
}


def build_convnext_encoder(variant: str = 'tiny', in_channels: int = 3) -> ConvNeXtEncoder:
    """
    Factory function to build ConvNeXt encoder variants.
    
    Args:
        variant: Model size ('tiny', 'small', 'base', 'large', 'xlarge')
        in_channels: Number of input channels
    
    Returns:
        ConvNeXtEncoder instance
    
    Example:
        >>> encoder = build_convnext_encoder('tiny', in_channels=1)
        >>> features = encoder(torch.randn(1, 1, 512, 512))
        >>> print([f.shape for f in features])
    """
    if variant not in CONVNEXT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(CONVNEXT_CONFIGS.keys())}")
    
    config = CONVNEXT_CONFIGS[variant]
    return ConvNeXtEncoder(in_channels=in_channels, **config)


if __name__ == "__main__":
    # Test encoder
    print("Testing ConvNeXt Encoder...")
    
    encoder = build_convnext_encoder('tiny', in_channels=3)
    x = torch.randn(2, 3, 512, 512)
    
    features = encoder(x)
    
    print(f"Input shape: {x.shape}")
    for i, feat in enumerate(features):
        print(f"C{i+1} shape: {feat.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {params:,}")
