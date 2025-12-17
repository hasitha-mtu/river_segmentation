"""
SAM Image Encoder for Semantic Segmentation
============================================

Adapted from Segment Anything Model (SAM) for semantic segmentation tasks.
Uses Vision Transformer (ViT) architecture with patch embeddings.

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, D) where N = (H/P) * (W/P)
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLPBlock(nn.Module):
    """MLP block with GELU activation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MLP.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Use bias in qkv projection
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SAMImageEncoder(nn.Module):
    """
    SAM-style Vision Transformer encoder.
    
    Extracts multi-scale features from images using a ViT architecture.
    
    Args:
        img_size: Input image size
        patch_size: Patch size for embedding
        in_channels: Input channels (3 for RGB)
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        out_channels: Output channels at each scale
    """
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        out_channels: List[int] = [96, 192, 384, 768]
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Multi-scale feature projections
        # Extract features at different depths for multi-scale output
        self.extract_layers = [depth // 4, depth // 2, depth * 3 // 4, depth - 1]
        
        # Project to output channels
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, out_ch, kernel_size=1)
            for out_ch in out_channels
        ])
        
        self._init_weights()
    
    def interpolate_pos_encoding(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Interpolate position embeddings for arbitrary input size.
        
        Args:
            x: Patch embeddings (B, N, D)
            H, W: Input image height and width
        
        Returns:
            Interpolated position embeddings (1, N, D)
        """
        N = x.shape[1]
        if N == self.pos_embed.shape[1]:
            return self.pos_embed
        
        # Calculate grid sizes
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        
        # Original grid size (from initialization)
        orig_size = self.patch_embed.grid_size
        
        # Current grid size (from input)
        h = H // self.patch_size
        w = W // self.patch_size
        
        # Reshape position embeddings to 2D grid
        patch_pos_embed = patch_pos_embed.reshape(
            1, orig_size, orig_size, dim
        ).permute(0, 3, 1, 2)  # (1, D, orig_size, orig_size)
        
        # Interpolate to current size
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h, w),
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape back to sequence
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, dim)
        
        return patch_pos_embed
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights_modules)
    
    def _init_weights_modules(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W)
        
        Returns:
            List of feature maps at different scales
            [f1, f2, f3, f4] with shapes:
            - f1: (B, out_channels[0], H/4, W/4)
            - f2: (B, out_channels[1], H/8, W/8)
            - f3: (B, out_channels[2], H/16, W/16)
            - f4: (B, out_channels[3], H/32, W/32)
        """
        B, _, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # ═══════════════════════════════════════════════════════════
        # FIX: Interpolate position embedding if input size differs
        # ═══════════════════════════════════════════════════════════
        pos_embed = self.interpolate_pos_encoding(x, H, W)
        
        # Add position embedding
        x = x + pos_embed
        
        # Extract features at different depths
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.extract_layers:
                features.append(x)
        
        # Apply normalization
        features = [self.norm(f) for f in features]
        
        # ═══════════════════════════════════════════════════════════
        # FIX: Calculate actual grid size from input, not initialization
        # ═══════════════════════════════════════════════════════════
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        # Reshape to spatial format
        spatial_features = []
        for f in features:
            # (B, N, D) -> (B, D, H', W')
            f = f.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
            spatial_features.append(f)
        
        # Project to output channels and create multi-scale pyramid
        outputs = []
        for i, (feat, conv) in enumerate(zip(spatial_features, self.fpn_convs)):
            # Project channels
            feat = conv(feat)
            
            # Resize to target scale
            # Target sizes: H/4, H/8, H/16, H/32
            target_size = (H // (4 * 2**i), W // (4 * 2**i))
            feat = F.interpolate(
                feat, size=target_size, mode='bilinear', align_corners=False
            )
            outputs.append(feat)
        
        return outputs


# Configuration for different SAM variants
SAM_CONFIGS = {
    'vit_b': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'out_channels': [96, 192, 384, 768]
    },
    'vit_l': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'out_channels': [128, 256, 512, 1024]
    },
    'vit_h': {
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'out_channels': [160, 320, 640, 1280]
    }
}


def build_sam_encoder(
    variant: str = 'vit_b',
    img_size: int = 1024,
    in_channels: int = 3
) -> SAMImageEncoder:
    """
    Build SAM encoder variant.
    
    Args:
        variant: Model size ('vit_b', 'vit_l', 'vit_h')
        img_size: Input image size
        in_channels: Input channels
    
    Returns:
        SAMImageEncoder
    """
    if variant not in SAM_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}")
    
    config = SAM_CONFIGS[variant]
    
    return SAMImageEncoder(
        img_size=img_size,
        patch_size=16,
        in_channels=in_channels,
        **config
    )


if __name__ == "__main__":
    print("Testing SAM Image Encoder")
    
    encoder = build_sam_encoder('vit_b', img_size=512, in_channels=3)
    x = torch.randn(2, 3, 512, 512)
    
    features = encoder(x)
    
    print(f"Input: {x.shape}")
    for i, f in enumerate(features):
        print(f"Feature {i+1}: {f.shape}")
    
    params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {params:,}")
