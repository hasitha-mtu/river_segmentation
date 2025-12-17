"""
DINOv2 Encoder for Semantic Segmentation
=========================================

Vision Transformer backbone trained with DINOv2 self-supervised learning.

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
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
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
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


class MLP(nn.Module):
    """MLP block."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    """Layer scale module for better training dynamics."""
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class Block(nn.Module):
    """
    Transformer block with LayerScale.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Use bias in qkv
        drop: Dropout rate
        attn_drop: Attention dropout
        init_values: LayerScale init value
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        init_values: float = 1e-5
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.ls1 = LayerScale(dim, init_values=init_values)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2Encoder(nn.Module):
    """
    DINOv2 Vision Transformer encoder.
    
    Self-supervised vision transformer with strong representation learning.
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        out_channels: Output channels for multi-scale features
    """
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
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
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding (with CLS token)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Extract features at different depths
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
        Handles CLS token separately.
        
        Args:
            x: Patch embeddings with CLS token (B, N+1, D)
            H, W: Input image height and width
        
        Returns:
            Interpolated position embeddings (1, N+1, D)
        """
        N = x.shape[1] - 1  # Exclude CLS token
        if N == self.pos_embed.shape[1] - 1:
            return self.pos_embed
        
        # Separate CLS token and patch embeddings
        pos_embed_cls = self.pos_embed[:, :1, :]  # (1, 1, D)
        pos_embed_patches = self.pos_embed[:, 1:, :]  # (1, N_orig, D)
        
        dim = x.shape[-1]
        
        # Original grid size (from initialization)
        orig_size = self.patch_embed.grid_size
        
        # Current grid size (from input)
        h = H // self.patch_size
        w = W // self.patch_size
        
        # Reshape position embeddings to 2D grid
        pos_embed_patches = pos_embed_patches.reshape(
            1, orig_size, orig_size, dim
        ).permute(0, 3, 1, 2)  # (1, D, orig_size, orig_size)
        
        # Interpolate to current size
        pos_embed_patches = F.interpolate(
            pos_embed_patches,
            size=(h, w),
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape back to sequence
        pos_embed_patches = pos_embed_patches.permute(0, 2, 3, 1).reshape(1, h * w, dim)
        
        # Concatenate CLS token back
        pos_embed = torch.cat([pos_embed_cls, pos_embed_patches], dim=1)
        
        return pos_embed
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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
            List of multi-scale features
        """
        B, _, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        
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
                # Remove CLS token for spatial features
                features.append(x[:, 1:, :])
        
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
            target_size = (H // (4 * 2**i), W // (4 * 2**i))
            feat = F.interpolate(
                feat, size=target_size, mode='bilinear', align_corners=False
            )
            outputs.append(feat)
        
        return outputs


# Configuration for different DINOv2 variants
DINOV2_CONFIGS = {
    'vit_s': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'out_channels': [48, 96, 192, 384]
    },
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
    'vit_g': {
        'embed_dim': 1536,
        'depth': 40,
        'num_heads': 24,
        'out_channels': [192, 384, 768, 1536]
    }
}


def build_dinov2_encoder(
    variant: str = 'vit_b',
    img_size: int = 518,
    in_channels: int = 3
) -> DINOv2Encoder:
    """
    Build DINOv2 encoder variant.
    
    Args:
        variant: Model size ('vit_s', 'vit_b', 'vit_l', 'vit_g')
        img_size: Input image size
        in_channels: Input channels
    
    Returns:
        DINOv2Encoder
    """
    if variant not in DINOV2_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}")
    
    config = DINOV2_CONFIGS[variant]
    
    return DINOv2Encoder(
        img_size=img_size,
        patch_size=14,
        in_channels=in_channels,
        **config
    )


if __name__ == "__main__":
    print("Testing DINOv2 Encoder")
    
    encoder = build_dinov2_encoder('vit_b', img_size=518, in_channels=3)
    x = torch.randn(2, 3, 518, 518)
    
    features = encoder(x)
    
    print(f"Input: {x.shape}")
    for i, f in enumerate(features):
        print(f"Feature {i+1}: {f.shape}")
    
    params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {params:,}")
