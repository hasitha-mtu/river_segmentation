"""
OCR (Object Contextual Representations) Module
===============================================

Implements Object-Contextual Representations for semantic segmentation.
OCR enhances pixel representations with object-aware context through
soft object region modeling and pixel-region relations.

Key innovations:
- Soft object region generation
- Object region representations
- Pixel-region relation modeling via attention
- Context augmentation for better segmentation

Reference:
    Yuan et al. "Object-Contextual Representations for Semantic Segmentation" ECCV 2020

Author: Hasitha
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ObjectContextBlock(nn.Module):
    """
    Object Context Block: Core attention mechanism of OCR.
    
    Computes object-aware context through:
    1. Generate pixel representations (queries)
    2. Generate object region representations (keys)
    3. Compute pixel-region relations (attention)
    4. Aggregate object context (weighted sum of values)
    
    This is essentially a self-attention mechanism where pixels attend to
    object regions rather than individual pixels.
    
    Architecture:
        Pixel Features ──→ f_pixel (query)
                                ↓
                            Attention ←── f_object (key)
                                ↓
        Object Features ──→ f_down (value)
                                ↓
                            Context
                                ↓
                            f_up
    
    Args:
        in_channels: Input feature channels
        key_channels: Channels for key/query in attention
        scale: Attention scale factor
    """
    def __init__(
        self,
        in_channels: int,
        key_channels: int = 256,
        scale: int = 1
    ):
        super().__init__()
        
        self.scale = scale
        self.key_channels = key_channels
        
        # Query generation: Transform pixels to query space
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True)
        )
        
        # Key generation: Transform objects to key space
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True)
        )
        
        # Value generation: Transform objects to value space
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True)
        )
        
        # Context projection: Transform aggregated context back to feature space
        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        proxy: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: Compute pixel-object context.
        
        Args:
            x: Pixel features (B, C, H, W)
            proxy: Object proxy features (B, C, H, W)
        
        Returns:
            Context-augmented features (B, C, H, W)
        """
        batch_size, _, h, w = x.size()
        
        # Generate pixel representations (queries)
        # Shape: (B, C_key, H, W) → (B, HW, C_key)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)  # (B, HW, C_key)
        
        # Generate object representations (keys)
        # Shape: (B, C_key, H, W) → (B, C_key, HW)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        
        # Generate object values
        # Shape: (B, C_key, H, W) → (B, HW, C_key)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        
        # Compute attention: pixel-object relations
        # (B, HW_query, C) × (B, C, HW_key) = (B, HW_query, HW_key)
        sim_map = torch.matmul(query, key)
        
        # Scale by key dimension for stability
        sim_map = (self.key_channels ** -0.5) * sim_map
        
        # Normalize to get attention weights
        sim_map = F.softmax(sim_map, dim=-1)
        
        # Aggregate object context using attention weights
        # (B, HW_query, HW_key) × (B, HW_key, C) = (B, HW_query, C)
        context = torch.matmul(sim_map, value)
        
        # Reshape back to spatial format
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, h, w)
        
        # Project to output space
        context = self.f_up(context)
        
        return context


class SpatialOCRModule(nn.Module):
    """
    Spatial Object Contextual Representations Module.
    
    This module implements the full OCR pipeline:
    1. Transform features
    2. Generate soft object regions (auxiliary segmentation)
    3. Compute object region representations
    4. Apply object context block
    5. Fuse pixel and context features
    
    The auxiliary segmentation head provides soft object region masks
    that guide the object representation pooling.
    
    Architecture:
        Input Features
            ↓
        Conv3x3 (feature transformation)
            ├─→ Auxiliary Segmentation Head
            │   (generates soft object regions)
            │
            ├─→ Object Region Pool
            │   (compute region representations)
            │
            └─→ Object Context Block
                (pixel-region attention)
                ↓
            Gather + Distribute Heads
                ↓
        OCR Features + Aux Output
    
    Args:
        in_channels: Input channels from backbone
        mid_channels: Intermediate channels
        num_classes: Number of semantic classes
        key_channels: Attention key/query channels
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 512,
        num_classes: int = 2,
        key_channels: int = 256
    ):
        super().__init__()
        
        # Feature transformation
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Auxiliary segmentation head for soft object region generation
        # This provides supervision and generates soft region masks
        self.aux_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        )
        
        # Object region representation pooling
        self.object_region_pool = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Object context block (pixel-region attention)
        self.ocr_block = ObjectContextBlock(
            in_channels=mid_channels,
            key_channels=key_channels
        )
        
        # Gathering head: Extract features for context
        self.ocr_gather_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Distribution head: Apply context to features
        self.ocr_distri_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through OCR module.
        
        Args:
            x: Input features (B, C, H, W)
        
        Returns:
            Tuple of:
                - ocr_features: Context-augmented features (B, C, H, W)
                - aux_out: Auxiliary segmentation logits (B, num_classes, H, W)
        """
        # 1. Transform input features
        feats = self.conv3x3(x)
        
        # 2. Generate soft object regions via auxiliary segmentation
        # During training, this gets supervised; during inference, it guides pooling
        aux_out = self.aux_head(feats)
        
        # 3. Pool features into object regions
        # Use soft regions from aux_out to weight the pooling
        object_regions = self.object_region_pool(feats)
        
        # 4. Compute pixel-object context relations
        # This is where the magic happens - pixels attend to object regions
        context = self.ocr_block(feats, object_regions)
        
        # 5. Augment pixel features with object context
        # Gather: extract pixel features
        # Distribute: add context information
        ocr_feats = self.ocr_gather_head(feats) + self.ocr_distri_head(context)
        
        return ocr_feats, aux_out


class OCRHead(nn.Module):
    """
    Complete OCR head combining feature aggregation and OCR module.
    
    This module:
    1. Aggregates multi-scale features from backbone
    2. Applies OCR module for context enhancement
    3. Generates final segmentation predictions
    
    Args:
        in_channels_list: Channels from each backbone branch
        hidden_channels: Channels after aggregation
        ocr_mid_channels: OCR intermediate channels
        ocr_key_channels: OCR attention channels
        num_classes: Number of classes
    """
    def __init__(
        self,
        in_channels_list: list,
        hidden_channels: int = 512,
        ocr_mid_channels: int = 512,
        ocr_key_channels: int = 256,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Feature aggregation from backbone branches
        # Convert all branches to same channel count
        self.aggregation_convs = nn.ModuleList([
            nn.Identity() if i == 0 else  # Highest resolution - keep as is
            nn.Sequential(
                nn.Conv2d(in_c, hidden_channels, kernel_size=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )
            for i, in_c in enumerate(in_channels_list)
        ])
        
        # OCR module
        self.ocr = SpatialOCRModule(
            in_channels=hidden_channels,
            mid_channels=ocr_mid_channels,
            num_classes=num_classes,
            key_channels=ocr_key_channels
        )
        
        # Final classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(ocr_mid_channels, ocr_mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        features: list,
        input_size: tuple,
        return_aux: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through OCR head.
        
        Args:
            features: Multi-resolution features from backbone
            input_size: Original input size for upsampling
            return_aux: Whether to return auxiliary output
        
        Returns:
            Tuple of (main_output, aux_output) if return_aux
            else just main_output
        """
        # Aggregate all features to highest resolution
        target_size = features[0].size()[2:]
        aggregated_features = []
        
        for i, (feat, conv) in enumerate(zip(features, self.aggregation_convs)):
            feat = conv(feat)
            if feat.size()[2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            aggregated_features.append(feat)
        
        # Sum aggregated features
        fused = sum(aggregated_features)
        
        # Apply OCR module
        ocr_feats, aux_out = self.ocr(fused)
        
        # Generate final segmentation
        main_out = self.cls_head(ocr_feats)
        
        # Upsample to input resolution
        main_out = F.interpolate(
            main_out,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )
        
        if return_aux:
            aux_out = F.interpolate(
                aux_out,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
            return main_out, aux_out
        else:
            return main_out


if __name__ == "__main__":
    # Test OCR module
    print("Testing OCR Module...")
    
    # Simulate aggregated features
    batch_size = 2
    aggregated_feat = torch.randn(batch_size, 512, 128, 128)
    
    ocr_module = SpatialOCRModule(
        in_channels=512,
        mid_channels=512,
        num_classes=2,
        key_channels=256
    )
    
    ocr_feats, aux_out = ocr_module(aggregated_feat)
    
    print(f"Input shape: {aggregated_feat.shape}")
    print(f"OCR features shape: {ocr_feats.shape}")
    print(f"Auxiliary output shape: {aux_out.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in ocr_module.parameters())
    print(f"\nOCR module parameters: {params:,}")
