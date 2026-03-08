"""
SAM Encoder + CNN Decoder
==========================
Uses SAM's powerful image encoder as a frozen feature extractor,
then adds a lightweight CNN decoder for segmentation.

SAM (Segment Anything Model) is pretrained on 1.1B masks, providing
strong segmentation-specific features. This architecture explores whether
SAM's mask pretraining transfers better than DINOv2's image-level pretraining
for water segmentation.

Reference: Kirillov et al. "Segment Anything" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from segment_anything import sam_model_registry

class SAMEncoderDecoder(nn.Module):
    """
    SAM encoder with custom CNN decoder
    
    Uses SAM's ViT encoder (frozen) to extract features, then applies
    a lightweight decoder for binary water segmentation.
    
    IMPORTANT: SAM's image encoder has fixed positional embeddings for 1024x1024 input.
    Input images are always resized to 1024x1024, producing 64x64 feature maps.
    The decoder then upsamples back to the desired output size.
    
    Args:
        sam_checkpoint: Path to SAM checkpoint (.pth file)
        model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
        freeze_encoder: Freeze SAM encoder weights
        decoder_channels: Base channels for decoder
    
    Note: Requires RGB input (3 channels) - SAM only works with RGB
    """
    
    def __init__(
        self,
        sam_checkpoint: str = 'checkpoints/sam_vit_b_01ec64.pth',
        model_type: str = 'vit_b',
        freeze_encoder: bool = True,
        decoder_channels: int = 256
    ):
        super().__init__()
        
        
        # Load SAM model
        print(f"Loading SAM model: {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder = self.sam.image_encoder
        
        print(f"✓ SAM encoder loaded from {sam_checkpoint}")
        
        # Freeze encoder
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            print("✓ SAM encoder frozen")
        
        # SAM encoder output: 256 channels at 64x64 for 1024x1024 input
        # MUST use 1024x1024 input due to fixed positional embeddings
        sam_out_channels = 256
        
        # Lightweight decoder: 64x64 -> 512x512 (8x upsampling, 3 stages)
        self.decoder = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(sam_out_channels, decoder_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, decoder_channels // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 512x512
            nn.ConvTranspose2d(decoder_channels // 2, decoder_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 4, decoder_channels // 4, 3, padding=1),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True),
            
            # Final prediction
            nn.Conv2d(decoder_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: RGB input (B, 3, H, W)
        
        Returns:
            Segmentation mask (B, 1, H, W)
        """
        if x.shape[1] != 3:
            raise ValueError(f"SAM requires RGB input (3 channels), got {x.shape[1]} channels")
        
        B, C, H, W = x.shape
        original_size = (H, W)
        
        # SAM REQUIRES 1024x1024 input (fixed positional embeddings)
        # Cannot use smaller sizes without modifying the encoder
        sam_input_size = 1024
        x_resized = F.interpolate(x, size=(sam_input_size, sam_input_size), mode='bilinear', align_corners=False)
        
        # SAM encoder (frozen, no gradients needed)
        with torch.no_grad():
            features = self.image_encoder(x_resized)  # B, 256, 64, 64
        
        # Decode: 64x64 -> 512x512
        output = self.decoder(features)  # B, 1, 512, 512
        
        # Resize to original size if needed
        if output.shape[2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        return output


class SAMFineTuned(nn.Module):
    """
    Fine-tuned SAM for direct segmentation
    
    Uses full SAM architecture but with:
    - Frozen image encoder
    - Frozen prompt encoder
    - Trainable mask decoder
    - Learnable prompt embeddings (no explicit prompts needed)
    
    This allows the mask decoder to adapt to water segmentation
    while leveraging SAM's powerful pretrained encoder.
    
    IMPORTANT: SAM's mask decoder was designed for single-image-multiple-prompts,
    not batch-of-images. We handle batching by processing images individually
    then concatenating results.
    
    Args:
        sam_checkpoint: Path to SAM checkpoint
        model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
    """
    
    def __init__(
        self,
        sam_checkpoint: str = 'checkpoints/sam_vit_b_01ec64.pth',
        model_type: str = 'vit_b'
    ):
        super().__init__()
        
        # Load full SAM model
        print(f"Loading full SAM model: {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        print(f"✓ SAM loaded from {sam_checkpoint}")
        
        # Freeze encoder and prompt encoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        
        print("✓ Image encoder and prompt encoder frozen")
        print("✓ Mask decoder trainable")
        
        # Learnable prompt embeddings (replace point/box prompts)
        # Shape: (1, 1, 256) - single prompt token
        self.learnable_prompt = nn.Parameter(torch.randn(1, 1, 256))
        
        # Output post-processing
        self.sigmoid = nn.Sigmoid()
    
    def _decode_single_image(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode mask for a single image embedding.
        
        SAM's mask decoder expects:
        - image_embeddings: (1, 256, H, W)
        - image_pe: (1, 256, H, W)
        - sparse_prompt_embeddings: (1, N_prompts, 256)
        - dense_prompt_embeddings: (1, 256, H, W)
        
        Args:
            image_embedding: Single image embedding (1, 256, H, W)
            image_pe: Positional encoding (1, 256, H, W)
        
        Returns:
            Low resolution mask (1, 1, H_low, W_low)
        """
        # Sparse embeddings: our learnable prompt (1, 1, 256)
        sparse_embeddings = self.learnable_prompt
        
        # Dense embeddings: zeros (no additional spatial guidance)
        dense_embeddings = torch.zeros(
            1,
            256,
            image_embedding.shape[2],
            image_embedding.shape[3],
            device=image_embedding.device,
            dtype=image_embedding.dtype
        )
        
        # Decode mask
        low_res_mask, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        return low_res_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        SAM's mask decoder doesn't support standard batched inference.
        It was designed for single-image-multiple-prompts, not batch-of-images.
        We handle this by encoding all images together (efficient), then
        decoding each image individually (necessary for correctness).
        
        Args:
            x: RGB input (B, 3, H, W)
        
        Returns:
            Segmentation mask (B, 1, H, W)
        """
        if x.shape[1] != 3:
            raise ValueError(f"SAM requires RGB input (3 channels), got {x.shape[1]} channels")
        
        B, C, H, W = x.shape
        original_size = (H, W)
        
        # Resize to 1024x1024 (SAM's native resolution)
        target_size = 1024
        x_resized = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        # Encode all images at once (batched encoding is supported)
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(x_resized)  # (B, 256, 64, 64)
        
        # Get positional encoding (same for all images)
        image_pe = self.sam.prompt_encoder.get_dense_pe()  # (1, 256, 64, 64)
        
        # Decode each image individually
        # SAM's mask decoder uses repeat_interleave internally which breaks with batch > 1
        low_res_masks_list: List[torch.Tensor] = []
        
        for i in range(B):
            # Extract single image embedding
            img_embed = image_embeddings[i:i+1]  # (1, 256, 64, 64)
            
            # Decode single image
            low_res_mask = self._decode_single_image(img_embed, image_pe)
            low_res_masks_list.append(low_res_mask)
        
        # Concatenate all masks
        low_res_masks = torch.cat(low_res_masks_list, dim=0)  # (B, 1, H_low, W_low)
        
        # Upsample to target size
        masks = F.interpolate(
            low_res_masks,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Apply sigmoid
        masks = self.sigmoid(masks)
        
        # Resize to original size
        if masks.shape[2:] != original_size:
            masks = F.interpolate(masks, size=original_size, mode='bilinear', align_corners=False)
        
        return masks

class SAM_FPN_Decoder(nn.Module):
    def __init__(self, encoder_embed_dim=768, out_channels=1):
        super().__init__()
        # 1. We keep 256 as our internal "Working Depth"
        self.internal_dims = 256 
        
        self.latlayer1 = nn.Conv2d(encoder_embed_dim, self.internal_dims, 1)
        self.latlayer2 = nn.Conv2d(encoder_embed_dim, self.internal_dims, 1)
        self.latlayer3 = nn.Conv2d(encoder_embed_dim, self.internal_dims, 1)
        
        self.up_scale2 = nn.ConvTranspose2d(self.internal_dims, self.internal_dims, 2, 2)
        self.up_scale1 = nn.ConvTranspose2d(self.internal_dims, self.internal_dims, 4, 4)

        # 2. Only the very last layer uses out_channels (num_classes)
        self.predict = nn.Sequential(
            nn.Conv2d(self.internal_dims, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1) # This is where '1' or 'num_classes' goes
        )

    def forward(self, features: List[torch.Tensor]):
        # features[0] is early (more spatial detail), features[2] is late (more semantic)
        f1, f2, f3 = features 

        # Map to 256 channels
        p3 = self.latlayer3(f3) # 64x64
        p2 = self.latlayer2(f2) # 64x64
        p1 = self.latlayer1(f1) # 64x64

        # Upsample and merge
        # We upsample the early layers to provide higher resolution detail
        m3 = p3 
        m2 = p2 + m3
        m1 = p1 + m2
        
        # Final upsampling to original 1024x1024 resolution
        # Combining features for the prediction
        feat = self.up_scale1(m1) # This brings us to 256x256
        out = F.interpolate(self.predict(feat), scale_factor=4, mode='bilinear', align_corners=False)
        
        return out

class SAMEncoderDecoderUpdated(nn.Module):
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = 'vit_b',
        freeze_encoder: bool = True,
        out_channels: int = 1
    ):
        super().__init__()
        # Load SAM
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.encoder = sam.image_encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Determine embedding dimension based on variant
        embed_dim = self.encoder.embed_dim
        
        # New FPN Decoder
        self.decoder = SAM_FPN_Decoder(encoder_embed_dim=embed_dim, out_channels=out_channels)

    def forward(self, x):
        # SAM expects 1024x1024
        if x.shape[-2:] != (1024, 1024):
            x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # Extract intermediate features
        # We manually run the blocks to catch intermediate outputs
        # This is the "hook-less" way to get multi-level features from SAM
        hidden_states = []
        
        # Indices of blocks to extract (e.g., for Vit-B with 12 blocks)
        # We take 4, 8, and 12
        extract_blocks = [3, 7, 11] 
        
        z = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            z = z + self.encoder.pos_embed

        for i, blk in enumerate(self.encoder.blocks):
            z = blk(z)
            if i in extract_blocks:
                # Permute from [B, L, C] to [B, C, H, W] for Conv layers
                # SAM ViT output is (Batch, 64, 64, Dim) -> (Batch, Dim, 64, 64)
                hidden_states.append(z.permute(0, 3, 1, 2))

        # Pass the pyramid of features to our FPN decoder
        output = self.decoder(hidden_states)
        return output
    
def build_sam_segmentation(variant, in_channels, num_classes):
    paths = {
        'vit_b': './checkpoints/sam/sam_vit_b_01ec64.pth',
        'vit_l': './checkpoints/sam/sam_vit_l_0b3195.pth',
        'vit_h': './checkpoints/sam/sam_vit_h_4b8939.pth'
    }
    checkpoint_path = paths.get(variant)
    return SAMEncoderDecoderUpdated(
            sam_checkpoint=checkpoint_path,
            model_type=variant,
            freeze_encoder=True,
            decoder_channels=num_classes
        )
    # return SAMFineTuned(
    #         sam_checkpoint=checkpoint_path,
    #         model_type=variant
    #     )


if __name__ == '__main__':
    print("SAM Models Test")
    print("="*70)
    print("\nNote: These models require:")
    print("1. segment_anything package installed")
    print("2. SAM checkpoint downloaded to checkpoints/")
    print("3. RGB input only (3 channels)")
    print("\nSkipping actual model creation in test mode")
    print("Use in training script with proper setup")
    
    # Test basic structure without loading actual SAM
    print("\n✓ SAM model definitions ready")
    print("\nTo use SAM models:")
    print("1. Download SAM checkpoint:")
    print("   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/")
    print("2. Install segment_anything:")
    print("   pip install git+https://github.com/facebookresearch/segment-anything.git")
    print("3. Use in training with RGB feature_config")
