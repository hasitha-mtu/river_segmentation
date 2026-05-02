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
from typing import List
from segment_anything import sam_model_registry
import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# # from torch.utils.data import Dataset
# from pathlib import Path
# from PIL import Image
# from datasets import Dataset
# import random

# def create_dataset(data_dir, split):
#     data_dir = Path(data_dir) / split
#     print(f'data_dir: {data_dir}')
#     images_dir = data_dir / "images"
#     print(f'images_dir: {images_dir}')
#     masks_dir = data_dir / "masks"
#     print(f'masks_dir: {masks_dir}')
#     image_paths = sorted(list(images_dir.glob("*.jpg")))
#     dataset_dict = {
#         "image": [np.array(Image.open(image_path).convert('RGB')) for image_path in image_paths],
#         "label": [np.array(Image.open(masks_dir / f"{image_path.stem}.png").convert('L')) for image_path in image_paths]
#         }
#     dataset = Dataset.from_dict(dataset_dict)
#     print(dataset.shape)
#     return dataset

# def view_sample_from_dataset(dataset):
#     img_num = random.randint(0, dataset.shape[0]-1)
#     example_image = dataset[img_num]["image"]
#     example_mask = dataset[img_num]["label"]

#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#     # Plot the first image on the left
#     axes[0].imshow(np.array(example_image), cmap='gray')  # Assuming the first image is grayscale
#     axes[0].set_title("Image")

#     # Plot the second image on the right
#     axes[1].imshow(example_mask, cmap='gray')  # Assuming the second image is grayscale
#     axes[1].set_title("Mask")

#     # Hide axis ticks and labels
#     for ax in axes:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])

#     # Display the images side by side
#     plt.show()
     

# class SAMDataset(Dataset):
#   """
#   This class is used to create a dataset that serves input images and masks.
#   It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
#   """
#   def __init__(self, dataset, processor):
#     self.dataset = dataset
#     self.processor = processor

#   def __len__(self):
#     return len(self.dataset)

#   def __getitem__(self, idx):
#     item = self.dataset[idx]
#     image = item["image"]
#     ground_truth_mask = np.array(item["label"])

#     # get bounding box prompt
#     prompt = get_bounding_box(ground_truth_mask)

#     # prepare image and prompt for the model
#     inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

#     # remove batch dimension which the processor adds by default
#     inputs = {k:v.squeeze(0) for k,v in inputs.items()}

#     # add ground truth segmentation
#     inputs["ground_truth_mask"] = ground_truth_mask

#     return inputs
  
# #Get bounding boxes from mask.
# def get_bounding_box(ground_truth_map):
#   # get bounding box from mask
#   y_indices, x_indices = np.where(ground_truth_map > 0)
#   # Check if mask is empty
#   if len(x_indices) == 0:
#     # Return a dummy box or a box covering the whole image
#     # Standard practice: return [0, 0, W, H] or a specific flag
#     return [0, 0, ground_truth_map.shape[1], ground_truth_map.shape[0]]
#   x_min, x_max = np.min(x_indices), np.max(x_indices)
#   y_min, y_max = np.min(y_indices), np.max(y_indices)
#   # add perturbation to bounding box coordinates
#   H, W = ground_truth_map.shape
#   x_min = max(0, x_min - np.random.randint(0, 20))
#   x_max = min(W, x_max + np.random.randint(0, 20))
#   y_min = max(0, y_min - np.random.randint(0, 20))
#   y_max = min(H, y_max + np.random.randint(0, 20))
#   bbox = [x_min, y_min, x_max, y_max]

#   return bbox

# def visualize_bbox_perturbation(image, mask, n_samples=3):
#     """
#     Visualizes the river image, mask, and the perturbed bounding box.
#     """
#     fig, axes = plt.subplots(1, n_samples, figsize=(18, 6))
    
#     for i in range(n_samples):
#         # Generate box using your function
#         bbox = get_bounding_box(mask) # [x_min, y_min, x_max, y_max]
        
#         axes[i].imshow(image)
#         # Overlay the mask with low opacity
#         axes[i].imshow(mask, alpha=0.3, cmap='Blues')
        
#         # Create a Rectangle patch
#         # Rectangle expects (x, y), width, height
#         rect = patches.Rectangle(
#             (bbox[0], bbox[1]), 
#             bbox[2] - bbox[0], 
#             bbox[3] - bbox[1], 
#             linewidth=2, edgecolor='r', facecolor='none', linestyle='--'
#         )
        
#         axes[i].add_patch(rect)
#         axes[i].set_title(f"Perturbed BBox Analysis {i+1}")
#         axes[i].axis('off')

#     plt.tight_layout()
#     plt.show()

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


class SAM_FPN_Decoder(nn.Module):
    """
    Improved FPN Decoder for SAM.
    Takes intermediate features from the ViT encoder to reconstruct 
    fine-grained boundaries for river segmentation.
    """
    def __init__(self, encoder_embed_dim=768, out_channels=1):
        super().__init__()
        
        # Lateral layers to normalize channels from different ViT blocks
        self.latlayer1 = nn.Conv2d(encoder_embed_dim, 256, kernel_size=1)
        self.latlayer2 = nn.Conv2d(encoder_embed_dim, 256, kernel_size=1)
        self.latlayer3 = nn.Conv2d(encoder_embed_dim, 256, kernel_size=1)
        
        # Upsampling layers for higher resolution levels
        # These create the 128x128 and 256x256 scales from SAM's 64x64 blocks
        self.up_scale2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up_scale1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4)

        # Final prediction head
        self.predict = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(128, out_channels, kernel_size=1)
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

class SAMEncoderDecoderFPN(nn.Module):
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
        
        # FIX: Robust way to get embedding dimension
        if hasattr(self.encoder, 'patch_embed'):
            self.embed_dim = self.encoder.patch_embed.proj.out_channels
        else:
            # Fallback for standard SAM variants
            mapping = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
            self.embed_dim = mapping.get(model_type, 768)
        
        # New FPN Decoder
        self.decoder = SAM_FPN_Decoder(encoder_embed_dim=self.embed_dim, out_channels=out_channels)

    def forward(self, x):
        # 1. Resize input to SAM's expected 1024x1024
        original_size = x.shape[-2:]
        if original_size != (1024, 1024):
            x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # 2. Extract intermediate features
        hidden_states = []
        # We pick 3 blocks to create the pyramid (Early, Mid, Late)
        # For ViT-B (12 blocks), we use 3, 7, 11
        # For ViT-L/H, these indices should be scaled, but 3/7/11 works well for all
        extract_blocks = [3, 7, 11] 
        
        # Manually trace the encoder forward pass
        z = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            z = z + self.encoder.pos_embed

        for i, blk in enumerate(self.encoder.blocks):
            z = blk(z)
            if i in extract_blocks:
                # SAM ViT output is [B, H, W, C]. We need [B, C, H, W] for the FPN
                # We use permute to move the Channel dimension to the front
                hidden_states.append(z.permute(0, 3, 1, 2))

        # 3. Pass through FPN decoder
        output = self.decoder(hidden_states)
        
        # 4. Resize back to the user's original input size if necessary
        if output.shape[-2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
            
        return output
    
def build_sam_fpn_segmentation(variant, in_channels, num_classes):
    print(f'Current working dir: {os.getcwd()}')
    paths = {
        'vit_b': r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam/sam_vit_b_01ec64.pth',
        'vit_l': r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam/sam_vit_l_0b3195.pth',
        'vit_h': r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam/sam_vit_h_4b8939.pth'
    }

    checkpoint_path = paths.get(variant)
    return SAMEncoderDecoderFPN(
            sam_checkpoint=checkpoint_path,
            model_type=variant,
            freeze_encoder=True,
            out_channels=num_classes
        )
    # return SAMFineTuned(
    #         sam_checkpoint=checkpoint_path,
    #         model_type=variant
    #     )

def build_sam_finetuned_segmentation(variant, in_channels, num_classes):
    paths = {
        'vit_b': r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam/sam_vit_b_01ec64.pth',
        'vit_l': r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam/sam_vit_l_0b3195.pth',
        'vit_h': r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam/sam_vit_h_4b8939.pth'
    }
    checkpoint_path = paths.get(variant)
    return SAMFineTuned(
            sam_checkpoint=checkpoint_path,
            model_type=variant
        )

def build_sam_segmentation(variant, in_channels, num_classes):
    paths = {
        'vit_b': r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam/sam_vit_b_01ec64.pth',
        'vit_l': r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam/sam_vit_l_0b3195.pth',
        'vit_h': r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam/sam_vit_h_4b8939.pth'
    }
    checkpoint_path = paths.get(variant)
    return SAMEncoderDecoder(
            sam_checkpoint=checkpoint_path,
            model_type=variant,
            freeze_encoder=True,
            decoder_channels=num_classes
        )
    # return SAMFineTuned(
    #         sam_checkpoint=checkpoint_path,
    #         model_type=variant
    #     )


