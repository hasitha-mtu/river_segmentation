import numpy as np
import torch  # Added missing import
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from datasets import Dataset as InitialDataset
from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader
from transformers import SamProcessor
import torch.nn.functional as F
from src.utils.losses import get_loss_function

def create_dataset(data_dir, split):
    data_dir = Path(data_dir) / split
    print(f'data_dir: {data_dir}')
    images_dir = data_dir / "images"
    print(f'images_dir: {images_dir}')
    masks_dir = data_dir / "masks"
    print(f'masks_dir: {masks_dir}')
    image_paths = sorted(list(images_dir.glob("*.jpg")))
    dataset_dict = {
        "image": [np.array(Image.open(image_path).convert('RGB')) for image_path in image_paths],
        "label": [np.array(Image.open(masks_dir / f"{image_path.stem}.png").convert('L')) for image_path in image_paths]
        }
    dataset = InitialDataset.from_dict(dataset_dict)
    print(dataset.shape)
    return dataset
    

def view_sample_from_dataset(dataset):
    img_num = random.randint(0, dataset.shape[0]-1)
    example_image = dataset[img_num]["image"]
    example_mask = dataset[img_num]["label"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first image on the left
    axes[0].imshow(np.array(example_image), cmap='gray')  # Assuming the first image is grayscale
    axes[0].set_title("Image")

    # Plot the second image on the right
    axes[1].imshow(example_mask, cmap='gray')  # Assuming the second image is grayscale
    axes[1].set_title("Mask")

    # Hide axis ticks and labels
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Display the images side by side
    plt.show()

def get_bounding_box(ground_truth_map):
    # Ensure it is a numpy array for calculations
    ground_truth_map = np.array(ground_truth_map)
    y_indices, x_indices = np.where(ground_truth_map > 0)
    
    # Handle images with no river pixels to prevent min/max errors
    if len(x_indices) == 0:
        return [0, 0, ground_truth_map.shape[1], ground_truth_map.shape[0]]
        
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Add perturbation
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    
    # Return as floats for the processor
    return [float(x_min), float(y_min), float(x_max), float(y_max)]

class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # FIX 1: Convert nested lists back to NumPy arrays to prevent processor recursion error
        image = np.array(item["image"], dtype=np.uint8)
        ground_truth_mask = np.array(item["label"], dtype=np.uint8)

        # Get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # Prepare image and prompt for the model
        # input_boxes expects (batch, num_boxes, 4). [[prompt]] provides (1, 1, 4)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # Remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add ground truth segmentation as a torch tensor
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask).float()

        return inputs

def visualize_bbox_perturbation(image, mask, n_samples=3):
    """
    Visualizes the river image, mask, and the perturbed bounding box.
    """
    fig, axes = plt.subplots(1, n_samples, figsize=(18, 6))
    
    for i in range(n_samples):
        # Generate box using your function
        bbox = get_bounding_box(mask) # [x_min, y_min, x_max, y_max]
        
        axes[i].imshow(image)
        # Overlay the mask with low opacity
        axes[i].imshow(mask, alpha=0.3, cmap='Blues')
        
        # Create a Rectangle patch
        # Rectangle expects (x, y), width, height
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1], 
            linewidth=2, edgecolor='r', facecolor='none', linestyle='--'
        )
        
        axes[i].add_patch(rect)
        axes[i].set_title(f"Perturbed BBox Analysis {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # FIX 2: Use raw string for Windows paths to avoid SyntaxWarning
    data_dir = r'dataset\processed_512_resized' 
    dataset = create_dataset(data_dir, 'train')
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDataset(dataset=dataset, processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # FIX 3: Correct unpacking of the dictionary batch
    batch = next(iter(train_dataloader))
    print(f"Batch keys: {batch.keys()}")
    
    train_features = batch['pixel_values']
    train_labels = batch['ground_truth_mask']
    
    print(f"Images shape: {train_features.shape}")
    print(f"Masks shape: {train_labels.shape}")

    from transformers import SamModel
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    from torch.optim import Adam
    # Initialize the optimizer and the loss function
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    criterion = get_loss_function(
                'combined',
                bce_weight      = 1.0,
                dice_weight     = 1.0,
                boundary_weight = 1.0,
                use_boundary    = False,
            )

    from tqdm import tqdm
    from statistics import mean
    import torch
    from torch.nn.functional import threshold, normalize

    #Training loop
    num_epochs = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks
            print(f'predicted_masks type: {type(predicted_masks)}')
            print(f'predicted_masks shape: {predicted_masks.shape}')

            predicted_masks = predicted_masks.squeeze(1).squeeze(1)
            print(f'predicted_masks type: {type(predicted_masks)}')
            print(f'predicted_masks shape: {predicted_masks.shape}')

            # 3. Upsample to your Ground Truth size (e.g., 512x512)
            # Resulting shape: [4, 512, 512]
            predicted_masks = F.interpolate(
                predicted_masks.unsqueeze(1), # interpolate expects [B, C, H, W], so add '1' channel
                size=(512, 512), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1) # Remove the channel dimension again if doing Binary Loss
            
            print(f'predicted_masks type: {type(predicted_masks)}')
            print(f'predicted_masks shape: {predicted_masks.shape}')

            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            print(f'ground_truth_masks type: {type(ground_truth_masks)}')
            print(f'ground_truth_masks shape: {ground_truth_masks.shape}')
            loss, loss_dict = criterion(predicted_masks, ground_truth_masks, None)

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

