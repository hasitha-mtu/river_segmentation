import os
import numpy as np
from tqdm import tqdm
from glob import glob
from numpy import zeros
from numpy.random import randint
import torch
import os
import cv2
from statistics import mean
from torch.nn.functional import threshold, normalize
from segment_anything import SamPredictor, sam_model_registry
import torch
from src.utils.losses import get_loss_function

# Data Viz
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# === Metric Computation Function ===
def binary_segmentation_metrics(predictions, targets):
    predictions = predictions.squeeze()
    targets = targets.squeeze()
    predictions_binary = (predictions > 0.5).astype(int)
    targets_binary = targets.astype(int)

    TP = np.sum((predictions_binary == 1) & (targets_binary == 1))
    FP = np.sum((predictions_binary == 1) & (targets_binary == 0))
    FN = np.sum((predictions_binary == 0) & (targets_binary == 1))
    TN = np.sum((predictions_binary == 0) & (targets_binary == 0))

    eps = 1e-5
    accuracy = (TP + TN + eps) / (TP + FP + FN + TN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)
    f_score = 2 * (precision * recall) / (precision + recall + eps)
    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    iou = (TP + eps) / (TP + FP + FN + eps)

    total = TP + FP + FN + TN
    p_o = (TP + TN) / total
    p_e = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total ** 2)
    kappa = (p_o - p_e) / (1 - p_e + eps)

    return accuracy, precision, recall, f_score, iou, kappa, FP, FN, TP, TN, dice

# === Categorization Function ===
def categorize_metric(value, metric):
    categories = {
        'IoU': [(0.90, 'Excellent'), (0.85, 'Good'), (0.75, 'Fair'), (0.65, 'Poor'), (0, 'Unacceptable')],
        'Precision': [(0.95, 'Excellent'), (0.85, 'Good'), (0.65, 'Moderate'), (0, 'Fail')],
        'Kappa': [(0.88, 'Excellent'), (0.78, 'Good'), (0.68, 'Moderate'), (0, 'Fail')],
        'F-Score': [(0.88, 'Excellent'), (0.78, 'Good'), (0.68, 'Moderate'), (0, 'Fail')],
        'Recall': [(0.88, 'Excellent'), (0.78, 'Good'), (0.68, 'Moderate'), (0, 'Fail')]
    }

    if metric in categories:
        for threshold, label in categories[metric]:
            if value >= threshold:
                return label
    return "Unknown"

# === Compute Metrics for All Images ===
def compute_all_metrics(predictions_list, targets_list):
    all_metrics = []
    for i in range(len(predictions_list)):
        metrics = binary_segmentation_metrics(predictions_list[i], targets_list[i])
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F-Score', 'IoU', 'Kappa', 'FP', 'FN', 'TP', 'TN', 'Dice']
        metric_dict = dict(zip(metric_names, metrics))

        # Add categorized versions
        for name in ['IoU', 'Precision', 'Kappa', 'F-Score', 'Recall']:
            metric_dict[f'{name}_Category'] = categorize_metric(metric_dict[name], name)

        all_metrics.append(metric_dict)
    return all_metrics

# === Summarize Counts and Percentages ===
def summarize_category_counts(metrics_list, metric_name, labels):
    counts = {label: 0 for label in labels}
    total = len(metrics_list)

    for m in metrics_list:
        category = m.get(f"{metric_name}_Category")
        if category in counts:
            counts[category] += 1

    # Print summary
    print(f"Counts and Percentages of Images in Each Category for {metric_name}:")
    for label in labels:
        count = counts[label]
        percentage = (count / total) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")
    print()

if __name__ == '__main__':
  model_type = "vit_b"  # or "vit_l", "vit_h", etc.

  # Path to the pretrained SAM checkpoint file (.pth)
  checkpoint = "./checkpoints/best_model_epoch5.pth" 
  # === Load Model ===
  # Use the model registry to initialize the correct SAM architecture
  sam_model = sam_model_registry[model_type](checkpoint=checkpoint)

  # Move model to GPU if available, otherwise CPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  sam_model.to(device)

  # === Set Model to Training Mode ===
  # Use `model.train()` when fine-tuning or training the model
  # For inference, use `model.eval()` instead
  predictor_tuned = SamPredictor(sam_model)
  sam_model.eval()

  # Set the paths to your test images and labels
  test_image_dir = "dataset/processed_512_resized/test/images"   # <-- Update this path
  test_label_dir = "dataset/processed_512_resized/test/masks"   # <-- Update this path

  desired_size=(512, 512)

  # === Load and Sort Test Image Paths ===
  # Collect all test image files (e.g., .jpg)
  all_test_image_paths = sorted(glob(os.path.join(test_image_dir, "*.jpg")))
  test_total_images = len(all_test_image_paths)
  print(f"Total Number of Test Images: {test_total_images}")

  # === Load and Sort Test Label Paths ===
  # Collect all test label files (e.g., .png masks)
  all_test_label_paths = sorted(glob(os.path.join(test_label_dir, "*.png")))
  test_total_labels = len(all_test_label_paths)
  print(f"Total Number of Test Labels: {test_total_labels}")

  # === Match Image and Label Paths ===
  # These lists can now be used for DataLoader or evaluation
  Test_image_paths = all_test_image_paths[:test_total_images]
  Test_label_paths = all_test_label_paths[:test_total_labels]

  # Optional: Print a few samples to verify
  print("Sample test image path:", Test_image_paths[0] if Test_image_paths else "No images found")
  print("Sample test label path:", Test_label_paths[0] if Test_label_paths else "No labels found")

  # Dictionary to hold ground truth binary masks for test data
  ground_truth_test_masks = {}

  # === Load and Process Each Test Mask ===
  for idx in range(len(Test_label_paths)):
      # Read label image in color (3-channel); expected mask is in the red channel
      gt_color = cv2.imread(Test_label_paths[idx])

      # Extract the red channel only and convert to binary mask
      # Note: OpenCV loads in BGR, so red is at index 2
      binary_mask = (gt_color[:, :, 2] > 0).astype(np.float32)

      # Resize if specified
      if desired_size is not None:
          binary_mask = cv2.resize(binary_mask, desired_size, interpolation=cv2.INTER_NEAREST)

      # Store in dictionary
      ground_truth_test_masks[idx] = binary_mask

  print(f"Loaded {len(ground_truth_test_masks)} ground truth test masks.")

  
  # === Inference with SAM Predictor on Test Set ===
  masks_tuned_list = {}   # Stores predicted binary masks
  images_tuned_list = {}  # Stores input images used during inference

  for idx in range(len(Test_image_paths)):
      # === Load and Preprocess Image ===
      image = cv2.imread(Test_image_paths[idx])
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      if desired_size is not None:
          image_rgb = cv2.resize(image_rgb, desired_size, interpolation=cv2.INTER_LINEAR)

      # === Set the image for SAM predictor ===
      predictor_tuned.set_image(image_rgb)

      # === Predict segmentation mask ===
      masks_tuned, _, _ = predictor_tuned.predict(
          point_coords=None,
          box=None,
          multimask_output=False,  # Only get the most confident mask
      )

      # === Extract and post-process the first predicted mask ===
      mask_np = masks_tuned[0, :, :]                 # Select first mask
      binary_mask = (mask_np > 0).astype(np.float32) # Convert to float binary mask

      # === Store results ===
      images_tuned_list[idx] = image_rgb
      masks_tuned_list[idx] = binary_mask

  print(f"Inference complete on {len(Test_image_paths)} test images.")

  import matplotlib.pyplot as plt
  import numpy as np

  # === Grid Configuration ===
  n_images = len(images_tuned_list)
  n_cols = 4  # Number of images per row
  n_rows = (n_images // n_cols) + (n_images % n_cols > 0)  # Auto-calculate rows

  # Create a figure with subplots
  fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

  # If axs is 1D (e.g., only 1 row), convert to 2D for consistency
  axs = np.atleast_2d(axs)

  # === Iterate and Plot ===
  for i in range(n_rows):
      for j in range(n_cols):
          index = i * n_cols + j
          ax = axs[i, j]

          if index < n_images:
              # Display the RGB image
              ax.imshow(images_tuned_list[index], interpolation='none')

              # Generate a blue mask overlay (R=0, G=0, B=1) for binary mask = 1
              mask = masks_tuned_list[index]
              blue_mask_rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
              blue_mask_rgb[..., 2] = mask  # Blue channel

              # Overlay the mask with transparency
              ax.imshow(blue_mask_rgb, alpha=0.5)

          # Remove axes ticks
          ax.axis('off')

  # === Final Layout ===
  plt.subplots_adjust(wspace=0.03, hspace=0.03)
  plt.tight_layout()
  plt.show()

  # === Run All ===
  metrics_list = compute_all_metrics(masks_tuned_list, ground_truth_test_masks)

  summarize_category_counts(metrics_list, 'IoU',        ['Excellent', 'Good', 'Fair', 'Poor', 'Unacceptable'])
  summarize_category_counts(metrics_list, 'Precision',  ['Excellent', 'Good', 'Moderate', 'Fail'])
  summarize_category_counts(metrics_list, 'Kappa',      ['Excellent', 'Good', 'Moderate', 'Fail'])
  summarize_category_counts(metrics_list, 'F-Score',    ['Excellent', 'Good', 'Moderate', 'Fail'])
  summarize_category_counts(metrics_list, 'Recall',     ['Excellent', 'Good', 'Moderate', 'Fail'])


def train():
  # === CONFIGURATION ===
  # Set the path to your training images and labels
  image_path = "dataset/processed_512_resized/train/images"   # <-- Replace with actual image folder path
  label_path = "dataset/processed_512_resized/train/masks"   # <-- Replace with actual label folder path

  # === Load Image Paths ===
  # Count total number of image files (e.g., .jpg format)
  all_image_paths = sorted(glob(os.path.join(image_path, "*.jpg")))  # Use .png if needed
  total_images = len(all_image_paths)
  print(f"Total Number of Images: {total_images}")

  # === Load Label Paths ===
  # Count total number of label files (e.g., .png format for segmentation masks)
  all_label_paths = sorted(glob(os.path.join(label_path, "*.png")))
  total_labels = len(all_label_paths)
  print(f"Total Number of Labels: {total_labels}")

  # === Match Images and Labels ===
  # Assuming both are in matching order and of equal length
  train_image_paths = all_image_paths[:total_images]
  train_label_paths = all_label_paths[:total_labels]

  # Preview label paths (for verification)
  print("Sample label paths:")
  for path in train_label_paths[:5]:
      print(path)

  # === CONFIGURATION ===
  # Set the path to your validation images and labels
  val_image_path = "dataset/processed_512_resized/val/images"   # <-- Replace with actual validation image folder
  val_label_path = "dataset/processed_512_resized/val/masks"   # <-- Replace with actual validation label folder

  # === Load Validation Image Paths ===
  # Collect and sort all .jpg image files in the validation folder
  val_all_image_paths = sorted(glob(os.path.join(val_image_path, "*.jpg")))
  val_total_images = len(val_all_image_paths)
  print(f"Total Number of Validation Images: {val_total_images}")

  # === Load Validation Label Paths ===
  # Collect and sort all .png label files in the validation folder
  val_all_label_paths = sorted(glob(os.path.join(val_label_path, "*.png")))
  val_total_labels = len(val_all_label_paths)
  print(f"Total Number of Validation Labels: {val_total_labels}")

  # === Match Images and Labels (by order) ===
  # This assumes one-to-one correspondence between image and label files
  Val1_image_paths = val_all_image_paths[:val_total_images]
  Val1_label_paths = val_all_label_paths[:val_total_labels]

  # Preview a few label paths to confirm loading
  print("Sample validation label paths:")
  for path in Val1_label_paths[:5]:
      print(path)

  # Please dont run this line if you would like to use the original size of input images.
  desired_size=(512, 512)

  
  # === Load and Process Ground Truth Masks ===
  # This dictionary will store binary masks where pixel > 0 is treated as True
  ground_truth_masks = {}

  for idx in range(len(train_label_paths)):
      # Read the label mask in grayscale
      gt_grayscale = cv2.imread(train_label_paths[idx], cv2.IMREAD_GRAYSCALE)

      # Resize the mask if desired_size is specified
      if desired_size is not None:
          gt_grayscale = cv2.resize(gt_grayscale, desired_size, interpolation=cv2.INTER_LINEAR)

      # Convert to binary mask (True where pixel > 0)
      ground_truth_masks[idx] = (gt_grayscale > 0)

  # Optional: Print number of masks and preview a sample
  print(f"Total ground truth masks loaded: {len(ground_truth_masks)}")
  print("Example binary mask shape:", ground_truth_masks[0].shape)

  # === Load and Process Validation Ground Truth Masks ===
  # This dictionary will store binary masks for validation data
  ground_truth_masksv = {}

  for idx in range(len(Val1_label_paths)):
      # Read the validation label mask in grayscale
      gt_grayscale = cv2.imread(Val1_label_paths[idx], cv2.IMREAD_GRAYSCALE)

      # Resize the mask if a desired size is specified
      if desired_size is not None:
          gt_grayscale = cv2.resize(gt_grayscale, desired_size, interpolation=cv2.INTER_LINEAR)

      # Convert to binary mask: True where pixel > 0
      ground_truth_masksv[idx] = (gt_grayscale > 0)

  # Print summary
  print(f"Total validation ground truth masks loaded: {len(ground_truth_masksv)}")
  print("Example validation mask shape:", ground_truth_masksv[0].shape)

  # === Configuration ===
  # Set the model type: "vit_b", "vit_l", or "vit_h" depending on your .pth file
  model_type = "vit_b"  # or "vit_l", "vit_h", etc.

  # Path to the pretrained SAM checkpoint file (.pth)
  checkpoint = "./checkpoints/sam/sam_vit_b_01ec64.pth"  # <-- Update with actual path

  # === Load Model ===
  # Use the model registry to initialize the correct SAM architecture
  sam_model = sam_model_registry[model_type](checkpoint=checkpoint)

  # Move model to GPU if available, otherwise CPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  sam_model.to(device)

  # === Set Model to Training Mode ===
  # Use `model.train()` when fine-tuning or training the model
  # For inference, use `model.eval()` instead
  sam_model.train()

  print(f"SAM model ({model_type}) loaded on {device} and set to training mode.")


  from collections import defaultdict
  from segment_anything.utils.transforms import ResizeLongestSide

  # Preprocessed image data will be stored in this dictionary
  transformed_data = defaultdict(dict)

  # Transformer that resizes image while preserving aspect ratio
  resize_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

  # === Image Preprocessing Loop ===
  for idx in range(len(train_image_paths)):
      # Load image from path
      image = cv2.imread(train_image_paths[idx])

      # Resize if a fixed input size is specified (e.g., for training consistency)
      if desired_size is not None:
          image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)

      # Convert BGR (OpenCV default) to RGB (SAM model expects RGB)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Apply SAM’s resizing transformation to match its input constraints
      input_image_np = resize_transform.apply_image(image_rgb)

      # Convert NumPy array to torch tensor and add batch dimension
      input_image_tensor = torch.as_tensor(input_image_np, device=device)
      input_image_tensor = input_image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]  # Shape: [1, 3, H, W]

      # Preprocess using SAM model’s preprocessing method (normalization, padding, etc.)
      input_tensor = sam_model.preprocess(input_image_tensor)

      # Store processed data
      transformed_data[idx]['image'] = input_tensor                          # Preprocessed image tensor
      transformed_data[idx]['input_size'] = input_image_tensor.shape[-2:]   # Input tensor size (H, W)
      transformed_data[idx]['original_image_size'] = image_rgb.shape[:2]    # Original image size (H, W)

  print(f"Processed {len(transformed_data)} training images for SAM input.")


  # === Training Hyperparameters ===
  lr = 1e-5                     # Learning rate for optimizer
  wd = 0                        # Weight decay (L2 regularization)
  batch_size = 32              # Number of samples per batch
  num_epochs = 5               # Total number of training epochs

  # === Optimizer Setup ===
  # Only the mask decoder parameters are being fine-tuned (others are frozen)
  optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

  # === Loss Function ===
  # Binary Cross Entropy with logits is commonly used for binary segmentation
  loss_fn = get_loss_function(
                'combined',
                bce_weight=1.0,
                dice_weight=1.0,
                boundary_weight=1.0,
                use_boundary=True
            )

  # === Device Setup ===
  # Automatically use GPU if available, otherwise fallback to CPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # === Ground Truth Mask Keys ===
  # These lists are used to index into your ground truth dictionaries
  keys_train = list(ground_truth_masks.keys())
  keys_valid = list(ground_truth_masksv.keys())

  print(f"Using device: {device}")
  print(f"Training on {len(keys_train)} images, validating on {len(keys_valid)} images")

  from torch.utils.data import DataLoader

  # === Validation DataLoader Setup ===
  # Here we're using a list of file paths as the dataset, which will later need to be wrapped in a proper Dataset class
  val_loader = DataLoader(Val1_image_paths, batch_size=batch_size, shuffle=False)

  # === Basic Validation Dataset Checks ===

  # Total number of validation examples
  num_val_examples = len(Val1_image_paths)
  print(f"Number of validation examples: {num_val_examples}")

  # Number of items returned by val_loader.dataset (same as above since it's a list)
  print(f"Number of examples in validation dataset (via DataLoader): {len(val_loader.dataset)}")

  # Number of batches in the validation DataLoader
  print(f"Number of batches in validation loader: {len(val_loader)}")

  # === Safety Check ===
  # Prevent training from continuing if validation data is empty
  if num_val_examples == 0:
      raise ValueError("The validation dataset is empty. Please check your data paths.")

  torch.cuda.empty_cache()

  # === Utility: Accuracy Calculation ===
  def calculate_accuracy(predictions, targets):
      binary_predictions = (predictions > 0.5).float()
      accuracy = (binary_predictions == targets).float().mean()
      return accuracy.item()

  # === Utility: One Batch Training ===
  def train_on_batch(keys, batch_start, batch_end):
      batch_losses = []
      batch_accuracies = []

      for k in keys[batch_start:batch_end]:
          input_image = transformed_data[k]['image'].to(device)
          input_size = transformed_data[k]['input_size']
          original_image_size = transformed_data[k]['original_image_size']

          # Freeze encoder and prompt embeddings during training
          with torch.no_grad():
              image_embedding = sam_model.image_encoder(input_image)
              sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=None, masks=None)

          # Forward pass through mask decoder
          low_res_masks, iou_predictions = sam_model.mask_decoder(
              image_embeddings=image_embedding,
              image_pe=sam_model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=False,
          )

          # Resize decoder output to original image size
          upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)

          # Convert predicted mask to binary format
          binary_mask = (torch.sigmoid(upscaled_masks) > 0.5).float()

          # Load and reshape ground truth mask
          gt_mask = ground_truth_masks[k]
          gt_tensor = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

          # Loss + Backprop
          loss = loss_fn(binary_mask, gt_tensor)
          optimizer.zero_grad()
          loss.requires_grad = True
          loss.backward()
          optimizer.step()

          batch_losses.append(loss.item())
          batch_accuracies.append(calculate_accuracy(binary_mask, gt_tensor))

      return batch_losses, batch_accuracies

  # === Training Configuration ===
  losses, val_losses = [], []
  accuracies, val_acc = [], []
  best_val_loss = float('inf')

  # === Training Loop ===
  for epoch in range(num_epochs):
      epoch_losses = []
      epoch_accuracies = []

      print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

      # === Training ===
      for batch_start in range(0, len(keys_train ), batch_size):
          batch_end = min(batch_start + batch_size, len(keys_train ))
          batch_losses, batch_accuracies = train_on_batch(keys_train , batch_start, batch_end)

          # Metrics
          epoch_losses.append(mean(batch_losses))
          epoch_accuracies.extend(batch_accuracies)

          print(f'Batch: [{batch_start}-{batch_end}]  Loss: {mean(batch_losses):.4f}  Accuracy: {mean(batch_accuracies):.4f}')

      mean_train_loss = mean(epoch_losses)
      mean_train_accuracy = mean(epoch_accuracies)
      losses.append(mean_train_loss)
      accuracies.append(mean_train_accuracy)

      print(f"Epoch {epoch+1} - Training Loss: {mean_train_loss:.4f}, Accuracy: {mean_train_accuracy:.4f}")

      # === Validation ===
      predictor_tuned = SamPredictor(sam_model)
      val_loss, val_accuracy = 0.0, 0.0
      num_val_examples = len(Val1_image_paths)

      with torch.no_grad():
          for s in range(num_val_examples):
              image = cv2.imread(Val1_image_paths[s])
              if desired_size is not None:
                  image = cv2.resize(image, desired_size)
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

              # Inference using tuned model
              predictor_tuned.set_image(image)
              masks_tuned, _, _ = predictor_tuned.predict(
                  point_coords=None,
                  box=None,
                  multimask_output=False,
              )

              pred_mask = torch.as_tensor((masks_tuned > 0)).float().unsqueeze(0).to(device)
              gt_val_mask = torch.from_numpy(ground_truth_masksv[s].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

              val_loss += loss_fn(pred_mask, gt_val_mask).item()
              val_accuracy += calculate_accuracy(pred_mask, gt_val_mask)

      val_loss /= num_val_examples
      val_acc_epoch = val_accuracy / num_val_examples
      val_losses.append(val_loss)
      val_acc.append(val_acc_epoch)

      print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc_epoch:.4f}")

  # === Save Best Model Checkpoint ===
  # Save model only if validation loss improves
  if val_loss < best_val_loss:
      best_val_loss = val_loss

      # Define a general path to save the model
      model_dir = "./checkpoints"  # <-- Change this to any directory you want
      model_name = f"best_model_epoch{epoch+1}.pth"  # Include epoch or keep static if preferred

      # Create directory if it doesn't exist
      os.makedirs(model_dir, exist_ok=True)

      # Save the model's state dictionary
      save_path = os.path.join(model_dir, model_name)
      torch.save(sam_model.state_dict(), save_path)

      print(f"Saved new best model to: {save_path}")

      torch.cuda.empty_cache()
  print(type(ground_truth_masksv))
  print(ground_truth_masksv.keys() if isinstance(ground_truth_masksv, dict) else len(ground_truth_masksv))