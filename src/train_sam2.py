import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random


def read_batch(data, visualize_data=False):
    # Select a random entry
    ent = data[np.random.randint(len(data))]

    print(ent)

    # Get full paths
    img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Read annotation as grayscale

    if img is None or ann_map is None:
       print(f"Error: Could not read image or mask from path {ent['image']} or {ent['annotation']}")
       return None, None, None, 0

    # Resize image and mask
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # Scaling factor
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)


    # Initialize a single binary mask
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []

    # Get binary masks and combine them into a single mask
    inds = np.unique(ann_map)[1:]  # Skip the background (index 0)
    for ind in inds:
       mask = (ann_map == ind).astype(np.uint8)  # Create binary mask for each unique index
       binary_mask = np.maximum(binary_mask, mask)  # Combine with the existing binary mask
 
    # Erode the combined binary mask to avoid boundary points
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)

    # Get all coordinates inside the eroded mask and choose a random point
    coords = np.argwhere(eroded_mask > 0)
    if len(coords) > 0:
       for _ in inds:  # Select as many points as there are unique labels
           yx = np.array(coords[np.random.randint(len(coords))])
           points.append([yx[1], yx[0]])

    points = np.array(points)

       ### Continuation of read_batch() ###

    if visualize_data:
        # Plotting the images and points
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')

        # Segmentation Mask (binary_mask)
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # Mask with Points in Different Colors
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # Plot points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')  # Corrected to plot y, x order

        # plt.legend()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Now shape is (1024, 1024, 1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)

    # Return the image, binarized mask, points, and number of masks
    return img, binary_mask, points, len(inds)

def train_model(predictor, train_data):
    # Train mask decoder.
    predictor.model.sam_mask_decoder.train(True)

    # Train prompt encoder.
    predictor.model.sam_prompt_encoder.train(True)

    # Configure optimizer.
    optimizer = torch.optim.AdamW(params = predictor.model.parameters(),
                                  lr = 0.0001,weight_decay=1e-4) #1e-5, weight_decay = 4e-5

    # Mix precision.
    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    # No. of steps to train the model.
    NO_OF_STEPS = 3000 # @param 

    # Fine-tuned model name.
    FINE_TUNED_MODEL_NAME = "fine_tuned_sam2"

    # # Initialize scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #                 optimizer,
    #                 step_size = 500,
    #                 gamma     = 0.2,
    #             )
    accumulation_steps = 4  # Number of steps to accumulate gradients before updating

    for step in range(1, NO_OF_STEPS + 1):
        with torch.amp.autocast('cuda'):
            print(step)
            image, mask, input_point, num_masks = read_batch(train_data, visualize_data=False)
            if image is None or mask is None or num_masks == 0:
                continue

            input_label = np.ones((num_masks, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue

            if input_point.size == 0 or input_label.size == 0:
                continue

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            # Apply gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()

            # # Update scheduler
            # scheduler.step()

            if step % 500 == 0:
                FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step) + ".torch"
                torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

            if step == 1:
                mean_iou = 0

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            if step % 100 == 0:
                print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)

def read_image(image_path, mask_path):  # read and resize image and mask
   img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
   mask = cv2.imread(mask_path, 0)
   r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
   img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
   mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
   return img, mask

def get_points(mask, num_points):  # Sample points inside the input mask
   points = []
   coords = np.argwhere(mask > 0)
   for i in range(num_points):
       yx = np.array(coords[np.random.randint(len(coords))])
       points.append([[yx[1], yx[0]]])
   return np.array(points)

def evaluate_model(model, predictor, image_path, mask_path):
    # Randomly select a test image from the test_data
    selected_entry = random.choice(test_data)
    image_path = selected_entry['image']
    mask_path = selected_entry['annotation']

    # Load the selected image and mask
    image, mask = read_image(image_path, mask_path)
    original_mask = mask

    # Generate random points for the input
    num_samples = 30  # Number of points per segment to sample
    input_points = get_points(mask, num_samples)

    # Perform inference and predict masks
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    # Process the predicted masks and sort by scores
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    # Initialize segmentation map and occupancy mask
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

    # Combine masks to create the final segmentation map
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue

    mask_bool = mask.astype(bool)
    mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
    seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
    occupancy_mask[mask_bool] = True  # Update occupancy_mask

    # Visualization: Show the original image, mask, and final segmentation side by side
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Test Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Original Mask')
    plt.imshow(original_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Final Segmentation')
    plt.imshow(seg_map, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# if __name__ == '__main__':
#     train_data_dir = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/dataset/processed_512_resized/sequential/train/'
#     train_data=[] 
#     for ff, name in enumerate(os.listdir(train_data_dir+"images/")):  # go over all folder annotation
#         train_data.append({"image":train_data_dir+"images/"+name,"annotation":train_data_dir+"masks/"+name[:-4]+".png"})

#     # Img1, masks1, points1, num_masks = read_batch(train_data, visualize_data=True)

#     sam2_checkpoint = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam2/sam2.1_hiera_small.pt'
#     model_cfg = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam2/sam2.1_hiera_s.yaml' 

#     sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model
#     predictor = SAM2ImagePredictor(sam2_model) # load net

#     train_model(predictor)

if __name__ == '__main__':
    test_data_dir = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/dataset/processed_512_resized/sequential/test/'
    test_data=[] 
    for ff, name in enumerate(os.listdir(test_data_dir+"images/")):  # go over all folder annotation
        test_data.append({"image":test_data_dir+"images/"+name,"annotation":test_data_dir+"masks/"+name[:-4]+".png"})

    # Load the fine-tuned model
    FINE_TUNED_MODEL_WEIGHTS = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/fine_tuned_sam2_1000.torch'

    sam2_checkpoint = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam2/sam2.1_hiera_small.pt'
    model_cfg = r'c:/Users/AdikariAdikari/PycharmProjects/river_segmentation/checkpoints/sam2/sam2.1_hiera_s.yaml' 

    # Build net and load weights
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))
    
    for test_item in test_data:
        image_path = test_item['image']
        mask_path = test_item['annotation']
        evaluate_model(sam2_model, predictor, image_path, mask_path)
