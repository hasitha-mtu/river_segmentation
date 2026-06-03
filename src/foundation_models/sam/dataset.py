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
import cv2
import os

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
        "label": [np.array(Image.open(masks_dir / f"{image_path.stem}.png").convert('L')) for image_path in image_paths],
        "image_path" : [os.path.basename(image_path) for image_path in image_paths],
        "image_full_path" : [str(image_path) for image_path in image_paths],
        }
    dataset = InitialDataset.from_dict(dataset_dict)
    print(dataset.shape)
    return dataset

# def create_sam2_dataset(data_dir, split, batch_size):
#     data_dir = f'{data_dir}/{split}'
#     print(f'data_dir: {data_dir}')
#     data=[]
#     for ff, name in enumerate(os.listdir(data_dir+"/images/")):  # go over all folder annotation
#         image_path = data_dir+"/images/"+name
#         mask_path = data_dir+"/masks/"+name[:-4]+".png"
#         image, mask, points, labels_size = format_dataset_item(image_path, mask_path)
#         data.append({
#             'image': image,
#             'mask': mask,
#             'points': points,
#             'labels_size': labels_size
#         })
#     return data


def create_sam2_dataset(data_dir, split, batch_size, img_size):
    data_dir = f'{data_dir}/{split}'
    print(f'data_dir: {data_dir} and split: {split}')

    data = []

    for ff, name in enumerate(os.listdir(data_dir + "/images/")):
        image_path = data_dir + "/images/" + name
        mask_path = data_dir + "/masks/" + name[:-4] + ".png"

        image, mask, points, labels_size = format_dataset_item(image_path, mask_path, img_size)

        data.append({
            'image': image,
            'mask': mask,
            'points': points,
            'labels_size': labels_size,
            'image_path': image_path
        })

    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return batches

def format_dataset_item(image_path,  mask_path, img_size):
    img = cv2.imread(image_path)[..., ::-1]  # read image
    ann_map = cv2.imread(mask_path)  # read annotation

    # resize image
    r = np.min([img_size / img.shape[1], img_size / img.shape[0]])  # scalling factor
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                         interpolation=cv2.INTER_NEAREST)

    # merge vessels and materials annotations
    mat_map = ann_map[:, :, 0]  # material annotation map
    ves_map = ann_map[:, :, 2]  # vessel  annotaion map

    offset = mat_map.max()
    merge_mask = (mat_map == 0) & (ves_map > 0)
    mat_map[merge_mask] = ves_map[merge_mask] + offset

    # Get binary masks and points
    inds = np.unique(mat_map)
    inds = inds[inds > 0]

    points = []
    masks = []
    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)  # make binary mask
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))])  # choose random point/coordinate
        points.append([[yx[1], yx[0]]])

    return img, np.array(masks), np.array(points), len(masks)
    

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



