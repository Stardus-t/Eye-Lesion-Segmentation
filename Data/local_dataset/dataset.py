import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import dotenv
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms.functional as F
import random
from torchvision.transforms import InterpolationMode
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

dotenv.load_dotenv()
anno_k_path = os.getenv('ANNO_K_PATH')
anno_images_path = os.getenv('ANNO_IMAGES_PATH')

def load_annotations(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def get_image_annotations(annotations, image_name):
    image_data = next((img for img in annotations['images'] if img['file_name'] == image_name), None)
    if not image_data:
        raise ValueError(f"Image {image_name} not found in annotations.")
    image_id = image_data['id']
    image_annotations = [anno for anno in annotations['annotations'] if anno['image_id'] == image_id]
    return image_data, image_annotations

def coco_polygons_to_mask(segmentation, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_image = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_image)
    for polygon in segmentation:
        poly = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        draw.polygon(poly.flatten().tolist(), outline=1, fill=1)
    mask = np.array(mask_image, dtype=np.uint8)
    return mask

def annotations_to_masks(image_annotations, image_data, sorted_category_ids):
    masks = {}
    H, W = image_data["height"], image_data["width"]
    for anno in image_annotations:
        category_id = anno['category_id']
        mask = coco_polygons_to_mask(anno['segmentation'], H, W)
        if category_id not in masks:
            masks[category_id] = mask
        else:
            masks[category_id] = np.maximum(masks[category_id], mask)
    masks_tensor = []
    for c in sorted_category_ids:
        _m = masks.get(c)
        if _m is None:
            masks_tensor.append(np.zeros((H, W), dtype=np.uint8))
        else:
            masks_tensor.append(_m)
    return np.stack(masks_tensor, axis=0)

def get_PCO_image_datalist(anno_k_path, anno_images_path):
    roi_annos = load_annotations(anno_k_path)
    im_fnames = [_['file_name'] for _ in roi_annos['images']]
    pco_data_list = []
    sorted_category_ids = sorted([c['id'] for c in roi_annos['categories']])

    for fn in im_fnames:
        im_info, im_annos = get_image_annotations(roi_annos, fn)
        mask_images = annotations_to_masks(im_annos, im_info, sorted_category_ids)
        # 构建完整的图像路径
        im = Image.open(os.path.join(anno_images_path, im_info['file_name']))
        pco_data_list.append({
            'image': np.array(im),
            'class_masks': mask_images,
            'image_info': im_info
        })
    return pco_data_list, roi_annos

class PairedTensorTransform:
    def __init__(self,
                 rotation_degrees=30,
                 resize_size=None,
                 crop_size=None,
                 hflip_prob=0.5):
        if isinstance(rotation_degrees, int):
            if rotation_degrees == 0:
                self.rotation_degrees = None
            else:
                self.rotation_degrees = (-rotation_degrees, rotation_degrees)
        else:
            self.rotation_degrees = rotation_degrees
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.hflip_prob = hflip_prob

    def random_rotate(self, img, mask):
        if self.rotation_degrees is None:
            return img, mask
        angle = random.uniform(*self.rotation_degrees)
        img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        return img, mask

    def paired_resize(self, img, mask):
        if self.resize_size is None:
            return img, mask
        img = F.resize(img, self.resize_size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.resize_size, interpolation=InterpolationMode.NEAREST)
        return img, mask

    def random_crop(self, img, mask):
        if self.crop_size is None:
            return img, mask
        _, h, w = img.shape
        th, tw = self.crop_size
        if w == tw and h == th:
            return img, mask
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        img = img[:, i:i+th, j:j+tw]
        mask = mask[:, i:i+th, j:j+tw]
        return img, mask

    def random_hflip(self, img, mask):
        if random.random() < self.hflip_prob:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[2])
        return img, mask

    def __call__(self, img, mask):
        img, mask = self.random_rotate(img, mask)
        img, mask = self.paired_resize(img, mask)
        img, mask = self.random_crop(img, mask)
        img, mask = self.random_hflip(img, mask)
        return img, mask

class AugSegDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        image_np = sample['image']
        mask_np = sample['class_masks']
        image_pil = Image.fromarray(image_np)
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        image_np = np.array(image_pil)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).float()
        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        return image_tensor, mask_tensor.to(torch.bool)

# 构建数据集
pco_data_list, roi_annos = get_PCO_image_datalist(anno_k_path, anno_images_path)

train_transform = PairedTensorTransform(
    resize_size=(256, 256),
    crop_size=(256, 256),
    rotation_degrees=20
)
test_transform = PairedTensorTransform(
    resize_size=(256, 256),
    rotation_degrees=0,
    crop_size=None,
    hflip_prob=0
)

dataset = AugSegDataset(pco_data_list, test_transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset.dataset.transform = train_transform

print(f'训练集大小：{len(train_dataset)}')
print(f'测试集大小：{len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

for image, mask in train_loader:
    print(f'图片形状：{image.size()}')
    print(f'掩码形状：{mask.size()}')
    print("Mask unique values:", torch.unique(mask))
    mask_int = mask.long()
    print("Mask value counts:", torch.bincount(mask_int.flatten()))
    single_image = image[0].cpu().permute(1, 2, 0).numpy()
    single_mask = mask[0, 0].cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(single_image)
    plt.axis('off')
    plt.title('原始图像')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(single_mask, cmap='gray')
    plt.axis('off')
    plt.title('掩码')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(single_image)
    plt.imshow(single_mask, alpha=0.4, cmap='Reds')
    plt.axis('off')
    plt.title('掩码叠加在原始图像上')
    plt.show()
    break

for image, mask in test_loader:
    print(f'图片形状：{image.size()}')
    print(f'掩码形状：{mask.size()}')
    print("Mask unique values:", torch.unique(mask))
    mask_int = mask.long()
    print("Mask value counts:", torch.bincount(mask_int.flatten()))
    single_image = image[0].cpu().permute(1, 2, 0).numpy()
    single_mask = mask[0, 0].cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(single_image)
    plt.axis('off')
    plt.title('原始图像')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(single_mask, cmap='gray')
    plt.axis('off')
    plt.title('掩码')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(single_image)
    plt.imshow(single_mask, alpha=0.4, cmap='Reds')
    plt.axis('off')
    plt.title('掩码叠加在原始图像上')
    plt.show()
    break