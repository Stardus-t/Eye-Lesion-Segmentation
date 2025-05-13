import os
import json
import torch
from PIL import Image
import random
import dotenv
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import numpy as np

dotenv.load_dotenv()
anno_k_path=os.getenv('ANNO_K_PATH')
anno_images_path=os.getenv('ANNO_IMAGES_PATH')

with open(anno_k_path,'r')as f:
    data=json.load(f)
filename_to_id={img['file_name']:img['id'] for img in data['images']}
image_files=[f for f in os.listdir(anno_images_path)if f.lower().endswith(('png','jpg'))]

for img_file in image_files:
    try:
        image_id=filename_to_id[img_file]
        image_path=os.path.join(anno_images_path,img_file)
        image=Image.open(image_path)
    except KeyError:
        print(f"Warning: Image{img_file} not exist")

class PairedTensorTransform_for_Train:
    def __init__(self,
                 rotation_degrees=30,
                 resize_size=None,
                 crop_size=None,
                 hflip_prob=0.5):
        if isinstance(rotation_degrees,int):
            if rotation_degrees == 0:
                self.rotation_degress = None
            else:
                self.rotation_degress = (-rotation_degrees, rotation_degrees)
        else:
            self.rotation_degress=rotation_degrees
        self.crop_size=crop_size
        self.resize_size=resize_size
        self.hflip_prob=hflip_prob
    def random_rotate(self,image,mask):
        if self.rotation_degress is None:
            return image,mask
        angle=random.uniform(*self.rotation_degress)
        image=F.rotate(image,angle,interpolation=InterpolationMode.BILINEAR,fill=(0,0,0))
        mask=F.rotate(mask, angle, interpolation=InterpolationMode.BILINEAR,fill=(0,))
        return image,mask
    def paied_resize(self,image,mask):
        if self.resize_size is None:
            return image,mask
        image=F.resize(image,self.resize_size,interpolation=InterpolationMode.BILINEAR)
        mask=F.resize(mask, self.resize_size, interpolation=InterpolationMode.BILINEAR)
        return image,mask
    def random_crop(self,image,mask):
        if self.crop_size is None:
            return image,mask
        _,h,w=image.shape
        th,tw=self.crop_size
        if h < th or w < tw:
            image = F.resize(image, (th, tw), interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, (th, tw), interpolation=InterpolationMode.BILINEAR)
            return image, mask
        if h==th and w==tw:
            return image,mask
        i=random.randint(0,h-th)
        j=random.randint(0,w-tw)
        image=image[:,i:i+th,j:j+tw]
        mask=mask[:,i:i+th,j:j+tw]
        return image,mask
    def random_hflip(self,image,mask):
        if random.random()<self.hflip_prob:
            image=torch.flip(image,dims=[2])
            mask=torch.flip(mask,dims=[2])
        return image,mask
    def __call__(self,image,mask):
        image,mask=self.random_rotate(image,mask)
        image,mask=self.paied_resize(image,mask)
        image,mask=self.random_crop(image,mask)
        image,mask=self.random_hflip(image,mask)

        return image,mask

class PairedTensorTransform_for_Test:
    def __init__(self,
                 rotation_degrees=30,
                 resize_size=None,
                 crop_size=None,
                 hflip_prob=0.5):
        self.resize_size=resize_size
    def paied_resize(self,image,mask):
        if self.resize_size is None:
            return image,mask
        image=F.resize(image,self.resize_size,interpolation=InterpolationMode.BILINEAR)
        mask=F.resize(mask, self.resize_size, interpolation=InterpolationMode.BILINEAR)
        return image,mask
    def __call__(self,image,mask):
        image,mask=self.paied_resize(image,mask)
        if mask.shape[0]>1:
            mask=mask[0:1,:,:]
        return image,mask

class AugSegDataset(Dataset):
    def __init__(self,data_list,transform=None):
        self.data_list=data_list
        self.transform=transform
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        sample=self.data_list[idx]
        image_np=sample['image']
        mask_np=sample['mask']
        image_pil=Image.fromarray(image_np)
        if image_pil.mode!='RGB':
            image_pil=image_pil.convert('RGB')
        image_np=np.array(image_pil)
        image_tensor=torch.from_numpy(image_np).permute(2,0,1).contiguous().float()/255.0
        mask_tensor=torch.from_numpy(mask_np).permute(2,0,1).contiguous().float()
        if self.transform:
            image_tensor,mask_tensor=self.transform(image_tensor,mask_tensor)
        return image_tensor,mask_tensor.to(torch.long)