import os
import json
import math
import numpy as np
from PIL import Image
import OpenEXR
import array

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank

def load_exr_image(filepath):
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    
    # Get image dimensions
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Read RGB channels
    if 'depth' in filepath:
        channels = exr_file.channels(['R', 'G', 'B'])
    elif 'normal' in filepath:
        channels = exr_file.channels(['X', 'Y', 'Z'])
    
    # Convert to numpy array
    rgb_data = []
    for channel in channels:
        channel_data = array.array('f', channel)
        rgb_data.append(np.frombuffer(channel_data, dtype=np.float32))
    
    # Reshape to image dimensions
    rgb_array = np.stack(rgb_data, axis=-1)
    rgb_array = rgb_array.reshape(height, width, 3)

    if 'depth' in filepath:
        rgb_array = rgb_array.mean(axis=-1)  # Convert to grayscale if needed
    
    exr_file.close()
    return torch.from_numpy(rgb_array)

class BlenderCuesDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        self.all_normals, self.all_depths = [], []

        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            self.all_fg_masks.append(img[..., -1]) # (h, w)
            self.all_images.append(img[...,:3])

            depth_path = os.path.join(self.config.root_dir, f"{frame['file_path']}_depth.exr")
            self.all_depths.append(load_exr_image(depth_path))

            normal_path = os.path.join(self.config.root_dir, f"{frame['file_path']}_normal.exr")
            normals_base = load_exr_image(normal_path)
            #print(normals_base.shape)
            normal_path2 = os.path.join(self.config.root_dir, f"{frame['file_path']}_normal.png")
            normals = Image.open(normal_path2)
            normals = normals.resize(self.img_wh, Image.BICUBIC)
            normals = TF.to_tensor(normals).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
            normals = normals.float() * 2.0 - 1.0
            normals = normals[:,:,:3]
            normals = torch.nn.functional.normalize(normals, p=2, dim=-1)
            #print(normals.shape)
            
            self.all_normals.append(normals)

        if split == 'train' and self.config.get('num_views', False):
            num_views = self.config.num_views
            jump = len(self.all_c2w) // num_views
            if jump == 0:
                jump = 1
            self.all_c2w = self.all_c2w[::jump]
            self.all_images = self.all_images[::jump]
            self.all_fg_masks = self.all_fg_masks[::jump]
            self.all_depths = self.all_depths[::jump]
            self.all_normals = self.all_normals[::jump]
            self.directions = self.directions[::jump]

        print("\n\n"+str(len(self.all_normals))+"\n\n")
        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
        
        
        self.all_depths = torch.stack(self.all_depths, dim=0).float().to(self.rank)
        self.all_normals = torch.stack(self.all_normals, dim=0).float().to(self.rank)
        

class BlenderCuesDataset(Dataset, BlenderCuesDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderCuesIterableDataset(IterableDataset, BlenderCuesDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('blendercues')
class BlenderCuesDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderCuesIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderCuesDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderCuesDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderCuesDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
