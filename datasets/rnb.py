import os
import json
import math
import numpy as np
from PIL import Image
import OpenEXR
import array
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


def crop_image_tensor_with_bbox(image_tensor: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
   
    if bbox.tolist() == [-1, -1, -1, -1]:
        print("Warning: Invalid bounding box (no foreground found in mask). Returning empty tensor.")
        return torch.empty(0) # Or handle this case as appropriate for your application

    x_min, y_min, x_max, y_max = bbox.tolist()

    # Calculate height and width of the crop
    crop_height = y_max - y_min + 1
    crop_width = x_max - x_min + 1

    # torchvision.transforms.functional.crop expects (img, top, left, height, width)
    # top = y_min, left = x_min
    cropped_image = TF.crop(image_tensor, top=y_min, left=x_min, height=crop_height, width=crop_width)

    return cropped_image



def compute_expanded_bbox_from_mask(mask: torch.Tensor) -> torch.Tensor:

    # Get mask dimensions
    H, W = mask.shape

    # Determine the reference image size for padding calculation
    ref_H, ref_W = H, W

    # Find coordinates of foreground pixels
    # torch.nonzero() returns indices where the mask is non-zero
    foreground_coords = torch.nonzero(mask, as_tuple=False) # shape (N, 2) where N is number of foreground pixels

    if foreground_coords.numel() == 0:
        # No foreground pixels found, return a default "invalid" bbox
        return torch.tensor([-1, -1, -1, -1], dtype=torch.float32)

    # Calculate initial bounding box
    y_min_initial = torch.min(foreground_coords[:, 0]).item()
    y_max_initial = torch.max(foreground_coords[:, 0]).item()
    x_min_initial = torch.min(foreground_coords[:, 1]).item()
    x_max_initial = torch.max(foreground_coords[:, 1]).item()

    # Calculate padding based on 10% of the biggest image dimension
    biggest_dim = max(ref_H, ref_W)
    padding = int(0.05 * biggest_dim) # Ensure padding is an integer

    # Expand bounding box
    x_min_expanded = x_min_initial - padding
    y_min_expanded = y_min_initial - padding
    x_max_expanded = x_max_initial + padding
    y_max_expanded = y_max_initial + padding

    # Clip bounding box to image boundaries (0 to W-1, 0 to H-1)
    x_min_clipped = int(max(0, x_min_expanded))
    y_min_clipped = int(max(0, y_min_expanded))
    x_max_clipped = int(min(W - 1, x_max_expanded))
    y_max_clipped = int(min(H - 1, y_max_expanded))

    # Return tensor with integer type
    return torch.tensor([x_min_clipped, y_min_clipped, x_max_clipped, y_max_clipped], dtype=torch.int32)




def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

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

'''
def gen_light_directions(c2w,normal=None):
        
        if normal is not None : 
            normal = normal.cpu().numpy()

        c2w = c2w.float().to("cuda") 

        nb_views = c2w.shape[0]
        # Extract the 3x3 rotation matrix component from c2w
        R_c2w = c2w[:, :3, :3] # Shape: (nb_views, 3, 3)

        tilt = np.radians([0, 120, 240])
        slant = np.radians([30, 30, 30]) if normal is None else np.radians([54.74, 54.74, 54.74])
        n_lights = tilt.shape[0]

        u = np.array([
            np.sin(slant) * np.cos(tilt),
            np.sin(slant) * np.sin(tilt),
            np.cos(slant)
        ]) # [3, 3(n_lights)]

        if normal is not None:
            n_images, n_rows, n_cols, _ = normal.shape # [n_images, H, W, 3]
            # normal_flat = normal.reshape(-1, 3) # [n_images*H*W, 3]
            # outer_prod = np.einsum('ij,ik->ijk', normal_flat, normal_flat) # [n_images*H*W, 3, 3]
            outer_prod = np.einsum('...j,...k->...jk', normal, normal) # [n_images, H, W, 3, 3]
            U, _, _ = np.linalg.svd(outer_prod)

            det_U = np.linalg.det(U)
            det_U_sign = np.where(det_U < 0, -1, 1)[..., np.newaxis, np.newaxis]

            R = np.where(det_U_sign < 0, 
                        np.einsum('...ij,jk->...ik', U, np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])), 
                        np.einsum('...ij,jk->...ik', U, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])))
            
            R_22 = (R[..., 2, 2] < 0)[..., np.newaxis, np.newaxis]
            R = np.where(R_22, np.einsum('...ij,jk->...ik', R, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])), R)

            light_directions_all = np.einsum('...lm,mn->...ln', R, u) # [n_images, H, W, 3, 3(n_lights)]
            light_directions = light_directions_all.transpose(0, 4, 1, 2, 3)
        else:
            light_directions = u.T

        light_directions = torch.from_numpy(light_directions).float().to("cuda")

        light_directions_world = None
        if light_directions.dim() == 2: 
            light_directions = light_directions.unsqueeze(0).expand(nb_views, -1, -1) 
            light_directions_world = torch.einsum('nij,nlj->nli', R_c2w, light_directions)

        elif light_directions.dim() == 5: 
            
            light_directions_world = torch.einsum('nij,nlhwj->nlhwi', R_c2w, light_directions)
    
        return light_directions_world
'''

class RNbDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True
        self.apply_light_opti = self.config.apply_light_opti

        cams = np.load(os.path.join(self.config.root_dir, 'cameras.npz'))

        img_sample = cv2.imread(os.path.join(self.config.root_dir, 'normal', '000.png'))
        H, W = img_sample.shape[0], img_sample.shape[1]

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = int(W / self.config.img_downscale), int(H / self.config.img_downscale)
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)
        self.factor = w / W
        print(self.w)
        print(self.h)

        #self.near, self.far = self.config.near_plane, self.config.far_plane

        #self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        #self.directions = \
        #    get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        self.all_normals = []
        all_normals_cam = []
        self.directions = []

        n_images = max([int(k.split('_')[-1]) for k in cams.keys()]) + 1

        crop_images = True
        if crop_images : 
            accumulated_mask = torch.zeros((self.img_wh[1], self.img_wh[0]), dtype=torch.float32)
            for i in range(n_images):

                mask_path = os.path.join(self.config.root_dir, 'mask', f'{i:03d}.png')
                mask = Image.open(mask_path).convert('L') # (H, W, 1)
                mask = mask.resize(self.img_wh, Image.BICUBIC)
                mask = TF.to_tensor(mask)[0]
                accumulated_mask = accumulated_mask + mask
            final_boolean_mask_for_bbox = accumulated_mask > 0.0 # Or > 0.5 if masks are strictly [0,1]
            bbox_mask = compute_expanded_bbox_from_mask(final_boolean_mask_for_bbox)
            print(final_boolean_mask_for_bbox.shape)
            print(bbox_mask)

        for i in range(n_images):

            mask_path = os.path.join(self.config.root_dir, 'mask', f'{i:03d}.png')
            mask = Image.open(mask_path).convert('L') # (H, W, 1)
            mask = mask.resize(self.img_wh, Image.BICUBIC)
            mask = TF.to_tensor(mask)[0]

            world_mat, scale_mat = cams[f'world_mat_{i}'], cams[f'scale_mat_{i}']
            P = (world_mat @ scale_mat)[:3,:4]
            K, c2w = load_K_Rt_from_P(P)
            fx, fy, cx, cy = K[0,0] * self.factor, K[1,1] * self.factor, K[0,2] * self.factor, K[1,2] * self.factor
            print(cx)
            print(cy)
            if crop_images:
                cx = cx - bbox_mask[0].item()
                cy = cy - bbox_mask[1].item()

            c2w = torch.from_numpy(c2w).float()
            c2w_ = c2w.clone()
            c2w_[:3,1:3] *= -1. # flip input sign
            self.all_c2w.append(c2w_[:3,:4]) 

            img_path = os.path.join(self.config.root_dir, 'albedo', f'{i:03d}.png')
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]

            

            
            #depth_path = os.path.join(self.config.root_dir, f"{frame['file_path']}_depth.exr")
            #self.all_depths.append(load_exr_image(depth_path))

            #normal_path = os.path.join(self.config.root_dir, f"{frame['file_path']}_normal.exr")
            #normals_base = load_exr_image(normal_path)
            #print(normals_base.shape)
            normal_path2 = os.path.join(self.config.root_dir, 'normal', f'{i:03d}.png')
            normals = Image.open(normal_path2)
            normals = normals.resize(self.img_wh, Image.BICUBIC)
            normals = TF.to_tensor(normals).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
            normals = normals.float() * 2.0 - 1.0
            normals = normals[:,:,:3]
            normals = torch.nn.functional.normalize(normals, p=2, dim=-1)
            #all_normals_cam.append(normals)
            #normals = torch.matmul(normals, c2w_[:3,:3].T)
            boolean_mask = mask > 0.5 # Shape (H, W), boolean tensor
            boolean_mask_expanded = boolean_mask.unsqueeze(-1).expand_as(normals)
            normals[~boolean_mask_expanded] = 0.0
            img[~boolean_mask_expanded] = 0.0

            if crop_images:

                img_to_crop = img.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
                normals_to_crop = normals.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

                img2 = crop_image_tensor_with_bbox(img_to_crop, bbox_mask)
                normals2 = crop_image_tensor_with_bbox(normals_to_crop, bbox_mask)
                img = img2.permute(1, 2, 0) # (C, H_cropped, W_cropped) -> (H_cropped, W_cropped, C)
                normals = normals2.permute(1, 2, 0) # (C, H_cropped, W_cropped) -> (H_cropped, W_cropped, C)
                mask = crop_image_tensor_with_bbox(mask,bbox_mask)
                print(mask.shape)

                self.img_wh = (mask.shape[1], mask.shape[0])
                self.h = mask.shape[0]
                self.w = mask.shape[1]

            add_05 = self.config.add_05
            directions = get_ray_directions(w, h, fx, fy, cx, cy, use_pixel_centers=add_05)
            self.directions.append(directions)
            
            self.all_fg_masks.append(mask) # (h, w)
            self.all_images.append(img[...,:3])

            self.all_normals.append(normals)

        if split == 'train' and self.config.get('num_views', False):
            num_views = self.config.num_views
            jump = len(self.all_c2w) // num_views
            if jump == 0:
                jump = 1
            self.all_c2w = self.all_c2w[::jump]
            self.all_images = self.all_images[::jump]
            self.all_fg_masks = self.all_fg_masks[::jump]
            #self.all_depths = self.all_depths[::jump]
            self.all_normals = self.all_normals[::jump]
            #all_normals_cam = all_normals_cam[::jump]

        #all_normals_cam = torch.stack(all_normals_cam, dim=0).float().to(self.rank)

        
        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
        self.directions = torch.stack(self.directions, dim=0)

        #if self.apply_light_opti:
        #    self.all_lights = gen_light_directions(self.all_c2w,all_normals_cam)
        #else : 
        #    self.all_lights = gen_light_directions(self.all_c2w)
        
        
        self.directions = self.directions.float().to(self.rank)
        #self.all_depths = torch.stack(self.all_depths, dim=0).float().to(self.rank)
        self.all_normals = torch.stack(self.all_normals, dim=0).float().to(self.rank)
        
        if self.split == 'test':
            self.test_render_combinations = []
            num_base_images = len(self.all_images)
            num_light_conditions = 3
            
            for img_idx in range(num_base_images):
                for light_idx in range(num_light_conditions):
                    self.test_render_combinations.append({'image_idx': img_idx, 'light_idx': light_idx})
            print(f"Test split: Prepared {len(self.test_render_combinations)} image-light combinations for rendering.")


class RNbDataset(Dataset, RNbDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        if self.split == 'test':
            return len(self.test_render_combinations)
        return len(self.all_images)
    
    def __getitem__(self, index):
        if self.split == 'test':
            # For test split, 'item_idx' maps to a specific (image, light) combination
            combination = self.test_render_combinations[index]
            return {
                'index': torch.tensor(combination['image_idx'], dtype=torch.long),
                'index_light': torch.tensor(combination['light_idx'], dtype=torch.long),
            }
        else:
            # For validation split (since IterableDataset handles train),
            # item_idx directly corresponds to the image index.
            return {
                'index': torch.tensor(index, dtype=torch.long)
            }


class RNbIterableDataset(IterableDataset, RNbDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('RNb')
class RNbDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = RNbIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = RNbDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = RNbDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = RNbDataset(self.config, self.config.train_split)

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
