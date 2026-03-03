import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
import numpy as np
import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy

def gen_light_directions(R_c2w,index_light,normal_world=None):
        


    # Define base tilt and slant angles for the light sources in a canonical frame.
    # Convert degrees to radians using torch.deg2rad
    tilt = torch.deg2rad(torch.tensor([0., 120., 240.], device='cuda')) # Azimuth angles for 3 lights
    
    # Slant angles (zenith angles) for light sources.
    # Ensure these are float tensors and on the correct device.
    slant_val = 30. if normal_world is None else 54.74
    slant = torch.deg2rad(torch.tensor([slant_val, slant_val, slant_val], device='cuda'))

    # Convert spherical coordinates (slant, tilt) to Cartesian coordinates (x, y, z)
    # These are initial light directions in a canonical local space.
    # Use torch.sin and torch.cos
    u = torch.stack([
        torch.sin(slant) * torch.cos(tilt),
        torch.sin(slant) * torch.sin(tilt),
        torch.cos(slant)
    ], dim=0) # Shape: (3, n_lights)

    if normal_world is not None:
        
        normal = torch.matmul(normal_world,R_c2w)
        
        outer_prod = torch.einsum('...j,...k->...jk', normal, normal) # Shape: (N, 3, 3)
        U = torch.linalg.svd(outer_prod).U # U: (N, 3, 3)
        det_U = torch.linalg.det(U) # Shape: (N,)
        det_U_sign = torch.where(det_U < 0, -1., 1.).unsqueeze(-1).unsqueeze(-1) # Shape: (N, 1, 1)
        mat1 = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.]], device='cuda', dtype=torch.float32)
        mat2 = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], device="cuda", dtype=torch.float32)
        
        # Apply the alignment rotation (R) to U using torch.einsum and torch.where.
        R = torch.where(det_U_sign < 0, 
                        torch.einsum('...ij,jk->...ik', U, mat1), 
                        torch.einsum('...ij,jk->...ik', U, mat2)) # R: (N, 3, 3)
        
        # Further adjustment based on the R[..., 2, 2] component (z-axis alignment).
        R_22 = (R[..., 2, 2] < 0).unsqueeze(-1).unsqueeze(-1) # Shape: (N, 1, 1)
        mat3 = torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], device='cuda', dtype=torch.float32)
        R = torch.where(R_22, torch.einsum('...ij,jk->...ik', R, mat3), R) # R: (N, 3, 3)

        light_directions_all = torch.einsum('...lm,mn->...ln', R, u) 
        light_directions = light_directions_all.permute(2, 0, 1) # Shape: (n_lights, N, 3)
    else:
        # If no normal is provided, light directions are simply the initial 'u' directions, permuted.
        # These are light directions in the camera's local coordinate system.
        light_directions = u.permute(1, 0) # Shape: (n_lights, 3)

    
    selected_light_direction = light_directions[index_light]

    #print(selected_light_direction.shape)
    light_directions_world = None
    if normal_world is None: 
        light_directions_world = torch.matmul(selected_light_direction,R_c2w.permute(1, 0))
        light_directions_world = light_directions_world.unsqueeze(0) # Output shape: (1, 3)
    else: 
        light_directions_world = torch.matmul(selected_light_direction,R_c2w.permute(1, 0))
    
    return light_directions_world


@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

    def on_fit_start(self):
        """Propagate scene scaling from dataset to geometry for inverse scaling at export.
        Runs unconditionally — any sfm dataset needs inverse scaling, even without albedo scaling."""
        import torch
        ds = self.trainer.datamodule.train_dataloader().dataset
        if hasattr(ds, 'scale_factor'):
            self.model.geometry.scene_center.copy_(
                torch.tensor(ds.scene_center, dtype=torch.float32))
            self.model.geometry.scale_factor.fill_(ds.scale_factor)
            rank_zero_info(f"[NeuS] Scene scaling propagated: center={ds.scene_center}, scale={ds.scale_factor}")

    def forward(self, batch):
        return self.model(batch['rays'],batch['lights'])
    
    def preprocess_data(self, batch, stage):

        # Get dataset instance directly from trainer for clarity and access to pre-loaded data
        dataset = self.dataset
        index_light = torch.randint(0, 3, (1,)).item()

        if stage in ['train']:
            # For training, randomly pick an image index and a random light index per batch
            # If batch_image_sampling is True, 'index' will be a batch of image indices.
            # Otherwise, it's a single image index.
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(dataset.all_images), size=(self.train_num_rays,), device=dataset.all_images.device)
            else:
                index = torch.randint(0, len(dataset.all_images), size=(1,), device=dataset.all_images.device)

            # Randomly select one of the light conditions for training
            # Your gen_light_directions generates N_lights conditions (default 3)
            
            
            # Make sure 'index' and 'index_light' are in the batch for potential downstream use
            # (though the main data fetching uses the directly computed `index` and `index_light` here)
            batch.update({'index': index, 'index_light': torch.tensor([index_light], dtype=torch.long, device=self.rank)})

            c2w = dataset.all_c2w[index]
            x = torch.randint(
                0, dataset.w, size=(self.train_num_rays,), device=dataset.all_images.device
            )
            y = torch.randint(
                0, dataset.h, size=(self.train_num_rays,), device=dataset.all_images.device
            )
            if dataset.directions.ndim == 3: # (H, W, 3)
                directions = dataset.directions[y, x]
            elif dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = dataset.all_images[index, y, x].view(-1, dataset.all_images.shape[-1]).to(self.rank)
            if self.config.model.no_albedo : 
                rgb = torch.ones_like(rgb)

            fg_mask = dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
            if hasattr(dataset, 'all_normals'):
                normals = dataset.all_normals[index, y, x].view(-1, 3).to(self.rank)

            #print("\n\n")
            #print(c2w[0,:3,:3].shape)
            #print(normals.shape)
            #print("\n\n")
            if self.config.dataset.apply_light_opti:
                lights = gen_light_directions(c2w[0,:3,:3],index_light,normals)
            else : 
                lights = gen_light_directions(c2w[0,:3,:3],index_light)


            if not self.config.dataset.apply_light_opti:
                lights = lights.expand(rgb.shape[0], -1) # (N_rays, 3)
            lights = lights.to(self.rank) # Ensure on correct device

            
            # Update batch with retrieved data
            batch.update({
                'rgb': rgb,
                'fg_mask': fg_mask,
                'normals': normals if hasattr(dataset, 'all_normals') else None,

                'lights': lights, # Pass the prepared lights
                'H': dataset.h, # Pass H,W from dataset for test_step
                'W': dataset.w,
            })

        else: # Validation or Testing
            # In validation/test, batch contains 'index' (image_idx) and optionally 'index_light' for test
            img_idx = batch['index'][0].item() # Get scalar image index
            
            # For validation, there's no 'index_light' from __getitem__; use first light
            # For test, 'index_light' is provided in the batch.
            if stage == 'test':
                light_idx = batch['index_light'][0].item()
            else: # val stage
                # For validation, pick the first light condition (index 0) for consistency
                light_idx = 0

            # --- Data Retrieval based on img_idx and light_idx ---
            c2w = dataset.all_c2w[img_idx]
            if dataset.directions.ndim == 3: # (H, W, 3)
                directions = dataset.directions # Directions are global for the image
            elif dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = dataset.directions[img_idx] # Directions specific to this image
            
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = dataset.all_images[img_idx].view(-1, dataset.all_images.shape[-1]).to(self.rank)

            if self.config.model.no_albedo : 
                rgb = torch.ones_like(rgb)

            fg_mask = dataset.all_fg_masks[img_idx].view(-1).to(self.rank)
            if hasattr(dataset, 'all_normals'):
                normals = dataset.all_normals[img_idx].view(-1, 3).to(self.rank)


            if self.config.dataset.apply_light_opti:
                lights = gen_light_directions(c2w[:,:3],index_light,normals)
            else : 
                lights = gen_light_directions(c2w[:,:3],index_light)


            if not self.config.dataset.apply_light_opti:
                lights = lights.expand(rgb.shape[0], -1) # (N_rays, 3)
            lights = lights.to(self.rank) # Ensure on correct device


            
            # Update batch with retrieved data
            batch.update({
                'index': torch.tensor([img_idx], dtype=torch.long, device=self.rank), # Ensure index is back in batch
                'rgb': rgb,
                'fg_mask': fg_mask,
                'normals': normals if hasattr(dataset, 'all_normals') else None,
                'lights': lights, # Pass the prepared lights
                'H': dataset.h, # Pass H,W from dataset for test_step
                'W': dataset.w,
            })
            # Also pass light_idx if it's the test stage, for saving filenames
            if stage == 'test':
                batch.update({'index_light': torch.tensor([light_idx], dtype=torch.long, device=self.rank)})

        # Common batch updates for both stages
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)
        batch.update({'rays': rays})

        plus_mse = torch.sqrt(3.0 - (rgb[...,0]**2 + rgb[...,1]**2 + rgb[...,2]**2)).unsqueeze(-1)
        plus_l1 = 3.0 - (rgb[...,0] + rgb[...,1] + rgb[...,2]).unsqueeze(-1)

        rgb_plus_mse = torch.cat([rgb, plus_mse], dim=-1)
        rgb_plus_l1 = torch.cat([rgb, plus_l1], dim=-1)

        # Shading and rendering calculation
        # Make sure normals is available before using it in shading
        if 'normals' in batch and batch['normals'] is not None:
            shading = torch.einsum('ij,ij->i', batch['normals'], batch['lights'])
            rendering = torch.einsum('i,ij->ij', shading, batch['rgb'])
            rendering_plus_mse = torch.einsum('i,ij->ij', shading, rgb_plus_mse)
            rendering_plus_l1 = torch.einsum('i,ij->ij', shading, rgb_plus_l1)
        else:
            # Fallback if normals are not available, or handle error as per your design
            rendering = batch['rgb'] # Or some default/error
        batch.update({'rendering': rendering})
        batch.update({'rendering_plus_mse': rendering_plus_mse})
        batch.update({'rendering_plus_l1': rendering_plus_l1})

        if stage in ['train']:
            if self.config.model.background_color == 'black':
                self.model.background_color = torch.zeros((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else: # Val and Test
            self.model.background_color = torch.zeros((3,), dtype=torch.float32, device=self.rank)

        background_color_plus = torch.zeros((1,), dtype=torch.float32, device=self.rank)
        self.model.background_color_plus = torch.cat([self.model.background_color, background_color_plus], dim=-1)
        
        if dataset.apply_mask: # Use dataset.apply_mask property
            batch['rgb'] = batch['rgb'] * batch['fg_mask'][...,None] + self.model.background_color * (1 - batch['fg_mask'][...,None])
            batch['rendering'] = batch['rendering'] * batch['fg_mask'][...,None] + self.model.background_color * (1 - batch['fg_mask'][...,None])
            batch['rendering_plus_l1'] = batch['rendering_plus_l1'] * batch['fg_mask'][...,None] + self.model.background_color_plus * (1 - batch['fg_mask'][...,None])
            batch['rendering_plus_mse'] = batch['rendering_plus_mse'] * batch['fg_mask'][...,None] + self.model.background_color_plus * (1 - batch['fg_mask'][...,None])

        if hasattr(self.dataset, 'all_normals'):
            batch.update({
                'normals': normals,
            })
        
        if hasattr(self.dataset, 'all_depths'):
            batch.update({
                'depths': depths,
            })
        
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            num_samples_full = out['num_samples_full'].sum().item()
            if num_samples_full > 0:
                train_num_rays = int(self.train_num_rays * (self.train_num_samples / num_samples_full))
                self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        '''
        # depth scale-invariant loss
        if self.C(self.config.system.loss.lambda_depth) > 0 and 'depth' in out and 'depths' in batch:
            pred_depth = out['depth']  # Ensure 1D: [N]
            gt_depth = batch['depths']   # Ensure 1D: [N]

            mask = gt_depth != 1e10  # Ignore infinite depths

            if self.config.system.loss.scale_depth:
                with torch.no_grad():
                    pred_depth_valid = pred_depth[mask]
                    gt_depth_valid = gt_depth[mask]
                    A_matrix = torch.stack([pred_depth_valid, torch.ones_like(pred_depth_valid)], dim=-1)  # [N, 2]
                    scale_params = torch.linalg.lstsq(A_matrix, gt_depth_valid)[0]  # [2]
            else:
                scale_params = torch.tensor([1.0, 0.0], device=self.rank)  # No scaling
            
            pred_depth_scaled = pred_depth * scale_params[0] + scale_params[1]
            loss_depth = F.mse_loss(pred_depth_scaled * mask, gt_depth * mask)
            
            self.log('train/loss_depth', loss_depth)
            loss += loss_depth * self.C(self.config.system.loss.lambda_depth)
        '''
        '''
        # normal loss l1
        if self.C(self.config.system.loss.lambda_normal_l1) > 0 and 'comp_normal' in out and 'normals' in batch:
            mask = (batch['depths'] != 1e10).squeeze() if 'depths' in batch else torch.ones_like(out['comp_normal'][..., 0], dtype=torch.bool)
            loss_normal_l1 = torch.abs(out['comp_normal'][mask] - batch['normals'][mask]).sum(dim=-1).mean()
            self.log('train/loss_normal_l1', loss_normal_l1)
            loss += loss_normal_l1 * self.C(self.config.system.loss.lambda_normal_l1)

        # normal loss cos
        if self.C(self.config.system.loss.lambda_normal_cos) > 0 and 'comp_normal' in out and 'normals' in batch:
            mask = (batch['depths'] != 1e10).squeeze() if 'depths' in batch else torch.ones_like(out['comp_normal'][..., 0], dtype=torch.bool)
            loss_normal_cos = (1.0 - torch.sum(out['comp_normal'][mask] * batch['normals'][mask], dim = -1)).mean()
            self.log('train/loss_normal_cos', loss_normal_cos)
            loss += loss_normal_cos * self.C(self.config.system.loss.lambda_normal_cos)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        
        '''

        if self.config.dataset.apply_rgb_plus:
            loss_rendering_mse = F.mse_loss(out['comp_rendering_plus_mse_full'][out['rays_valid_full'][...,0]], batch['rendering_plus_mse'][out['rays_valid_full'][...,0]])
            self.log('train/loss_rgb_mse', loss_rendering_mse)
            loss += loss_rendering_mse * self.C(self.config.system.loss.lambda_rendering_mse)

            loss_rendering_l1 = F.l1_loss(out['comp_rendering_plus_l1_full'][out['rays_valid_full'][...,0]], batch['rendering_plus_l1'][out['rays_valid_full'][...,0]])
            self.log('train/loss_rgb', loss_rendering_l1)
            loss += loss_rendering_l1 * self.C(self.config.system.loss.lambda_rendering_l1)

        else:
            loss_rendering_mse = F.mse_loss(out['comp_rendering_full'][out['rays_valid_full'][...,0]], batch['rendering'][out['rays_valid_full'][...,0]])
            self.log('train/loss_rgb_mse', loss_rendering_mse)
            loss += loss_rendering_mse * self.C(self.config.system.loss.lambda_rendering_mse)

            loss_rendering_l1 = F.l1_loss(out['comp_rendering_full'][out['rays_valid_full'][...,0]], batch['rendering'][out['rays_valid_full'][...,0]])
            self.log('train/loss_rgb', loss_rendering_l1)
            loss += loss_rendering_l1 * self.C(self.config.system.loss.lambda_rendering_l1)

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def on_validation_epoch_start(self):
        self._validation_outputs = []

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh

        rgb = out['comp_rgb_full'].to("cpu")
        normals = out['comp_normal'].to("cpu")
        lights = batch['lights'].to("cpu")
        shading = torch.einsum('ij,ij->i', normals, lights)
        rendering = torch.einsum('i,ij->ij', shading, rgb)

        if 'depths' in batch:
            depths = batch['depths'].clone().view(H, W)
            depths[depths == depths.max()] = 0.0
        if self.config.system.get('save_images', True):
            self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [[
                {'type': 'rgb', 'img': batch['rendering'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': batch['normals'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
            ],[
                {'type': 'rgb', 'img': rendering.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rgb.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
            ]])

        step_out = {'psnr': psnr, 'index': batch['index']}
        self._validation_outputs.append(step_out)
        return step_out
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def on_validation_epoch_end(self):
        out = self.all_gather(self._validation_outputs)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)

    def on_test_epoch_start(self):
        self._test_outputs = []

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        if self.config.system.render_all_lights and self.config.system.get('save_images', True):
            self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}_{batch['index_light'][0].item()}.png", [[
                {'type': 'rgb', 'img': batch['rendering'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': batch['normals'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
            ],[
                {'type': 'rgb', 'img': out['comp_rendering'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
            ]])

        step_out = {'psnr': psnr, 'index': batch['index']}
        self._test_outputs.append(step_out)
        return step_out

    def on_test_epoch_end(self):
        out = self.all_gather(self._test_outputs)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)

            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.ply",
            **mesh
        )        
