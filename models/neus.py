import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect
import kornia.geometry.conversions as kornia_geo # Import kornia for matrix-to-quaternion conversion

from lietorch import SE3,SO3 # Use SE3 for rigid transformations

class CameraPoseOptimizer(nn.Module):
    def __init__(self, num_images, initial_c2w_matrices):
        super().__init__()
        self.num_images = num_images

        self.delta_se3_logmap = nn.Parameter(torch.zeros(num_images, 6))

        # --- Pre-compute initial SE3 data and store it as a buffer ---
        initial_R = initial_c2w_matrices[:, :3, :3]
        #initial_R = initial_c2w_matrices[:, :3, :3].transpose(-1, -2)
        initial_T = initial_c2w_matrices[:, :3, 3]

        # 1. Convertir la matrice de rotation 3x3 en quaternion (w, x, y, z) avec Kornia
        kornia_quaternions = kornia_geo.rotation_matrix_to_quaternion(initial_R) # Forme : (num_images, 4) au format (w, x, y, z)

        # AJOUT DU BRUIT ICI
        quaternion_noise_std = -1#0.001
        if quaternion_noise_std > 0:

            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            # Générer du bruit gaussien de la même forme que les quaternions
            noise = torch.randn_like(kornia_quaternions) * quaternion_noise_std
            torch.set_rng_state(rng_state)

            noisy_kornia_quaternions = kornia_quaternions + noise
            # Très important : re-normaliser le quaternion après l'ajout de bruit
            # pour qu'il reste un quaternion unitaire valide pour la rotation.
            noisy_kornia_quaternions = F.normalize(noisy_kornia_quaternions, p=2, dim=-1)
            quaternions_to_use = noisy_kornia_quaternions
        else:
            quaternions_to_use = kornia_quaternions
        # FIN DE L'AJOUT DU BRUIT

        # 2. Réordonner le quaternion de (w, x, y, z) à (x, y, z, w) pour Lietorch
        initial_quaternions_lietorch_format = torch.cat([
            quaternions_to_use[:, 1:4], # composantes x, y, z
            quaternions_to_use[:, 0:1]  # composante w
        ], dim=-1) # Forme : (num_images, 4) au format (x, y, z, w)


        # 3. Concatenate translation and reordered quaternion to form the 7-element data vector for SE3
        # SE3 expects data in the format: [tx, ty, tz, qx, qy, qz, qw]
        initial_se3_data = torch.cat([initial_T, initial_quaternions_lietorch_format], dim=-1) # Shape: (num_images, 7)

        # Store the raw tensor data in the buffer
        self.register_buffer('initial_se3_data_buffer', initial_se3_data)


    def forward(self, image_idx):
        # Retrieve the raw tensor data from the buffer
        initial_se3_data_batch = self.initial_se3_data_buffer[image_idx]

        # Reconstruct the lietorch SE3 object from the data
        initial_se3_batch = SE3(initial_se3_data_batch) # Reconstruct SE3 object here

        delta_se3_logmap_batch = self.delta_se3_logmap[image_idx]

        delta_transform = SE3.exp(delta_se3_logmap_batch)

        corrected_se3 = initial_se3_batch * delta_transform

        corrected_c2w_matrices = corrected_se3.matrix()

        return corrected_c2w_matrices

class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


@models.register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.AABB

        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01    

        if self.config.no_albedo : 
            print("\nFreezing self.model.texture parameters by setting requires_grad=False:")
            for name, param in self.texture.named_parameters():
                param.requires_grad = False
                print(f"  - Parameter '{name}' from texture: requires_grad={param.requires_grad}")
            print("--------------------------------------------------")
        

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=256,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
    
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[...,None] * self.render_step_size_bg
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
            if self.config.learned_background:
                self.occupancy_grid_bg.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn_bg, occ_thre=self.config.get('grid_prune_occ_thre_bg', 0.01))

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    
    def forward_(self, rays, lights, c2w, mask):
        

        n_rays = rays.shape[0]

        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )

        #print("\n\n")
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        if self.config.geometry.grad_type == 'finite_difference':
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
        else:
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
        normal = F.normalize(sdf_grad, p=2, dim=-1)

        if ray_indices.shape[0]> 0:
            if torch.isnan(normal).any() : 
                print("\n\nNan in normal\n\n")
        #normal = sdf_grad


        alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]

        #rgb = self.texture(feature, t_dirs, normal)
        
        
        if self.config.no_albedo : 
            rgb = torch.ones_like(normal)
        else:
            rgb = self.texture(feature, normal)

        if ray_indices.shape[0]> 0:
            if torch.isnan(rgb).any() : 
                print("\n\nNan in rgb\n\n")

        torch.clamp(rgb,0.0,1.0)
        #print(rgb)
        if ray_indices.shape[0]> 0:
            if torch.isnan(rgb).any() : 
                print("\n\nNan in rgb\n\n")

        if self.config.no_albedo : 
                rgb = torch.ones_like(rgb)

        plus_mse = torch.sqrt(3.0 - (rgb[...,0]**2 + rgb[...,1]**2 + rgb[...,2]**2)).unsqueeze(-1)
        plus_l1 = 3.0 - (rgb[...,0] + rgb[...,1] + rgb[...,2]).unsqueeze(-1)

        rgb_plus_mse = torch.cat([rgb, plus_mse], dim=-1)
        rgb_plus_l1 = torch.cat([rgb, plus_l1], dim=-1)

        if ray_indices.shape[0]> 0:
            if torch.isnan(rgb).any() : 
                print("\n\nNan in rgb\n\n")

        #print(normal.shape)
        #print(c2w.shape)
        #c2w_all = c2w[ray_indices]
        #c2w_all = c2w_all[:,:,:3].permute(0, 2, 1)
        

        # Unsqueeze normals_world to (M, 3, 1) for batch matrix multiplication
        #normals_world_expanded = normal.unsqueeze(-1) # Shape: (M, 3, 1)

        # Perform batch matrix multiplication
        #normal = (c2w_all @ normals_world_expanded).squeeze(-1)        

        lights_rays = lights[ray_indices]
        shading = torch.einsum('ij,ij->i', normal, lights_rays)
        rendering = torch.einsum('i,ij->ij', shading, rgb)
        rendering_plus_mse = torch.einsum('i,ij->ij', shading, rgb_plus_mse)
        rendering_plus_l1 = torch.einsum('i,ij->ij', shading, rgb_plus_l1)



        if ray_indices.shape[0]> 0:
            if torch.isnan(normal).any() : 
                print("\n\nNan in normal\n\n")

        if ray_indices.shape[0]> 0:
            if torch.isnan(rgb).any() : 
                print("Nan in rgb\n\n")



        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        torch.clamp(comp_rgb,0.0,1.0)
        if self.config.no_albedo : 
                comp_rgb = torch.ones_like(comp_rgb)
        comp_rendering = accumulate_along_rays(weights, ray_indices, values=rendering, n_rays=n_rays)
        comp_rendering_plus_mse = accumulate_along_rays(weights, ray_indices, values=rendering_plus_mse, n_rays=n_rays)
        comp_rendering_plus_l1 = accumulate_along_rays(weights, ray_indices, values=rendering_plus_l1, n_rays=n_rays)

        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        #comp_normal = F.normalize(comp_normal, p=2, dim=-1)
        #comp_normal_mask = comp_normal[mask]
        #comp_normal_mask = F.normalize(comp_normal_mask, p=2, dim=-1)
        #comp_normal[mask] = comp_normal_mask

        #comp_shading = torch.einsum('ij,ij->i', comp_normal, lights)
        #comp_rendering = torch.einsum('i,ij->ij', comp_shading, comp_rgb)

        #comp_plus_mse = torch.sqrt(3.0 - (comp_rgb[...,0]**2 + comp_rgb[...,1]**2 + comp_rgb[...,2]**2)).unsqueeze(-1)
        #comp_plus_l1 = 3.0 - (comp_rgb[...,0] + comp_rgb[...,1] + comp_rgb[...,2]).unsqueeze(-1)

        #comp_rgb_plus_mse = torch.cat([comp_rgb, comp_plus_mse], dim=-1)
        #comp_rgb_plus_l1 = torch.cat([comp_rgb, comp_plus_l1], dim=-1)

        #comp_rendering_plus_mse = torch.einsum('i,ij->ij', comp_shading, comp_rgb_plus_mse)
        #comp_rendering_plus_l1 = torch.einsum('i,ij->ij', comp_shading, comp_rgb_plus_l1)



        out = {
            'comp_rgb': comp_rgb,
            'comp_rendering': comp_rendering,
            'comp_rendering_plus_mse': comp_rendering_plus_mse,
            'comp_rendering_plus_l1': comp_rendering_plus_l1,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1)                
            })
            if self.config.geometry.grad_type == 'finite_difference':
                out.update({
                    'sdf_laplace_samples': sdf_laplace
                })


        out_bg = {
            'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
            'comp_rendering': self.background_color[None,:].expand(*comp_rendering.shape),
            'comp_rendering_plus_mse': self.background_color_plus[None,:].expand(*comp_rendering_plus_mse.shape),
            'comp_rendering_plus_l1': self.background_color_plus[None,:].expand(*comp_rendering_plus_l1.shape),
            'num_samples': torch.zeros_like(out['num_samples']),
            'rays_valid': torch.zeros_like(out['rays_valid'])
        }

        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'comp_rendering': out['comp_rendering'] + out_bg['comp_rendering'] * (1.0 - out['opacity']),
            'comp_rendering_plus_mse': out['comp_rendering_plus_mse'] + out_bg['comp_rendering_plus_mse'] * (1.0 - out['opacity']),
            'comp_rendering_plus_l1': out['comp_rendering_plus_l1'] + out_bg['comp_rendering_plus_l1'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }

    def forward(self, rays, lights, c2w, mask):
        if self.training:
            out = self.forward_(rays, lights, c2w, mask)
        else:

            out = chunk_batch(self.forward_, self.config.ray_chunk, True, True, rays, lights, c2w, mask)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses
    


    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, True, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            #rgb = self.texture(feature, -normal, normal) # set the viewing directions to the normal to get "albedo"
            rgb = self.texture(feature, normal)
            mesh['v_rgb'] = rgb.cpu()
        return mesh
    
    
