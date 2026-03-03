"""
Albedo Scaling — multi-view consistency-based albedo ratio computation.

Adapted from RNb_NeuS2/scripts/utils/albedo_scaling_lib.py.
This version works directly with tensors/arrays (no disk I/O during training).
"""

import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator


def compute_albedo_scale_ratios(albedo_images, camera_Ks, camera_c2ws,
                                mesh_path=None, tri_mesh=None, n_samples=2000):
    """
    Compute per-view albedo scaling ratios via multi-view consistency.

    For each view:
      1. Sample n_samples masked pixels
      2. Cast rays from camera origin through pixels -> find 3D intersections on mesh
      3. For each neighbor view:
         a. Check visibility (recast ray from neighbor camera to 3D point)
         b. Project 3D points to neighbor image plane
         c. Bilinear interpolate neighbor albedo at projected locations
         d. Compute ratio: albedo_i / albedo_j (per-channel)
      4. Take median ratio per neighbor pair
    Propagate ratios sequentially, normalize by mean.

    Args:
        albedo_images: list of np.array (H, W, 3) — albedo images in [0, 1]
        camera_Ks: list of (3,3) intrinsic matrices
        camera_c2ws: list of (4,4) or (3,4) cam-to-world matrices
        mesh_path: path to OBJ mesh for ray tracing (used if tri_mesh not given)
        tri_mesh: pre-built trimesh.Trimesh object (avoids OBJ reload)
        n_samples: number of pixel samples per view

    Returns:
        scale_ratios: np.array (N, 3) — per-view RGB scale factors
    """
    if tri_mesh is not None:
        mesh = tri_mesh
    elif mesh_path is not None:
        mesh = trimesh.load_mesh(mesh_path)
    else:
        raise ValueError("Either tri_mesh or mesh_path must be provided")
    n_views = len(albedo_images)

    # Build masks from albedo images (non-zero pixels)
    masks = []
    for img in albedo_images:
        mask = np.any(img > 0, axis=-1)
        masks.append(mask)

    # Extract camera centers and R_c2w from c2w matrices
    centers = []
    R_c2ws = []
    for c2w in camera_c2ws:
        c2w = np.array(c2w, dtype=np.float64)
        R_c2ws.append(c2w[:3, :3])
        centers.append(c2w[:3, 3:4])  # (3, 1)

    h, w = albedo_images[0].shape[:2]

    # Storage for ratios between neighboring views
    ratios = np.zeros((n_views, n_samples, 3, 2), dtype=np.float32)
    intersection_found = np.zeros((n_views, n_samples, 2), dtype=np.bool_)

    for cam_id in range(n_views):
        mask = masks[cam_id]
        ind_mask = np.where(mask)
        pixels = np.stack([ind_mask[1], ind_mask[0]], axis=1)  # (N, 2) [x, y]
        albedo_vals = albedo_images[cam_id][ind_mask[0], ind_mask[1], :]

        current_K = np.array(camera_Ks[cam_id], dtype=np.float64)
        current_R_c2w = R_c2ws[cam_id]
        current_center = centers[cam_id]

        n_valid = min(n_samples, pixels.shape[0])
        if n_valid == 0:
            continue

        idx = np.random.choice(pixels.shape[0], n_valid, replace=False)
        pixels = pixels[idx]
        albedo_vals = albedo_vals[idx]

        # Create rays: origin at camera center, direction through pixel
        rays_origin = np.tile(current_center.T, (n_valid, 1))
        K_inv = np.linalg.inv(current_K)
        pixel_homo = np.concatenate([pixels, np.ones((n_valid, 1))], axis=1).T  # (3, N)
        point_on_rays = (current_R_c2w @ (K_inv @ pixel_homo) + current_center).T  # (N, 3)
        rays_direction = point_on_rays - rays_origin
        rays_direction /= np.linalg.norm(rays_direction, axis=1, keepdims=True)

        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=rays_origin,
            ray_directions=rays_direction,
            multiple_hits=False
        )

        if len(index_ray) == 0:
            continue

        hit_pixels = pixels[index_ray]
        hit_albedo_vals = albedo_vals[index_ray]

        # Check both neighbors (circular)
        right_cam_id = (cam_id + 1) % n_views
        left_cam_id = (cam_id - 1) % n_views

        for kk, neigh_cam_id in enumerate([right_cam_id, left_cam_id]):
            neighbor_K = np.array(camera_Ks[neigh_cam_id], dtype=np.float64)
            neighbor_R_c2w = R_c2ws[neigh_cam_id]
            neighbor_center = centers[neigh_cam_id]

            # Check visibility from neighbor: ray from intersection to neighbor camera
            neigh_dir = neighbor_center.T - locations
            neigh_dir /= np.linalg.norm(neigh_dir, axis=1, keepdims=True)
            neigh_origin = locations + 1e-3 * neigh_dir

            hit = mesh.ray.intersects_any(
                ray_origins=neigh_origin,
                ray_directions=neigh_dir
            )

            # Keep only visible points
            visible = ~hit
            vis_locations = locations[visible]
            vis_index_ray = index_ray[visible]
            vis_albedo = hit_albedo_vals[visible]

            if len(vis_index_ray) == 0:
                continue

            # Project to neighbor camera
            neighbor_R_w2c = neighbor_R_c2w.T
            pts_in_cam = (neighbor_R_w2c @ (vis_locations.T - neighbor_center))  # (3, N)
            pts_projected = (neighbor_K @ pts_in_cam).T  # (N, 3)
            pts_projected /= pts_projected[:, 2:3]
            pts_2d = pts_projected[:, :2]  # (N, 2) [x, y]

            # Filter points inside image
            valid = (
                (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w - 1) &
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h - 1)
            )

            pts_2d = pts_2d[valid]
            vis_index_ray = vis_index_ray[valid]
            vis_albedo = vis_albedo[valid]

            if len(vis_index_ray) == 0:
                continue

            # Bilinear interpolate neighbor albedo
            neigh_img = albedo_images[neigh_cam_id].astype(np.float32)
            rows_inds = np.arange(h)
            cols_inds = np.arange(w)
            interp_r = RegularGridInterpolator((rows_inds, cols_inds), neigh_img[:, :, 0])
            interp_g = RegularGridInterpolator((rows_inds, cols_inds), neigh_img[:, :, 1])
            interp_b = RegularGridInterpolator((rows_inds, cols_inds), neigh_img[:, :, 2])

            pts_yx = np.stack([pts_2d[:, 1], pts_2d[:, 0]], axis=1)
            neigh_albedo = np.stack([interp_r(pts_yx), interp_g(pts_yx), interp_b(pts_yx)], axis=1)

            # Filter out zero values
            nonzero = ~np.any(neigh_albedo == 0, axis=1)
            vis_index_ray = vis_index_ray[nonzero]
            vis_albedo = vis_albedo[nonzero]
            neigh_albedo = neigh_albedo[nonzero]

            if len(vis_index_ray) == 0:
                continue

            # Compute ratios
            ratios[cam_id, vis_index_ray, :, kk] = vis_albedo / neigh_albedo
            intersection_found[cam_id, vis_index_ray, kk] = True

    # Compute median ratios per view
    median_ratios = np.ones((n_views, 3))
    right_ratios = ratios[:, :, :, 0]
    right_ind = intersection_found[:, :, 0]
    left_ratios = np.roll(ratios[:, :, :, 1], -1, axis=0)
    left_ind = np.roll(intersection_found[:, :, 1], -1, axis=0)

    for cam_id in range(n_views):
        right_r = right_ratios[cam_id, right_ind[cam_id]]
        left_r = 1.0 / left_ratios[cam_id, left_ind[cam_id]]
        all_r = np.concatenate([right_r, left_r], axis=0)
        if len(all_r) > 0:
            median_ratios[cam_id] = np.median(all_r, axis=0)

    # Propagate ratios sequentially
    median_ratio_prop = np.ones((n_views, 3))
    for i in range(n_views - 1):
        median_ratio_prop[i + 1] = median_ratio_prop[i] * median_ratios[i]

    # Normalize by mean
    mean_prop = np.mean(median_ratio_prop, axis=0)
    mean_prop[mean_prop < 1e-8] = 1.0
    scale_ratios = median_ratio_prop / mean_prop

    return scale_ratios


def scale_albedo_images(albedo_images, scale_ratios):
    """
    Apply scale ratios to albedo images tensor.

    Args:
        albedo_images: tensor (N, H, W, 3)
        scale_ratios: np.array (N, 3)

    Returns:
        scaled: tensor (N, H, W, 3)
    """
    import torch
    ratios = torch.tensor(scale_ratios, dtype=albedo_images.dtype, device=albedo_images.device)
    return albedo_images * ratios[:, None, None, :]  # broadcast (N,1,1,3)
