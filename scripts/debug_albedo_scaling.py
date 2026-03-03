#!/usr/bin/env python3
"""
Debug script for albedo scaling pipeline.
Saves all intermediate data as .npy + diagnostic images.

Usage:
    python scripts/debug_albedo_scaling.py --mode idr
    python scripts/debug_albedo_scaling.py --mode sfm
    python scripts/debug_albedo_scaling.py --mode both
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_experiment(mode, cli_args=None):
    """Load mesh, cameras, and albedos. Returns dict with all raw data."""
    import trimesh
    from utils.misc import load_config

    if mode == 'idr':
        config_path = str(ROOT / 'configs' / 'idr.yaml')
        if cli_args is None:
            cli_args = [
                'dataset.scene=golden_snail',
                'dataset.root_dir=/media/bbrument/T9/skoltech3d_data/golden_snail/sdmunips',
            ]
        exp_dirs = sorted((ROOT / 'exp' / 'idr-golden_snail').glob('@*'))
    else:
        config_path = str(ROOT / 'configs' / 'sfm.yaml')
        if cli_args is None:
            cli_args = [
                'dataset.scene=golden_snail',
                'dataset.normal_sfm=data/golden_snail/normalSfm.json',
                'dataset.albedo_sfm=data/golden_snail/albedoSfm.json',
                'dataset.mask_sfm=data/golden_snail/maskSfm.json',
            ]
        exp_dirs = sorted((ROOT / 'exp' / 'sfm-golden_snail').glob('@*'))

    if not exp_dirs:
        print(f"[{mode}] No experiment directory found, skipping.")
        return None

    exp_dir = exp_dirs[-1]
    mesh_path = exp_dir / 'save' / 'intermediate_mesh.ply'
    if not mesh_path.exists():
        print(f"[{mode}] No intermediate mesh at {mesh_path}, skipping.")
        return None

    print(f"[{mode}] Loading mesh: {mesh_path}")
    tri_mesh = trimesh.load(str(mesh_path))

    # Load dataset
    config = load_config(config_path, cli_args=cli_args)
    if mode == 'idr':
        from datasets.idr import IDRIterableDataset
        ds = IDRIterableDataset(config.dataset, 'train')
    else:
        from datasets.sfm import SfMIterableDataset
        ds = SfMIterableDataset(config.dataset, 'train')

    # Collect raw data
    albedo_images = [img.cpu().numpy() for img in ds.all_images]
    camera_Ks = [np.array(K, dtype=np.float64) for K in ds.camera_Ks]
    camera_c2ws = [np.array(c, dtype=np.float64) for c in ds.camera_c2ws]
    all_c2w = ds.all_c2w.cpu().numpy()  # (N, 3, 4) — normalized space
    scale_factor = float(ds.scale_factor)
    scene_center = np.array(ds.scene_center, dtype=np.float64)

    n_views = len(albedo_images)
    h, w = albedo_images[0].shape[:2]

    # Mesh vertices are in WORLD space (already inverse-scaled at export)
    mesh_verts_world = tri_mesh.vertices.copy()
    mesh_faces = tri_mesh.faces.copy()

    # Convert mesh back to NORMALIZED space for comparison
    mesh_verts_norm = (mesh_verts_world - scene_center) * scale_factor

    # Camera centers in both spaces
    cam_centers_world = np.array([c[:3, 3] for c in camera_c2ws])
    cam_centers_norm = all_c2w[:, :3, 3]  # from training poses

    # IDR-specific: save scale_mat
    scale_mat = None
    if mode == 'idr':
        import cv2
        cams = np.load(os.path.join(config.dataset.root_dir, 'cameras.npz'))
        scale_mat = cams['scale_mat_0']

    out_dir = ROOT / 'debug_albedo_scaling' / mode
    os.makedirs(out_dir, exist_ok=True)

    data = {
        'mode': mode,
        'tri_mesh': tri_mesh,
        'mesh_verts_world': mesh_verts_world,
        'mesh_verts_norm': mesh_verts_norm,
        'mesh_faces': mesh_faces,
        'albedo_images': albedo_images,
        'camera_Ks': camera_Ks,
        'camera_c2ws': camera_c2ws,
        'all_c2w': all_c2w,
        'cam_centers_world': cam_centers_world,
        'cam_centers_norm': cam_centers_norm,
        'scale_factor': scale_factor,
        'scene_center': scene_center,
        'scale_mat': scale_mat,
        'h': h, 'w': w,
        'n_views': n_views,
        'out_dir': out_dir,
    }
    return data


def save_npy_data(data):
    """Save all raw data as .npy files."""
    out = data['out_dir'] / 'npy'
    os.makedirs(out, exist_ok=True)

    np.save(out / 'mesh_verts_world.npy', data['mesh_verts_world'])
    np.save(out / 'mesh_verts_norm.npy', data['mesh_verts_norm'])
    np.save(out / 'mesh_faces.npy', data['mesh_faces'])
    np.save(out / 'cam_centers_world.npy', data['cam_centers_world'])
    np.save(out / 'cam_centers_norm.npy', data['cam_centers_norm'])
    np.save(out / 'camera_Ks.npy', np.array(data['camera_Ks']))
    np.save(out / 'camera_c2ws.npy', np.array(data['camera_c2ws']))
    np.save(out / 'all_c2w.npy', data['all_c2w'])
    np.save(out / 'scale_factor.npy', np.array(data['scale_factor']))
    np.save(out / 'scene_center.npy', data['scene_center'])
    np.save(out / 'albedo_images.npy', np.array(data['albedo_images']))

    if data['scale_mat'] is not None:
        np.save(out / 'scale_mat.npy', data['scale_mat'])

    # Summary JSON
    summary = {
        'mode': data['mode'],
        'n_views': data['n_views'],
        'image_size': [data['w'], data['h']],
        'scale_factor': data['scale_factor'],
        'scene_center': data['scene_center'].tolist(),
        'mesh_world': {
            'n_verts': data['mesh_verts_world'].shape[0],
            'n_faces': data['mesh_faces'].shape[0],
            'bbox_min': data['mesh_verts_world'].min(axis=0).tolist(),
            'bbox_max': data['mesh_verts_world'].max(axis=0).tolist(),
            'center': ((data['mesh_verts_world'].max(axis=0) + data['mesh_verts_world'].min(axis=0)) / 2).tolist(),
        },
        'mesh_norm': {
            'bbox_min': data['mesh_verts_norm'].min(axis=0).tolist(),
            'bbox_max': data['mesh_verts_norm'].max(axis=0).tolist(),
            'center': ((data['mesh_verts_norm'].max(axis=0) + data['mesh_verts_norm'].min(axis=0)) / 2).tolist(),
        },
        'cam_world': {
            'center_min': data['cam_centers_world'].min(axis=0).tolist(),
            'center_max': data['cam_centers_world'].max(axis=0).tolist(),
            'center_mean': data['cam_centers_world'].mean(axis=0).tolist(),
        },
        'cam_norm': {
            'center_min': data['cam_centers_norm'].min(axis=0).tolist(),
            'center_max': data['cam_centers_norm'].max(axis=0).tolist(),
            'center_mean': data['cam_centers_norm'].mean(axis=0).tolist(),
        },
    }
    if data['scale_mat'] is not None:
        summary['scale_mat'] = data['scale_mat'].tolist()

    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved .npy data to {out}/")
    return summary


def print_summary(summary):
    """Print readable summary."""
    mode = summary['mode'].upper()
    print(f"\n  === {mode} COORDINATE SYSTEMS ===")
    print(f"  scale_factor = {summary['scale_factor']}")
    print(f"  scene_center = {summary['scene_center']}")
    print(f"")
    print(f"  MESH (world):  bbox = {summary['mesh_world']['bbox_min']} → {summary['mesh_world']['bbox_max']}")
    print(f"                 center = {summary['mesh_world']['center']}")
    print(f"  MESH (norm):   bbox = {summary['mesh_norm']['bbox_min']} → {summary['mesh_norm']['bbox_max']}")
    print(f"                 center = {summary['mesh_norm']['center']}")
    print(f"")
    print(f"  CAMERAS (world, camera_c2ws): mean = {summary['cam_world']['center_mean']}")
    print(f"  CAMERAS (norm, all_c2w):      mean = {summary['cam_norm']['center_mean']}")

    # Distance comparison
    mesh_center_world = np.array(summary['mesh_world']['center'])
    mesh_center_norm = np.array(summary['mesh_norm']['center'])
    cam_mean_world = np.array(summary['cam_world']['center_mean'])
    cam_mean_norm = np.array(summary['cam_norm']['center_mean'])

    d_world = np.linalg.norm(cam_mean_world - mesh_center_world)
    d_norm = np.linalg.norm(cam_mean_norm - mesh_center_norm)
    print(f"")
    print(f"  dist(cam_world, mesh_world) = {d_world:.3f}")
    print(f"  dist(cam_norm,  mesh_norm)  = {d_norm:.3f}")


def make_overview_plot(data, space='both'):
    """3D + 2D overview of cameras vs mesh in both coordinate systems."""
    out_dir = data['out_dir']
    mode = data['mode'].upper()

    # Subsample mesh
    n_sub = min(8000, data['mesh_verts_world'].shape[0])
    idx = np.random.RandomState(42).choice(data['mesh_verts_world'].shape[0], n_sub, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    for row, (space_label, mesh_v, cam_c) in enumerate([
        ('WORLD', data['mesh_verts_world'], data['cam_centers_world']),
        ('NORMALIZED', data['mesh_verts_norm'], data['cam_centers_norm']),
    ]):
        mv = mesh_v[idx]
        for col, (d1, d2, lbl) in enumerate([(0, 2, 'XZ (top)'), (0, 1, 'XY (front)'), (1, 2, 'YZ (side)')]):
            ax = axes[row, col]
            ax.scatter(mv[:, d1], mv[:, d2], s=0.05, alpha=0.15, c='gray', rasterized=True)
            ax.scatter(cam_c[:, d1], cam_c[:, d2], s=40, c='red', marker='^', zorder=5)
            for i, cc in enumerate(cam_c):
                ax.text(cc[d1], cc[d2], str(i), fontsize=5, color='red')
            mc = (mesh_v.max(axis=0) + mesh_v.min(axis=0)) / 2
            ax.scatter(mc[d1], mc[d2], s=80, c='blue', marker='x', zorder=5, label='mesh center')
            ax.scatter(0, 0, s=80, c='green', marker='o', zorder=5, label='origin')
            ax.set_title(f'{space_label} — {lbl}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            if col == 0 and row == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f'{mode} — Cameras vs Mesh: World (top) vs Normalized (bottom)', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / 'overview_both_spaces.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir / 'overview_both_spaces.png'}")


def test_ray_intersections(data, n_samples=2000):
    """
    Test ray-mesh intersection in BOTH world and normalized space.
    Save per-view stats + debug images.
    """
    import trimesh
    from scipy.interpolate import RegularGridInterpolator

    out_dir = data['out_dir']
    mode = data['mode']
    n_views = data['n_views']
    h, w = data['h'], data['w']

    # Build meshes for both spaces
    mesh_world = data['tri_mesh']  # already in world
    mesh_norm = trimesh.Trimesh(vertices=data['mesh_verts_norm'], faces=data['mesh_faces'])

    # Build masks
    masks = [np.any(img > 0, axis=-1) for img in data['albedo_images']]

    # ==================================================
    # Test both spaces
    # ==================================================
    results = {}
    norm_c2ws, norm_Ks = _build_norm_c2ws_and_Ks(data)
    spaces = [
        ('world', mesh_world, data['camera_c2ws'], data['camera_Ks']),
        ('norm', mesh_norm, norm_c2ws, norm_Ks),
    ]

    for space_label, mesh, c2ws, Ks in spaces:

        centers = []
        R_c2ws = []
        for c2w in c2ws:
            c2w = np.array(c2w, dtype=np.float64)
            if c2w.shape == (3, 4):
                c2w_full = np.eye(4, dtype=np.float64)
                c2w_full[:3, :4] = c2w
                c2w = c2w_full
            R_c2ws.append(c2w[:3, :3])
            centers.append(c2w[:3, 3:4])

        stats = []
        # Storage for ratio computation (same as albedo_scaling.py)
        ratios_arr = np.zeros((n_views, n_samples, 3, 2), dtype=np.float32)
        intersection_found_arr = np.zeros((n_views, n_samples, 2), dtype=np.bool_)

        view_dir = out_dir / f'per_view_{space_label}'
        os.makedirs(view_dir, exist_ok=True)

        for cam_id in range(n_views):
            mask = masks[cam_id]
            ind_mask = np.where(mask)
            pixels = np.stack([ind_mask[1], ind_mask[0]], axis=1)
            albedo_vals = data['albedo_images'][cam_id][ind_mask[0], ind_mask[1], :]

            n_masked = pixels.shape[0]
            current_K = np.array(Ks[cam_id], dtype=np.float64)
            current_R = R_c2ws[cam_id]
            current_center = centers[cam_id]

            n_valid = min(n_samples, n_masked)
            if n_valid == 0:
                stats.append({'cam_id': cam_id, 'masked_px': 0, 'sampled': 0,
                              'hits': 0, 'hit_rate': 0})
                continue

            np.random.seed(42 + cam_id)
            idx = np.random.choice(n_masked, n_valid, replace=False)
            sampled_pixels = pixels[idx]
            sampled_albedo = albedo_vals[idx]

            # Ray construction
            rays_origin = np.tile(current_center.T, (n_valid, 1))
            K_inv = np.linalg.inv(current_K)
            pixel_homo = np.concatenate([sampled_pixels, np.ones((n_valid, 1))], axis=1).T
            point_on_rays = (current_R @ (K_inv @ pixel_homo) + current_center).T
            rays_direction = point_on_rays - rays_origin
            rays_direction /= np.linalg.norm(rays_direction, axis=1, keepdims=True)

            locations, index_ray, _ = mesh.ray.intersects_location(
                ray_origins=rays_origin,
                ray_directions=rays_direction,
                multiple_hits=False
            )

            n_hits = len(index_ray)
            stats.append({
                'cam_id': cam_id,
                'masked_px': n_masked,
                'sampled': n_valid,
                'hits': n_hits,
                'hit_rate': n_hits / n_valid if n_valid > 0 else 0,
            })

            # ---- Per-view debug image ----
            fig, axes_fig = plt.subplots(1, 3, figsize=(20, 6))

            axes_fig[0].imshow(data['albedo_images'][cam_id])
            miss_mask = np.ones(n_valid, dtype=bool)
            if n_hits > 0:
                miss_mask[index_ray] = False
            axes_fig[0].scatter(sampled_pixels[miss_mask, 0], sampled_pixels[miss_mask, 1],
                                s=0.5, c='red', alpha=0.3)
            if n_hits > 0:
                axes_fig[0].scatter(sampled_pixels[index_ray, 0], sampled_pixels[index_ray, 1],
                                    s=0.5, c='lime', alpha=0.5)
            axes_fig[0].set_title(f'View {cam_id}: {n_hits}/{n_valid} hits ({100*n_hits/max(n_valid,1):.0f}%)')

            # Neighbor projections
            right_id = (cam_id + 1) % n_views
            left_id = (cam_id - 1) % n_views

            for panel, (kk, neigh_id, nlbl) in enumerate([(0, right_id, 'right'), (1, left_id, 'left')]):
                ax = axes_fig[panel + 1]
                ax.imshow(data['albedo_images'][neigh_id])

                if n_hits == 0:
                    ax.set_title(f'Neigh {nlbl} (v{neigh_id}): no hits from source')
                    continue

                hit_albedo = sampled_albedo[index_ray]
                neigh_K = np.array(Ks[neigh_id], dtype=np.float64)
                neigh_R = R_c2ws[neigh_id]
                neigh_center = centers[neigh_id]

                # Visibility
                neigh_dir = neigh_center.T - locations
                neigh_dir /= np.linalg.norm(neigh_dir, axis=1, keepdims=True)
                neigh_origin = locations + 1e-3 * neigh_dir
                hit = mesh.ray.intersects_any(ray_origins=neigh_origin, ray_directions=neigh_dir)
                visible = ~hit
                n_vis = visible.sum()

                vis_loc = locations[visible]
                vis_idx = index_ray[visible]
                vis_alb = hit_albedo[visible]

                if n_vis == 0:
                    ax.set_title(f'Neigh {nlbl} (v{neigh_id}): 0/{n_hits} visible')
                    continue

                # Project
                neigh_Rw2c = neigh_R.T
                pts_cam = neigh_Rw2c @ (vis_loc.T - neigh_center)
                pts_proj = (neigh_K @ pts_cam).T
                pts_proj /= pts_proj[:, 2:3]
                pts_2d = pts_proj[:, :2]

                valid = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w - 1) & \
                        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h - 1)
                n_inb = valid.sum()

                # Out of bounds
                oob = pts_2d[~valid]
                if len(oob) > 0:
                    ax.scatter(np.clip(oob[:, 0], -50, w+50), np.clip(oob[:, 1], -50, h+50),
                               s=2, c='yellow', alpha=0.4, label=f'OOB ({len(oob)})')

                pts_2d_v = pts_2d[valid]
                vis_idx_v = vis_idx[valid]
                vis_alb_v = vis_alb[valid]

                if n_inb > 0:
                    ax.scatter(pts_2d_v[:, 0], pts_2d_v[:, 1], s=2, c='cyan', alpha=0.5,
                               label=f'in-bounds ({n_inb})')

                    # Compute ratios
                    neigh_img = data['albedo_images'][neigh_id].astype(np.float32)
                    interps = [RegularGridInterpolator((np.arange(h), np.arange(w)), neigh_img[:,:,ch])
                               for ch in range(3)]
                    pts_yx = np.stack([pts_2d_v[:, 1], pts_2d_v[:, 0]], axis=1)
                    neigh_alb = np.stack([interp(pts_yx) for interp in interps], axis=1)

                    nonzero = ~np.any(neigh_alb == 0, axis=1)
                    n_nz = nonzero.sum()

                    if n_nz > 0:
                        r = vis_alb_v[nonzero] / neigh_alb[nonzero]
                        ratios_arr[cam_id, vis_idx_v[nonzero], :, kk] = r
                        intersection_found_arr[cam_id, vis_idx_v[nonzero], kk] = True
                        med = np.median(r, axis=0)
                        info = f'med=[{med[0]:.3f},{med[1]:.3f},{med[2]:.3f}]'
                    else:
                        info = 'all neigh=0'
                        n_nz = 0
                else:
                    info = 'no in-bounds'
                    n_nz = 0

                ax.set_title(f'Neigh {nlbl} (v{neigh_id}): {n_vis} vis, {n_inb} inb, {n_nz} used\n{info}')
                ax.legend(fontsize=7, loc='upper right')

            fig.suptitle(f'{mode.upper()} [{space_label}] — View {cam_id}', fontsize=13)
            fig.tight_layout()
            fig.savefig(view_dir / f'view_{cam_id:02d}.png', dpi=100, bbox_inches='tight')
            plt.close(fig)

        # Save stats
        results[space_label] = stats

        # ---- Compute final ratios ----
        median_ratios = np.ones((n_views, 3))
        right_ratios = ratios_arr[:, :, :, 0]
        right_ind = intersection_found_arr[:, :, 0]
        left_ratios = np.roll(ratios_arr[:, :, :, 1], -1, axis=0)
        left_ind = np.roll(intersection_found_arr[:, :, 1], -1, axis=0)

        for cam_id in range(n_views):
            right_r = right_ratios[cam_id, right_ind[cam_id]]
            left_r = 1.0 / left_ratios[cam_id, left_ind[cam_id]]
            all_r = np.concatenate([right_r, left_r], axis=0)
            if len(all_r) > 0:
                median_ratios[cam_id] = np.median(all_r, axis=0)

        median_prop = np.ones((n_views, 3))
        for i in range(n_views - 1):
            median_prop[i + 1] = median_prop[i] * median_ratios[i]
        mean_prop = np.mean(median_prop, axis=0)
        mean_prop[mean_prop < 1e-8] = 1.0
        scale_ratios = median_prop / mean_prop

        # Save ratios
        np.save(out_dir / 'npy' / f'scale_ratios_{space_label}.npy', scale_ratios)
        np.save(out_dir / 'npy' / f'median_ratios_{space_label}.npy', median_ratios)

        # ---- Hit rates bar chart ----
        fig_hr, ax_hr = plt.subplots(figsize=(12, 4))
        cam_ids = [s['cam_id'] for s in stats]
        hit_rates = [s['hit_rate'] * 100 for s in stats]
        colors = ['green' if hr > 50 else 'orange' if hr > 10 else 'red' for hr in hit_rates]
        ax_hr.bar(cam_ids, hit_rates, color=colors)
        ax_hr.set_xlabel('View ID')
        ax_hr.set_ylabel('Hit rate (%)')
        ax_hr.set_title(f'{mode.upper()} [{space_label}] — Ray-mesh hit rates')
        ax_hr.set_ylim(0, 105)
        for cid, hr in zip(cam_ids, hit_rates):
            ax_hr.text(cid, hr + 1, f'{hr:.0f}%', ha='center', fontsize=7)
        fig_hr.tight_layout()
        fig_hr.savefig(out_dir / f'hit_rates_{space_label}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_hr)
        print(f"  Saved: hit_rates_{space_label}.png")

        # ---- Ratios plot ----
        fig_r, axes_r = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        x = np.arange(n_views)
        for ch, color, label in [(0, 'red', 'R'), (1, 'green', 'G'), (2, 'blue', 'B')]:
            axes_r[0].plot(x, median_ratios[:, ch], 'o-', color=color, label=f'{label}', ms=4)
            axes_r[1].plot(x, scale_ratios[:, ch], 's-', color=color, label=f'{label}', ms=4)
        axes_r[0].axhline(1.0, color='gray', ls='--', alpha=0.5)
        axes_r[0].set_ylabel('Pairwise median ratio')
        axes_r[0].set_title(f'{mode.upper()} [{space_label}] — Pairwise ratios')
        axes_r[0].legend()
        axes_r[1].axhline(1.0, color='gray', ls='--', alpha=0.5)
        axes_r[1].set_ylabel('Final scale')
        axes_r[1].set_xlabel('View ID')
        axes_r[1].set_title('Propagated + normalized')
        axes_r[1].legend()
        fig_r.tight_layout()
        fig_r.savefig(out_dir / f'ratios_{space_label}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_r)
        print(f"  Saved: ratios_{space_label}.png")

    # ---- Summary table (both spaces side by side) ----
    np.save(out_dir / 'npy' / 'ray_stats.npy', {
        space: [{'cam_id': s['cam_id'], 'hits': s['hits'], 'sampled': s['sampled'],
                 'hit_rate': s['hit_rate'], 'masked_px': s['masked_px']}
                for s in stats_list]
        for space, stats_list in results.items()
    })

    print(f"\n  {'='*90}")
    print(f"  {mode.upper()} RAY INTERSECTION SUMMARY")
    print(f"  {'='*90}")
    print(f"  {'View':>4} | {'World hits':>10} | {'World %':>8} | {'Norm hits':>10} | {'Norm %':>8}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")

    for i in range(n_views):
        w_stat = results.get('world', [{}] * n_views)[i]
        n_stat = results.get('norm', [{}] * n_views)[i]
        w_hits = w_stat.get('hits', '-')
        w_rate = f"{w_stat.get('hit_rate', 0)*100:.0f}%" if 'hit_rate' in w_stat else '-'
        n_hits = n_stat.get('hits', '-')
        n_rate = f"{n_stat.get('hit_rate', 0)*100:.0f}%" if 'hit_rate' in n_stat else '-'
        print(f"  {i:4d} | {w_hits:>10} | {w_rate:>8} | {n_hits:>10} | {n_rate:>8}")

    print(f"  {'='*90}")

    return results


def _build_norm_c2ws_and_Ks(data):
    """
    Build cameras in normalized space from all_c2w.
    all_c2w has Y/Z flipped — we need to un-flip for ray casting
    (albedo_scaling uses unflipped cameras).

    For intrinsics, we reuse camera_Ks (they don't depend on the coordinate system).
    """
    all_c2w = data['all_c2w']  # (N, 3, 4) — normalized, flipped
    n_views = data['n_views']

    c2ws_norm = []
    for i in range(n_views):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :4] = all_c2w[i]
        # Un-flip Y/Z (the flip was: c2w[:3, 1:3] *= -1)
        c2w[:3, 1:3] *= -1
        c2ws_norm.append(c2w)

    # For intrinsics in normalized space: same pixel-space K
    # (intrinsics don't change with coordinate system)
    return c2ws_norm, data['camera_Ks']


def main():
    parser = argparse.ArgumentParser(description='Debug albedo scaling pipeline')
    parser.add_argument('--mode', choices=['idr', 'sfm', 'both'], default='both')
    parser.add_argument('--n_samples', type=int, default=2000)
    args, extras = parser.parse_known_args()

    modes = ['idr', 'sfm'] if args.mode == 'both' else [args.mode]

    for mode in modes:
        print(f"\n{'#'*60}")
        print(f"# {mode.upper()}")
        print(f"{'#'*60}")

        data = load_experiment(mode, cli_args=extras if extras else None)
        if data is None:
            continue

        summary = save_npy_data(data)
        print_summary(summary)
        make_overview_plot(data)
        test_ray_intersections(data, n_samples=args.n_samples)

    print(f"\nAll output saved to: {ROOT / 'debug_albedo_scaling'}/")


if __name__ == '__main__':
    main()
