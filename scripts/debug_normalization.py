"""
Diagnostic: normalisation de scène — comparaison IDR vs SFM.

Pour chaque dataset :
  - Phase 1 : normalisation initiale (scale_mat pour IDR, silhouettes pour SFM)
  - Phase 2 : renormalisation depuis le mesh intermédiaire

Outputs dans OUTDIR :
  {dataset}_phase1_norm.ply   mesh normalisé P1 + sphère unité + caméras
  {dataset}_phase2_norm.ply   mesh normalisé P2 + sphère unité + caméras
  {dataset}_overview.png      vues 2D (top/front/side) des deux phases
  {dataset}_histogram.png     distribution des distances vertex-origine
  report.txt                  métriques quantitatives
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# ── paths ─────────────────────────────────────────────────────────────────────
import glob as _glob

T9 = '/media/bbrument/T9'
OUTDIR = f'{T9}/RNb-NeuS_exp/normalization_debug'

IDR_ROOT = f'{T9}/skoltech3d_data/golden_snail/sdmunips'


def _latest_run(exp_name):
    """Return path to the most recent run directory under RNb-NeuS_exp/<exp_name>/."""
    pattern = f'{T9}/RNb-NeuS_exp/{exp_name}/@*'
    dirs = sorted(_glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f'No run found matching {pattern}')
    return dirs[-1]  # lexicographic = chronological for @YYYYMMDD-HHMMSS


def _latest_ckpt(run_dir):
    """Return the phase-1 checkpoint (epoch=0-step=N with lowest N)."""
    matches = sorted(_glob.glob(f'{run_dir}/ckpt/epoch=0-step=*.ckpt'))
    if not matches:
        raise FileNotFoundError(f'No phase-1 checkpoint in {run_dir}/ckpt/')
    return matches[0]  # lowest step = end of phase 1


def _final_mesh(run_dir):
    """Return the final exported mesh (it<N>-mc<R>.ply with highest N)."""
    matches = sorted(_glob.glob(f'{run_dir}/save/it*-mc*.ply'))
    if not matches:
        raise FileNotFoundError(f'No final mesh in {run_dir}/save/')
    return matches[-1]  # highest step number


_idr_run = _latest_run('idr-golden_snail')
IDR_MESH_P1 = f'{_idr_run}/save/intermediate_mesh.ply'
IDR_MESH_P2 = _final_mesh(_idr_run)
IDR_CKPT_P1 = _latest_ckpt(_idr_run)

SFM_NORMAL  = 'data/golden_snail/normalSfm.json'
SFM_ALBEDO  = 'data/golden_snail/albedoSfm.json'
SFM_MASK    = 'data/golden_snail/maskSfm.json'
_sfm_run = _latest_run('sfm-golden_snail')
SFM_MESH_P1 = f'{_sfm_run}/save/intermediate_mesh.ply'
SFM_MESH_P2 = _final_mesh(_sfm_run)
SFM_CKPT_P1 = _latest_ckpt(_sfm_run)

SPHERE_SCALE_P1 = 1.0   # phase-1 silhouette target (matches configs/sfm.yaml sphere_scale)
SPHERE_SCALE_P2 = 1.5   # phase-2 mesh renorm target (matches launch.py sphere_scale_p2)
MODEL_RADIUS    = 1.5   # NeuS model sphere radius

os.makedirs(OUTDIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def renorm_from_mesh(mesh_path):
    """Compute phase-2 scene_center + scale_factor from intermediate mesh (world space)."""
    m = trimesh.load(mesh_path, process=False)
    v = np.array(m.vertices)
    center = (v.max(0) + v.min(0)) / 2
    max_dist = np.linalg.norm(v - center, axis=1).max()
    scale = SPHERE_SCALE_P2 / max_dist
    return center, scale


def load_mesh(path):
    m = trimesh.load(path, process=False)
    return np.array(m.vertices), np.array(m.faces)


def p1_from_checkpoint(ckpt_path):
    """Read scene_center and scale_factor from a phase-1 PL checkpoint."""
    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    center = sd['model.geometry.scene_center'].numpy().astype(np.float64)
    scale  = float(sd['model.geometry.scale_factor'].item())
    return center, scale


def to_norm(verts_w, center, scale):
    return (verts_w - center) * scale


def cam_unflip(all_c2w_np):
    """Extract world camera positions from all_c2w (undo NeuS Y/Z flip)."""
    positions = []
    for c2w34 in all_c2w_np:
        c44 = np.eye(4)
        c44[:3, :4] = c2w34
        c44[:3, 1:3] *= -1.
        positions.append(c44[:3, 3])
    return np.array(positions)


def metrics(verts_n, cams_n):
    d = np.linalg.norm(verts_n, axis=1)
    center = (verts_n.max(0) + verts_n.min(0)) / 2
    bs_c = np.linalg.norm(verts_n - center, axis=1).max()
    bs_o = d.max()
    cd = np.linalg.norm(cams_n, axis=1)
    return dict(
        dists=d, bs_centroid=bs_c, bs_origin=bs_o,
        pct_in_09=100*(d < 0.9).mean(),
        pct_in_10=100*(d < 1.0).mean(),
        pct_in_15=100*(d < 1.5).mean(),
        cam_min=cd.min(), cam_max=cd.max(), cam_mean=cd.mean(),
    )


def make_sphere(r, color, subdiv=3):
    s = trimesh.creation.icosphere(subdivisions=subdiv, radius=r)
    s.visual.vertex_colors = color
    return s


def colorize_mesh(verts, faces, clip_r):
    d = np.linalg.norm(verts, axis=1)
    t = np.clip(d / clip_r, 0, 1)
    c = np.zeros((len(verts), 4), np.uint8)
    c[:, 0] = (t * 255).astype(np.uint8)
    c[:, 1] = ((1 - t) * 200).astype(np.uint8)
    c[:, 3] = 255
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=c)
    return m


def cam_spheres(positions, r=0.04):
    parts = []
    for pos in positions:
        s = trimesh.creation.icosphere(subdivisions=1, radius=r)
        s.apply_translation(pos)
        s.visual.vertex_colors = [255, 200, 0, 255]
        parts.append(s)
    return parts


def save_ply(verts_n, cams_n, faces, path, clip_r=2.0):
    parts = [
        make_sphere(1.0, [100, 100, 255, 50]),
        make_sphere(0.9, [60,  60,  220, 40]),
        make_sphere(MODEL_RADIUS, [200, 200, 200, 20]),
        colorize_mesh(verts_n, faces, clip_r),
    ] + cam_spheres(cams_n)
    trimesh.util.concatenate(parts).export(path)
    print(f"  → {path}")


def plot_overview(verts_n, cams_n, m1, m2, title, path):
    """4-panel: top/front/side 2D + histogram."""
    fig = plt.figure(figsize=(20, 5))
    planes = [(0, 2, 'Top X-Z'), (0, 1, 'Front X-Y'), (1, 2, 'Side Y-Z')]
    lim = max(np.abs(cams_n).max(), m2['bs_origin']) * 1.15
    theta = np.linspace(0, 2*np.pi, 300)
    labels = ['X', 'Y', 'Z']

    for col, (i0, i1, ptitle) in enumerate(planes):
        ax = fig.add_subplot(1, 4, col + 1)
        ax.fill(np.cos(theta), np.sin(theta), color='blue', alpha=0.05)
        ax.plot(np.cos(theta), np.sin(theta), 'b-', lw=1.2, alpha=0.5, label='R=1 (unit)')
        ax.plot(np.cos(theta)*0.9, np.sin(theta)*0.9, 'b--', lw=0.8, alpha=0.4, label='R=0.9 (target)')
        ax.plot(np.cos(theta)*MODEL_RADIUS, np.sin(theta)*MODEL_RADIUS,
                color='gray', lw=0.8, ls=':', alpha=0.4, label=f'R={MODEL_RADIUS} (model)')
        step = max(1, len(verts_n)//4000)
        mv = verts_n[::step]
        ax.scatter(mv[:, i0], mv[:, i1], s=0.8, c='green', alpha=0.25)
        mn, mx = verts_n.min(0), verts_n.max(0)
        ax.plot([mn[i0], mx[i0], mx[i0], mn[i0], mn[i0]],
                [mn[i1], mn[i1], mx[i1], mx[i1], mn[i1]],
                'g--', lw=1, alpha=0.6, label=f'BBox')
        ax.scatter(cams_n[:, i0], cams_n[:, i1], s=70, c='orange',
                   zorder=5, marker='^', label='Cameras')
        for j, p in enumerate(cams_n):
            ax.annotate(str(j), (p[i0], p[i1]), fontsize=5, ha='center', va='bottom')
        ax.set_xlabel(labels[i0]); ax.set_ylabel(labels[i1])
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
        ax.set_title(ptitle, fontweight='bold')
        if col == 0:
            ax.legend(fontsize=6, loc='upper left')

    # histogram
    ax_h = fig.add_subplot(1, 4, 4)
    ax_h.hist(m1['dists'], bins=80, alpha=0.55, color='steelblue',  label='Phase 1', edgecolor='none')
    ax_h.hist(m2['dists'], bins=80, alpha=0.55, color='darkorange', label='Phase 2', edgecolor='none')
    for r, c, ls in [(0.9, 'red', '--'), (1.0, 'purple', '--'), (MODEL_RADIUS, 'gray', ':')]:
        ax_h.axvline(r, color=c, lw=1.5, ls=ls, label=f'R={r}')
    ax_h.set_xlabel('Distance from origin'); ax_h.set_ylabel('Vertex count')
    ax_h.set_title('Vertex distance histogram')
    ax_h.legend(fontsize=7)

    fig.suptitle(
        f"{title}\n"
        f"P1: bsphere={m1['bs_centroid']:.3f}, {m1['pct_in_09']:.0f}% < 0.9  |  "
        f"P2: bsphere={m2['bs_centroid']:.3f}, {m2['pct_in_09']:.0f}% < 0.9",
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")


# ── IDR ───────────────────────────────────────────────────────────────────────
print("\n=== IDR ===")
import numpy as np
# P1 normalization: read from checkpoint (ground truth — independent of img_downscale)
IDR_P1_CENTER, IDR_P1_SCALE = p1_from_checkpoint(IDR_CKPT_P1)
IDR_P2_CENTER, IDR_P2_SCALE = renorm_from_mesh(IDR_MESH_P1)

print(f"  P1: center={IDR_P1_CENTER.round(4)}, scale={IDR_P1_SCALE:.6f}")
print(f"  P2: center={IDR_P2_CENTER.round(4)}, scale={IDR_P2_SCALE:.6f}")

# IDR cameras from dataset
cfg_idr = OmegaConf.create({
    'root_dir': IDR_ROOT, 'scene': 'golden_snail',
    'img_downscale': 4.0, 'train_split': 'train', 'val_split': 'val',
    'test_split': 'test', 'apply_light_opti': False, 'apply_rgb_plus': False,
})
from datasets.idr import IDRDataset
ds_idr = IDRDataset(cfg_idr, 'train')
cam_norm_p1_idr = cam_unflip(ds_idr.all_c2w.cpu().numpy())          # already in P1 norm space
cam_world_idr   = cam_norm_p1_idr / IDR_P1_SCALE + IDR_P1_CENTER   # world space
cam_norm_p2_idr = to_norm(cam_world_idr, IDR_P2_CENTER, IDR_P2_SCALE)

# Phase 1: intermediate mesh in P1 coords  — what NeuS actually built in phase 1
vw_idr_p1, fc_idr_p1 = load_mesh(IDR_MESH_P1)
vn1_idr = to_norm(vw_idr_p1, IDR_P1_CENTER, IDR_P1_SCALE)
# Phase 2: final mesh in P2 coords         — what NeuS built in phase 2
vw_idr_p2, fc_idr_p2 = load_mesh(IDR_MESH_P2)
vn2_idr = to_norm(vw_idr_p2, IDR_P2_CENTER, IDR_P2_SCALE)

m1_idr = metrics(vn1_idr, cam_norm_p1_idr)
m2_idr = metrics(vn2_idr, cam_norm_p2_idr)

save_ply(vn1_idr, cam_norm_p1_idr, fc_idr_p1,
         f'{OUTDIR}/idr_phase1_norm.ply', clip_r=max(m1_idr['bs_origin']*0.8, 2))
save_ply(vn2_idr, cam_norm_p2_idr, fc_idr_p2,
         f'{OUTDIR}/idr_phase2_norm.ply', clip_r=max(m2_idr['bs_origin']*0.8, 1.5))
plot_overview(vn2_idr, cam_norm_p2_idr, m1_idr, m2_idr,
              'IDR — golden_snail', f'{OUTDIR}/idr_overview.png')

# ── SFM ───────────────────────────────────────────────────────────────────────
print("\n=== SFM ===")
# P1 normalization: read from checkpoint (matches img_downscale=1.0 used in training)
SFM_P1_CENTER, SFM_P1_SCALE = p1_from_checkpoint(SFM_CKPT_P1)
SFM_P2_CENTER, SFM_P2_SCALE = renorm_from_mesh(SFM_MESH_P1)

print(f"  P1: center={SFM_P1_CENTER.round(4)}, scale={SFM_P1_SCALE:.6f}")
print(f"  P2: center={SFM_P2_CENTER.round(4)}, scale={SFM_P2_SCALE:.6f}")

# Load cameras using img_downscale=1.0 (matches training) to get correct P1 all_c2w
cfg_sfm = OmegaConf.create({
    'name': 'sfm', 'scene': 'golden_snail',
    'normal_sfm': SFM_NORMAL, 'albedo_sfm': SFM_ALBEDO, 'mask_sfm': SFM_MASK,
    'scaling_mode': 'auto', 'sphere_scale': SPHERE_SCALE_P1, 'fg_area_ratio': 5,
    'img_downscale': 1.0, 'train_split': 'train', 'val_split': 'val',
    'test_split': 'test', 'apply_light_opti': False, 'apply_rgb_plus': False,
})
from datasets.sfm import SfMDataset
ds_sfm = SfMDataset(cfg_sfm, 'train')
cam_norm_p1_sfm = cam_unflip(ds_sfm.all_c2w.cpu().numpy())
cam_world_sfm   = cam_norm_p1_sfm / SFM_P1_SCALE + SFM_P1_CENTER
cam_norm_p2_sfm = to_norm(cam_world_sfm, SFM_P2_CENTER, SFM_P2_SCALE)

# Phase 1: intermediate mesh in P1 coords
vw_sfm_p1, fc_sfm_p1 = load_mesh(SFM_MESH_P1)
vn1_sfm = to_norm(vw_sfm_p1, SFM_P1_CENTER, SFM_P1_SCALE)
# Phase 2: final mesh in P2 coords
vw_sfm_p2, fc_sfm_p2 = load_mesh(SFM_MESH_P2)
vn2_sfm = to_norm(vw_sfm_p2, SFM_P2_CENTER, SFM_P2_SCALE)

m1_sfm = metrics(vn1_sfm, cam_norm_p1_sfm)
m2_sfm = metrics(vn2_sfm, cam_norm_p2_sfm)

save_ply(vn1_sfm, cam_norm_p1_sfm, fc_sfm_p1,
         f'{OUTDIR}/sfm_phase1_norm.ply', clip_r=max(m1_sfm['bs_origin']*0.8, 2))
save_ply(vn2_sfm, cam_norm_p2_sfm, fc_sfm_p2,
         f'{OUTDIR}/sfm_phase2_norm.ply', clip_r=max(m2_sfm['bs_origin']*0.8, 1.5))
plot_overview(vn2_sfm, cam_norm_p2_sfm, m1_sfm, m2_sfm,
              'SFM — golden_snail', f'{OUTDIR}/sfm_overview.png')

# ── rapport ───────────────────────────────────────────────────────────────────
report = f"""
=== Normalization Diagnostic — IDR vs SFM ===

┌─────────────────────────────────────────────────────────────────────┐
│ IDR                                                                  │
├──────────────────┬──────────────────────┬───────────────────────────┤
│                  │ Phase 1 (scale_mat)  │ Phase 2 (renorm mesh)     │
├──────────────────┼──────────────────────┼───────────────────────────┤
│ scene_center     │ {str(IDR_P1_CENTER.round(3)):20s} │ {str(IDR_P2_CENTER.round(3)):25s} │
│ scale_factor     │ {IDR_P1_SCALE:20.6f} │ {IDR_P2_SCALE:25.6f} │
│ bsphere (centr)  │ {m1_idr['bs_centroid']:20.4f} │ {m2_idr['bs_centroid']:25.4f} │
│ % verts < 0.9    │ {m1_idr['pct_in_09']:19.1f}% │ {m2_idr['pct_in_09']:24.1f}% │
│ % verts < 1.5    │ {m1_idr['pct_in_15']:19.1f}% │ {m2_idr['pct_in_15']:24.1f}% │
│ cam dist range   │ [{m1_idr['cam_min']:.2f}, {m1_idr['cam_max']:.2f}]          │ [{m2_idr['cam_min']:.2f}, {m2_idr['cam_max']:.2f}]                │
└──────────────────┴──────────────────────┴───────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ SFM                                                                  │
├──────────────────┬──────────────────────┬───────────────────────────┤
│                  │ Phase 1 (silhouette) │ Phase 2 (renorm mesh)     │
├──────────────────┼──────────────────────┼───────────────────────────┤
│ scene_center     │ {str(SFM_P1_CENTER.round(3)):20s} │ {str(SFM_P2_CENTER.round(3)):25s} │
│ scale_factor     │ {SFM_P1_SCALE:20.6f} │ {SFM_P2_SCALE:25.6f} │
│ bsphere (centr)  │ {m1_sfm['bs_centroid']:20.4f} │ {m2_sfm['bs_centroid']:25.4f} │
│ % verts < 0.9    │ {m1_sfm['pct_in_09']:19.1f}% │ {m2_sfm['pct_in_09']:24.1f}% │
│ % verts < 1.5    │ {m1_sfm['pct_in_15']:19.1f}% │ {m2_sfm['pct_in_15']:24.1f}% │
│ cam dist range   │ [{m1_sfm['cam_min']:.2f}, {m1_sfm['cam_max']:.2f}]         │ [{m2_sfm['cam_min']:.2f}, {m2_sfm['cam_max']:.2f}]                │
└──────────────────┴──────────────────────┴───────────────────────────┘

Scale ratio P1/P2:
  IDR : {IDR_P1_SCALE/IDR_P2_SCALE:.3f}x  → P1 était déjà bien calibrée (scale_mat)
  SFM : {SFM_P1_SCALE/SFM_P2_SCALE:.3f}x  → P1 silhouette sous-estimait l'objet de {SFM_P1_SCALE/SFM_P2_SCALE:.1f}x
"""
print(report)
with open(f'{OUTDIR}/report.txt', 'w') as f:
    f.write(report)
print(f"\nAll outputs in {OUTDIR}/")
