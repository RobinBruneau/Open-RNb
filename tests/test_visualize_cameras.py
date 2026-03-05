"""Visualize cameras around the unit sphere after SfM loading.

Generates an interactive 3D HTML plot showing:
- The unit sphere (wireframe)
- Camera positions (after scene scaling)
- Camera axes (RGB = XYZ: X=right, Y=up, Z=forward toward scene)

Uses mask-based silhouette scaling when masks are available.

Run directly: .venv/bin/python tests/test_visualize_cameras.py
Or via pytest: .venv/bin/python -m pytest tests/test_visualize_cameras.py -v -s
"""
import os
import sys
import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIXTURE_DIR = os.path.join(ROOT, "tests", "data", "golden_snail_mini")
NORMAL_SFM = os.path.join(FIXTURE_DIR, "normalSfm.json")
MASK_SFM = os.path.join(FIXTURE_DIR, "maskSfm.json")
OUTPUT_HTML = os.path.join(ROOT, "tests", "cameras_3d.html")


def load_masks(mask_cameras):
    """Load mask images from camera dicts, return list of (H, W) numpy arrays."""
    masks = []
    for cam in mask_cameras:
        mask_img = Image.open(cam['image_path']).convert('L')
        masks.append(np.array(mask_img).astype(np.float32) / 255.0)
    return masks


def build_camera_plot(cameras, scene_center, scale_factor, radius=1.5, title="Cameras & Unit Sphere"):
    """Build a Plotly figure with unit sphere + camera frustums.

    Shows cameras in corrected world frame (after WORLD_CORRECTION):
    X=right (red), Y=up (green), Z=forward (blue).
    No NeuS flip — this is the natural camera orientation for debugging.

    Args:
        cameras: list of camera dicts from load_sfm (WORLD_CORRECTION already applied)
        scene_center: (3,) array in corrected world frame
        scale_factor: float
        radius: model radius (bounding sphere)
        title: plot title

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # --- Unit sphere wireframe ---
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    r = radius
    x_sphere = r * np.outer(np.cos(u), np.sin(v))
    y_sphere = r * np.outer(np.sin(u), np.sin(v))
    z_sphere = r * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.1, colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        showscale=False, name='Unit sphere'
    ))

    # --- Camera positions and axes ---
    axis_len = 0.3 * radius
    colors = {'X': 'red', 'Y': 'green', 'Z': 'blue'}

    cam_positions = []
    for i, cam in enumerate(cameras):
        R = cam['R_cam2world']  # Already has WORLD_CORRECTION applied
        pos = scale_factor * (cam['center'] - scene_center)

        cam_positions.append(pos)

        # Camera axes: X=right(red), Y=up(green), Z=forward(blue)
        for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
            direction = R[:, axis_idx]
            end = pos + axis_len * direction
            fig.add_trace(go.Scatter3d(
                x=[pos[0], end[0]], y=[pos[1], end[1]], z=[pos[2], end[2]],
                mode='lines',
                line=dict(color=colors[axis_name], width=4),
                name=f'Cam {i} {axis_name}' if i == 0 else None,
                showlegend=(i == 0),
                legendgroup=axis_name,
            ))

    # Camera center points
    cam_positions = np.array(cam_positions)
    fig.add_trace(go.Scatter3d(
        x=cam_positions[:, 0], y=cam_positions[:, 1], z=cam_positions[:, 2],
        mode='markers+text',
        marker=dict(size=5, color='black'),
        text=[str(i) for i in range(len(cam_positions))],
        textposition='top center',
        name='Cameras',
    ))

    # --- Scene center ---
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='orange', symbol='diamond'),
        name='Origin (scaled)',
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
        ),
        width=900, height=700,
    )
    return fig


def compute_scaling(normal_cameras, mask_sfm_path=None):
    """Compute scene scaling using masks when available, else camera centers."""
    from datasets.sfm import load_sfm_json, match_views_by_id, cameras_by_view_id
    from datasets.utils import compute_scaling_from_silhouettes, compute_scaling_from_cameras

    if mask_sfm_path and os.path.isfile(mask_sfm_path):
        mask_cameras, _ = load_sfm_json(mask_sfm_path)
        common_ids = match_views_by_id([normal_cameras, mask_cameras])
        normal_by_id = cameras_by_view_id(normal_cameras)
        mask_by_id = cameras_by_view_id(mask_cameras)

        ordered_cams = [normal_by_id[vid] for vid in common_ids]
        masks = load_masks([mask_by_id[vid] for vid in common_ids])

        center, scale = compute_scaling_from_silhouettes(ordered_cams, masks)
        return center, scale, "silhouettes"
    else:
        center, scale = compute_scaling_from_cameras(normal_cameras)
        return center, scale, "camera centers"


class TestVisualizeCameras:
    def test_generate_3d_plot(self):
        from datasets.sfm import load_sfm_json
        cameras, _ = load_sfm_json(NORMAL_SFM)
        center, scale, method = compute_scaling(cameras, MASK_SFM)

        fig = build_camera_plot(cameras, center, scale)
        fig.write_html(OUTPUT_HTML)
        assert os.path.isfile(OUTPUT_HTML)
        size_kb = os.path.getsize(OUTPUT_HTML) / 1024
        print(f"\n  -> Wrote {OUTPUT_HTML} ({size_kb:.0f} KB)")
        print(f"     Scaling method: {method}")
        print(f"     Open in browser to inspect cameras.")


# Allow running directly
if __name__ == "__main__":
    from datasets.sfm import load_sfm_json

    # Try full-res data first, fall back to mini fixture
    full_res_normal = os.path.join(ROOT, "data", "golden_snail", "normalSfm.json")
    full_res_mask = os.path.join(ROOT, "data", "golden_snail", "maskSfm.json")

    if os.path.isfile(full_res_normal):
        sfm_path = full_res_normal
        mask_path = full_res_mask
        title = "Golden Snail — 20 views (full-res)"
    else:
        sfm_path = NORMAL_SFM
        mask_path = MASK_SFM
        title = "Golden Snail — 3 views (mini fixture)"

    cameras, _ = load_sfm_json(sfm_path)
    center, scale, method = compute_scaling(cameras, mask_path)

    print(f"Loaded {len(cameras)} cameras")
    print(f"Scene center: {center}")
    print(f"Scale factor: {scale:.4f}")
    print(f"Scaling method: {method}")

    fig = build_camera_plot(cameras, center, scale, title=title)
    fig.write_html(OUTPUT_HTML)
    print(f"Wrote {OUTPUT_HTML}")

    # Also try to open in browser
    import webbrowser
    webbrowser.open(f"file://{OUTPUT_HTML}")
