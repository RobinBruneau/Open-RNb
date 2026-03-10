"""Scene normalization utilities for Open-RNb datasets.

All functions are numpy-only (no torch, no PyTorch Lightning).
"""
import numpy as np


# ---------------------------------------------------------------------------
# Constants — use these instead of raw strings to avoid silent typos.
# ---------------------------------------------------------------------------

SCALE_MAT = 'scale_mat'
PCD = 'pcd'
SILHOUETTES = 'silhouettes'
CAMERAS = 'cameras'
AUTO = 'auto'
NONE = 'none'

VALID_SCALING_MODES = {SCALE_MAT, PCD, SILHOUETTES, CAMERAS, AUTO, NONE}

SPACE_WORLD = 'world'
SPACE_NORMALIZED = 'normalized'


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def make_K(fx, fy, cx, cy, dtype=np.float32):
    """Build a 3x3 camera intrinsic matrix."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype)


def neus_c2w_to_standard(c2w34):
    """Convert a NeuS (3,4) c2w (Y/Z flipped) to a standard (4,4) cam-to-world."""
    c2w44 = np.eye(4, dtype=np.float64)
    c2w44[:3, :4] = c2w34
    c2w44[:3, 1:3] *= -1.
    return c2w44


def scale_camera_intrinsics(cameras, factor):
    """Return a copy of each camera dict with fx/fy/cx/cy scaled by *factor*."""
    return [{**cam,
             'fx': cam['fx'] * factor, 'fy': cam['fy'] * factor,
             'cx': cam['cx'] * factor, 'cy': cam['cy'] * factor}
            for cam in cameras]


def compute_scaling_from_scale_mat(scale_mat_0):
    """Extract (scene_center, scale_factor) from a scale_mat (RNb/cameras.npz format).

    RNb scale_mat convention:
        p_norm = s * p_world + t   =>   p_world = (p_norm - t) / s

    so:  scene_center = -t / s,   scale_factor = s

    Args:
        scale_mat_0: (4, 4) numpy array — the scale_mat for view 0.

    Returns:
        scene_center: (3,) numpy array
        scale_factor: float
    """
    s = float(scale_mat_0[0, 0])   # uniform scale on diagonal
    t = scale_mat_0[:3, 3]          # translation column
    scene_center = (-t / s).astype(np.float64)
    scale_factor = s
    return scene_center, scale_factor


def compute_scaling_from_pcd(pcd, sphere_scale=1.0):
    """Compute (scene_center, scale_factor) from a point cloud (N, 3).

    Uses centroid + 99th-percentile distance for robustness against outliers.
    Recomputes centroid on inliers (points within the 99th-percentile radius).

    Args:
        pcd:          (N, 3) numpy array of 3D points.
        sphere_scale: target radius after normalisation.

    Returns:
        center:      (3,) numpy array
        scale:       float — multiply world positions by scale to fit inside
                     a sphere of radius sphere_scale centred at center.
    """
    pcd = np.asarray(pcd, dtype=np.float64)
    if len(pcd) == 0:
        return np.zeros(3), 1.0

    # Initial centroid
    center = np.mean(pcd, axis=0)
    distances = np.linalg.norm(pcd - center, axis=1)

    if len(distances) == 1:
        # Single point — degenerate; return that point as center, scale=1
        max_dist = 1.0
        return center, float(sphere_scale / max_dist)

    max_dist_99 = np.percentile(distances, 99)

    # Recompute center on inliers
    inliers = pcd[distances <= max_dist_99]
    center = np.mean(inliers, axis=0)
    max_dist = np.max(np.linalg.norm(inliers - center, axis=1))

    if max_dist < 1e-8:
        max_dist = 1.0

    scale = float(sphere_scale / max_dist)
    return center, scale


def compute_scaling_from_silhouettes(cameras, masks, sphere_scale=1.0, fg_area_ratio=5):
    """Compute (scene_center, scale_factor) from silhouettes (MVSCPS method).

    Center is estimated via mask center-of-mass triangulation (least-squares).
    Radius is estimated via projected sphere area matching.

    Args:
        cameras:       list of dicts with keys fx, fy, cx, cy,
                       R_cam2world (3x3), center (3,).
        masks:         list of (H, W) float arrays (values in [0, 1]).
        sphere_scale:  target sphere radius relative to model.radius.
        fg_area_ratio: ratio of sphere area to foreground area.

    Returns:
        center: (3,) array
        scale:  float
    """
    from scipy.ndimage import center_of_mass

    A = np.zeros((3, 3))
    b = np.zeros(3)

    cam_data = []
    for cam, mask in zip(cameras, masks):
        fx, fy = cam['fx'], cam['fy']
        cx, cy = cam['cx'], cam['cy']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K_inv = np.linalg.inv(K)

        R_c2w = cam['R_cam2world']
        center_cam = cam['center']

        # Mask center-of-mass in pixel coordinates (row, col) -> (x_pixel, y_pixel)
        com = center_of_mass(mask.astype(np.float64))
        com_pixel = np.array([com[1], com[0], 1.0])  # (x, y, 1)

        # Ray direction in camera space
        dir_cam = K_inv @ com_pixel
        dir_cam = dir_cam / np.linalg.norm(dir_cam)

        # Direction in world space
        m = R_c2w @ dir_cam
        o = center_cam

        # Accumulate least-squares: (I - m*m^T) * d = (I - m*m^T) * o
        I_mmT = np.eye(3) - np.outer(m, m)
        A += I_mmT
        b += I_mmT @ o

        cam_data.append((fx, fy, R_c2w, center_cam, mask))

    # Solve for scene center
    scene_center = np.linalg.lstsq(A, b, rcond=None)[0]

    # Compute radius from projected sphere area
    total_fg_area = 0
    sum_fz2 = 0
    for fx, fy, R_c2w, center_cam, mask in cam_data:
        total_fg_area += mask.sum()
        R_w2c = R_c2w.T
        center_in_cam = R_w2c @ (scene_center - center_cam)
        Z = center_in_cam[2]
        if abs(Z) < 1e-8:
            Z = 1e-8
        sum_fz2 += (fx / Z) ** 2

    radius = np.sqrt(fg_area_ratio * total_fg_area / (np.pi * sum_fz2))
    if radius < 1e-8:
        radius = 1.0
    scale = float(sphere_scale / radius)

    return scene_center, scale


def compute_scaling_from_cameras(cameras, sphere_scale=1.0):
    """Fallback: compute (scene_center, scale_factor) from camera centers only.

    Args:
        cameras:      list of dicts, each with key 'center' (3,).
        sphere_scale: target sphere radius.

    Returns:
        center: (3,) array
        scale:  float
    """
    centers = np.array([c['center'] for c in cameras])
    return compute_scaling_from_pcd(centers, sphere_scale)


def compute_scaling_from_mesh(vertices, sphere_scale=1.5):
    """Compute (scene_center, scale_factor) from mesh bounding box.

    Used for phase-2 renormalization. Default sphere_scale=1.5 fills the
    model sphere.

    Args:
        vertices:     (N, 3) numpy array of mesh vertex positions.
        sphere_scale: target sphere radius (default 1.5).

    Returns:
        center: (3,) array — bounding-box midpoint.
        scale:  float — multiply world positions by scale to fit inside
                a sphere of radius sphere_scale.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    max_dist = np.linalg.norm(vertices - center, axis=1).max()
    if max_dist < 1e-8:
        max_dist = 1.0
    return center, float(sphere_scale / max_dist)


def compute_scene_scaling(scaling_mode, sphere_scale, scale_mat=None, pcd=None,
                           cameras=None, masks=None, fg_area_ratio=5.0):
    """Universal dispatcher for scene normalisation.

    Args:
        scaling_mode: one of:
            'scale_mat'   — use RNb-style scale_mat (requires scale_mat arg).
            'pcd'         — use 3D point cloud (requires pcd arg).
            'silhouettes' — use per-view silhouettes (requires cameras + masks).
            'cameras'     — use camera centres as a point cloud.
            'auto'        — try in order: pcd -> silhouettes -> cameras.
            'none'        — identity (zeros center, scale 1.0).
        sphere_scale: target sphere radius passed to the chosen sub-function.
        scale_mat:    (4, 4) numpy array; required for 'scale_mat' mode.
        pcd:          (N, 3) numpy array; used by 'pcd' and 'auto' modes.
        cameras:      list of camera dicts; used by 'silhouettes', 'cameras',
                      and 'auto' modes.
        masks:        list of (H, W) arrays; used by 'silhouettes' and 'auto'.
        fg_area_ratio: ratio passed to compute_scaling_from_silhouettes.

    Returns:
        scene_center: (3,) numpy array
        scale_factor: float
    """
    if scaling_mode not in VALID_SCALING_MODES:
        raise ValueError(
            f"Unknown scaling_mode={scaling_mode!r}. "
            f"Expected one of: {VALID_SCALING_MODES}."
        )

    if scaling_mode == NONE:
        return np.zeros(3), 1.0

    if scaling_mode == SCALE_MAT:
        if scale_mat is None:
            raise ValueError("scale_mat must be provided when scaling_mode='scale_mat'")
        return compute_scaling_from_scale_mat(scale_mat)

    if scaling_mode == PCD:
        return compute_scaling_from_pcd(pcd, sphere_scale)

    if scaling_mode == SILHOUETTES:
        return compute_scaling_from_silhouettes(
            cameras, masks, sphere_scale=sphere_scale, fg_area_ratio=fg_area_ratio
        )

    if scaling_mode == CAMERAS:
        return compute_scaling_from_cameras(cameras, sphere_scale)

    if scaling_mode == AUTO:
        # Try silhouettes first (more reliable for PS/neural reconstruction)
        has_silhouettes = (cameras is not None and masks is not None
                           and len(cameras) > 0 and len(masks) > 0)
        if has_silhouettes:
            return compute_scaling_from_silhouettes(
                cameras, masks, sphere_scale=sphere_scale, fg_area_ratio=fg_area_ratio
            )

        # Fall back to pcd
        has_pcd = pcd is not None and len(pcd) > 0
        if has_pcd:
            return compute_scaling_from_pcd(pcd, sphere_scale)

        # Fall back to camera centres
        if cameras is not None and len(cameras) > 0:
            return compute_scaling_from_cameras(cameras, sphere_scale)

        # Nothing available
        return np.zeros(3), 1.0
