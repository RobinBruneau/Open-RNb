import os
import json
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


# =============================================================================
# Dual-backend SfM loading
# =============================================================================

def load_sfm_pyalicevision(sfm_path):
    """Primary: load via pyalicevision C++ bindings.

    Returns:
        cameras: list of dicts with keys:
            view_id, pose_id, image_path, R_cam2world (3x3), center (3,),
            fx, fy, cx, cy, width, height
        landmarks: (N, 3) array or None
    """
    from pyalicevision import sfmData, sfmDataIO

    sfm = sfmData.SfMData()
    if not sfmDataIO.load(sfm, str(sfm_path), sfmDataIO.ALL):
        raise ValueError(f"Failed to load SfMData: {sfm_path}")

    # Extract via JSON dict (avoids pyalicevision buffer reuse issues)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    try:
        sfmDataIO.save(sfm, temp_path, sfmDataIO.ALL)
        with open(temp_path, 'r') as f:
            data = json.load(f)
    finally:
        os.unlink(temp_path)

    return _parse_sfm_json_data(data)


def load_sfm_json(sfm_path):
    """Fallback: parse .sfm/.json file directly (no pyalicevision needed)."""
    with open(sfm_path, 'r') as f:
        data = json.load(f)
    return _parse_sfm_json_data(data)


def _parse_sfm_json_data(data):
    """Parse SfM JSON dict into standardized camera list + landmarks.

    Returns:
        cameras: list of dicts
        landmarks: (N, 3) array or None
    """
    intrinsics = {i['intrinsicId']: i for i in data.get('intrinsics', [])}
    poses = {p['poseId']: p['pose']['transform'] for p in data.get('poses', [])}

    cameras = []
    for view in data.get('views', []):
        view_id = view['viewId']
        intr_id = view['intrinsicId']
        pose_id = view['poseId']

        if intr_id not in intrinsics or pose_id not in poses:
            continue

        intr = intrinsics[intr_id]
        transform = poses[pose_id]

        width = int(intr['width'])
        height = int(intr['height'])

        # Focal length: pxFocalLength [fx, fy] or focalLength (mm) + sensorWidth (mm)
        if 'pxFocalLength' in intr:
            pxf = intr['pxFocalLength']
            if isinstance(pxf, list):
                fx, fy = float(pxf[0]), float(pxf[1])
            else:
                fx = fy = float(pxf)
        else:
            focal_mm = float(intr['focalLength'])
            sensor_width = float(intr.get('sensorWidth', 36.0))
            if 'sensorWidth' not in intr:
                import warnings
                warnings.warn(f"sensorWidth not found in intrinsics, using default 36.0mm")
            fx = fy = focal_mm * width / sensor_width

        # Principal point: offset from image center
        pp = intr.get('principalPoint', ['0', '0'])
        cx = width / 2.0 + float(pp[0])
        cy = height / 2.0 + float(pp[1])

        # Rotation cam2world (row-major, 9 strings)
        rotation_flat = [float(r) for r in transform['rotation']]
        R_cam2world = np.array(rotation_flat).reshape(3, 3)

        # Camera center in world
        center = np.array([float(c) for c in transform['center']])

        cameras.append({
            'view_id': view_id,
            'pose_id': pose_id,
            'image_path': view.get('path', ''),
            'R_cam2world': R_cam2world,
            'center': center,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'width': width, 'height': height,
        })

    # Extract 3D landmarks from structure
    landmarks = None
    structure = data.get('structure', [])
    if structure:
        pts = []
        for s in structure:
            coord = s.get('X', None)
            if coord is not None:
                pts.append([float(coord[0]), float(coord[1]), float(coord[2])])
        if pts:
            landmarks = np.array(pts)

    return cameras, landmarks


def load_sfm(sfm_path):
    """Try pyalicevision, fall back to JSON parsing."""
    try:
        return load_sfm_pyalicevision(sfm_path)
    except ImportError:
        return load_sfm_json(sfm_path)


# =============================================================================
# Scene scaling
# =============================================================================

def compute_scaling_from_landmarks(points_3d, sphere_scale=0.9):
    """Compute scene center and scale from 3D landmarks.

    Uses centroid + 99th percentile distance for robustness.

    Returns:
        center: (3,) array
        scale: float (multiply positions by this to fit in unit sphere * sphere_scale)
    """
    center = np.mean(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - center, axis=1)
    max_dist = np.percentile(distances, 99)
    # Recompute center on inliers
    inliers = points_3d[distances <= max_dist]
    center = np.mean(inliers, axis=0)
    max_dist = np.max(np.linalg.norm(inliers - center, axis=1))
    if max_dist < 1e-8:
        max_dist = 1.0
    scale = sphere_scale / max_dist
    return center, scale


def compute_scaling_from_silhouettes(cameras, masks, sphere_scale=0.9, fg_area_ratio=5):
    """Compute scene center and scale from silhouettes (MVSCPS method).

    Center is estimated via mask center-of-mass triangulation (least-squares).
    Radius is estimated via projected sphere area matching.

    Args:
        cameras: list of camera dicts (from load_sfm)
        masks: list of (H, W) binary masks
        sphere_scale: target sphere radius relative to model.radius
        fg_area_ratio: ratio of sphere area to foreground area

    Returns:
        center: (3,) array
        scale: float
    """
    from scipy.ndimage import center_of_mass

    A = np.zeros((3, 3))
    b = np.zeros(3)

    cam_data = []
    for cam, mask in zip(cameras, masks):
        # Camera intrinsics
        fx, fy = cam['fx'], cam['fy']
        cx, cy = cam['cx'], cam['cy']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K_inv = np.linalg.inv(K)

        R_c2w = cam['R_cam2world']
        center_cam = cam['center']

        # Mask center of mass (row, col) -> (x_pixel, y_pixel)
        com = center_of_mass(mask.astype(np.float64))
        com_pixel = np.array([com[1], com[0], 1.0])  # (x, y, 1)

        # Direction in camera space
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
        # Depth of scene center in this camera
        R_w2c = R_c2w.T
        center_in_cam = R_w2c @ (scene_center - center_cam)
        Z = center_in_cam[2]
        if abs(Z) < 1e-8:
            Z = 1e-8
        sum_fz2 += (fx / Z) ** 2

    radius = np.sqrt(fg_area_ratio * total_fg_area / (np.pi * sum_fz2))
    if radius < 1e-8:
        radius = 1.0
    scale = sphere_scale / radius

    return scene_center, scale


def compute_scaling_from_cameras(cameras, sphere_scale=0.9):
    """Fallback: compute scaling from camera centers only."""
    centers = np.array([c['center'] for c in cameras])
    return compute_scaling_from_landmarks(centers, sphere_scale)


# =============================================================================
# View matching across multiple SfMData files
# =============================================================================

def match_views_by_id(camera_lists):
    """Find common view IDs across multiple camera lists.

    Args:
        camera_lists: list of camera lists (from load_sfm)

    Returns:
        common_ids: sorted list of common view IDs (as strings)
    """
    id_sets = []
    for cam_list in camera_lists:
        id_sets.append(set(c['view_id'] for c in cam_list))
    common = id_sets[0]
    for s in id_sets[1:]:
        common = common & s
    return sorted(common)


def cameras_by_view_id(camera_list):
    """Build a dict mapping view_id -> camera dict."""
    return {c['view_id']: c for c in camera_list}


# =============================================================================
# Dataset classes
# =============================================================================

class SfMDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True
        self.apply_light_opti = self.config.apply_light_opti

        # Load normal SfMData (required)
        normal_cameras, landmarks = load_sfm(self.config.normal_sfm)

        # Load optional albedo / mask SfMData
        albedo_cameras = None
        if self.config.get('albedo_sfm', ''):
            albedo_cameras, _ = load_sfm(self.config.albedo_sfm)

        mask_cameras = None
        if self.config.get('mask_sfm', ''):
            mask_cameras, _ = load_sfm(self.config.mask_sfm)

        # Match views across SfMData files by viewId
        all_cam_lists = [normal_cameras]
        if albedo_cameras is not None:
            all_cam_lists.append(albedo_cameras)
        if mask_cameras is not None:
            all_cam_lists.append(mask_cameras)

        if len(all_cam_lists) > 1:
            common_ids = match_views_by_id(all_cam_lists)
        else:
            common_ids = sorted(c['view_id'] for c in normal_cameras)

        if not common_ids:
            raise ValueError(
                "No common views found across SfMData files. "
                "Check that viewIds match between normal_sfm, albedo_sfm, and mask_sfm."
            )

        normal_by_id = cameras_by_view_id(normal_cameras)
        albedo_by_id = cameras_by_view_id(albedo_cameras) if albedo_cameras else {}
        mask_by_id = cameras_by_view_id(mask_cameras) if mask_cameras else {}

        # Determine image dimensions from first normal view (all views must match)
        first_cam = normal_by_id[common_ids[0]]
        W_orig = first_cam['width']
        H_orig = first_cam['height']
        for vid in common_ids[1:]:
            cam = normal_by_id[vid]
            if cam['width'] != W_orig or cam['height'] != H_orig:
                raise ValueError(
                    f"View {vid} has resolution {cam['width']}x{cam['height']} "
                    f"but expected {W_orig}x{H_orig} (from first view). "
                    "All views must have the same resolution."
                )

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
        elif 'img_downscale' in self.config:
            w, h = int(W_orig / self.config.img_downscale), int(H_orig / self.config.img_downscale)
        else:
            w, h = W_orig, H_orig

        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)
        self.factor = w / W_orig

        # Compute scene scaling
        scaling_mode = self.config.get('scaling_mode', 'auto')
        sphere_scale = self.config.get('sphere_scale', 0.9)

        if scaling_mode == 'landmarks' or (scaling_mode == 'auto' and landmarks is not None and len(landmarks) > 0):
            scene_center, scene_scale = compute_scaling_from_landmarks(landmarks, sphere_scale)
        elif scaling_mode == 'none':
            scene_center = np.zeros(3)
            scene_scale = 1.0
        else:
            # silhouettes mode or auto without landmarks — need masks first
            # Will compute after loading masks below
            scene_center = None
            scene_scale = None

        # Load images, masks, normals per view
        self.all_c2w = []
        self.all_images = []
        self.all_fg_masks = []
        self.all_normals = []
        self.directions = []

        # Store metadata for albedo scaling
        self.albedo_paths = []
        self.camera_Ks = []
        self.camera_c2ws = []  # c2w after Y/Z flip, before scaling (world coords)

        loaded_masks_for_scaling = []
        loaded_cameras_for_scaling = []

        for vid in common_ids:
            cam = normal_by_id[vid]
            fx = cam['fx'] * self.factor
            fy = cam['fy'] * self.factor
            cx = cam['cx'] * self.factor
            cy = cam['cy'] * self.factor

            directions = get_ray_directions(w, h, fx, fy, cx, cy)
            self.directions.append(directions)

            # Build c2w from R_cam2world + center
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = cam['R_cam2world']
            c2w[:3, 3] = cam['center']
            c2w = torch.from_numpy(c2w).float()

            # Apply Y/Z flip to match NeuS convention
            c2w_flipped = c2w.clone()
            c2w_flipped[:3, 1:3] *= -1.

            # Store UNFLIPPED c2w for albedo scaling — mesh is exported in original
            # world coords (inverse scaling undoes the flip), so rays must match.
            self.camera_c2ws.append(c2w[:4, :4].numpy())
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            self.camera_Ks.append(K)

            # Load normal image
            normal_path = cam['image_path']
            normals = Image.open(normal_path)
            normals = normals.resize(self.img_wh, Image.BICUBIC)
            normals = TF.to_tensor(normals).permute(1, 2, 0)  # (H, W, C)
            normals = normals.float() * 2.0 - 1.0
            normals = normals[:, :, :3]
            normals = torch.nn.functional.normalize(normals, p=2, dim=-1)
            # Convert normals to world space
            normals = torch.matmul(normals, c2w_flipped[:3, :3].T)

            # Load albedo image
            if vid in albedo_by_id:
                albedo_path = albedo_by_id[vid]['image_path']
                self.albedo_paths.append(albedo_path)
                img = Image.open(albedo_path)
                img = img.resize(self.img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0)[..., :3]
            else:
                self.albedo_paths.append(None)
                img = torch.zeros(h, w, 3)

            # Load mask
            if vid in mask_by_id:
                mask_path = mask_by_id[vid]['image_path']
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize(self.img_wh, Image.BICUBIC)
                mask = TF.to_tensor(mask)[0]
            else:
                mask = torch.ones(h, w)

            # Apply mask to normals and images
            boolean_mask = mask > 0.5
            boolean_mask_expanded = boolean_mask.unsqueeze(-1).expand_as(normals)
            normals[~boolean_mask_expanded] = 0.0
            img[~boolean_mask_expanded] = 0.0

            self.all_fg_masks.append(mask)
            self.all_images.append(img[..., :3])
            self.all_normals.append(normals)

            # Store mask + cam data for silhouette scaling
            loaded_masks_for_scaling.append(mask.numpy())
            loaded_cameras_for_scaling.append(cam)

            # Store unscaled c2w (will apply scaling after all views loaded)
            self.all_c2w.append(c2w_flipped[:3, :4])

        # Compute silhouette-based scaling if needed
        if scene_center is None:
            fg_area_ratio = self.config.get('fg_area_ratio', 5)
            scene_center, scene_scale = compute_scaling_from_silhouettes(
                loaded_cameras_for_scaling, loaded_masks_for_scaling,
                sphere_scale, fg_area_ratio
            )

        # Apply scaling to camera positions
        scene_center_t = torch.tensor(scene_center, dtype=torch.float32)
        for i in range(len(self.all_c2w)):
            self.all_c2w[i][:3, 3] = scene_scale * (self.all_c2w[i][:3, 3] - scene_center_t)

        # Store scaling info for mesh export + albedo scaling
        self.scale_factor = scene_scale
        self.scene_center = scene_center

        # Subsample views if requested
        if split == 'train' and self.config.get('num_views', False):
            num_views = self.config.num_views
            jump = len(self.all_c2w) // num_views
            if jump == 0:
                jump = 1
            self.all_c2w = self.all_c2w[::jump]
            self.all_images = self.all_images[::jump]
            self.all_fg_masks = self.all_fg_masks[::jump]
            self.all_normals = self.all_normals[::jump]
            self.albedo_paths = self.albedo_paths[::jump]
            self.camera_Ks = self.camera_Ks[::jump]
            self.camera_c2ws = self.camera_c2ws[::jump]

        # Stack into tensors
        self.all_c2w = torch.stack(self.all_c2w, dim=0).float().to(self.rank)
        self.all_images = torch.stack(self.all_images, dim=0).float().to(self.rank)
        self.all_fg_masks = torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
        self.directions = torch.stack(self.directions, dim=0).float().to(self.rank)
        self.all_normals = torch.stack(self.all_normals, dim=0).float().to(self.rank)

        if self.split == 'test':
            self.test_render_combinations = []
            num_base_images = len(self.all_images)
            num_light_conditions = 3
            for img_idx in range(num_base_images):
                for light_idx in range(num_light_conditions):
                    self.test_render_combinations.append({'image_idx': img_idx, 'light_idx': light_idx})
            print(f"Test split: Prepared {len(self.test_render_combinations)} image-light combinations for rendering.")

    def update_albedos(self, scaled_images_tensor):
        """Called by two-phase training to replace albedos in-place."""
        self.all_images = scaled_images_tensor.to(self.all_images.device)


class SfMDataset(Dataset, SfMDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        if self.split == 'test':
            return len(self.test_render_combinations)
        return len(self.all_images)

    def __getitem__(self, index):
        if self.split == 'test':
            combination = self.test_render_combinations[index]
            return {
                'index': torch.tensor(combination['image_idx'], dtype=torch.long),
                'index_light': torch.tensor(combination['light_idx'], dtype=torch.long),
            }
        else:
            return {
                'index': torch.tensor(index, dtype=torch.long)
            }


class SfMIterableDataset(IterableDataset, SfMDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('sfm')
class SfMDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        # IDEMPOTENT: skip re-creation if datasets already exist.
        # Critical for two-phase training — stage 2's trainer.fit()
        # calls setup() again, which must NOT overwrite scaled albedos.
        if stage in [None, 'fit'] and not hasattr(self, 'train_dataset'):
            self.train_dataset = SfMIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate'] and not hasattr(self, 'val_dataset'):
            self.val_dataset = SfMDataset(self.config, self.config.val_split)
        if stage in [None, 'test'] and not hasattr(self, 'test_dataset'):
            self.test_dataset = SfMDataset(self.config, self.config.test_split)
        if stage in [None, 'predict'] and not hasattr(self, 'predict_dataset'):
            self.predict_dataset = SfMDataset(self.config, self.config.train_split)

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
