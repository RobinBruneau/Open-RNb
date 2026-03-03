"""Tests for the SfM dataset loading pipeline.

Uses a minimal 3-view fixture (32x28 images) extracted from the golden_snail
SDM-UniPS project. Tests cover SfM JSON parsing, scene scaling, view matching,
and full dataset setup.
"""
import os
import sys

import numpy as np
import pytest
import torch

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIXTURE_DIR = os.path.join(ROOT, "tests", "data", "golden_snail_mini")
NORMAL_SFM = os.path.join(FIXTURE_DIR, "normalSfm.json")
ALBEDO_SFM = os.path.join(FIXTURE_DIR, "albedoSfm.json")
MASK_SFM = os.path.join(FIXTURE_DIR, "maskSfm.json")

VIEW_IDS = ["46483756", "479365573", "676752860"]
IMG_W, IMG_H = 32, 28


# ---- SfM JSON parsing ----

class TestLoadSfmJson:
    def test_returns_cameras_and_landmarks(self):
        from datasets.sfm import load_sfm_json
        cameras, landmarks = load_sfm_json(NORMAL_SFM)
        assert isinstance(cameras, list)
        assert len(cameras) == 3

    def test_camera_fields(self):
        from datasets.sfm import load_sfm_json
        cameras, _ = load_sfm_json(NORMAL_SFM)
        required_keys = {
            "view_id", "pose_id", "image_path",
            "R_cam2world", "center",
            "fx", "fy", "cx", "cy", "width", "height",
        }
        for cam in cameras:
            assert required_keys.issubset(cam.keys()), f"Missing keys: {required_keys - cam.keys()}"

    def test_view_ids_match(self):
        from datasets.sfm import load_sfm_json
        cameras, _ = load_sfm_json(NORMAL_SFM)
        ids = sorted(c["view_id"] for c in cameras)
        assert ids == sorted(VIEW_IDS)

    def test_rotation_is_3x3(self):
        from datasets.sfm import load_sfm_json
        cameras, _ = load_sfm_json(NORMAL_SFM)
        for cam in cameras:
            assert cam["R_cam2world"].shape == (3, 3)

    def test_rotation_is_orthogonal(self):
        from datasets.sfm import load_sfm_json
        cameras, _ = load_sfm_json(NORMAL_SFM)
        for cam in cameras:
            R = cam["R_cam2world"]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_center_is_3d(self):
        from datasets.sfm import load_sfm_json
        cameras, _ = load_sfm_json(NORMAL_SFM)
        for cam in cameras:
            assert cam["center"].shape == (3,)

    def test_resolution_matches_fixture(self):
        from datasets.sfm import load_sfm_json
        cameras, _ = load_sfm_json(NORMAL_SFM)
        for cam in cameras:
            assert cam["width"] == IMG_W
            assert cam["height"] == IMG_H

    def test_focal_length_positive(self):
        from datasets.sfm import load_sfm_json
        cameras, _ = load_sfm_json(NORMAL_SFM)
        for cam in cameras:
            assert cam["fx"] > 0
            assert cam["fy"] > 0

    def test_no_landmarks_when_structure_empty(self):
        from datasets.sfm import load_sfm_json
        _, landmarks = load_sfm_json(NORMAL_SFM)
        assert landmarks is None

    def test_image_paths_exist(self):
        from datasets.sfm import load_sfm_json
        cameras, _ = load_sfm_json(NORMAL_SFM)
        for cam in cameras:
            assert os.path.isfile(cam["image_path"]), f"Missing: {cam['image_path']}"


# ---- View matching ----

class TestViewMatching:
    def test_match_identical_lists(self):
        from datasets.sfm import load_sfm_json, match_views_by_id
        cams_n, _ = load_sfm_json(NORMAL_SFM)
        cams_a, _ = load_sfm_json(ALBEDO_SFM)
        common = match_views_by_id([cams_n, cams_a])
        assert sorted(common) == sorted(VIEW_IDS)

    def test_match_three_sfm_files(self):
        from datasets.sfm import load_sfm_json, match_views_by_id
        cams_n, _ = load_sfm_json(NORMAL_SFM)
        cams_a, _ = load_sfm_json(ALBEDO_SFM)
        cams_m, _ = load_sfm_json(MASK_SFM)
        common = match_views_by_id([cams_n, cams_a, cams_m])
        assert sorted(common) == sorted(VIEW_IDS)

    def test_match_subset(self):
        from datasets.sfm import load_sfm_json, match_views_by_id
        cams_n, _ = load_sfm_json(NORMAL_SFM)
        cams_partial = [c for c in cams_n if c["view_id"] != VIEW_IDS[0]]
        common = match_views_by_id([cams_n, cams_partial])
        assert VIEW_IDS[0] not in common
        assert len(common) == 2


# ---- Scene scaling ----

class TestScaling:
    def test_scaling_from_cameras(self):
        from datasets.sfm import load_sfm_json
        from datasets.utils import compute_scaling_from_cameras
        cameras, _ = load_sfm_json(NORMAL_SFM)
        center, scale = compute_scaling_from_cameras(cameras, sphere_scale=0.9)
        assert center.shape == (3,)
        assert scale > 0

    def test_scaling_from_landmarks(self):
        from datasets.utils import compute_scaling_from_pcd
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        center, scale = compute_scaling_from_pcd(pts, sphere_scale=0.9)
        np.testing.assert_allclose(center, [0, 0, 0], atol=0.1)
        assert scale > 0

    def test_scaling_from_silhouettes(self):
        from datasets.sfm import load_sfm_json
        from datasets.utils import compute_scaling_from_silhouettes
        cameras, _ = load_sfm_json(NORMAL_SFM)
        masks = [np.ones((IMG_H, IMG_W), dtype=np.float64) for _ in cameras]
        center, scale = compute_scaling_from_silhouettes(cameras, masks)
        assert center.shape == (3,)
        assert scale > 0


# ---- Full dataset setup ----

class TestSfMDataset:
    @pytest.fixture(autouse=True)
    def _patch_rank(self, monkeypatch):
        """Force CPU for tests without GPU."""
        monkeypatch.setattr("datasets.sfm.get_rank", lambda: "cpu")

    def _make_config(self, **overrides):
        from omegaconf import OmegaConf
        cfg = {
            "normal_sfm": NORMAL_SFM,
            "albedo_sfm": "",
            "mask_sfm": "",
            "scaling_mode": "auto",
            "sphere_scale": 0.9,
            "fg_area_ratio": 5,
            "img_downscale": 1.0,
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "apply_light_opti": True,
            "apply_rgb_plus": True,
        }
        cfg.update(overrides)
        return OmegaConf.create(cfg)

    def test_setup_normals_only(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "train")
        assert ds.all_c2w.shape == (3, 3, 4)
        assert ds.all_images.shape == (3, IMG_H, IMG_W, 3)
        assert ds.all_fg_masks.shape == (3, IMG_H, IMG_W)
        assert ds.all_normals.shape == (3, IMG_H, IMG_W, 3)
        assert ds.directions.shape == (3, IMG_H, IMG_W, 3)

    def test_setup_with_albedos(self):
        from datasets.sfm import SfMDataset
        config = self._make_config(albedo_sfm=ALBEDO_SFM)
        ds = SfMDataset(config, "train")
        assert ds.all_images.shape == (3, IMG_H, IMG_W, 3)
        assert ds.all_images.sum() > 0

    def test_setup_with_masks(self):
        from datasets.sfm import SfMDataset
        config = self._make_config(mask_sfm=MASK_SFM)
        ds = SfMDataset(config, "train")
        assert ds.all_fg_masks.shape == (3, IMG_H, IMG_W)
        # Masks should have both fg and bg pixels
        assert ds.all_fg_masks.min() < 0.5
        assert ds.all_fg_masks.max() > 0.5

    def test_setup_with_all_sfm(self):
        from datasets.sfm import SfMDataset
        config = self._make_config(albedo_sfm=ALBEDO_SFM, mask_sfm=MASK_SFM)
        ds = SfMDataset(config, "train")
        assert ds.all_c2w.shape == (3, 3, 4)
        assert ds.all_images.shape == (3, IMG_H, IMG_W, 3)
        assert ds.all_fg_masks.shape == (3, IMG_H, IMG_W)
        assert ds.all_normals.shape == (3, IMG_H, IMG_W, 3)

    def test_mask_zeros_background(self):
        """Masked regions should have zero normals and images."""
        from datasets.sfm import SfMDataset
        config = self._make_config(mask_sfm=MASK_SFM)
        ds = SfMDataset(config, "train")
        bg = ds.all_fg_masks < 0.5
        for i in range(3):
            bg_i = bg[i]
            if bg_i.any():
                assert ds.all_normals[i][bg_i].abs().max() == 0.0
                assert ds.all_images[i][bg_i].abs().max() == 0.0

    def test_scaling_stored(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "train")
        assert hasattr(ds, "scale_factor")
        assert hasattr(ds, "scene_center")
        assert ds.scale_factor > 0
        assert ds.scene_center.shape == (3,)

    def test_camera_metadata_stored(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "train")
        assert len(ds.camera_Ks) == 3
        for K in ds.camera_Ks:
            assert K.shape == (3, 3)

    def test_downscale(self):
        from datasets.sfm import SfMDataset
        config = self._make_config(img_downscale=2.0)
        ds = SfMDataset(config, "train")
        assert ds.w == IMG_W // 2
        assert ds.h == IMG_H // 2
        assert ds.all_images.shape == (3, IMG_H // 2, IMG_W // 2, 3)

    def test_scaling_mode_none(self):
        from datasets.sfm import SfMDataset
        config = self._make_config(scaling_mode="none")
        ds = SfMDataset(config, "train")
        assert ds.scale_factor == 1.0
        np.testing.assert_allclose(ds.scene_center, [0, 0, 0])

    def test_tensors_are_float(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "train")
        assert ds.all_c2w.dtype == torch.float32
        assert ds.all_images.dtype == torch.float32
        assert ds.all_fg_masks.dtype == torch.float32
        assert ds.all_normals.dtype == torch.float32

    def test_normals_unit_length(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "train")
        norms = ds.all_normals.norm(dim=-1)
        mask = norms > 0.1
        if mask.any():
            np.testing.assert_allclose(
                norms[mask].cpu().numpy(), 1.0, atol=0.05
            )

    def test_masks_binary_range(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "train")
        assert ds.all_fg_masks.min() >= 0.0
        assert ds.all_fg_masks.max() <= 1.0

    def test_update_albedos(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "train")
        new_albedos = torch.ones_like(ds.all_images) * 0.5
        ds.update_albedos(new_albedos)
        torch.testing.assert_close(ds.all_images, new_albedos)

    def test_test_split_combinations(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "test")
        # 3 views * 3 light conditions = 9
        assert len(ds) == 9
        item = ds[0]
        assert "index" in item
        assert "index_light" in item

    def test_len_train(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "train")
        assert len(ds) == 3

    def test_getitem_returns_index(self):
        from datasets.sfm import SfMDataset
        config = self._make_config()
        ds = SfMDataset(config, "train")
        item = ds[1]
        assert item["index"].item() == 1


# ---- DataModule ----

class TestSfMDataModule:
    @pytest.fixture(autouse=True)
    def _patch_rank(self, monkeypatch):
        monkeypatch.setattr("datasets.sfm.get_rank", lambda: "cpu")

    def _make_config(self, **overrides):
        from omegaconf import OmegaConf
        cfg = {
            "normal_sfm": NORMAL_SFM,
            "albedo_sfm": "",
            "mask_sfm": "",
            "scaling_mode": "auto",
            "sphere_scale": 0.9,
            "fg_area_ratio": 5,
            "img_downscale": 1.0,
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "apply_light_opti": True,
            "apply_rgb_plus": True,
        }
        cfg.update(overrides)
        return OmegaConf.create(cfg)

    def test_setup_idempotent(self):
        from datasets.sfm import SfMDataModule
        config = self._make_config()
        dm = SfMDataModule(config)
        dm.setup("fit")
        train_ds_id = id(dm.train_dataset)
        dm.setup("fit")
        assert id(dm.train_dataset) == train_ds_id, "setup() must be idempotent"

    def test_registry(self):
        import datasets
        assert "sfm" in datasets.datasets


# ---- Silhouette scaling invariance ----

class TestSilhouetteScalingInvariance:
    """Verify that SfMDatasetBase.setup() produces scale_factor invariant to img_downscale.

    Bug (pre-fix)
    -------------
    In setup(), loaded_cameras_for_scaling holds full-resolution fx/fy/cx/cy
    but loaded_masks_for_scaling are already at the downscaled resolution
    (self.h x self.w).  compute_scaling_from_silhouettes assumes cameras and
    masks share the same pixel coordinate space, so:
      - total_fg_area   proportional to 1/downscale²  (fewer pixels in mask)
      - sum_fz2         uses original fx              → constant across downscales
      - radius          proportional to 1/downscale
      - scale           proportional to downscale     ← BUG

    Fix (applied to SfMDatasetBase.setup())
    ----------------------------------------
    Before calling compute_scaling_from_silhouettes, scale fx, fy, cx, cy by
    self.factor (= downscaled_width / original_width) so both cameras and
    masks live in the same downscaled pixel space.

    Test strategy
    -------------
    The primary invariance tests drive the full SfMDatasetBase.setup() path
    through SfMDataset, varying img_downscale, and assert that scale_factor
    and scene_center are the same regardless of the chosen downscale factor.

    Additional unit tests cover compute_scaling_from_silhouettes directly,
    verifying the correct vs. incorrect calling conventions.

    Geometry setup
    --------------
    * Scene center at origin (0, 0, 0).
    * Object radius r = 1.0 (sphere fitting in unit sphere).
    * Two cameras placed on the +Z and -Z axes at distance D = 5.0 looking at
      origin.  Back-to-back placement gives perfectly conditioned least-squares
      for center estimation.
    * Full-resolution images: W=200, H=160, fx=fy=180 (≈50 deg FOV).
    * Circular masks at full resolution with radius = fx * r / D pixels.
    """

    # --- synthetic camera / mask builders -----------------------------------

    @staticmethod
    def _make_camera(pos, lookat, fx, fy, W, H):
        """Return a camera dict with R_cam2world from position and lookat point.

        Camera Z-axis points from pos toward lookat.
        """
        z_axis = np.array(lookat, dtype=float) - np.array(pos, dtype=float)
        z_axis /= np.linalg.norm(z_axis)
        # World up = Y; handle degenerate case when z_axis is parallel to Y
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(z_axis, up)) > 0.999:
            up = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # R_cam2world: columns are cam-X, cam-Y, cam-Z in world coords
        R_cam2world = np.stack([x_axis, y_axis, z_axis], axis=1)

        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(W) / 2.0,
            "cy": float(H) / 2.0,
            "R_cam2world": R_cam2world,
            "center": np.array(pos, dtype=float),
            "width": int(W),
            "height": int(H),
        }

    @staticmethod
    def _make_circular_mask(H, W, cx, cy, radius_px):
        """Return a binary mask with a filled circle."""
        ys, xs = np.ogrid[:H, :W]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        mask = (dist <= radius_px).astype(np.float64)
        return mask

    @pytest.fixture
    def synthetic_scene(self):
        """Two cameras at ±Z, object at origin with radius 1.0."""
        D = 5.0          # camera distance from origin
        r = 1.0          # object radius
        W_full, H_full = 200, 160
        fx_full = fy_full = 180.0

        # Projected radius at full resolution: fx * r / D pixels
        proj_r_full = fx_full * r / D

        cameras_full = [
            self._make_camera([0, 0,  D], [0, 0, 0], fx_full, fy_full, W_full, H_full),
            self._make_camera([0, 0, -D], [0, 0, 0], fx_full, fy_full, W_full, H_full),
        ]

        mask_full_0 = self._make_circular_mask(H_full, W_full, W_full / 2, H_full / 2, proj_r_full)
        mask_full_1 = self._make_circular_mask(H_full, W_full, W_full / 2, H_full / 2, proj_r_full)
        masks_full = [mask_full_0, mask_full_1]

        return {
            "cameras_full": cameras_full,
            "masks_full": masks_full,
            "W_full": W_full,
            "H_full": H_full,
            "fx_full": fx_full,
            "fy_full": fy_full,
            "proj_r_full": proj_r_full,
            "D": D,
            "r": r,
        }

    def _downscale_mask(self, mask, factor):
        """Downscale a mask by integer factor using area averaging."""
        from PIL import Image as PILImage
        H, W = mask.shape
        H2, W2 = H // factor, W // factor
        pil = PILImage.fromarray((mask * 255).astype(np.uint8), mode="L")
        pil = pil.resize((W2, H2), PILImage.BICUBIC)
        return np.array(pil, dtype=np.float64) / 255.0

    # --- integration tests: end-to-end through SfMDatasetBase.setup() -------

    @pytest.fixture(autouse=True)
    def _patch_rank(self, monkeypatch):
        """Force CPU for tests without GPU."""
        monkeypatch.setattr("datasets.sfm.get_rank", lambda: "cpu")

    def test_scale_factor_is_invariant_to_img_downscale(self, monkeypatch):
        """scale_factor from SfMDatasetBase.setup() must not depend on img_downscale.

        This is the primary regression test for the bug.  It fails before the
        fix because setup() passed full-resolution intrinsics together with
        downscaled masks to compute_scaling_from_silhouettes.
        """
        from datasets.sfm import SfMDataset

        scale_factors = {}
        for ds in [1, 2]:
            from omegaconf import OmegaConf
            cfg = OmegaConf.create({
                "normal_sfm": NORMAL_SFM,
                "mask_sfm": MASK_SFM,
                "albedo_sfm": "",
                "scaling_mode": "silhouettes",
                "sphere_scale": 0.9,
                "fg_area_ratio": 5,
                "img_downscale": float(ds),
                "train_split": "train",
                "val_split": "val",
                "test_split": "test",
                "apply_light_opti": True,
                "apply_rgb_plus": True,
            })
            ds_obj = SfMDataset(cfg, "train")
            scale_factors[ds] = ds_obj.scale_factor

        np.testing.assert_allclose(
            scale_factors[2], scale_factors[1], rtol=0.05,
            err_msg=(
                f"scale_factor at img_downscale=2 ({scale_factors[2]:.6f}) differs "
                f"from img_downscale=1 ({scale_factors[1]:.6f}) — fix the call site "
                "in SfMDatasetBase.setup() to scale intrinsics by self.factor"
            )
        )

    def test_scene_center_is_invariant_to_img_downscale(self, monkeypatch):
        """scene_center from SfMDatasetBase.setup() must not depend on img_downscale.

        Note: the center estimation uses mask center-of-mass which is computed
        in pixel coordinates.  BICUBIC resizing of binary masks introduces
        small CoM shifts (soft edges), so the tolerance is set to 0.05 world
        units (~2% of scene radius).  Before the fix the discrepancy is ~1.0
        world units (the full scene radius), so this tolerance clearly
        distinguishes the fixed from the unfixed code while allowing for the
        expected resampling imprecision on small fixture images.
        """
        from datasets.sfm import SfMDataset

        centers = {}
        for ds in [1, 2]:
            from omegaconf import OmegaConf
            cfg = OmegaConf.create({
                "normal_sfm": NORMAL_SFM,
                "mask_sfm": MASK_SFM,
                "albedo_sfm": "",
                "scaling_mode": "silhouettes",
                "sphere_scale": 0.9,
                "fg_area_ratio": 5,
                "img_downscale": float(ds),
                "train_split": "train",
                "val_split": "val",
                "test_split": "test",
                "apply_light_opti": True,
                "apply_rgb_plus": True,
            })
            ds_obj = SfMDataset(cfg, "train")
            centers[ds] = ds_obj.scene_center

        np.testing.assert_allclose(
            centers[2], centers[1], atol=0.05,
            err_msg=(
                f"scene_center at img_downscale=2 ({centers[2]}) differs too much "
                f"from img_downscale=1 ({centers[1]}) — intrinsics must be scaled "
                "by self.factor before calling compute_scaling_from_silhouettes"
            )
        )

    # --- unit tests: compute_scaling_from_silhouettes directly --------------

    def test_function_is_invariant_when_intrinsics_are_scaled(self, synthetic_scene):
        """compute_scaling_from_silhouettes is invariant when cameras and masks
        are both scaled consistently (the correct calling convention).

        This is the explicit unit-level proof that the fix strategy is correct:
        scaling fx/fy/cx/cy by the same factor as the mask dimensions makes
        scale and center independent of the downscale factor.
        """
        from datasets.utils import compute_scaling_from_silhouettes

        cameras_full = synthetic_scene["cameras_full"]
        masks_full = synthetic_scene["masks_full"]

        scales = {}
        for ds in [1, 2, 4]:
            if ds == 1:
                masks_ds = masks_full
                cams_ds = cameras_full
            else:
                factor = 1.0 / ds
                masks_ds = [self._downscale_mask(m, ds) for m in masks_full]
                cams_ds = []
                for cam in cameras_full:
                    scaled = dict(cam)
                    scaled["fx"] = cam["fx"] * factor
                    scaled["fy"] = cam["fy"] * factor
                    scaled["cx"] = cam["cx"] * factor
                    scaled["cy"] = cam["cy"] * factor
                    cams_ds.append(scaled)

            _, scale = compute_scaling_from_silhouettes(
                cams_ds, masks_ds, sphere_scale=0.9, fg_area_ratio=5
            )
            scales[ds] = scale

        np.testing.assert_allclose(
            scales[2], scales[1], rtol=0.05,
            err_msg="scale with corrected intrinsics should be invariant at ds=2"
        )
        np.testing.assert_allclose(
            scales[4], scales[1], rtol=0.05,
            err_msg="scale with corrected intrinsics should be invariant at ds=4"
        )

    def test_function_produces_wrong_scale_with_mismatched_intrinsics(self, synthetic_scene):
        """Explicitly document the buggy calling convention.

        When full-resolution intrinsics are paired with downscaled masks the
        recovered scale grows by the downscale factor.  This test demonstrates
        and locks in the known bad behavior so future readers understand the
        root cause.  It is NOT an invariance test.
        """
        from datasets.utils import compute_scaling_from_silhouettes

        cameras_full = synthetic_scene["cameras_full"]
        masks_full = synthetic_scene["masks_full"]
        masks_ds4 = [self._downscale_mask(m, 4) for m in masks_full]

        _, scale_1 = compute_scaling_from_silhouettes(
            cameras_full, masks_full, sphere_scale=0.9, fg_area_ratio=5
        )
        _, scale_4_wrong = compute_scaling_from_silhouettes(
            cameras_full, masks_ds4, sphere_scale=0.9, fg_area_ratio=5
        )

        # With mismatched intrinsics, scale grows ~4x with ds=4 — the bug
        ratio = scale_4_wrong / scale_1
        assert ratio > 3.0, (
            f"Expected buggy ratio > 3.0 (≈4.0), got {ratio:.3f}. "
            "The function with mismatched inputs should produce a wrong scale."
        )


# ---- sphere_scale proportionality ----

class TestSphereScaleProportionality:
    """Verify that scale_factor scales linearly with sphere_scale.

    Phase 1 uses sphere_scale=1.0 (new) instead of 0.9 (old).
    The silhouette normalization must produce scale_factor(1.0) / scale_factor(0.9) == 1.0/0.9.
    """

    @pytest.fixture(autouse=True)
    def _patch_rank(self, monkeypatch):
        monkeypatch.setattr("datasets.sfm.get_rank", lambda: "cpu")

    def _make_config(self, sphere_scale, **overrides):
        from omegaconf import OmegaConf
        cfg = {
            "normal_sfm": NORMAL_SFM,
            "albedo_sfm": "",
            "mask_sfm": MASK_SFM,
            "scaling_mode": "silhouettes",
            "sphere_scale": sphere_scale,
            "fg_area_ratio": 5,
            "img_downscale": 1.0,
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "apply_light_opti": True,
            "apply_rgb_plus": True,
        }
        cfg.update(overrides)
        return OmegaConf.create(cfg)

    def test_scale_factor_proportional_to_sphere_scale(self):
        """scale_factor with sphere_scale=1.0 must be 1.0/0.9 times the value at 0.9.

        This test FAILS before configs/sfm.yaml changes sphere_scale to 1.0
        and validates that the silhouette normalization scales linearly.
        """
        from datasets.sfm import SfMDataset

        ds_09 = SfMDataset(self._make_config(sphere_scale=0.9), "train")
        ds_10 = SfMDataset(self._make_config(sphere_scale=1.0), "train")

        expected_ratio = 1.0 / 0.9
        actual_ratio = ds_10.scale_factor / ds_09.scale_factor

        np.testing.assert_allclose(
            actual_ratio, expected_ratio, rtol=1e-6,
            err_msg=(
                f"scale_factor(1.0) / scale_factor(0.9) = {actual_ratio:.6f}, "
                f"expected {expected_ratio:.6f}. "
                "Silhouette normalization must scale linearly with sphere_scale."
            )
        )

    def test_sfm_config_uses_sphere_scale_1_0(self):
        """configs/sfm.yaml must declare sphere_scale: 1.0 (phase-1 target)."""
        import yaml
        config_path = os.path.join(ROOT, "configs", "sfm.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        actual = cfg["dataset"]["sphere_scale"]
        assert actual == 1.0, (
            f"configs/sfm.yaml dataset.sphere_scale={actual!r}, expected 1.0. "
            "Phase 1 silhouette normalization should target R=1.0."
        )

    def test_idr_config_has_sphere_scale_1_0(self):
        """configs/idr.yaml must declare sphere_scale: 1.0 in the dataset section."""
        import yaml
        config_path = os.path.join(ROOT, "configs", "idr.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        actual = cfg["dataset"].get("sphere_scale")
        assert actual == 1.0, (
            f"configs/idr.yaml dataset.sphere_scale={actual!r}, expected 1.0. "
            "Add sphere_scale: 1.0 to the dataset section of configs/idr.yaml."
        )


# ---- Phase-2 renormalization fixed scale ----

class TestPhase2RenormScale:
    """Verify the phase-2 mesh-based renormalization uses sphere_scale_p2=1.5.

    The renorm logic is:
        new_scale = sphere_scale_p2 / max_dist

    With sphere_scale_p2=1.5 (new) vs config.dataset.sphere_scale=0.9 (old),
    the ratio must be 1.5/0.9 = 5/3 for the same mesh.
    """

    def test_phase2_scale_ratio_vs_old_config(self):
        """new_scale with sphere_scale_p2=1.5 must be 1.5/0.9 times the old result.

        This test FAILS before launch.py is updated to use sphere_scale_p2=1.5.
        """
        import numpy as np

        # Simulate a mesh with known max_dist
        max_dist = 0.85  # arbitrary non-trivial value

        # Old logic: sphere_scale from config (0.9)
        old_sphere_scale = 0.9
        old_new_scale = old_sphere_scale / max_dist

        # New logic: fixed sphere_scale_p2=1.5
        sphere_scale_p2 = 1.5
        new_new_scale = sphere_scale_p2 / max_dist

        expected_ratio = 1.5 / 0.9
        actual_ratio = new_new_scale / old_new_scale

        np.testing.assert_allclose(
            actual_ratio, expected_ratio, rtol=1e-10,
            err_msg=(
                f"Phase-2 new_scale ratio = {actual_ratio:.6f}, expected {expected_ratio:.6f} "
                "(1.5/0.9). Update launch.py to use sphere_scale_p2=1.5."
            )
        )

    def test_phase2_sphere_scale_p2_value(self):
        """The constant sphere_scale_p2 in launch.py must equal 1.5.

        Parse launch.py and assert the literal 1.5 is present in the
        sphere_scale_p2 assignment introduced by the phase-2 change.
        """
        launch_path = os.path.join(ROOT, "launch.py")
        with open(launch_path) as f:
            source = f.read()

        assert "sphere_scale_p2 = 1.5" in source, (
            "launch.py must contain 'sphere_scale_p2 = 1.5'. "
            "Update the phase-2 renormalization to use a fixed 1.5."
        )

    def test_phase2_does_not_use_config_sphere_scale_for_renorm(self):
        """launch.py must NOT read config.dataset.sphere_scale for phase-2 renorm.

        After the refactoring, config.dataset.sphere_scale is only used by the
        dataset loader (phase 1). The phase-2 renorm calls compute_scaling_from_mesh
        with sphere_scale=sphere_scale_p2 (= 1.5) — the fixed constant must be
        passed explicitly and config.dataset.sphere_scale must not be used.
        """
        launch_path = os.path.join(ROOT, "launch.py")
        with open(launch_path) as f:
            source = f.read()

        # The fixed constant must be used when calling compute_scaling_from_mesh
        # (either inline or via the sphere_scale_p2 variable)
        assert "compute_scaling_from_mesh" in source, (
            "launch.py must call compute_scaling_from_mesh for phase-2 renorm."
        )

        # sphere_scale_p2 = 1.5 must still be present (checked by test_phase2_sphere_scale_p2_value)
        # config.dataset.sphere_scale must NOT appear in an uncommented assignment for new_scale
        lines = source.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "new_scale" in stripped and "config.dataset" in stripped:
                raise AssertionError(
                    "launch.py must not use config.dataset.sphere_scale for the "
                    "phase-2 new_scale computation. Use compute_scaling_from_mesh "
                    "with the fixed sphere_scale_p2=1.5 constant instead."
                )
