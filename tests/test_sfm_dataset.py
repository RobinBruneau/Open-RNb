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
        from datasets.sfm import load_sfm_json, compute_scaling_from_cameras
        cameras, _ = load_sfm_json(NORMAL_SFM)
        center, scale = compute_scaling_from_cameras(cameras, sphere_scale=0.9)
        assert center.shape == (3,)
        assert scale > 0

    def test_scaling_from_landmarks(self):
        from datasets.sfm import compute_scaling_from_landmarks
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        center, scale = compute_scaling_from_landmarks(pts, sphere_scale=0.9)
        np.testing.assert_allclose(center, [0, 0, 0], atol=0.1)
        assert scale > 0

    def test_scaling_from_silhouettes(self):
        from datasets.sfm import load_sfm_json, compute_scaling_from_silhouettes
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
        assert len(ds.camera_c2ws) == 3
        for K in ds.camera_Ks:
            assert K.shape == (3, 3)
        for c2w in ds.camera_c2ws:
            assert c2w.shape == (4, 4)

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
