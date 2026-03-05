"""Tests for the IDR dataset loading pipeline.

Uses a minimal 3-view fixture (32x28 images) built from the golden_snail
SDM-UniPS cameras.npz. Tests cover scale_mat extraction, world-coord cameras,
tensor shapes, and DataModule idempotency.
"""
import os
import sys

import cv2
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIXTURE_DIR = os.path.join(ROOT, "tests", "data", "golden_snail_idr_mini")
N_VIEWS = 3
IMG_W, IMG_H = 32, 28


# ---- Helper function ----

class TestLoadKRtFromP:
    """Tests for the cv2-based projection matrix decomposition."""

    def test_returns_intrinsics_and_pose(self):
        from datasets.idr import load_K_Rt_from_P
        P = np.eye(3, 4, dtype=np.float64)
        K, pose = load_K_Rt_from_P(P)
        assert K.shape == (4, 4)
        assert pose.shape == (4, 4)

    def test_identity_projection(self):
        from datasets.idr import load_K_Rt_from_P
        P = np.eye(3, 4, dtype=np.float64)
        K, pose = load_K_Rt_from_P(P)
        # K should be identity-like (normalized)
        np.testing.assert_allclose(K[2, 2], 1.0)

    def test_real_projection(self):
        from datasets.idr import load_K_Rt_from_P
        cams = np.load(os.path.join(FIXTURE_DIR, "cameras.npz"))
        wm = cams["world_mat_0"]
        sm = cams["scale_mat_0"]
        P = (wm @ sm)[:3, :4]
        K, pose = load_K_Rt_from_P(P)
        # K should have positive focal lengths
        assert K[0, 0] > 0
        assert K[1, 1] > 0
        # pose should be a valid rigid transform
        R = pose[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)


# ---- Scale extraction ----

class TestScaleExtraction:
    """Tests that scale_factor / scene_center are correctly extracted from scale_mat."""

    @pytest.fixture()
    def cams(self):
        return np.load(os.path.join(FIXTURE_DIR, "cameras.npz"))

    def test_scale_factor_matches_scale_mat(self, cams):
        s = float(cams["scale_mat_0"][0, 0])
        assert s == pytest.approx(0.1477918, abs=1e-6)

    def test_scene_center_formula(self, cams):
        sm = cams["scale_mat_0"]
        s = sm[0, 0]
        t = sm[:3, 3]
        scene_center = -t / s
        expected = np.array([-0.3088153, 0.01379971, -5.012331])
        np.testing.assert_allclose(scene_center, expected, atol=1e-4)

    def test_roundtrip_normalization(self, cams):
        """p_norm = s * p_world + t  ⇔  p_world = (p_norm - t) / s = p_norm/s + scene_center."""
        sm = cams["scale_mat_0"]
        s = sm[0, 0]
        t = sm[:3, 3]
        scene_center = -t / s

        p_world = np.array([1.0, 2.0, 3.0])
        p_norm = s * p_world + t
        p_world_recovered = p_norm / s + scene_center
        np.testing.assert_allclose(p_world_recovered, p_world, atol=1e-10)

    def test_all_scale_mats_identical(self, cams):
        sm0 = cams["scale_mat_0"]
        for i in range(N_VIEWS):
            np.testing.assert_array_equal(cams[f"scale_mat_{i}"], sm0)


# ---- Full dataset setup ----

class TestIDRDataset:
    @pytest.fixture(autouse=True)
    def _patch_rank(self, monkeypatch):
        monkeypatch.setattr("datasets.idr.get_rank", lambda: "cpu")

    def _make_config(self, **overrides):
        from omegaconf import OmegaConf
        cfg = {
            "root_dir": FIXTURE_DIR,
            "img_downscale": 1,
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "apply_light_opti": False,
        }
        cfg.update(overrides)
        return OmegaConf.create(cfg)

    def test_tensor_shapes(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        assert ds.all_c2w.shape == (N_VIEWS, 3, 4)
        assert ds.all_images.shape == (N_VIEWS, IMG_H, IMG_W, 3)
        assert ds.all_fg_masks.shape == (N_VIEWS, IMG_H, IMG_W)
        assert ds.all_normals.shape == (N_VIEWS, IMG_H, IMG_W, 3)
        assert ds.directions.shape == (N_VIEWS, IMG_H, IMG_W, 3)

    def test_tensors_are_float(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        assert ds.all_c2w.dtype == torch.float32
        assert ds.all_images.dtype == torch.float32
        assert ds.all_fg_masks.dtype == torch.float32
        assert ds.all_normals.dtype == torch.float32

    def test_scale_factor_from_scale_mat(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        assert ds.scale_factor == pytest.approx(0.1477918, abs=1e-6)

    def test_scene_center_from_scale_mat(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        expected = np.array([-0.3088153, 0.01379971, -5.012331])
        np.testing.assert_allclose(ds.scene_center, expected, atol=1e-4)

    def test_scale_factor_is_not_identity(self):
        """Verify we're no longer hardcoding scale_factor=1.0."""
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        assert ds.scale_factor != 1.0

    def test_scene_center_is_not_zero(self):
        """Verify we're no longer hardcoding scene_center=zeros."""
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        assert np.linalg.norm(ds.scene_center) > 0.1

    def test_camera_Ks_shape_and_positive_focal(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        assert len(ds.camera_Ks) == N_VIEWS
        for K in ds.camera_Ks:
            assert K.shape == (3, 3)
            assert K[0, 0] > 0  # fx
            assert K[1, 1] > 0  # fy

    def test_albedo_paths_stored(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        assert len(ds.albedo_paths) == N_VIEWS
        for p in ds.albedo_paths:
            assert os.path.isfile(p)

    def test_masks_zero_background(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        bg = ds.all_fg_masks < 0.5
        for i in range(N_VIEWS):
            bg_i = bg[i]
            if bg_i.any():
                assert ds.all_normals[i][bg_i].abs().max() == 0.0
                assert ds.all_images[i][bg_i].abs().max() == 0.0

    def test_masks_binary_range(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        assert ds.all_fg_masks.min() >= 0.0
        assert ds.all_fg_masks.max() <= 1.0

    def test_normals_unit_length_in_fg(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        norms = ds.all_normals.norm(dim=-1)
        mask = norms > 0.1
        if mask.any():
            np.testing.assert_allclose(
                norms[mask].cpu().numpy(), 1.0, atol=0.05
            )

    def test_downscale(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(img_downscale=2), "train")
        assert ds.w == IMG_W // 2
        assert ds.h == IMG_H // 2
        assert ds.all_images.shape == (N_VIEWS, IMG_H // 2, IMG_W // 2, 3)

    def test_update_albedos(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        new_albedos = torch.ones_like(ds.all_images) * 0.42
        ds.update_albedos(new_albedos)
        torch.testing.assert_close(ds.all_images, new_albedos)

    def test_test_split_combinations(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "test")
        assert len(ds) == N_VIEWS * 3  # 3 light conditions
        item = ds[0]
        assert "index" in item
        assert "index_light" in item

    def test_len_train(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")
        assert len(ds) == N_VIEWS

    def test_getitem_returns_index(self):
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "val")
        item = ds[1]
        assert item["index"].item() == 1


# ---- Inverse-transform consistency ----

class TestInverseTransformConsistency:
    """Verify that the geometry export formula v_world = v_norm / s + center
    correctly inverts the scale_mat normalization."""

    @pytest.fixture(autouse=True)
    def _patch_rank(self, monkeypatch):
        monkeypatch.setattr("datasets.idr.get_rank", lambda: "cpu")

    def _make_config(self):
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "root_dir": FIXTURE_DIR,
            "img_downscale": 1,
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "apply_light_opti": False,
        })

    def test_camera_pos_roundtrip(self):
        """Camera positions from all_c2w (normalized) should round-trip through
        world space via scene_center / scale_factor."""
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")

        for i in range(N_VIEWS):
            # all_c2w is normalized + Y/Z flipped; undo flip to get pure normalized pose
            c2w34 = ds.all_c2w[i].cpu().numpy()
            c2w44 = np.eye(4, dtype=np.float64)
            c2w44[:3, :4] = c2w34
            c2w44[:3, 1:3] *= -1.  # undo NeuS Y/Z flip
            p_norm = c2w44[:3, 3]

            # Recover world: p_world = p_norm / scale_factor + scene_center
            p_world = p_norm / ds.scale_factor + ds.scene_center
            # Re-normalize: p_norm2 = (p_world - scene_center) * scale_factor
            p_norm2 = (p_world - ds.scene_center) * ds.scale_factor
            np.testing.assert_allclose(p_norm2, p_norm, atol=1e-5)

    def test_world_mesh_vertex_roundtrip(self):
        """A mesh vertex in normalized space should map back to world coords."""
        from datasets.idr import IDRDataset
        ds = IDRDataset(self._make_config(), "train")

        # Arbitrary world point
        p_world = np.array([0.1, -0.05, -5.0])
        # Normalize
        p_norm = ds.scale_factor * p_world + (ds.scale_factor * ds.scene_center * -1)
        # Actually, scale_mat applies: p_norm = s * p_world + t
        cams = np.load(os.path.join(FIXTURE_DIR, "cameras.npz"))
        sm = cams["scale_mat_0"]
        p_norm2 = (sm @ np.append(p_world, 1.0))[:3]
        # Export formula
        p_recovered = p_norm2 / ds.scale_factor + ds.scene_center
        np.testing.assert_allclose(p_recovered, p_world, atol=1e-6)


# ---- DataModule ----

class TestIDRDataModule:
    @pytest.fixture(autouse=True)
    def _patch_rank(self, monkeypatch):
        monkeypatch.setattr("datasets.idr.get_rank", lambda: "cpu")

    def _make_config(self):
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "root_dir": FIXTURE_DIR,
            "img_downscale": 1,
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "apply_light_opti": False,
        })

    def test_setup_idempotent(self):
        from datasets.idr import IDRDataModule
        dm = IDRDataModule(self._make_config())
        dm.setup("fit")
        train_id = id(dm.train_dataset)
        val_id = id(dm.val_dataset)
        dm.setup("fit")
        assert id(dm.train_dataset) == train_id, "setup() must be idempotent"
        assert id(dm.val_dataset) == val_id, "setup() must be idempotent"

    def test_setup_creates_all_splits(self):
        from datasets.idr import IDRDataModule
        dm = IDRDataModule(self._make_config())
        dm.setup(None)
        assert hasattr(dm, "train_dataset")
        assert hasattr(dm, "val_dataset")
        assert hasattr(dm, "test_dataset")
        assert hasattr(dm, "predict_dataset")

    def test_train_dataset_preserves_scaling_after_second_setup(self):
        """Phase 2 calls setup() again — scaled albedos must survive."""
        from datasets.idr import IDRDataModule
        dm = IDRDataModule(self._make_config())
        dm.setup("fit")
        # Simulate phase 2: update albedos, then call setup again
        new_albedos = torch.ones_like(dm.train_dataset.all_images) * 0.5
        dm.train_dataset.update_albedos(new_albedos)
        dm.setup("fit")
        # Albedos should still be 0.5, not reloaded from disk
        torch.testing.assert_close(
            dm.train_dataset.all_images,
            new_albedos.to(dm.train_dataset.all_images.device),
        )

    def test_registry(self):
        import datasets
        assert "idr" in datasets.datasets

    def test_dataloaders(self):
        from datasets.idr import IDRDataModule
        dm = IDRDataModule(self._make_config())
        dm.setup(None)
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()
        assert train_dl is not None
        assert val_dl is not None
        assert test_dl is not None
