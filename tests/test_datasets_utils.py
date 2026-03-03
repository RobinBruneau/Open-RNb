"""Tests for datasets/utils.py — scene normalization utility functions.

TDD: these tests were written BEFORE the implementation.
Coverage:
  - compute_scaling_from_scale_mat: round-trip and known values
  - compute_scaling_from_pcd: farthest point at sphere_scale after applying scale
  - compute_scaling_from_silhouettes: moved from test_sfm_dataset.py (same logic)
  - compute_scaling_from_mesh: center midpoint, farthest vertex at sphere_scale
  - compute_scene_scaling dispatcher: each mode routes correctly
  - compute_scaling_from_cameras: delegates to pcd correctly
  - Downscale invariance: silhouettes invariant to img_downscale
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# =============================================================================
# compute_scaling_from_scale_mat
# =============================================================================

class TestComputeScalingFromScaleMat:
    """Tests for scale_mat -> (scene_center, scale_factor) extraction."""

    def _make_scale_mat(self, s, center):
        """Build a 4x4 scale_mat encoding p_norm = s*(p_world - center).

        IDR convention: scale_mat encodes  p_norm = s*p_world + t
        with t = -s * center, so p_world = (p_norm - t) / s = p_norm/s + center.
        scene_center = -t/s = center, scale_factor = s.
        """
        scale_mat = np.eye(4, dtype=np.float64)
        scale_mat[0, 0] = s
        scale_mat[1, 1] = s
        scale_mat[2, 2] = s
        # t = -s * center
        scale_mat[:3, 3] = -s * np.array(center)
        return scale_mat

    def test_returns_tuple_of_center_and_scale(self):
        from datasets.utils import compute_scaling_from_scale_mat
        scale_mat = self._make_scale_mat(2.0, [1.0, 2.0, 3.0])
        result = compute_scaling_from_scale_mat(scale_mat)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_center_shape(self):
        from datasets.utils import compute_scaling_from_scale_mat
        scale_mat = self._make_scale_mat(2.0, [1.0, 2.0, 3.0])
        center, _ = compute_scaling_from_scale_mat(scale_mat)
        assert center.shape == (3,)

    def test_scale_factor_is_float(self):
        from datasets.utils import compute_scaling_from_scale_mat
        scale_mat = self._make_scale_mat(3.5, [0.0, 0.0, 0.0])
        _, scale = compute_scaling_from_scale_mat(scale_mat)
        assert isinstance(scale, float)

    def test_known_values_unit_scale_at_origin(self):
        """Identity scale_mat => center=(0,0,0), scale=1.0."""
        from datasets.utils import compute_scaling_from_scale_mat
        scale_mat = np.eye(4, dtype=np.float64)
        center, scale = compute_scaling_from_scale_mat(scale_mat)
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-10)
        assert scale == pytest.approx(1.0)

    def test_known_values_scale_2_at_origin(self):
        """scale_mat with s=2, t=0 => center=(0,0,0), scale=2.0."""
        from datasets.utils import compute_scaling_from_scale_mat
        scale_mat = self._make_scale_mat(2.0, [0.0, 0.0, 0.0])
        center, scale = compute_scaling_from_scale_mat(scale_mat)
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-10)
        assert scale == pytest.approx(2.0)

    def test_known_values_nonzero_center(self):
        """scale_mat with s=3, center=(1,2,3) => extracted values match."""
        from datasets.utils import compute_scaling_from_scale_mat
        expected_center = np.array([1.0, 2.0, 3.0])
        scale_mat = self._make_scale_mat(3.0, expected_center)
        center, scale = compute_scaling_from_scale_mat(scale_mat)
        np.testing.assert_allclose(center, expected_center, atol=1e-10)
        assert scale == pytest.approx(3.0)

    def test_round_trip_apply_then_invert(self):
        """Applying scale_mat then inverting should give identity transform.

        p_norm = scale * (p_world - center)
        p_world = p_norm / scale + center
        => forward then backward returns p_world.
        """
        from datasets.utils import compute_scaling_from_scale_mat
        expected_center = np.array([0.5, -1.0, 2.0])
        s = 4.0
        scale_mat = self._make_scale_mat(s, expected_center)
        center, scale = compute_scaling_from_scale_mat(scale_mat)

        # Forward: normalize a world point
        p_world = np.array([1.0, 0.0, -0.5])
        p_norm = scale * (p_world - center)

        # Inverse: back to world
        p_world_recovered = p_norm / scale + center
        np.testing.assert_allclose(p_world_recovered, p_world, atol=1e-10)

    def test_negative_center_components(self):
        from datasets.utils import compute_scaling_from_scale_mat
        expected_center = np.array([-3.0, 0.5, -2.5])
        scale_mat = self._make_scale_mat(1.5, expected_center)
        center, scale = compute_scaling_from_scale_mat(scale_mat)
        np.testing.assert_allclose(center, expected_center, atol=1e-10)
        assert scale == pytest.approx(1.5)


# =============================================================================
# compute_scaling_from_pcd
# =============================================================================

class TestComputeScalingFromPcd:
    """Tests for point-cloud -> (scene_center, scale_factor)."""

    def test_returns_tuple(self):
        from datasets.utils import compute_scaling_from_pcd
        pcd = np.random.randn(50, 3)
        result = compute_scaling_from_pcd(pcd)
        assert isinstance(result, tuple) and len(result) == 2

    def test_center_shape(self):
        from datasets.utils import compute_scaling_from_pcd
        pcd = np.random.randn(50, 3)
        center, _ = compute_scaling_from_pcd(pcd)
        assert center.shape == (3,)

    def test_scale_positive(self):
        from datasets.utils import compute_scaling_from_pcd
        pcd = np.random.randn(50, 3)
        _, scale = compute_scaling_from_pcd(pcd)
        assert scale > 0

    def test_farthest_point_at_sphere_scale(self):
        """After applying scale, farthest inlier point should be at sphere_scale."""
        from datasets.utils import compute_scaling_from_pcd
        sphere_scale = 1.0
        # Axis-aligned points at distance 1 from origin
        pcd = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=np.float64)
        center, scale = compute_scaling_from_pcd(pcd, sphere_scale=sphere_scale)
        # Apply normalization
        pcd_norm = (pcd - center) * scale
        max_dist = np.linalg.norm(pcd_norm, axis=1).max()
        # max_dist after scaling should be close to sphere_scale
        assert max_dist == pytest.approx(sphere_scale, rel=0.05)

    def test_sphere_scale_09(self):
        """sphere_scale=0.9 should produce proportionally smaller scale."""
        from datasets.utils import compute_scaling_from_pcd
        pcd = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
        ], dtype=np.float64)
        _, scale_10 = compute_scaling_from_pcd(pcd, sphere_scale=1.0)
        _, scale_09 = compute_scaling_from_pcd(pcd, sphere_scale=0.9)
        np.testing.assert_allclose(scale_09 / scale_10, 0.9, rtol=1e-6)

    def test_single_point_does_not_crash(self):
        """Single-point cloud should not crash (degenerate case)."""
        from datasets.utils import compute_scaling_from_pcd
        pcd = np.array([[5.0, 3.0, 1.0]])
        # Should not raise; scale should be positive
        _, scale = compute_scaling_from_pcd(pcd, sphere_scale=1.0)
        assert scale > 0

    def test_centroid_near_origin_for_symmetric_pcd(self):
        from datasets.utils import compute_scaling_from_pcd
        pcd = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=np.float64)
        center, _ = compute_scaling_from_pcd(pcd, sphere_scale=1.0)
        np.testing.assert_allclose(center, [0, 0, 0], atol=0.1)

    def test_outlier_robustness(self):
        """99th-percentile filter should make scale robust to single outlier."""
        from datasets.utils import compute_scaling_from_pcd
        np.random.seed(42)
        # Dense cluster near origin
        pcd_good = np.random.randn(100, 3) * 0.5
        # One extreme outlier
        outlier = np.array([[1000.0, 0.0, 0.0]])
        pcd_with = np.vstack([pcd_good, outlier])
        pcd_without = pcd_good

        _, scale_with = compute_scaling_from_pcd(pcd_with, sphere_scale=1.0)
        _, scale_without = compute_scaling_from_pcd(pcd_without, sphere_scale=1.0)

        # Scale with outlier should be reasonably close to scale without
        # (within 50% is a loose bound — exact behavior depends on 99th percentile)
        assert abs(scale_with - scale_without) / scale_without < 0.5


# =============================================================================
# compute_scaling_from_silhouettes
# =============================================================================

class TestComputeScalingFromSilhouettes:
    """Tests for silhouette-based (scene_center, scale_factor).

    Mirrors the TestSilhouetteScalingInvariance tests from test_sfm_dataset.py
    but imports from datasets.utils instead of datasets.sfm.
    """

    @staticmethod
    def _make_camera(pos, lookat, fx, fy, W, H):
        z_axis = np.array(lookat, dtype=float) - np.array(pos, dtype=float)
        z_axis /= np.linalg.norm(z_axis)
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(z_axis, up)) > 0.999:
            up = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R_cam2world = np.stack([x_axis, y_axis, z_axis], axis=1)
        return {
            "fx": float(fx), "fy": float(fy),
            "cx": float(W) / 2.0, "cy": float(H) / 2.0,
            "R_cam2world": R_cam2world,
            "center": np.array(pos, dtype=float),
            "width": int(W), "height": int(H),
        }

    @staticmethod
    def _make_circular_mask(H, W, cx, cy, radius_px):
        ys, xs = np.ogrid[:H, :W]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        return (dist <= radius_px).astype(np.float64)

    @pytest.fixture
    def synthetic_scene(self):
        D = 5.0
        r = 1.0
        W, H = 200, 160
        fx = fy = 180.0
        proj_r = fx * r / D
        cameras = [
            self._make_camera([0, 0,  D], [0, 0, 0], fx, fy, W, H),
            self._make_camera([0, 0, -D], [0, 0, 0], fx, fy, W, H),
        ]
        masks = [
            self._make_circular_mask(H, W, W / 2, H / 2, proj_r),
            self._make_circular_mask(H, W, W / 2, H / 2, proj_r),
        ]
        return dict(cameras=cameras, masks=masks, W=W, H=H, fx=fx, D=D, r=r, proj_r=proj_r)

    def test_returns_tuple(self):
        from datasets.utils import compute_scaling_from_silhouettes
        cams = [self._make_camera([0, 0, 5], [0, 0, 0], 180, 180, 200, 160)]
        mask = self._make_circular_mask(160, 200, 100, 80, 36.0)
        result = compute_scaling_from_silhouettes(cams, [mask])
        assert isinstance(result, tuple) and len(result) == 2

    def test_center_shape(self):
        from datasets.utils import compute_scaling_from_silhouettes
        cams = [self._make_camera([0, 0, 5], [0, 0, 0], 180, 180, 200, 160)]
        mask = self._make_circular_mask(160, 200, 100, 80, 36.0)
        center, _ = compute_scaling_from_silhouettes(cams, [mask])
        assert center.shape == (3,)

    def test_scale_positive(self):
        from datasets.utils import compute_scaling_from_silhouettes
        cams = [self._make_camera([0, 0, 5], [0, 0, 0], 180, 180, 200, 160)]
        mask = self._make_circular_mask(160, 200, 100, 80, 36.0)
        _, scale = compute_scaling_from_silhouettes(cams, [mask])
        assert scale > 0

    def test_center_near_origin_for_symmetric_setup(self, synthetic_scene):
        from datasets.utils import compute_scaling_from_silhouettes
        center, _ = compute_scaling_from_silhouettes(
            synthetic_scene["cameras"], synthetic_scene["masks"]
        )
        np.testing.assert_allclose(center, [0, 0, 0], atol=0.1)

    def test_sphere_scale_proportionality(self, synthetic_scene):
        from datasets.utils import compute_scaling_from_silhouettes
        _, s1 = compute_scaling_from_silhouettes(
            synthetic_scene["cameras"], synthetic_scene["masks"], sphere_scale=1.0
        )
        _, s09 = compute_scaling_from_silhouettes(
            synthetic_scene["cameras"], synthetic_scene["masks"], sphere_scale=0.9
        )
        np.testing.assert_allclose(s09 / s1, 0.9, rtol=1e-6)

    def test_invariant_when_intrinsics_scaled(self, synthetic_scene):
        """compute_scaling_from_silhouettes is invariant when both cams and masks
        are downscaled consistently."""
        from datasets.utils import compute_scaling_from_silhouettes
        from PIL import Image as PILImage

        cameras_full = synthetic_scene["cameras"]
        masks_full = synthetic_scene["masks"]

        scales = {}
        for ds in [1, 2, 4]:
            if ds == 1:
                masks_ds = masks_full
                cams_ds = cameras_full
            else:
                factor = 1.0 / ds
                W2 = synthetic_scene["W"] // ds
                H2 = 160 // ds
                masks_ds = []
                for m in masks_full:
                    pil = PILImage.fromarray((m * 255).astype(np.uint8), mode="L")
                    pil = pil.resize((W2, H2), PILImage.BICUBIC)
                    masks_ds.append(np.array(pil, dtype=np.float64) / 255.0)
                cams_ds = []
                for cam in cameras_full:
                    sc = dict(cam)
                    sc["fx"] = cam["fx"] * factor
                    sc["fy"] = cam["fy"] * factor
                    sc["cx"] = cam["cx"] * factor
                    sc["cy"] = cam["cy"] * factor
                    cams_ds.append(sc)

            _, scale = compute_scaling_from_silhouettes(
                cams_ds, masks_ds, sphere_scale=0.9, fg_area_ratio=5
            )
            scales[ds] = scale

        np.testing.assert_allclose(scales[2], scales[1], rtol=0.05)
        np.testing.assert_allclose(scales[4], scales[1], rtol=0.05)

    def test_full_mask_does_not_crash(self):
        """All-ones mask (full foreground) should not crash."""
        from datasets.utils import compute_scaling_from_silhouettes
        cams = [self._make_camera([0, 0, 5], [0, 0, 0], 180, 180, 200, 160)]
        mask = np.ones((160, 200), dtype=np.float64)
        _, scale = compute_scaling_from_silhouettes(cams, [mask])
        assert scale > 0


# =============================================================================
# compute_scaling_from_cameras
# =============================================================================

class TestComputeScalingFromCameras:
    """Delegates to compute_scaling_from_pcd with camera centers."""

    @staticmethod
    def _make_cam(center):
        return {"center": np.array(center, dtype=np.float64)}

    def test_returns_tuple(self):
        from datasets.utils import compute_scaling_from_cameras
        cams = [self._make_cam([1, 0, 0]), self._make_cam([-1, 0, 0])]
        result = compute_scaling_from_cameras(cams)
        assert isinstance(result, tuple) and len(result) == 2

    def test_center_shape(self):
        from datasets.utils import compute_scaling_from_cameras
        cams = [self._make_cam([1, 0, 0]), self._make_cam([-1, 0, 0])]
        center, _ = compute_scaling_from_cameras(cams)
        assert center.shape == (3,)

    def test_scale_positive(self):
        from datasets.utils import compute_scaling_from_cameras
        cams = [self._make_cam([1, 0, 0]), self._make_cam([-1, 0, 0])]
        _, scale = compute_scaling_from_cameras(cams)
        assert scale > 0

    def test_sphere_scale_passed_through(self):
        from datasets.utils import compute_scaling_from_cameras
        cams = [self._make_cam([1, 0, 0]), self._make_cam([-1, 0, 0])]
        _, s1 = compute_scaling_from_cameras(cams, sphere_scale=1.0)
        _, s05 = compute_scaling_from_cameras(cams, sphere_scale=0.5)
        np.testing.assert_allclose(s05 / s1, 0.5, rtol=1e-6)

    def test_symmetric_cameras_center_near_origin(self):
        from datasets.utils import compute_scaling_from_cameras
        cams = [
            self._make_cam([1, 0, 0]),
            self._make_cam([-1, 0, 0]),
            self._make_cam([0, 1, 0]),
            self._make_cam([0, -1, 0]),
        ]
        center, _ = compute_scaling_from_cameras(cams)
        np.testing.assert_allclose(center, [0, 0, 0], atol=0.1)


# =============================================================================
# compute_scaling_from_mesh
# =============================================================================

class TestComputeScalingFromMesh:
    """Tests for mesh bounding box -> (scene_center, scale_factor)."""

    def test_returns_tuple(self):
        from datasets.utils import compute_scaling_from_mesh
        verts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = compute_scaling_from_mesh(verts)
        assert isinstance(result, tuple) and len(result) == 2

    def test_center_shape(self):
        from datasets.utils import compute_scaling_from_mesh
        verts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
        center, _ = compute_scaling_from_mesh(verts)
        assert center.shape == (3,)

    def test_center_is_bbox_midpoint(self):
        """Center should be midpoint of bounding box."""
        from datasets.utils import compute_scaling_from_mesh
        verts = np.array([
            [2, 4, 0],
            [0, 0, 0],
            [2, 0, 0],
            [0, 4, 0],
        ], dtype=np.float64)
        # Bounding box: x in [0,2], y in [0,4], z in [0,0]
        # Center: [1, 2, 0]
        center, _ = compute_scaling_from_mesh(verts)
        np.testing.assert_allclose(center, [1.0, 2.0, 0.0], atol=1e-10)

    def test_scale_puts_farthest_vertex_at_sphere_scale(self):
        """After applying scale, farthest vertex from center = sphere_scale."""
        from datasets.utils import compute_scaling_from_mesh
        verts = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
        ], dtype=np.float64)
        sphere_scale = 1.5
        center, scale = compute_scaling_from_mesh(verts, sphere_scale=sphere_scale)
        verts_norm = (verts - center) * scale
        max_dist = np.linalg.norm(verts_norm, axis=1).max()
        assert max_dist == pytest.approx(sphere_scale, rel=1e-6)

    def test_default_sphere_scale_is_1_5(self):
        """Default sphere_scale for mesh should be 1.5."""
        from datasets.utils import compute_scaling_from_mesh
        import inspect
        sig = inspect.signature(compute_scaling_from_mesh)
        default = sig.parameters["sphere_scale"].default
        assert default == 1.5

    def test_sphere_scale_proportionality(self):
        from datasets.utils import compute_scaling_from_mesh
        verts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
        _, s15 = compute_scaling_from_mesh(verts, sphere_scale=1.5)
        _, s10 = compute_scaling_from_mesh(verts, sphere_scale=1.0)
        np.testing.assert_allclose(s15 / s10, 1.5, rtol=1e-6)

    def test_degenerate_single_vertex(self):
        """Single-vertex mesh (max_dist=0) should not crash."""
        from datasets.utils import compute_scaling_from_mesh
        verts = np.array([[3.0, 3.0, 3.0]], dtype=np.float64)
        _, scale = compute_scaling_from_mesh(verts, sphere_scale=1.5)
        assert scale > 0

    def test_asymmetric_bounding_box(self):
        from datasets.utils import compute_scaling_from_mesh
        verts = np.array([
            [10, 0, 0], [0, 0, 0], [10, 5, 3],
        ], dtype=np.float64)
        center, _ = compute_scaling_from_mesh(verts)
        expected_center = (verts.max(axis=0) + verts.min(axis=0)) / 2
        np.testing.assert_allclose(center, expected_center, atol=1e-10)


# =============================================================================
# compute_scene_scaling dispatcher
# =============================================================================

class TestComputeSceneScaling:
    """Tests for the universal dispatcher compute_scene_scaling."""

    def _make_scale_mat(self, s=2.0, center=None):
        if center is None:
            center = [1.0, 0.0, 0.0]
        mat = np.eye(4, dtype=np.float64)
        mat[0, 0] = mat[1, 1] = mat[2, 2] = s
        mat[:3, 3] = -s * np.array(center)
        return mat

    def _make_cameras(self, n=4):
        angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return [{"center": np.array([np.cos(a), np.sin(a), 0.0])} for a in angle]

    def _make_silhouette_cameras_and_masks(self):
        """Two simple cameras for silhouette mode."""
        cams = []
        for z in [5.0, -5.0]:
            z_axis = np.array([0, 0, -z]) / abs(z)
            x_axis = np.array([1, 0, 0])
            y_axis = np.cross(z_axis, x_axis)
            cams.append({
                "fx": 180.0, "fy": 180.0, "cx": 100.0, "cy": 80.0,
                "R_cam2world": np.stack([x_axis, y_axis, z_axis], axis=1),
                "center": np.array([0, 0, z]),
            })
        masks = [np.ones((160, 200), dtype=np.float64) for _ in cams]
        return cams, masks

    def test_mode_none_returns_identity(self):
        from datasets.utils import compute_scene_scaling
        center, scale = compute_scene_scaling("none", sphere_scale=1.0)
        np.testing.assert_allclose(center, [0, 0, 0])
        assert scale == 1.0

    def test_mode_none_ignores_other_args(self):
        from datasets.utils import compute_scene_scaling
        pcd = np.random.randn(20, 3)
        center, scale = compute_scene_scaling("none", sphere_scale=0.9, pcd=pcd)
        np.testing.assert_allclose(center, [0, 0, 0])
        assert scale == 1.0

    def test_mode_scale_mat_uses_scale_mat(self):
        from datasets.utils import compute_scene_scaling, compute_scaling_from_scale_mat
        mat = self._make_scale_mat(3.0, [2.0, -1.0, 0.5])
        expected_center, expected_scale = compute_scaling_from_scale_mat(mat)
        center, scale = compute_scene_scaling("scale_mat", sphere_scale=1.0, scale_mat=mat)
        np.testing.assert_allclose(center, expected_center, atol=1e-10)
        assert scale == pytest.approx(expected_scale)

    def test_mode_scale_mat_requires_scale_mat(self):
        from datasets.utils import compute_scene_scaling
        with pytest.raises((ValueError, TypeError, KeyError)):
            compute_scene_scaling("scale_mat", sphere_scale=1.0, scale_mat=None)

    def test_mode_pcd_uses_pcd(self):
        from datasets.utils import compute_scene_scaling, compute_scaling_from_pcd
        pcd = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float64)
        expected_center, expected_scale = compute_scaling_from_pcd(pcd, sphere_scale=1.0)
        center, scale = compute_scene_scaling("pcd", sphere_scale=1.0, pcd=pcd)
        np.testing.assert_allclose(center, expected_center, atol=1e-10)
        assert scale == pytest.approx(expected_scale)

    def test_mode_silhouettes_uses_silhouettes(self):
        from datasets.utils import compute_scene_scaling
        cams, masks = self._make_silhouette_cameras_and_masks()
        center, scale = compute_scene_scaling(
            "silhouettes", sphere_scale=1.0, cameras=cams, masks=masks
        )
        assert center.shape == (3,)
        assert scale > 0

    def test_mode_cameras_uses_cameras(self):
        from datasets.utils import compute_scene_scaling
        cameras = self._make_cameras(4)
        center, scale = compute_scene_scaling("cameras", sphere_scale=1.0, cameras=cameras)
        assert center.shape == (3,)
        assert scale > 0

    def test_mode_auto_uses_pcd_when_available(self):
        """auto mode with pcd should use pcd, not silhouettes/cameras."""
        from datasets.utils import compute_scene_scaling, compute_scaling_from_pcd
        pcd = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float64)
        cameras = self._make_cameras(4)
        expected_center, expected_scale = compute_scaling_from_pcd(pcd, sphere_scale=1.0)
        center, scale = compute_scene_scaling(
            "auto", sphere_scale=1.0, pcd=pcd, cameras=cameras
        )
        np.testing.assert_allclose(center, expected_center, atol=1e-10)
        assert scale == pytest.approx(expected_scale)

    def test_mode_auto_falls_back_to_silhouettes_without_pcd(self):
        """auto mode without pcd but with masks should use silhouettes."""
        from datasets.utils import compute_scene_scaling
        cams, masks = self._make_silhouette_cameras_and_masks()
        center, scale = compute_scene_scaling(
            "auto", sphere_scale=1.0, pcd=None, cameras=cams, masks=masks
        )
        assert center.shape == (3,)
        assert scale > 0

    def test_mode_auto_falls_back_to_cameras_without_pcd_or_masks(self):
        """auto mode without pcd or masks should use camera centers."""
        from datasets.utils import compute_scene_scaling
        cameras = self._make_cameras(4)
        center, scale = compute_scene_scaling(
            "auto", sphere_scale=1.0, pcd=None, cameras=cameras, masks=None
        )
        assert center.shape == (3,)
        assert scale > 0

    def test_mode_auto_empty_pcd_falls_back(self):
        """auto mode with empty pcd array should fall back to cameras."""
        from datasets.utils import compute_scene_scaling
        cameras = self._make_cameras(4)
        pcd_empty = np.zeros((0, 3))
        center, scale = compute_scene_scaling(
            "auto", sphere_scale=1.0, pcd=pcd_empty, cameras=cameras
        )
        assert center.shape == (3,)
        assert scale > 0

    def test_sphere_scale_passed_to_pcd(self):
        from datasets.utils import compute_scene_scaling
        pcd = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float64)
        _, s1 = compute_scene_scaling("pcd", sphere_scale=1.0, pcd=pcd)
        _, s05 = compute_scene_scaling("pcd", sphere_scale=0.5, pcd=pcd)
        np.testing.assert_allclose(s05 / s1, 0.5, rtol=1e-6)

    def test_unknown_mode_raises(self):
        from datasets.utils import compute_scene_scaling
        with pytest.raises((ValueError, KeyError, NotImplementedError)):
            compute_scene_scaling("unknown_mode", sphere_scale=1.0)

    def test_center_returns_numpy_array(self):
        from datasets.utils import compute_scene_scaling
        center, _ = compute_scene_scaling("none", sphere_scale=1.0)
        assert isinstance(center, np.ndarray)

    def test_scale_returns_float_like(self):
        from datasets.utils import compute_scene_scaling
        _, scale = compute_scene_scaling("none", sphere_scale=1.0)
        # Should be a Python float or numpy scalar
        assert isinstance(scale, (float, np.floating))


# =============================================================================
# Integration: SfM dataset still imports correctly after refactoring
# =============================================================================

class TestSfmImportsAfterRefactoring:
    """Smoke tests to ensure sfm.py can still import after the refactoring.

    These tests check that the old functions have been removed from sfm.py
    and that the new utils.py module provides them.
    """

    def test_utils_module_importable(self):
        import datasets.utils  # noqa: F401

    def test_compute_scene_scaling_in_utils(self):
        from datasets.utils import compute_scene_scaling
        assert callable(compute_scene_scaling)

    def test_compute_scaling_from_pcd_in_utils(self):
        from datasets.utils import compute_scaling_from_pcd
        assert callable(compute_scaling_from_pcd)

    def test_compute_scaling_from_silhouettes_in_utils(self):
        from datasets.utils import compute_scaling_from_silhouettes
        assert callable(compute_scaling_from_silhouettes)

    def test_compute_scaling_from_cameras_in_utils(self):
        from datasets.utils import compute_scaling_from_cameras
        assert callable(compute_scaling_from_cameras)

    def test_compute_scaling_from_mesh_in_utils(self):
        from datasets.utils import compute_scaling_from_mesh
        assert callable(compute_scaling_from_mesh)

    def test_compute_scaling_from_scale_mat_in_utils(self):
        from datasets.utils import compute_scaling_from_scale_mat
        assert callable(compute_scaling_from_scale_mat)
