# Bug fixes (branch `bench-normals-only`)

Two bugs found while benchmarking Open-RNb against RNb-NeuS2 on DiLiGenT-MV /
LUCES-MV / Martine. Both are fixed on this branch.

## 1. Loader crashes on normals-only datasets (no `albedo/` folder)

**File:** `datasets/rnb.py` (commit `03361be`)

**Symptom:** training aborts immediately when the dataset has no `albedo/`
folder (e.g. the `unimsps` photometric-stereo variant, which provides normals
but no reliable reflectance):
`FileNotFoundError: .../albedo/000.png`.

**Root cause:** the per-view loop opened `albedo/{i}.png` unconditionally.

**Fix:** guard the albedo read with `os.path.isdir(albedo_dir)`. When absent,
append `None` to `albedo_paths` and use a neutral gray image
(`torch.full((H, W, 3), 0.5)`) as the color target — matching the `None`-append
pattern already used by `datasets/sfm.py`. Two-phase albedo scaling auto-disables
via the same `isdir` check, so the gray target is never used for supervision.

## 2. `scale_mat` scenes export a degenerate (collapsed) mesh

**File:** `datasets/utils.py::compute_scaling_from_scale_mat` (commit `3fabe0c`)

**Symptom:** for `cameras.npz` (RNb format) scenes, the exported mesh collapses
to a tiny blob centered on `scene_center` (e.g. bounds `[-1.03,-0.98,-0.43] ±
0.01`) instead of spanning the object in world coordinates. Chamfer evaluation
then fails ("zero-size array" after the z-threshold removes every vertex).

**Root cause:** the mesh export (`models/geometry.py`) reconstructs world coords
as `world = norm / scale_factor + scene_center` — the same convention the
pcd/silhouette scaling functions use (world-space center, world→norm scale). But
`compute_scaling_from_scale_mat` returned the **opposite** convention
(`scene_center = -t/s`, `scale_factor = s`) based on a docstring that had the
`scale_mat` relationship backwards. The IDR/NeuS convention is
`p_world = scale_mat @ p_norm = s·p_norm + t`, so the export needs
`scale_factor = 1/s`, `scene_center = t`.

**Fix:** return `scene_center = t`, `scale_factor = 1/s`. Verified by a
camera-center round-trip (`export(normalized_centers) == world_centers`, max
error 0.0) on DiLiGenT-MV and LUCES-MV. `scale_mat`-mode training does not use
these values (poses are pre-normalized), so only the export path changes — no
effect on the silhouette/pcd modes, which were already correct.
