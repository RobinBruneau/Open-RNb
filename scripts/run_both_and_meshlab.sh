#!/bin/bash
# Run IDR then SFM training from scratch, then open outputs in MeshLab.
set -e
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
EXP_DIR="/media/bbrument/T9/RNb-NeuS_exp"

echo "============================================================"
echo "PHASE IDR — training"
echo "============================================================"
$VENV launch.py \
    --config configs/idr.yaml \
    --gpu 0 \
    --train \
    --exp_dir "$EXP_DIR" \
    dataset.scene=golden_snail \
    "dataset.root_dir=/media/bbrument/T9/skoltech3d_data/golden_snail/sdmunips"

echo "============================================================"
echo "PHASE SFM — training"
echo "============================================================"
$VENV launch.py \
    --config configs/sfm.yaml \
    --gpu 0 \
    --train \
    --exp_dir "$EXP_DIR" \
    dataset.scene=golden_snail \
    dataset.normal_sfm=data/golden_snail/normalSfm.json \
    dataset.albedo_sfm=data/golden_snail/albedoSfm.json \
    dataset.mask_sfm=data/golden_snail/maskSfm.json

echo "============================================================"
echo "Generating normalization debug PLYs"
echo "============================================================"
$VENV scripts/debug_normalization.py

# Find the final world-space meshes
IDR_WORLD=$(ls -t "$EXP_DIR"/idr-golden_snail/*/save/it20000-mc512.ply 2>/dev/null | head -1)
SFM_WORLD=$(ls -t "$EXP_DIR"/sfm-golden_snail/*/save/it20000-mc512.ply 2>/dev/null | head -1)
IDR_NORM="$EXP_DIR/normalization_debug/idr_phase2_norm.ply"
SFM_NORM="$EXP_DIR/normalization_debug/sfm_phase2_norm.ply"

echo "============================================================"
echo "Opening MeshLab — world space"
echo "  IDR : $IDR_WORLD"
echo "  SFM : $SFM_WORLD"
echo "============================================================"
meshlab "$IDR_WORLD" "$SFM_WORLD" &

echo "============================================================"
echo "Opening MeshLab — normalized space (model sphere)"
echo "  IDR : $IDR_NORM"
echo "  SFM : $SFM_NORM"
echo "============================================================"
meshlab "$IDR_NORM" "$SFM_NORM" &

echo "Done."
