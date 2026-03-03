#!/bin/bash
# Quick sanity test: few iterations to verify P1/P2 scaling after refactoring.
set -e
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
EXP_DIR="/media/bbrument/T9/RNb-NeuS_exp"

# 10000 steps total → P1 = 1000 steps (warmup_ratio=0.1), P2 = 10000 steps
# isosurface resolution=256 to speed up marching cubes
MAX_STEPS=10000
ISO_RES=256

echo "============================================================"
echo "QUICK IDR — ${MAX_STEPS} steps, iso=${ISO_RES}"
echo "============================================================"
$VENV launch.py \
    --config configs/idr.yaml \
    --gpu 0 \
    --train \
    --exp_dir "$EXP_DIR" \
    dataset.scene=golden_snail \
    "dataset.root_dir=/media/bbrument/T9/skoltech3d_data/golden_snail/sdmunips" \
    trainer.max_steps=$MAX_STEPS \
    trainer.val_check_interval=$MAX_STEPS \
    model.geometry.isosurface.resolution=$ISO_RES

echo "============================================================"
echo "QUICK SFM — ${MAX_STEPS} steps, iso=${ISO_RES}"
echo "============================================================"
$VENV launch.py \
    --config configs/sfm.yaml \
    --gpu 0 \
    --train \
    --exp_dir "$EXP_DIR" \
    dataset.scene=golden_snail \
    dataset.normal_sfm=data/golden_snail/normalSfm.json \
    dataset.albedo_sfm=data/golden_snail/albedoSfm.json \
    dataset.mask_sfm=data/golden_snail/maskSfm.json \
    trainer.max_steps=$MAX_STEPS \
    trainer.val_check_interval=$MAX_STEPS \
    model.geometry.isosurface.resolution=$ISO_RES

echo "============================================================"
echo "Normalization debug + PLY generation"
echo "============================================================"
$VENV scripts/debug_normalization.py

# Locate outputs
IDR_WORLD=$(ls -t "$EXP_DIR"/idr-golden_snail/*/save/it${MAX_STEPS}-mc${ISO_RES}.ply 2>/dev/null | head -1)
SFM_WORLD=$(ls -t "$EXP_DIR"/sfm-golden_snail/*/save/it${MAX_STEPS}-mc${ISO_RES}.ply 2>/dev/null | head -1)
IDR_INTER=$(ls -t "$EXP_DIR"/idr-golden_snail/*/save/intermediate_mesh.ply 2>/dev/null | head -1)
SFM_INTER=$(ls -t "$EXP_DIR"/sfm-golden_snail/*/save/intermediate_mesh.ply 2>/dev/null | head -1)
IDR_NORM="$EXP_DIR/normalization_debug/idr_phase2_norm.ply"
SFM_NORM="$EXP_DIR/normalization_debug/sfm_phase2_norm.ply"

echo ""
echo "======= REPÈRE MONDE ======================================="
echo "  IDR final  : $IDR_WORLD"
echo "  SFM final  : $SFM_WORLD"
echo "============================================================"
meshlab "$IDR_WORLD" "$SFM_WORLD" &

echo ""
echo "======= REPÈRE NORMALISÉ (spheres) ========================="
echo "  IDR P2 norm : $IDR_NORM"
echo "  SFM P2 norm : $SFM_NORM"
echo "============================================================"
meshlab "$IDR_NORM" "$SFM_NORM" &

echo "Done."
