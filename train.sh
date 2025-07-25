#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH --output=neus_%j.out
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=rtx_3090:1
#SBATCH -c 16

# Check if a scene argument is provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <scene_path> <no_albedo_value>"
    echo "  <no_albedo_value> should be 'true' or 'false'"
    exit 1
fi

CONFIG="configs/rnb.yaml"
DATASET_NAME="RNb"

ROOT_DIR="./data/"
# Take the scene argument from the command line
SCENE="$1"
TAG="light_opti_rgb_plus"

VAL_INTERVAL=1000
MAX_STEPS=20000
NUM_VIEWS=20

NO_ALBEDO="$2"
APPLY_LIGHT_OPTI="true"
APPLY_RGB_PLUS="true"
OPTIMIZE_CAMERAS="false"

python launch.py \
        --config "$CONFIG" \
        --gpu 0 \
        --train \
        dataset.scene="$SCENE" \
        tag="$TAG" \
        dataset.name="$DATASET_NAME" \
        dataset.num_views=$NUM_VIEWS \
        dataset.apply_light_opti=$APPLY_LIGHT_OPTI \
        dataset.apply_rgb_plus=$APPLY_RGB_PLUS \
        trainer.val_check_interval=$VAL_INTERVAL \
        trainer.max_steps=$MAX_STEPS \
        model.background_color=black \
        model.no_albedo=$NO_ALBEDO \
        system.optimize_camera_poses=$OPTIMIZE_CAMERAS