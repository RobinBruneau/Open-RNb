#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH --output=neus_%j.out
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=rtx_3090:1
#SBATCH -c 16

CONFIG="configs/rnb.yaml"
DATASET_NAME="RNb"

ROOT_DIR="./data/"
SCENE="buddha_sdm"
TAG="rgbplus_opti_lights"

VAL_INTERVAL=1000
MAX_STEPS=20000
NUM_VIEWS=20

NO_ALBEDO="false"
APPLY_LIGHT_OPTI="true"
APPLY_RGB_PLUS="true"


python launch.py \
        --config "$CONFIG" \
        --gpu 0 \
        --train \
        root_dir: $ROOT_DIR/$SCENE\
        dataset.scene="$SCENE" \
        tag="$TAG" \
        dataset.name="$DATASET_NAME" \
        dataset.num_views=$NUM_VIEWS \
        dataset.apply_light_opti: $APPLY_LIGHT_OPTI \
        dataset.apply_rgb_plus: $APPLY_RGB_PLUS \
        trainer.val_check_interval=$VAL_INTERVAL \
        trainer.max_steps=$MAX_STEPS \
        model.background_color=black \
        model.no_albedo=$NO_ALBEDO \
        
        
        
