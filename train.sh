#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH --output=neus_%j.out
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=rtx_3090:1
#SBATCH -c 16

CONFIG="configs/rnb.yaml"
DATASET_NAME="RNb"
SCENE="buddha_sdm"
TAG="rgbplus_opti_lights"
VAL_INTERVAL=5000
MAX_STEPS=20000
NORMAL_LOSS=1.0
NUM_VIEWS=20
NO_ALBEDO="false"

python launch.py \
        --config "$CONFIG" \
        --gpu 0 \
        --train dataset.scene="$SCENE" \
        tag="$TAG" \
        trainer.val_check_interval=$VAL_INTERVAL \
        trainer.max_steps=$MAX_STEPS \
        dataset.name="$DATASET_NAME" \
        model.background_color=black \
        dataset.num_views=$NUM_VIEWS \
        system.loss.lambda_normal_l1=$NORMAL_LOSS \
        system.loss.lambda_normal_cos=$NORMAL_LOSS \
        model.no_albedo=$NO_ALBEDO \
        
