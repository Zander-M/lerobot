#!/usr/bin/env bash 
set -euo pipefail 

# Training Script

# Move to correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Multi GPU training script using accelerate.
# GPU Config
export CUDA_VISIBLE_DEVICES=1,2,3,4
export NUM_PROCESSES=4

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Experiment Configs

export CONFIG_PATH=/localhome/zma40/Desktop/project/generative_models_course_project/models/pi05_imle_checkpoint_libero/train_config.json
export PRETRAINED_PATH=/localhome/zma40/Desktop/project/generative_models_course_project/models/pi05_imle_checkpoint_libero

# Launch eval
# policy.compile_model set to false due to limited shared memory
conda run --no-capture-output -n lerobot \
accelerate launch \
  --multi_gpu \
  --num_processes=$NUM_PROCESSES \
  $(which lerobot-train) \
  --config_path=$CONFIG_PATH \
  --policy.pretrained_path=$PRETRAINED_PATH \
  --wandb.enable=true
