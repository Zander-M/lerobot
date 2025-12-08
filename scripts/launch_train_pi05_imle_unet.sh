#!/usr/bin/env bash 
set -euo pipefail 

# Training Script

# Move to correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# GPU Config
export CUDA_VISIBLE_DEVICES=1

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

TS=$(date +"%Y%m%d-%H%M%S")
export TS

# Train Config
export DATASET_ROOT=../data/dataset/libero
export STEPS=100
export LOG_FREQ=10

# Launch Train
# policy.compile_model set to false due to limited shared memory

conda run --no-capture-output -n lerobot lerobot-train\
  --dataset.repo_id=libero \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=pi05_imle_unet \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=1 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --save_checkpoint=true \
  --steps=${STEPS} --log_freq=${LOG_FREQ}