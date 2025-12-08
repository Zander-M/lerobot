#!/usr/bin/env bash 
set -euo pipefail 

# Training Script

# Move to correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# San check evaluation script. Just to make sure the finetuned pi05 model works
# GPU Config
export CUDA_VISIBLE_DEVICES=7

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Experiment Configs

# Change the POLICY path according to the model type

export POLICY_PATH=/localhome/zma40/Desktop/project/generative_models_course_project/models/pi05_imle_unet_libero_fintuned
export EVAL_TASK=libero_spatial

# Launch eval
# policy.compile_model set to false due to limited shared memory

conda run --no-capture-output -n lerobot lerobot-eval \
    --env.type=libero \
    --env.task=$EVAL_TASK \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.path=$POLICY_PATH \
    --policy.compile_model=false \
    --policy.device=cuda \
    --env.max_parallel_tasks=1 \
