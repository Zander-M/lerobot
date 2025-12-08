#!/usr/bin/env bash 
set -euo pipefail 

# Training Script

# Move to correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# San check evaluation script. Just to make sure the finetuned pi05 model works
# GPU Config
export CUDA_VISIBLE_DEVICES=0

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Experiment Configs

export POLICY_PATH=/localhome/zma40/Desktop/project/generative_models_course_project/lerobot/outputs/train/2025-12-03/08-54-53_pi05_imle_checkpoint_finetuning/checkpoints/last/pretrained_model
export EVAL_TASK=libero_spatial

# Launch eval
# policy.compile_model set to false due to limited shared memory

conda run --no-capture-output -n lerobot lerobot-eval \
    --env.type=libero \
    --env.task=$EVAL_TASK \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --policy.path=$POLICY_PATH \
    --policy.compile_model=false \
    --policy.device=cuda \
    --env.max_parallel_tasks=1 \
