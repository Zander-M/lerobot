#!/usr/bin/env bash 
set -euo pipefail 

# Move to correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# GPU Config
export CUDA_VISIBLE_DEVICES=7

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false


# Launch eval

export POLICY_PATH=/localhome/zma40/Desktop/project/generative_models_course_project/lerobot/scripts/outputs/imle_policy_20251124-100458/checkpoints/last/pretrained_model

conda run --no-capture-output -n lerobot lerobot-eval \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.path=${POLICY_PATH} \
    --policy.device=cuda \
