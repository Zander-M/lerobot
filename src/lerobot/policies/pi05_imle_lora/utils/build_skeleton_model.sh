#!/usr/bin/env bash
# Build Pi05 IMLE LoRA checkpoint from pretrained checkpoint
set -euo pipefail

export PRETRAINED_PATH=/localhome/zma40/Desktop/project/generative_models_course_project/lerobot/outputs/train/2025-12-03/08-54-53_pi05_imle_checkpoint_finetuning/checkpoints/last/pretrained_model
export OUTPUT_PATH=/localhome/zma40/Desktop/project/generative_models_course_project/models/pi05_imle_lora_checkpoint_libero_finetuned 

conda run -n lerobot python build_skeleton_model.py \
--pretrained_path $PRETRAINED_PATH \
--output_path $OUTPUT_PATH \
--lora_config_path lora_config.json

# copy pre/post processors (normalizer/unnormalizer) to the new checkpoint
cp "$PRETRAINED_PATH"/policy_preprocessor*.json "$OUTPUT_PATH"/
cp "$PRETRAINED_PATH"/policy_preprocessor_step_*_normalizer_processor.safetensors "$OUTPUT_PATH"/ 
cp "$PRETRAINED_PATH"/policy_postprocessor*.json "$OUTPUT_PATH"/
cp "$PRETRAINED_PATH"/policy_postprocessor_step_*_unnormalizer_processor.safetensors "$OUTPUT_PATH"/