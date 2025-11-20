# Training Script
# GPU Config
export CUDA_VISIBLE_DEVICES=0

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Launch eval
# policy.compile_model set to false due to limited shared memory

conda run --no-capture-output -n lerobot lerobot-train\
  --dataset.repo_id=/localhome/zma40/Desktop/project/generative_models_course_project/dataset/libero \
  --policy.type=pi05_imle_lora \
  --output_dir=outputs/pi05_imle_lora_test \
  --policy.push_to_hub=false \
  --policy.pretrained_path=/localhome/zma40/Desktop/project/generative_models_course_project/pretrained_models/pi05_libero_finetuned \
  --policy.compile_model=false \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.use_lora=true \
  --batch_size=1 \
  --policy.chunk_size=8 --policy.n_action_steps=8 \
  --steps 10 --log_freq 1  # tiny smoke run

