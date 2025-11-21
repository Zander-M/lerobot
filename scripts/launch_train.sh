# Training Script
# GPU Config
export CUDA_VISIBLE_DEVICES=0

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

TS=${data +"%Y%m%d-%H%M%S"}
# Launch Train
# policy.compile_model set to false due to limited shared memory

conda run --no-capture-output -n lerobot lerobot-train\
  --dataset.repo_id=libero \
  --dataset.root=/localhome/zma40/Desktop/project/generative_models_course_project/dataset/libero \
  --policy.type=pi05_imle_lora \
  --output_dir=outputs/pi05_imle_lora_finetune_${TS} \
  --policy.push_to_hub=false \
  --policy.pretrained_path=/localhome/zma40/Desktop/project/generative_models_course_project/pretrained_models/pi05_libero_finetuned \
  --policy.compile_model=true \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=false \
  --policy.device=cuda \
  --policy.use_lora=true \
  --batch_size=8 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --save_checkpoint=true \
  --policy.chunk_size=16 --policy.n_action_steps=10 \
  --steps 1000 --log_freq 10  
