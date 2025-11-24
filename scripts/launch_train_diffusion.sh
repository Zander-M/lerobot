# Training Script
# GPU Config
export CUDA_VISIBLE_DEVICES=0

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

export TS=$(date +"%Y%m%d-%H%M%S")
# Launch Train
# policy.compile_model set to false due to limited shared memory

conda run --no-capture-output -n lerobot lerobot-train\
  --dataset.repo_id=libero \
  --dataset.root=/localhome/zma40/Desktop/project/generative_models_course_project/dataset/libero \
  --policy.type=diffusion \
  --output_dir=outputs/diffusion_${TS} \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=8 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --save_checkpoint=true \
  --steps 10 --log_freq 1  
