# Training Script
# GPU Config
export CUDA_VISIBLE_DEVICES=0

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

export TS=$(date +"%Y%m%d-%H%M%S")
# Launch Train
# policy.compile_model set to false due to limited shared memory

conda run --no-capture-output -n lerobot lerobot-train\
  --dataset.repo_id=zak1040/libero_spatial_image_v3 \
  --policy.type=imle_policy\
  --output_dir=outputs/imle_policy_${TS} \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=8 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --save_checkpoint=true \
  --steps 10 --log_freq 1  
