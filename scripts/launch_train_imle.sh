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
  --dataset.root=/localhome/zma40/Desktop/project/generative_models_course_project/dataset/lerobot_libero_v3/libero_spatial_image_v3 \
  --policy.type=imle_policy\
  --output_dir=outputs/imle_policy_${TS} \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=128 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --save_checkpoint=true \
  --steps 500000 --log_freq 100
