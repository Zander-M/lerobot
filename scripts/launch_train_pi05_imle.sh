# Training Script
# GPU Config
export CUDA_VISIBLE_DEVICES=1

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

TS=$(date +"%Y%m%d-%H%M%S")
export TS

# Train Config
export DATASET_ROOT=/localhome/zma40/Desktop/project/generative_models_course_project/dataset/libero
export STEPS=100
export LOG_FREQ=10

# Launch Train
# policy.compile_model set to false due to limited shared memory

conda run --no-capture-output -n lerobot lerobot-train\
  --dataset.repo_id=libero \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=pi05_imle \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=1 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --save_checkpoint=true \
  --steps=${STEPS} --log_freq=${LOG_FREQ}