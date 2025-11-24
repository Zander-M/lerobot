# GPU Config
export CUDA_VISIBLE_DEVICES=0

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Launch eval

conda run --no-capture-output -n lerobot lerobot-eval \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.path=/localhome/zma40/Desktop/project/generative_models_course_project/models/pi05_imle_lora_libero_finetuned \
    --policy.compile_model=false \
    --policy.device=cuda \
    --env.max_parallel_tasks=1 \
    --policy.use_lora=false\