# GPU Config
export CUDA_VISIBLE_DEVICES=4

# suppress tokenizer parallelism
export TOKENIZERS_PARALLELISM=false


# Launch eval
# policy.compile_model set to false due to limited shared memory

lerobot-eval \
    --output_dir=/localhome/zma40/Desktop/project/generative_models_course_project/lerobot/logs \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.path=../pretrained_models/pi05_libero_finetuned \
    --policy.n_action_steps=10 \
    --policy.compile_model=false \
    --env.max_parallel_tasks=1
