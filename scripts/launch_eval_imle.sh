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
    --policy.path=/localhome/zma40/Desktop/project/generative_models_course_project/lerobot/scripts/outputs/imle_policy_20251123-210331/checkpoints/000010/pretrained_model \
    --policy.device=cuda \