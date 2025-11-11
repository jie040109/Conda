
torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr 0.001 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --betas 0.9 0.999 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_dir checkpoints/llama_350m_adamw_lr_0.001_betas_0.9_0.999_wd_0 \
    --optimizer adamw \
    --wandb_name llama_350m_adamw_lr_0.001_betas_0.9_0.999_wd_0 \