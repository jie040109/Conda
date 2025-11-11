
torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --betas 0.9 0.999 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_dir checkpoints/llama_130m_adamw_lr_0.001_betas_0.9_0.999_wd_0 \
    --optimizer adamw \
    --wandb_name llama_130m_adamw_lr_0.001_betas_0.9_0.999_wd_0 \