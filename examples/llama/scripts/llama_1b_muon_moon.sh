
torchrun --standalone --nproc_per_node 8 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --lr 0.001 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0.1 \
    --grad_clipping 1.0 \
    --betas 0.9 0.95 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_dir checkpoints/llama_1b_muon_lr_0.001_wd_0.1_adamw_betas_0.9_0.95_grad_clip_1.0 \
    --optimizer muon \
    --wandb_name llama_1b_muon_lr_0.001_wd_0.1_adamw_betas_0.9_0.95_grad_clip_1.0 \