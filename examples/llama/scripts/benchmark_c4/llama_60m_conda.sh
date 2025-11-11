
torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.01 \
    --conda_scale 0.25 \
    --update_proj_gap 2000 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --betas 0.9 0.99 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_dir checkpoints/llama_60m_conda_lr_0.01_betas_0.9_0.99_wd_0_update_proj_gap_2000_conda_scale_0.25 \
    --optimizer conda \
    --wandb_name llama_60m_conda_lr_0.01_betas_0.9_0.99_wd_0_update_proj_gap_2000_conda_scale_0.25 \