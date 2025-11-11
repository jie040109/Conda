
torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.0 \
    --betas 0.9 0.95 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_dir checkpoints/llama_60m_muon_moonlight_lr_0.001_wd_0_adamw_betas_0.9_0.95 \
    --optimizer muon \
    --wandb_name llama_60m_muon_moonlight_lr_0.001_wd_0_adamw_betas_0.9_0.95 \