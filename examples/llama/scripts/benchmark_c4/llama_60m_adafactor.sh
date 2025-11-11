
torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.003 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_dir checkpoints/llama_60m_adafactor_lr_0.003_beta1_0.9_wd_0 \
    --optimizer adafactor \
    --beta1 0.9 \
    --wandb_name llama_60m_adafactor_lr_0.003_beta1_0.9_wd_0