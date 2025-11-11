
torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr 0.001 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_dir checkpoints/llama_350m_adafactor_lr_0.001_beta1_0.9_wd_0 \
    --optimizer adafactor \
    --beta1 0.9  \
    --wandb_name llama_350m_adafactor_lr_0.001_beta1_0.9_wd_0 \