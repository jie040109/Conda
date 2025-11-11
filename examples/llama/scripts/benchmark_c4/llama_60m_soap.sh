
torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.002 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.0 \
    --betas 0.9 0.99 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --precondition_frequency 10 \
    --save_dir checkpoints/llama_60m_soap_lr_2e-3_wd_0_betas_0.9_0.99_precondition_frequency_10 \
    --optimizer soap \
    --wandb_name llama_60m_soap_lr_2e-3_wd_0_betas_0.9_0.99_precondition_frequency_10 \