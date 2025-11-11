
torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr 0.001 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --betas 0.9 0.99 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --precondition_frequency 10 \
    --save_dir checkpoints/llama_350m_soap_lr_0.001_betas_0.9_0.99_wd_0_precondition_frequency_10 \
    --optimizer soap \
    --wandb_name llama_350m_soap_lr_0.001_betas_0.9_0.99_wd_0_precondition_frequency_10  \