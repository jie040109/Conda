# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

out_dir = 'out/gpt2_355M_conda_lr_0.003_betas_0.9_0.99_wd_0.01_scale_0.25_update_2000'
wandb_log = True
wandb_project = 'gpt2_owt_pretraining'
wandb_run_name='gpt2_355M_conda_lr_0.003_betas_0.9_0.99_wd_0.01_scale_0.25_update_2000'
 
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 30
block_size = 1024
gradient_accumulation_steps = 16

#model
n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# optimizer
algorithm = 'galore_adamw'
scale =0.25
update_proj_gap=2000
learning_rate = 0.003
weight_decay = 0.01
betas = (0.9, 0.99)
grad_clip = 1.0 

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
max_iters = 100000
lr_decay_iters = 100000
min_lr = 6e-5

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

