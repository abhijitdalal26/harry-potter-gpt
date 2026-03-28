import time

out_dir = 'out-harry-potter'
eval_interval = 50
eval_iters = 20
wandb_log = False
wandb_project = 'harry-potter'
wandb_run_name = 'sft-' + str(time.time())

dataset = 'harry_potter_sft'     # points to data/harry_potter_sft/
init_from = 'resume'             # load your harry potter finetuned ckpt
sft_mode = True                  # turns on loss masking in get_batch

# only save if val loss improves
always_save_checkpoint = False

# adjust these based on how many conversations you have
batch_size = 4
gradient_accumulation_steps = 8  # effective batch = 32
max_iters = 400

# lower LR for SFT — don't blast away what was learned in pretraining
learning_rate = 1e-5
decay_lr = False
dropout = 0.1