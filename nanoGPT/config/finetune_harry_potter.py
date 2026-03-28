import time

out_dir = 'out-harry-potter'
eval_interval = 5
eval_iters = 40
wandb_log = False
wandb_project = 'harry-potter'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'harry_potter'
init_from = 'gpt2'  # gpt2-xl will OOM on colab T4, use gpt2 (124M)

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# harry potter has ~1.5M tokens, so 1 epoch ~= 45 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 200  # ~4-5 epochs over the full dataset

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

# colab safe settings
dtype = 'float16'   # T4 doesn't support bfloat16
compile = False     # torch.compile is flaky on colab