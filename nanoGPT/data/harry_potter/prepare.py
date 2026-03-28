import os
import tiktoken
import numpy as np

# Load the combined harry potter file
input_file_path = os.path.join(os.path.dirname(__file__), 'harry_potter.txt')

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"Total characters: {len(data):,}")

# 90/10 train/val split
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
# encode_ordinary ignores special tokens like <|endoftext|>
# we use encode instead so that <|endoftext|> separator between books is recognized
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode(train_data, allowed_special={"<|endoftext|>"})
val_ids = enc.encode(val_data, allowed_special={"<|endoftext|>"})

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Total characters: 6,430,081
# train has 1,797,593 tokens
# val has 198,524 tokens