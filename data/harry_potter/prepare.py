# prepare.py for Qwen2.5
import os
import numpy as np
from transformers import AutoTokenizer

# ── config ──────────────────────────────────────────────
input_file  = "harry_potter.txt"
model_name  = "Qwen/Qwen2.5-1.5B"   # base model, not instruct
val_split   = 0.1                    # 90% train, 10% val
# ────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("reading text...")
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

print(f"total characters: {len(text):,}")

# tokenize entire text
print("tokenizing...")
token_ids = tokenizer.encode(
    text,
    add_special_tokens=False   # <|endoftext|> already in your text
)

print(f"total tokens: {len(token_ids):,}")

# train / val split
split_idx  = int(len(token_ids) * (1 - val_split))
train_ids  = token_ids[:split_idx]
val_ids    = token_ids[split_idx:]

print(f"train tokens : {len(train_ids):,}")
print(f"val tokens   : {len(val_ids):,}")

# save as uint32 (Qwen vocab size is 151,936 — bigger than uint16 max of 65535)
train_arr = np.array(train_ids, dtype=np.uint32)
val_arr   = np.array(val_ids,   dtype=np.uint32)

train_arr.tofile("train.bin")
val_arr.tofile("val.bin")

print("done. saved train.bin and val.bin")