"""
Prepares DPO preference data from JSON into HuggingFace Dataset format.
Input:  data/harry_potter_dpo/dpo_data.json
Output: data/harry_potter_dpo/train/ and data/harry_potter_dpo/val/ (Arrow datasets)

Usage: python data/harry_potter_dpo/prepare_dpo.py
"""

import json
import os
import random

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "dpo_data.json")
TRAIN_DIR = os.path.join(SCRIPT_DIR, "train")
VAL_DIR = os.path.join(SCRIPT_DIR, "val")
VAL_SPLIT = 0.1
SEED = 42

# ── load and validate ────────────────────────────────────────────────────────
print(f"Loading DPO data from: {INPUT_FILE}")

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} preference pairs")

# Validate format
required_keys = {"prompt", "chosen", "rejected"}
valid_samples = []
for i, sample in enumerate(data):
    if not isinstance(sample, dict):
        print(f"  WARNING: Sample {i} is not a dict, skipping")
        continue
    missing = required_keys - set(sample.keys())
    if missing:
        print(f"  WARNING: Sample {i} missing keys {missing}, skipping")
        continue
    if not sample["prompt"].strip() or not sample["chosen"].strip() or not sample["rejected"].strip():
        print(f"  WARNING: Sample {i} has empty fields, skipping")
        continue
    valid_samples.append(sample)

print(f"Valid samples: {len(valid_samples)} / {len(data)}")

if len(valid_samples) == 0:
    raise ValueError("No valid samples found! Check your dpo_data.json format.")

# ── split ────────────────────────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(valid_samples)

n_val = max(1, int(len(valid_samples) * VAL_SPLIT))
val_data = valid_samples[:n_val]
train_data = valid_samples[n_val:]

print(f"Train: {len(train_data)} samples | Val: {len(val_data)} samples")

# ── save as HuggingFace datasets ─────────────────────────────────────────────
try:
    from datasets import Dataset
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    train_dataset.save_to_disk(TRAIN_DIR)
    val_dataset.save_to_disk(VAL_DIR)
    
    print(f"\nSaved HuggingFace datasets:")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Val:   {VAL_DIR}")
    print(f"\nTrain columns: {train_dataset.column_names}")
    print(f"Sample train[0]:")
    print(f"  prompt:   {train_dataset[0]['prompt'][:80]}...")
    print(f"  chosen:   {train_dataset[0]['chosen'][:80]}...")
    print(f"  rejected: {train_dataset[0]['rejected'][:80]}...")

except ImportError:
    # Fallback: save as JSON splits for environments without datasets lib
    print("\n'datasets' library not installed. Saving as JSON splits instead.")
    
    train_path = os.path.join(SCRIPT_DIR, "train.json")
    val_path = os.path.join(SCRIPT_DIR, "val.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Train: {train_path} ({len(train_data)} samples)")
    print(f"  Val:   {val_path} ({len(val_data)} samples)")

print("\nDone! Next step: python train_dpo.py")
