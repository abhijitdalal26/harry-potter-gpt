"""
DPO Training Script for Harry Potter GPT using TRL.

This script takes the SFT-trained HuggingFace model and applies
Direct Preference Optimization (DPO) using preference pairs.

Usage (on Kaggle with GPU):
    python train_dpo.py

Prerequisites:
    pip install trl transformers datasets peft accelerate
    python data/harry_potter_dpo/prepare_dpo.py  (prepare data first)
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk, load_dataset
from trl import DPOTrainer, DPOConfig

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_DIR = "./harry-potter-hf"              # SFT model (HuggingFace format)
DPO_DATA_DIR = "./data/harry_potter_dpo"     # Prepared DPO dataset
OUTPUT_DIR = "./harry-potter-hf-dpo"         # Where to save DPO model

# DPO Hyperparameters (tuned for RTX 3050 6GB)
BETA = 0.1                    # KL penalty coefficient
LEARNING_RATE = 5e-6          # Low LR to preserve SFT knowledge
BATCH_SIZE = 2                # Per-device batch size (small for 6GB VRAM)
GRAD_ACCUM_STEPS = 8          # Effective batch = 16
NUM_EPOCHS = 3                # Training epochs
MAX_LENGTH = 384              # Max total sequence length (reduced for VRAM)
MAX_PROMPT_LENGTH = 192       # Max prompt length
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 50
EVAL_STEPS = 50

# ── Load Model & Tokenizer ──────────────────────────────────────────────────
print("=" * 60)
print("DPO Training for Harry Potter GPT")
print("=" * 60)

print(f"\nLoading SFT model from: {MODEL_DIR}")
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)

# GPT2 needs a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print(f"  Parameters: {model.num_parameters():,}")
print(f"  Vocab size: {model.config.vocab_size}")

# ── Load Reference Model (frozen copy of SFT model) ─────────────────────────
print(f"\nLoading reference model (frozen copy)...")
ref_model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

# ── Load Dataset ─────────────────────────────────────────────────────────────
train_dir = os.path.join(DPO_DATA_DIR, "train")
val_dir = os.path.join(DPO_DATA_DIR, "val")

if os.path.exists(train_dir) and os.path.isdir(train_dir):
    print(f"\nLoading datasets from disk...")
    train_dataset = load_from_disk(train_dir)
    eval_dataset = load_from_disk(val_dir)
else:
    print(f"\nLoading datasets from JSON...")
    train_path = os.path.join(DPO_DATA_DIR, "train.json")
    val_path = os.path.join(DPO_DATA_DIR, "val.json")
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    eval_dataset = load_dataset("json", data_files=val_path, split="train")

print(f"  Train samples: {len(train_dataset)}")
print(f"  Eval samples:  {len(eval_dataset)}")
print(f"  Columns: {train_dataset.column_names}")

# ── DPO Training Config ─────────────────────────────────────────────────────
print(f"\nSetting up DPO training...")
print(f"  Beta (KL penalty): {BETA}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Epochs: {NUM_EPOCHS}")

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=BETA,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    max_length=MAX_LENGTH,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    remove_unused_columns=False,
    report_to="none",
    gradient_checkpointing=True,           # saves VRAM by recomputing activations
    precompute_ref_log_probs=True,         # pre-compute ref model probs to free its VRAM during training
)

# ── Initialize DPO Trainer ───────────────────────────────────────────────────
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# ── Train ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Starting DPO Training...")
print("=" * 60 + "\n")

trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"\nSaving DPO model to: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── Quick comparison ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Quick Test: SFT vs DPO outputs")
print("=" * 60)

sft_model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
dpo_model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
sft_model = sft_model.to(device).eval()
dpo_model = dpo_model.to(device).eval()

USER_TAG = "<" + "|user|" + ">"
ASST_TAG = "<" + "|assistant|" + ">"

test_prompts = [
    f"{USER_TAG} Who is Harry Potter?\n{ASST_TAG}",
    f"{USER_TAG} What is the Mirror of Erised?\n{ASST_TAG}",
]

for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        sft_out = sft_model.generate(input_ids, max_new_tokens=150, temperature=0.8,
                                      top_k=200, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        dpo_out = dpo_model.generate(input_ids, max_new_tokens=150, temperature=0.8,
                                      top_k=200, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    print(f"\nPrompt: {prompt}")
    print(f"\n  SFT: {tokenizer.decode(sft_out[0], skip_special_tokens=False)[:300]}")
    print(f"\n  DPO: {tokenizer.decode(dpo_out[0], skip_special_tokens=False)[:300]}")
    print("-" * 60)

print("\nDPO Training Complete!")
