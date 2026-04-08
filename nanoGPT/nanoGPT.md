# nanoGPT — Harry Potter GPT: Project Guide

> **Purpose of this file:** A complete map of this codebase so that any developer or LLM
> opening this project can instantly understand every file, every directory, and how everything fits together.

---

## What Is This Project?

A **GPT-2 (124M)** language model fine-tuned to chat like an engaged Harry Potter fan.
It was trained through a **3-stage NLP pipeline**:

```
Base GPT-2 (OpenAI)
    │
    ▼  Stage 1: Continued Pretraining
    │  Train on HP book text to learn HP-specific language
    │
    ▼  Stage 2: Supervised Fine-Tuning (SFT)
    │  Train on HP fan Q&A to learn conversational format
    │
    ▼  Stage 3: DPO (Direct Preference Optimization)
    │  Align to prefer high-quality, engaging fan-style responses
    │
    ▼
Harry Potter GPT — DPO aligned model
```

---

## Directory Structure (Every File Explained)

```
nanoGPT/
├── CORE MODEL & TRAINING
│   ├── model.py                    ← GPT-2 model definition (Transformer, attention, MLP)
│   ├── train.py                    ← Main nanoGPT training loop (pretraining + SFT)
│   ├── train_dpo.py                ← DPO fine-tuning script using TRL library
│   ├── configurator.py             ← Utility: loads config files and overrides via CLI
│   └── sample.py                   ← Text generation/sampling from nanoGPT checkpoints
│
├── INFERENCE
│   ├── predict_dpo.py              ← Run inference: compares SFT vs DPO side by side
│   └── check_loss.py               ← Quick script to check train/val loss of a checkpoint
│
├── NOTEBOOK
│   └── harry-potter-gpt.ipynb      ← Full training notebook (run on Kaggle)
│                                      Contains: pretraining, SFT, HF conversion, DPO inference
│
├── CONFIG (training hyperparameters for each stage)
│   ├── config/finetune_harry_potter.py   ← Stage 1: HP continued pretraining config
│   ├── config/harry_potter_sft.py        ← Stage 2: SFT config (loss masking ON)
│   ├── config/train_gpt2.py              ← OpenWebText GPT-2 reproduction config
│   ├── config/finetune_shakespeare.py    ← Shakespeare example config
│   ├── config/train_shakespeare_char.py  ← Shakespeare char-level config
│   ├── config/eval_gpt2.py               ← Eval base GPT-2
│   ├── config/eval_gpt2_medium.py        ← Eval GPT-2 medium
│   ├── config/eval_gpt2_large.py         ← Eval GPT-2 large
│   └── config/eval_gpt2_xl.py            ← Eval GPT-2 XL
│
├── DATA
│   ├── data/harry_potter/
│   │   ├── harry_potter.txt         ← Full HP book text (~6.8MB, all 7 books)
│   │   ├── hp_books/                ← Individual book files
│   │   ├── prepare.py               ← Tokenizes hp text → train.bin + val.bin
│   │   ├── process_hp_book.py       ← Cleans and merges book files into one text
│   │   ├── train.bin                ← Tokenized training data (uint16 numpy memmap)
│   │   └── val.bin                  ← Tokenized validation data
│   │
│   ├── data/harry_potter_sft/       ← SFT conversation dataset (used in Stage 2)
│   │   └── (hp_sft_data.txt)        ← Fan Q&A conversations in chat format
│   │
│   ├── data/harry_potter_dpo/
│   │   ├── dpo_data.json            ← 347 preference pairs (prompt/chosen/rejected)
│   │   ├── prepare_dpo.py           ← Validates + converts JSON → HuggingFace Dataset
│   │   ├── train/                   ← 313 training samples (HF Dataset format)
│   │   └── val/                     ← 34 validation samples (HF Dataset format)
│   │
│   └── data/shakespeare/            ← Shakespeare example dataset (from original nanoGPT)
│
├── MODEL OUTPUTS (NOT in git — too large)
│   ├── out-harry-potter/            ← nanoGPT checkpoints (.pt files, ~1.5GB)
│   ├── out-harry-potter-finetune/   ← SFT checkpoints
│   ├── harry-potter-hf/             ← SFT model in HuggingFace format (~475MB)
│   └── harry-potter-hf-dpo/         ← DPO model in HuggingFace format (~498MB)
│
├── DOCS
│   ├── nanoGPT.md                   ← THIS FILE — full project guide
│   └── dpo_data_prompt.md           ← Prompt to generate DPO data via ChatGPT/Claude
│
└── PYTHON ENV (NOT in git)
    ├── Lib/                         ← Conda env packages
    ├── Scripts/                     ← Conda env executables
    └── share/                       ← Package metadata
```

---

## Key Files Deep Dive

### `model.py` — The GPT-2 Brain
Implements GPT-2 from scratch using PyTorch:
- `CausalSelfAttention` — multi-head attention with causal mask
- `MLP` — feed-forward block (GELU activation)
- `Block` — Transformer block = attention + MLP + LayerNorm
- `GPT` — full model with embedding, 12 blocks, LM head
- Supports loading pretrained OpenAI GPT-2 weights with `from_pretrained()`

### `train.py` — The Training Engine
The central training script. Works for both pretraining and SFT:
- Reads data from `train.bin` / `val.bin` numpy memmaps
- Supports `init_from = 'gpt2' | 'resume' | 'scratch'`
- **SFT mode:** when `sft_mode=True`, masks user-token loss (only trains on assistant tokens)
- Mixed precision (fp16/bf16), gradient accumulation, torch.compile
- Saves checkpoints to `out_dir/ckpt.pt`

**How to run:**
```bash
# Stage 1: HP pretraining
python train.py config/finetune_harry_potter.py

# Stage 2: SFT
python train.py config/harry_potter_sft.py
```

### `train_dpo.py` — The DPO Trainer
Uses HuggingFace TRL's `DPOTrainer` to do preference optimization:
- Loads SFT model from `harry-potter-hf/`
- Loads frozen reference model (copy of SFT, used for KL constraint)
- Loads preference dataset from `data/harry_potter_dpo/`
- Saves DPO-aligned model to `harry-potter-hf-dpo/`

**Key hyperparameters:**
| Param | Value | Why |
|-------|-------|-----|
| beta | 0.1 | Low = don't stray too far from SFT |
| lr | 5e-6 | Very small to preserve SFT knowledge |
| batch | 2×8=16 | Fits in 6GB VRAM |
| epochs | 3 | ~60 steps on 313 samples |

### `predict_dpo.py` — Inference & Comparison
Loads both SFT and DPO models and answers the same questions with each,
so you can directly compare the effect of DPO alignment.

```bash
conda activate rl_env
python predict_dpo.py
```

### `harry-potter-gpt.ipynb` — The Full Notebook
Jupyter notebook meant to run on **Kaggle T4 GPU**.
Contains all stages end-to-end:
1. Data preparation
2. Pretraining (Stage 1)
3. SFT (Stage 2)
4. Checkpoint → HuggingFace conversion (Stage 3)
5. DPO inference comparison (Stage 4)

---

## Data Formats

### Stage 1 & 2: Binary token files
```python
# train.py reads data like this:
data = np.memmap('data/harry_potter/train.bin', dtype=np.uint16, mode='r')
# Each value is a GPT-2 BPE token ID (0–50256)
```

### Stage 2: SFT Chat Format
```
<|user|> Why is Snape important?
<|assistant|> He's the most complex character in the series because...
<|endoftext|>
<|user|> What about Dumbledore?
<|assistant|> Dumbledore is fascinating because on re-read...
<|endoftext|>
```
Loss is computed ONLY on the `<|assistant|>` tokens.

### Stage 4: DPO Preference Format (dpo_data.json)
```json
{
  "prompt":   "<|user|> Why did Voldemort fail to kill Harry?",
  "chosen":   "<|assistant|> Because of ancient magic — Lily's sacrifice created a...",
  "rejected": "<|assistant|> Voldemort failed because Harry was protected by love."
}
```
- **chosen** = engaging, detailed, fan-discussion style
- **rejected** = correct but short, generic, low-engagement

---

## Training Pipeline: Reproduce From Scratch

### Step 1 — Prepare HP Pretraining Data
```bash
cd data/harry_potter
python prepare.py
# Creates: train.bin, val.bin
```

### Step 2 — Pretrain on HP Books (Kaggle T4 recommended)
```bash
python train.py config/finetune_harry_potter.py
# Output: out-harry-potter/ckpt.pt
```

### Step 3 — SFT Fine-Tuning
```bash
python train.py config/harry_potter_sft.py
# Loads: out-harry-potter/ckpt.pt
# Output: updated out-harry-potter/ckpt.pt
```

### Step 4 — Convert to HuggingFace Format
```bash
# Run the "Convert to HF" section in harry-potter-gpt.ipynb
# OR see the conversion code in the notebook
# Output: harry-potter-hf/
```

### Step 5 — Prepare DPO Data
```bash
python data/harry_potter_dpo/prepare_dpo.py
# Reads: dpo_data.json (347 pairs)
# Output: data/harry_potter_dpo/train/ and val/
```

### Step 6 — DPO Training
```bash
conda activate rl_env
python train_dpo.py
# Output: harry-potter-hf-dpo/
```

### Step 7 — Inference
```bash
python predict_dpo.py
# Compares SFT vs DPO outputs
```

---

## Environment Setup

**Recommended:** `rl_env` conda environment (has all packages pre-installed)

```bash
conda activate rl_env
python predict_dpo.py  # should work immediately
```

**If setting up fresh:**
```bash
# PyTorch with CUDA (pick your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# HuggingFace stack for DPO
pip install transformers trl datasets accelerate

# Jupyter
pip install jupyter
```

**Python:** 3.10–3.13 (avoid 3.14+, PyTorch not supported yet)

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2 |
| Parameters | 124M |
| Layers (n_layer) | 12 |
| Attention heads | 12 |
| Embedding dim | 768 |
| Context length | 1024 tokens |
| Vocab size | 50,257 (GPT-2 BPE) |
| MLP hidden dim | 3072 (4 × 768) |

---

## Results Summary

| Stage | Model | Train Loss | Val Loss |
|-------|-------|-----------|---------|
| Stage 1 | HP Pretrained | — | — |
| Stage 2 | SFT | 1.97 | 2.33 |
| Stage 4 | DPO | — | — |

**Qualitative DPO improvement:** The DPO model produces longer, more engaged responses,
uses "on re-read..." phrasing, and ends with follow-up questions — matching the style
of the `chosen` examples in the training data.

---

## What Is NOT in Git

Large binary files are excluded (see `.gitignore`):

| Path | Size | How to get it |
|------|------|----------------|
| `harry-potter-hf/` | ~475 MB | Run the HF conversion in the notebook |
| `harry-potter-hf-dpo/` | ~498 MB | Run `python train_dpo.py` |
| `out-harry-potter/` | ~1.5 GB | Run Stage 1+2 training |
| `data/**/*.bin` | varies | Run `prepare.py` |
| `data/**/train/`, `val/` | varies | Run `prepare_dpo.py` |

To push models to HuggingFace Hub:
```bash
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/harry-potter-gpt-dpo ./harry-potter-hf-dpo
```
