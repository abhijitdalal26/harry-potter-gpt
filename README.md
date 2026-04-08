# Harry Potter GPT

A GPT-2 (124M) model fine-tuned to chat like a Harry Potter fan, trained through a full NLP pipeline: **Pretrain -> SFT -> DPO (RLHF)**.

---

## Training Pipeline

### Stage 1 - Continued Pretraining
Fine-tuned base GPT-2 on Harry Potter books to learn domain language.
- Data: HP book text (~5M tokens)
- Hardware: Kaggle T4 x 2
- Output: out-harry-potter/ckpt.pt

### Stage 2 - Supervised Fine-Tuning (SFT)
Taught the model to follow a fan-discussion conversational format.
- Data: hp_sft_data.txt (4,321 lines, ~540KB)
- Format: <|user|> question\n<|assistant|> answer<|endoftext|>
- Loss masking: only compute loss on assistant tokens
- 600 iterations on Kaggle T4 x 2 (~47 min)
- Final loss: 1.97 train / 2.33 val

### Stage 3 - HuggingFace Conversion
Converted nanoGPT checkpoint to HuggingFace format for TRL compatibility.
- Input: out-harry-potter/ckpt.pt (1.49 GB)
- Output: harry-potter-hf/ (474 MB safetensors)
- Key: Weight transposition (nanoGPT nn.Linear vs HF Conv1D)

### Stage 4 - DPO (Direct Preference Optimization)
Aligned the model to prefer high-quality fan-style responses over generic ones.
- Data: 347 preference pairs (prompt / chosen / rejected)
- Library: TRL DPOTrainer
- 3 epochs, 60 total steps (~1 min on RTX 3050 6GB)
- Output: harry-potter-hf-dpo/

---

## Quick Start

### Run inference (compare SFT vs DPO)
`ash
conda activate rl_env
python nanoGPT/predict_dpo.py
`

### Generate DPO data
Use the prompt in 
anoGPT/dpo_data_prompt.md with ChatGPT or Claude.
Save output as 
anoGPT/data/harry_potter_dpo/dpo_data.json, then:
`ash
python nanoGPT/data/harry_potter_dpo/prepare_dpo.py
`

### Run DPO training
`ash
conda activate rl_env
python nanoGPT/train_dpo.py
`

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2 |
| Parameters | 124M |
| Layers | 12 |
| Heads | 12 |
| Embedding dim | 768 |
| Context length | 1024 tokens |
| Vocab size | 50,257 |

---

## Requirements

- Python 3.10-3.13
- 	orch with CUDA
- 	ransformers, 	rl, datasets, ccelerate

Install:
`ash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers trl datasets accelerate
`

---

## Files NOT in Git

Model weights are large (400-500MB each) and excluded from git:
- 
anoGPT/harry-potter-hf/ - SFT model
- 
anoGPT/harry-potter-hf-dpo/ - DPO model
- 
anoGPT/out-harry-potter/ - nanoGPT checkpoints
- 
anoGPT/data/harry_potter/*.bin - tokenized binary data
