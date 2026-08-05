# Harry Potter GPT

Project write-up: [abhijitdalal.vercel.app/projects/harry-potter-gpt](https://abhijitdalal.vercel.app/projects/harry-potter-gpt)

A GPT-2 (124M) model fine-tuned to chat like a Harry Potter fan, trained through a full NLP pipeline: **Pretrain → SFT → DPO (RLHF)**. Built from nanoGPT to understand each stage of a modern LLM training pipeline hands-on, rather than just calling an API.

---

## Training Pipeline

### Stage 1 — Continued Pretraining
Fine-tuned base GPT-2 on Harry Potter books to learn domain language.
- Data: HP book text, all 7 books (~2M tokens)
- Hardware: Kaggle T4 x 2
- Output: `out-harry-potter/ckpt.pt`

### Stage 2 — Supervised Fine-Tuning (SFT)
Taught the model to follow a fan-discussion conversational format.
- Data: `hp_sft_data.txt` (4,321 lines, ~540KB)
- Format: `<|user|> question\n<|assistant|> answer<|endoftext|>`
- Loss masking: only compute loss on assistant tokens
- 600 iterations on Kaggle T4 x 2 (~47 min)
- Final loss: 1.97 train / 2.33 val

### Stage 3 — HuggingFace Conversion
Converted the nanoGPT checkpoint to HuggingFace format for TRL compatibility.
- Input: `out-harry-potter/ckpt.pt` (1.49 GB)
- Output: `harry-potter-hf/` (474 MB safetensors)
- Key detail: weight transposition (nanoGPT `nn.Linear` vs HF `Conv1D`)

### Stage 4 — DPO (Direct Preference Optimization)
Aligned the model to prefer high-quality fan-style responses over generic ones.
- Data: 347 preference pairs (prompt / chosen / rejected)
- Library: TRL `DPOTrainer`
- 3 epochs, 60 total steps (~1 min on RTX 3050 6GB)
- Output: `harry-potter-hf-dpo/`

---

## Pretrained Models

All three stages are published on Hugging Face Hub:
- [abhijit26/harry-potter-gpt-base](https://huggingface.co/abhijit26/harry-potter-gpt-base) — continued-pretraining stage
- [abhijit26/harry-potter-gpt-sft](https://huggingface.co/abhijit26/harry-potter-gpt-sft) — supervised fine-tuned stage
- [abhijit26/harry-potter-gpt-dpo](https://huggingface.co/abhijit26/harry-potter-gpt-dpo) — DPO-aligned final stage

`nanoGPT/kaggle_push_to_hf.ipynb` re-runs the checkpoint-to-HF conversion and upload from inside Kaggle's datacenter network (uploading the large model files from a home connection was the original bottleneck).

---

## Quick Start

**Run the full pipeline end-to-end (Colab/Kaggle)**

`nanoGPT/colab_pipeline.ipynb` and `nanoGPT/kaggle_pipeline.ipynb` re-run pretrain → SFT → DPO on a fresh GPU instance, with Google Drive checkpoint backup/restore (checksum-validated, resumable if a stage already completed) and `torchrun` DDP across all visible GPUs.

**Run inference (compare SFT vs DPO)**
```bash
conda activate rl_env
python nanoGPT/predict_dpo.py
```

**Generate DPO data**

Use the prompt in `nanoGPT/dpo_data_prompt.md` with ChatGPT or Claude. Save the output as `nanoGPT/data/harry_potter_dpo/dpo_data.json`, then:
```bash
python nanoGPT/data/harry_potter_dpo/prepare_dpo.py
```

**Run DPO training**
```bash
conda activate rl_env
python nanoGPT/train_dpo.py
```

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

- Python 3.10–3.13
- `torch` with CUDA
- `transformers`, `trl`, `datasets`, `accelerate`

Install:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers trl datasets accelerate
```

---

## Files not in Git

Model weights are large (400–500MB each) and excluded from git:
- `nanoGPT/harry-potter-hf/` — SFT model
- `nanoGPT/harry-potter-hf-dpo/` — DPO model
- `nanoGPT/out-harry-potter/` — nanoGPT checkpoints
- `nanoGPT/data/harry_potter/*.bin` — tokenized binary data
