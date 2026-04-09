# nanoGPT to Harry Potter GPT

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

## Directory Structure

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
│   ├── compare_models.py              ← Run inference: compares Base vs SFT vs DPO side by side
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
│   ├── config/eval_gpt2.py               ← Eval base GPT-2
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
│   |
├── MODEL OUTPUTS (NOT in git — too large)
    ├── out-harry-potter/            ← nanoGPT checkpoints (.pt files, ~1.5GB)
    ├── out-harry-potter-finetune/   ← SFT checkpoints
    ├── harry-potter-hf/             ← SFT model in HuggingFace format (~475MB)
    └── harry-potter-hf-dpo/         ← DPO model in HuggingFace format (~498MB)


```

---

## Example Model Evolution

Here is an example of how the generated answers for the same prompt improved across the different stages of training. You can reproduce this by running `compare_models.py`.

### Question: What is the significance of the Deathly Hallows?

**[1. Base Model]**

> Umbridge is the ultimate "secretary" for the wizarding world. Her "wizarding history" is "stealing" gold for her inventions, which we now use to create our next "superman" food line. I missed that she’s actually the one who actually "sees" the Deathly Hallows while the others were being "destroyed." Do you think Dumbledore was more arrogant than she looks, or was he just trying to keep quiet?
> <|endoftext|>The Department for Transport has admitted that it is still investigating a "mass murderer" who stole a train ticket on Tuesday morning...

**[2. SFT Model]**

> The Deathly Hallows are the ultimate symbol of wizarding progress. They're the foundation of everything that came before them. Dumbledore's prophecy to the Potters was about to be fulfilled. The Deathly Hallows are the foundation of the wizarding order's future. Do you think the Deathly Hallows represent the future of the wizarding world or just a myth?

**[3. DPO Model]**

> It’s a very specific kind of magic—it’s about finding the power within the dead, but also finding something else to transform into. The idea of using the power in an old body to make a new one seems so much more powerful than just a potion. I didn’t realize how much it ties into the plot until later. Do you think the Deathly Hallows were actually "true" or just a "game-changer" that eventually died?

### Question: Is Dumbledore a good person?

**[1. Base Model]**

> I think so, especially when he’s trying to save Hermione. He’s a lot more honest about himself and his own failings. Do you think he did the right thing or just made too much of himself?
> <|endoftext|>Gotham is a magical place and it’s weird that it treats you like a criminal even when it actually means something useful. And then there’s this persistent problem of magic being too dangerous for the well-organized...

**[2. SFT Model]**

> He is a true champion of reason and a great man who is also brave but also kind. He is also the perfect foil to Voldemort because he is both. I missed that Dumbledore was the only one who had the courage to say yes to Harry's "death" in a way Voldemort never could, which is why he is so different from Voldemort. Do you think Dumbledore was brave enough to tell Harry the truth?

**[3. DPO Model]**

> He has a lot of good qualities — like kindness, patience, and a good sense of humor — but he’s also got a heavy, dark side, which makes him a little too dangerous. I didn’t notice until later that Dumbledore actually’s a bit of a jerk sometimes, but he’s also not exactly the kind of man who’s easy to manipulate. Do you think he ever really got his own "good" side?

### Question: Who is Severus Snape?

**[1. Base Model]**

> Severus Snape’s sister, Lily. He was the first person I knew of who actually saw him as a man — he was wearing a tight suit, but he was holding a very heavy object, which was holding a ball of lightning. I caught that he’s the first person to recognize the ball and actually pull it out of the ground. Do you think he was the first to notice a snake on top of the ball?
> <|endoftext|>Kotobukiya has finally found its answer to the mystery of Harry Potter’s Deathly Hallows. It finally has a clue — it has a book called The Thousand and One Curse. The world’s biggest horror novel has been making the same mistake for years already.

**[2. SFT Model]**

> Severus Snape is a brilliant wizard who has been studying the Dark Arts for over fifty years. His entire career is spent studying Occlumency, Transfiguration, and Occlumency magic. What makes you think he could have been the one to steal the Stone of Gryffindor?

**[3. DPO Model]**

> He’s the "least dangerous" of the three. I didn't realize until later that the first book actually shows he’s the one who actually saves the world from the Ministry's "Gryffindor" (and later Snape’s) wizarding blood—he’s so clever at hiding that! I missed that he’s actually the one who actually saves Harry’s life in Order of the Phoenix, which is why he’s the only one who actually saves Harry’s life in Deathly Hallows. Do you think the "bad guys" were actually more dangerous than the good guys, or was it more of the "cool kids" who were just too
