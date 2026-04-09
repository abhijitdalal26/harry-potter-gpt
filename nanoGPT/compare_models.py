import os
import torch
import tiktoken
from model import GPTConfig, GPT
import sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import GPT2LMHeadModel, GPT2Tokenizer


BASE_DIR = 'out-harry-potter'
SFT_DIR = 'harry-potter-hf'
DPO_DIR = 'harry-potter-hf-dpo'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 1. Load Base Model (nanoGPT format)
print(f"Loading Base model from {BASE_DIR}...")
ckpt_path = os.path.join(BASE_DIR, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
base_model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
base_model.load_state_dict(state_dict)
base_model.eval()
base_model.to(device)

# 2. Load Tokenizers & HF Models
print("Loading Tokenizer and HF models...")
tokenizer = GPT2Tokenizer.from_pretrained(DPO_DIR)
tokenizer.pad_token = tokenizer.eos_token

sft_model = GPT2LMHeadModel.from_pretrained(SFT_DIR).to(device).eval()
dpo_model = GPT2LMHeadModel.from_pretrained(DPO_DIR).to(device).eval()

print("All models loaded!\n")

def generate_base(model, prompt, max_new_tokens=150):
    enc = tiktoken.get_encoding("gpt2")
    # For the base model, `<|user|>` and `<|assistant|>` are just regular sequences of characters
    start_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=0.8, top_k=50)
    
    # decode only the newly generated part
    tokens = y[0].tolist()[len(start_ids):]
    return enc.decode(tokens).strip()

def generate_hf(model, prompt, max_new_tokens=150):
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    n = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()

questions = [
    "Who is Severus Snape?",
    "Why did Voldemort fail to kill Harry as a baby?",
    "What is the significance of the Deathly Hallows?",
    "Is Dumbledore a good person?",
    "Was Severus Snape trying to save Harry?"
]

# We construct the tokens carefully
USER = chr(60) + "|user|" + chr(62)
ASST = chr(60) + "|assistant|" + chr(62)

print("Starting inference comparison...\n")
for q in questions:
    prompt = f"{USER} {q}\n{ASST}"
    print("=" * 80)
    print("Q:", q)
    print("=" * 80)
    
    ans_base = generate_base(base_model, prompt)
    print("[1. Base Model]")
    print(ans_base)
    print("-" * 80)
    
    ans_sft = generate_hf(sft_model, prompt)
    print("[2. SFT Model (HF)]")
    print(ans_sft)
    print("-" * 80)
    
    ans_dpo = generate_hf(dpo_model, prompt)
    print("[3. DPO Model (HF)]")
    print(ans_dpo)
    print("\n\n")
