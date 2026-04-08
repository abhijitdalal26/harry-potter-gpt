import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SFT_DIR = "./harry-potter-hf"
DPO_DIR = "./harry-potter-hf-dpo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
tokenizer = GPT2Tokenizer.from_pretrained(DPO_DIR)
tokenizer.pad_token = tokenizer.eos_token

print("Loading SFT model...")
sft = GPT2LMHeadModel.from_pretrained(SFT_DIR).to(DEVICE).eval()
print("Loading DPO model...")
dpo = GPT2LMHeadModel.from_pretrained(DPO_DIR).to(DEVICE).eval()
print("Ready!\n")

def generate(model, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    n = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()

USER = chr(60) + "|user|" + chr(62)
ASST = chr(60) + "|assistant|" + chr(62)

questions = [
    "Who is Severus Snape?",
    "Why did Voldemort fail to kill Harry as a baby?",
    "What is the significance of the Deathly Hallows?",
    "Is Dumbledore a good person?",
    "What is your favourite Harry Potter moment?",
]

for q in questions:
    prompt = USER + " " + q + "\n" + ASST
    print("=" * 60)
    print("Q: " + q)
    print("-" * 30)
    sft_ans = generate(sft, prompt)
    print("[SFT]  " + sft_ans)
    print("-" * 30)
    dpo_ans = generate(dpo, prompt)
    print("[DPO]  " + dpo_ans)
    print()
