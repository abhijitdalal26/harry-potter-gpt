"""
Runs the same question set against the Base (pretrain-only), SFT, and DPO
checkpoints and writes a markdown transcript, so it's obvious how the model
actually changed at each stage rather than trusting a metric alone.

Usage:
    python showcase.py --base_dir out-harry-potter --sft_dir harry-potter-hf \
        --dpo_dir harry-potter-hf-dpo --output results/stage_comparison.md
"""
import argparse
import os

import tiktoken
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from model import GPT, GPTConfig

USER = chr(60) + "|user|" + chr(62)
ASST = chr(60) + "|assistant|" + chr(62)

# basic -> advanced, mirrors the DPO training data's own category spread
QUESTIONS = [
    "Who is Harry Potter?",
    "What is the Mirror of Erised?",
    "Why did Voldemort fail to kill Harry as a baby?",
    "Is Dumbledore a good person?",
    "Does Snape try to save Harry Potter?",
    "Why didn't they just use Veritaserum on Sirius to prove he was innocent?",
    "I just finished Prisoner of Azkaban and I am destroyed about Sirius",
    "I have a theory that Dumbledore is actually Death from the Three Brothers tale",
]


def load_base_model(ckpt_dir: str, device: str):
    ckpt = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location=device)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)
    return model


def gen_base(model, enc, prompt: str, device: str, max_new_tokens: int = 150) -> str:
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=0.8, top_k=50)
    return enc.decode(y[0].tolist()[len(ids):]).strip()


def gen_hf(model, tok, prompt: str, device: str, max_new_tokens: int = 150) -> str:
    inp = tok(prompt, return_tensors="pt").to(device)
    n = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


def main(base_dir: str, sft_dir: str, dpo_dir: str, output_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_model = load_base_model(base_dir, device)
    enc = tiktoken.get_encoding("gpt2")

    tok = GPT2Tokenizer.from_pretrained(sft_dir)
    tok.pad_token = tok.eos_token
    sft_model = GPT2LMHeadModel.from_pretrained(sft_dir).to(device).eval()
    dpo_model = GPT2LMHeadModel.from_pretrained(dpo_dir).to(device).eval()

    lines = ["# Harry Potter GPT — stage comparison transcript\n"]
    for q in QUESTIONS:
        prompt = f"{USER} {q}\n{ASST}"
        base_ans = gen_base(base_model, enc, prompt, device)
        sft_ans = gen_hf(sft_model, tok, prompt, device)
        dpo_ans = gen_hf(dpo_model, tok, prompt, device)
        block = f"\n## Q: {q}\n\n**Base (pretrain only):** {base_ans}\n\n**SFT:** {sft_ans}\n\n**DPO:** {dpo_ans}\n"
        print(block)
        lines.append(block)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved transcript to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="out-harry-potter")
    parser.add_argument("--sft_dir", default="harry-potter-hf")
    parser.add_argument("--dpo_dir", default="harry-potter-hf-dpo")
    parser.add_argument("--output", default="results/stage_comparison.md")
    args = parser.parse_args()
    main(args.base_dir, args.sft_dir, args.dpo_dir, args.output)
