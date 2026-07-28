"""
Converts a nanoGPT checkpoint (ckpt.pt) to a HuggingFace GPT2LMHeadModel.

nanoGPT stores attn/mlp weights as Conv1D-style (transposed relative to
HuggingFace's nn.Linear), so those four weight matrices get transposed on
the way across. Everything else copies straight across by name.

Usage:
    python convert_to_hf.py --ckpt_path out-harry-potter-sft/ckpt.pt --output_dir harry-potter-hf
"""
import argparse

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

TRANSPOSED_KEYS = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]


def build_key_mapping(n_layer: int) -> dict[str, str]:
    mapping = {
        "transformer.wte.weight": "transformer.wte.weight",
        "transformer.wpe.weight": "transformer.wpe.weight",
        "transformer.ln_f.weight": "transformer.ln_f.weight",
        "transformer.ln_f.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    for i in range(n_layer):
        mapping.update(
            {
                f"transformer.h.{i}.ln_1.weight": f"transformer.h.{i}.ln_1.weight",
                f"transformer.h.{i}.ln_1.bias": f"transformer.h.{i}.ln_1.bias",
                f"transformer.h.{i}.ln_2.weight": f"transformer.h.{i}.ln_2.weight",
                f"transformer.h.{i}.ln_2.bias": f"transformer.h.{i}.ln_2.bias",
                f"transformer.h.{i}.attn.c_attn.weight": f"transformer.h.{i}.attn.c_attn.weight",
                f"transformer.h.{i}.attn.c_attn.bias": f"transformer.h.{i}.attn.c_attn.bias",
                f"transformer.h.{i}.attn.c_proj.weight": f"transformer.h.{i}.attn.c_proj.weight",
                f"transformer.h.{i}.attn.c_proj.bias": f"transformer.h.{i}.attn.c_proj.bias",
                f"transformer.h.{i}.mlp.c_fc.weight": f"transformer.h.{i}.mlp.c_fc.weight",
                f"transformer.h.{i}.mlp.c_fc.bias": f"transformer.h.{i}.mlp.c_fc.bias",
                f"transformer.h.{i}.mlp.c_proj.weight": f"transformer.h.{i}.mlp.c_proj.weight",
                f"transformer.h.{i}.mlp.c_proj.bias": f"transformer.h.{i}.mlp.c_proj.bias",
            }
        )
    return mapping


def convert(ckpt_path: str, output_dir: str) -> None:
    print(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_args = ckpt["model_args"]
    state_dict = ckpt["model"]
    print("Model args:", model_args)

    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    key_mapping = build_key_mapping(model_args["n_layer"])

    new_state_dict = {}
    for nano_key, hf_key in key_mapping.items():
        if nano_key not in state_dict:
            print(f"WARNING: missing key {nano_key}")
            continue
        tensor = state_dict[nano_key]
        if any(nano_key.endswith(k) for k in TRANSPOSED_KEYS):
            tensor = tensor.t()
        new_state_dict[hf_key] = tensor

    hf_config = GPT2Config(
        vocab_size=model_args["vocab_size"],
        n_positions=model_args["block_size"],
        n_embd=model_args["n_embd"],
        n_layer=model_args["n_layer"],
        n_head=model_args["n_head"],
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
    )

    hf_model = GPT2LMHeadModel(hf_config)
    missing, unexpected = hf_model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print("WARNING: missing keys in converted model:", missing)
    if unexpected:
        print("WARNING: unexpected keys in converted model:", unexpected)

    hf_model.save_pretrained(output_dir)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)

    print(f"Saved HF model to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert(args.ckpt_path, args.output_dir)
