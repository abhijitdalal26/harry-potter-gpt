import os
import re
import numpy as np
import tiktoken

INPUT_FILE = os.path.join(os.path.dirname(__file__), 'hp_sft_data.txt')
OUTPUT_DIR = os.path.dirname(__file__)
VAL_SPLIT  = 0.05

enc = tiktoken.get_encoding("gpt2")
EOT = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]  # 50256

def parse_conversations(text):
    all_convos = []
    
    # try both formats
    if '<|endoftext|>' in text:
        raw_convos = re.split(r'<\|endoftext\|>', text)
    else:
        raw_convos = re.split(r'<endoftext\|>', text)
    
    print(f"Raw splits found: {len(raw_convos)}")

    for convo in raw_convos:
        convo = convo.strip()
        if not convo:
            continue

        turns = re.split(r'(<\|user\|>|<\|assistant\|>)', convo)
        parsed = []
        current_role = None
        for chunk in turns:
            chunk = chunk.strip()
            if chunk == '<|user|>':
                current_role = 'user'
            elif chunk == '<|assistant|>':
                current_role = 'assistant'
            elif chunk and current_role:
                parsed.append((current_role, chunk))

        if parsed:
            all_convos.append(parsed)

    return all_convos


def build_conversation(turns):
    all_ids  = []
    all_mask = []

    for role, content in turns:
        tag_ids     = enc.encode(f"<|{role}|> ", allowed_special={"<|user|>", "<|assistant|>"})
        content_ids = enc.encode(
            content.strip() + "\n",
            allowed_special={"<|endoftext|>"},
            disallowed_special=()          # handles emojis + any weird unicode
        )
        turn_ids = tag_ids + content_ids

        if role == 'user':
            all_ids.extend(turn_ids)
            all_mask.extend([0] * len(turn_ids))
        else:  # assistant
            all_ids.extend(turn_ids)
            all_mask.extend([1] * len(turn_ids))

    # closing EOT, compute loss on it
    all_ids.append(EOT)
    all_mask.append(1)

    assert len(all_ids) == len(all_mask)
    return all_ids, all_mask


def pack_conversations(convos):
    all_ids, all_mask = [], []
    for turns in convos:
        ids, mask = build_conversation(turns)
        all_ids.extend(ids)
        all_mask.extend(mask)
    return all_ids, all_mask


def save_split(ids, mask, prefix):
    np.array(ids,  dtype=np.uint16).tofile(os.path.join(OUTPUT_DIR, f'{prefix}.bin'))
    np.array(mask, dtype=np.int8).tofile(os.path.join(OUTPUT_DIR,   f'{prefix}_mask.bin'))
    n_loss = sum(mask)
    print(f"{prefix}: {len(ids):,} tokens | {n_loss:,} loss tokens ({100*n_loss/len(ids):.1f}%)")


# ── main ─────────────────────────────────────────────────────────────────────

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"File loaded, {len(text):,} characters")

convos = parse_conversations(text)
print(f"Loaded {len(convos):,} conversations")

if len(convos) == 0:
    raise ValueError("No conversations parsed — check your separator format in sft_data.txt")

# shuffle + split
rng = np.random.default_rng(seed=42)
idx = rng.permutation(len(convos))
n_val        = max(1, int(len(convos) * VAL_SPLIT))
train_convos = [convos[i] for i in idx[n_val:]]
val_convos   = [convos[i] for i in idx[:n_val]]
print(f"Train conversations: {len(train_convos):,} | Val conversations: {len(val_convos):,}")

# tokenise, pack, save
print("\nTokenising and packing...")
train_ids, train_mask = pack_conversations(train_convos)
val_ids,   val_mask   = pack_conversations(val_convos)

save_split(train_ids, train_mask, 'train')
save_split(val_ids,   val_mask,   'val')


# train: 123,792 tokens | 95,704 loss tokens (77.3%)
# val: 6,282 tokens | 4,837 loss tokens (77.0%)