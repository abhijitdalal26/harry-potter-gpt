import os
import torch
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.stdout.reconfigure(encoding='utf-8')

# Directory where the DPO model is stored
DPO_DIR = 'harry-potter-hf-dpo'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print(f"Loading Tokenizer and Model from {DPO_DIR}...")
try:
    tokenizer = GPT2Tokenizer.from_pretrained(DPO_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(DPO_DIR).to(device).eval()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print("Model loaded successfully!\n")

def generate_response(model, prompt, max_new_tokens=400):
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

# Formats to prompt the model correctly
USER = chr(60) + "|user|" + chr(62)
ASST = chr(60) + "|assistant|" + chr(62)

print("==================================================")
print("Welcome to the Harry Potter DPO Chatbot!")
print("Type 'exit' or 'quit' to end the conversation.")
print("==================================================\n")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break
        if not user_input.strip():
            continue
            
        prompt = f"{USER} {user_input.strip()}\n{ASST}"
        
        response = generate_response(model, prompt, max_new_tokens=500)
        
        print(f"\nChatbot: {response}\n")
        print("-" * 50)
        
    except KeyboardInterrupt:
        print("\nChatbot: Goodbye!")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
