import torch
import os

# Path to your checkpoint
ckpt_path = 'out-harry-potter/ckpt.pt'

if os.path.exists(ckpt_path):
    # Load the checkpoint (map to CPU)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    print("-" * 30)
    print(" HARRY POTTER GPT - STATUS ")
    print("-" * 30)
    print(f"Best Validation Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print(f"Training Iterations:  {checkpoint.get('iter_num', 'N/A')}")
    
    if 'config' in checkpoint:
        print(f"Dataset Used:         {checkpoint['config'].get('dataset', 'N/A')}")
    print("-" * 30)
else:
    print(f"Error: {ckpt_path} not found.")
