import torch
import yaml
import numpy as np
from PIL import Image
# taming-transformers imports pytorch_lightning.utilities.distributed, which was
# removed in newer Lightning versions; alias it to the current location.
import sys
import pytorch_lightning.utilities.rank_zero as _pl_rank_zero
sys.modules.setdefault("pytorch_lightning.utilities.distributed", _pl_rank_zero)

from taming.models.vqgan import VQModel
from transformers import GPT2TokenizerFast
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_imaginer import GPT, GPTConfig 

# --- CONFIG ---
MODEL_PATH = "imaginer_final.pth"      # Your trained model
DATA_PATH = "train_data_final.pt"      # To get the exact vocab IDs
VQGAN_CONFIG = "vqgan_imagenet_f16_16384.yaml"
VQGAN_CKPT = "vqgan_imagenet_f16_16384.ckpt"

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

def load_vqgan():
    print("Loading VQGAN...")
    config = yaml.safe_load(open(VQGAN_CONFIG, 'r'))
    model = VQModel(**config['model']['params'])
    state = torch.load(VQGAN_CKPT, map_location='cpu', weights_only=False)
    # Remove loss/discriminator weights to avoid mismatch (we only need the encoder/decoder)
    new_state_dict = {k: v for k, v in state['state_dict'].items() if not k.startswith('loss.')}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval().to(DEVICE)
    return model

def generate():
    # 1. Load Metadata (The Truth)
    print(f"Loading metadata from {DATA_PATH}...")
    data = torch.load(DATA_PATH, weights_only=False)
    meta = data['meta']
    vocab_size = meta['vocab_size_total']
    img_start_id = meta['img_start_id']
    img_end_id = meta['img_end_id']
    vocab_offset = meta['vocab_offset']
    
    # 2. Load Model
    print("Loading Imaginer Model...")
    config = GPTConfig(vocab_size=vocab_size, block_size=512)
    model = GPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval().to(DEVICE)
    
    # 3. Setup Tokenizer & Prompt
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # A test prompt that requires logic
    # "Imagine a red square at (2,2) and blue circle at (6,6). Move red 2 right. Overlap?"
    # The Red Square should end up at (4,2). The Blue is at (6,6). They should NOT touch.
    prompt = "Imagine a red square at grid (2, 2) and a blue circle at (6, 6). Move red square 2 right, 0 down. Overlap?"
    
    print(f"Prompt: {prompt}")
    
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids + [img_start_id], dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 4. Generate Visual Tokens
    print("Dreaming...")
    generated = []
    
    with torch.no_grad():
        # Generate exactly 256 tokens (16x16 grid)
        for _ in range(256):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :]
            
            # Greedy decoding (pick the most likely pixel)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)
            generated.append(next_token.item())
            
    # 5. Decode Image
    print("Decoding pixels...")
    vqgan = load_vqgan()
    
    # Shift back to VQGAN range (0-16383)
    valid_tokens = []
    for t in generated:
        if t >= vocab_offset:
            valid_tokens.append(t - vocab_offset)
        else:
            valid_tokens.append(0) # Black pixel if it predicted a text token by mistake

    z_indices = torch.tensor(valid_tokens, dtype=torch.long).to(DEVICE).unsqueeze(0)
    
    # VQGAN Decode
    x_recon = vqgan.decode(vqgan.quantize.get_codebook_entry(z_indices, shape=(1, 16, 16, 256)))
    
    # Save
    x_recon = torch.clamp(x_recon, -1., 1.)
    x_recon = (x_recon + 1.0) / 2.0 * 255.0
    x_recon = x_recon.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    
    img = Image.fromarray(x_recon[0])
    img.save("final_dream.png")
    print("Saved 'final_dream.png' - Go look at it!")

if __name__ == "__main__":
    generate()