import os
import json
import torch
import yaml
import numpy as np  # The speed booster
from PIL import Image
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from taming.models.vqgan import VQModel

# --- CONFIG ---
DATA_DIR = "dataset_imagination_balanced"
OUTPUT_FILE = "train_data_final.pt"
VQGAN_CONFIG = "vqgan_imagenet_f16_16384.yaml"
VQGAN_CKPT = "vqgan_imagenet_f16_16384.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_vqgan():
    print(f"Loading VQGAN on {DEVICE}...")
    config = yaml.safe_load(open(VQGAN_CONFIG, 'r'))
    model = VQModel(**config['model']['params'])

    # Lightning checkpoint: must be weights_only=False
    state = torch.load(VQGAN_CKPT, map_location='cpu', weights_only=False)

    # IMPORTANT: drop loss/discriminator weights
    sd = state["state_dict"]
    sd = {k: v for k, v in sd.items() if not k.startswith("loss.")}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    print("Example missing keys:", missing[:5])

    model.eval().to(DEVICE)
    return model

def preprocess_image(img_path, model):
    # FAST WAY: Use NumPy
    try:
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        arr = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(arr)
        img_tensor = img_tensor / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            z, _, info = model.encode(img_tensor)
            indices = info[2].reshape(-1)
            
            # SAFETY CHECK
            assert indices.numel() == 256, f"Expected 256 tokens, got {indices.numel()}"
            
        return indices.cpu().tolist()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None # Handle bad images gracefully

def main():
    # 1. Setup Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<IMG_START>', '<IMG_END>']})
    
    IMG_START_ID = tokenizer.convert_tokens_to_ids('<IMG_START>')
    IMG_END_ID = tokenizer.convert_tokens_to_ids('<IMG_END>')
    VOCAB_OFFSET = len(tokenizer) 
    
    print(f"IDs: START={IMG_START_ID}, END={IMG_END_ID}, OFFSET={VOCAB_OFFSET}")
    
    # 2. Process Data
    vqgan = load_vqgan()
    processed_samples = []
    
    with open(os.path.join(DATA_DIR, "dataset.jsonl"), "r") as f:
        lines = f.readlines()
        
    print(f"Processing {len(lines)} samples...")
    for line in tqdm(lines):
        entry = json.loads(line)
        
        # Helper to skip bad images
        img_indices = preprocess_image(entry['image_path'], vqgan)
        if img_indices is None: continue 
        
        prompt_ids = tokenizer.encode(entry['prompt'])
        answer_ids = tokenizer.encode(" " + entry['answer'])
        
        shifted_img_indices = [idx + VOCAB_OFFSET for idx in img_indices]
        
        full_sequence = (
            prompt_ids + 
            [IMG_START_ID] + 
            shifted_img_indices + 
            [IMG_END_ID] + 
            answer_ids
        )
        processed_samples.append(torch.tensor(full_sequence, dtype=torch.long))
        
        # --- SANITY CHECKS (First Sample Only) ---
        if len(processed_samples) == 1:
            print(f"\n--- SANITY CHECK: First Sample ---")
            print(f"Full Sequence Length: {len(full_sequence)}")
            
            # Find image block
            try:
                start_idx = full_sequence.index(IMG_START_ID) + 1
                end_idx = full_sequence.index(IMG_END_ID)
                img_block = full_sequence[start_idx:end_idx]
                
                print(f"Image Tokens: {len(img_block)}")
                img_min, img_max = min(img_block), max(img_block)
                print(f"Image Token Range: [{img_min}, {img_max}]")
                
                # Assertions
                assert len(img_block) == 256, f"Expected 256 image tokens, got {len(img_block)}"
                assert img_min >= VOCAB_OFFSET, f"Image token {img_min} is below offset {VOCAB_OFFSET}"
                assert img_max < VOCAB_OFFSET + 16384, f"Image token {img_max} is out of range"
                print("All checks passed for first sample!\n")
            except Exception as e:
                print(f"Sanity check failed: {e}")
                import sys; sys.exit(1)

    # 3. Save with METADATA
    print("Saving to .pt file...")
    torch.save({
        "samples": processed_samples,
        "meta": {
            "vocab_size_total": VOCAB_OFFSET + 16384,
            "img_start_id": IMG_START_ID,
            "img_end_id": IMG_END_ID,
            "vocab_offset": VOCAB_OFFSET,
            "block_image_tokens": 256
        }
    }, OUTPUT_FILE)
    print(f"Saved {len(processed_samples)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
