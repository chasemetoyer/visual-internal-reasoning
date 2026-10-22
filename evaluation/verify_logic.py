import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_imaginer import load_checkpoint

# --- CONFIG ---
IMG_TOKENS = 256
ANSWER_TOKENS = 10
IGNORE_VAL = -1e10

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

def check_logic_strict():
    parser = argparse.ArgumentParser(description="Imagine a picture for one prompt, then answer it.")
    parser.add_argument("--ckpt", required=True, help="image-format checkpoint from src/train_imaginer.py")
    args = parser.parse_args()

    # 1. Load Model (the vocabulary IDs are stored in the checkpoint)
    print(f"Loading {args.ckpt}...")
    model, ckpt = load_checkpoint(args.ckpt, DEVICE, expected_format="image")
    meta = ckpt['meta']
    vocab_size = meta['vocab_size_total']
    img_start_id = meta['img_start_id']
    img_end_id = meta['img_end_id']
    vocab_offset = meta['vocab_offset']
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # 2. Setup Prompt
    prompt = "Imagine a red square at grid (2, 2) and a blue circle at (6, 6). Move red square 2 right, 0 down. Overlap?"
    print(f"QUESTION: {prompt}")
    
    input_ids = tokenizer.encode(prompt) + [img_start_id]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 3. Generate EXACTLY 256 Image Tokens
    print("Generating internal image...", end="")
    with torch.no_grad():
        for _ in range(IMG_TOKENS):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :].clone()
            
            # FORCE it to stay in "Image Mode" (Optional but good for stability)
            # Block all text tokens so it MUST pick a visual token
            logits[:, :vocab_offset] = IGNORE_VAL
            
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat((input_ids, next_token), dim=1)
    print(" Done.")

    # 4. Force <IMG_END>
    # We manually append the end token so the model knows the dream is over
    end_token_tensor = torch.tensor([[img_end_id]], device=DEVICE)
    input_ids = torch.cat((input_ids, end_token_tensor), dim=1)

    # 5. Generate Answer (STRICT TEXT ONLY)
    print("Generating answer...", end="")
    output = input_ids
    with torch.no_grad():
        for _ in range(ANSWER_TOKENS):
            logits, _ = model(output)
            logits = logits[:, -1, :].clone()
            
            # CRITICAL FIX: BAN IMAGE TOKENS
            # The model cannot output a visual token here. It must speak English.
            logits[:, vocab_offset:] = IGNORE_VAL 
            logits[:, img_start_id] = IGNORE_VAL # Don't start dreaming again
            
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            output = torch.cat((output, next_token), dim=1)
            
    # 6. Decode
    # We grab everything after the <IMG_END> token we appended
    seq = output[0].tolist()
    try:
        # Find the LAST occurrence of IMG_END (which we added)
        # Note: We manually added it at index: len(prompt) + 1 + 256
        end_pos = len(input_ids[0]) - 1 - ANSWER_TOKENS # Roughly here
        
        # Safer search:
        end_indices = [i for i, x in enumerate(seq) if x == img_end_id]
        real_end = end_indices[-1]
        
        answer_tokens = seq[real_end+1:]
        # Double check we don't have invalid tokens
        clean_tokens = [t for t in answer_tokens if t < vocab_offset]
        
        answer_text = tokenizer.decode(clean_tokens)
        print(f"\n\nSTRICT MODEL ANSWER: {answer_text}")
        
    except Exception as e:
        print(f"\nError decoding: {e}")

if __name__ == "__main__":
    check_logic_strict()