import os
import json
import argparse
import torch
import yaml
import numpy as np  # The speed booster
from PIL import Image
from tqdm import tqdm
from transformers import GPT2TokenizerFast

# taming-transformers imports pytorch_lightning.utilities.distributed, which was
# removed in newer Lightning versions; alias it to the current location.
import sys
import pytorch_lightning.utilities.rank_zero as _pl_rank_zero
sys.modules.setdefault("pytorch_lightning.utilities.distributed", _pl_rank_zero)

from taming.models.vqgan import VQModel

# --- CONFIG ---
DATA_DIR = "dataset_imagination_v2"  # Default --data (written by datafactory.py)
OUTPUT_DIR = "processed_v2"          # Default --out: gets train.pt, val.pt, test.pt
SPLITS = ("train", "val", "test")
FORMATS = ("image", "text", "filler")  # Sequence layouts; train_imaginer.py --format picks one
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
    parser = argparse.ArgumentParser(description="Tokenize prompts and VQGAN-encode images, one .pt per split.")
    parser.add_argument("--data", default=DATA_DIR, help="dataset directory written by datafactory.py")
    parser.add_argument("--out", default=OUTPUT_DIR, help="output directory for train.pt / val.pt / test.pt")
    args = parser.parse_args()

    # 1. Setup Tokenizer (one vocabulary shared by all three formats)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<IMG_START>', '<IMG_END>', '<THINK>']})

    IMG_START_ID = tokenizer.convert_tokens_to_ids('<IMG_START>')
    IMG_END_ID = tokenizer.convert_tokens_to_ids('<IMG_END>')
    THINK_ID = tokenizer.convert_tokens_to_ids('<THINK>')
    VOCAB_OFFSET = len(tokenizer)

    print(f"IDs: START={IMG_START_ID}, END={IMG_END_ID}, THINK={THINK_ID}, OFFSET={VOCAB_OFFSET}")

    entries = {}
    for split in SPLITS:
        with open(os.path.join(args.data, f"{split}.jsonl"), "r") as f:
            entries[split] = [json.loads(line) for line in f]

    # LEAKAGE CHECK: datafactory splits by configuration, so no test prompt may be in train/val
    prompts = {split: {e['prompt'] for e in rows} for split, rows in entries.items()}
    assert not prompts["test"] & (prompts["train"] | prompts["val"]), "Test prompt found in train/val!"

    # 2. Process Data
    vqgan = load_vqgan()
    os.makedirs(args.out, exist_ok=True)

    for split in SPLITS:
        processed_samples = []
        print(f"Processing {len(entries[split])} {split} samples...")
        for entry in tqdm(entries[split]):

            # Helper to skip bad images
            img_indices = preprocess_image(os.path.join(args.data, entry['image_path']), vqgan)
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
            # Same prompt and answer in every format; only what sits between them differs.
            # filler has the image format's length and compute, but no visual content.
            sequences = {
                "image": full_sequence,
                "text": prompt_ids + answer_ids,
                "filler": prompt_ids + [IMG_START_ID] + [THINK_ID] * len(shifted_img_indices) + [IMG_END_ID] + answer_ids,
            }
            processed_samples.append({
                "id": entry['id'],
                "tokens": {fmt: torch.tensor(seq, dtype=torch.long) for fmt, seq in sequences.items()},
                # Explicit indices into each format's tokens, so nothing downstream searches for <IMG_START>
                "loss_start": {  # first predicted token
                    "image": len(prompt_ids) + 1,  # the image tokens are generated, so they are trained
                    "text": len(prompt_ids),  # answer only
                    "filler": len(sequences["filler"]) - len(answer_ids),  # answer only: <THINK> is input, not predicted
                },
                "answer_start": {fmt: len(seq) - len(answer_ids) for fmt, seq in sequences.items()},
                "config": entry['config'],
                "label": entry['answer'],
                "near_miss": entry['near_miss'],
            })

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
                    assert len(sequences["filler"]) == len(full_sequence), "Filler and image formats differ in length"
                    assert sequences["text"] == prompt_ids + answer_ids
                    print("All checks passed for first sample!\n")
                except Exception as e:
                    print(f"Sanity check failed: {e}")
                    import sys; sys.exit(1)

        # 3. Save with METADATA
        out_path = os.path.join(args.out, f"{split}.pt")
        print(f"Saving to {out_path}...")
        torch.save({
            "samples": processed_samples,
            "meta": {
                "vocab_size_total": VOCAB_OFFSET + 16384,
                "img_start_id": IMG_START_ID,
                "img_end_id": IMG_END_ID,
                "think_id": THINK_ID,
                "vocab_offset": VOCAB_OFFSET,
                "block_image_tokens": 256,
                "formats": list(FORMATS)
            }
        }, out_path)
        print(f"Saved {len(processed_samples)} samples to {out_path}")

if __name__ == "__main__":
    main()
