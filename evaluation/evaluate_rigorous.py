import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import sys
import os
import math
import argparse
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_imaginer import GPT, GPTConfig 
import random
from tqdm import tqdm

# --- CONFIG ---
DATA_DIR = "processed_v2"  # Default --data: only test.pt is read
NUM_SAMPLES = 1000  # Default --n: distinct test configurations to score
IMG_TOKENS = 256
BLOCK_SIZE = 512
TEMPERATURE = 0.8  # For "Creative" Imagination
Z_95 = 1.96  # Normal quantile for 95% confidence intervals

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

def crop_to_block(x):
    if x.size(1) > BLOCK_SIZE: return x[:, -BLOCK_SIZE:]
    return x

def get_answer_probs_stable(model, input_ids, yes_id, no_id):
    input_ids = crop_to_block(input_ids)
    with torch.no_grad():
        logits, _ = model(input_ids)
        next_logits = logits[:, -1, :] 
        relevant_logits = torch.stack([next_logits[0, yes_id], next_logits[0, no_id]])
        probs = torch.softmax(relevant_logits, dim=0)
        p_yes, p_no = probs[0].item(), probs[1].item()
        pred = yes_id if p_yes > p_no else no_id
        return pred

def wilson_ci(k, n, z=Z_95):
    """95% Wilson score interval for k correct answers out of n."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

def config_key(sample):
    c = sample['config']
    return (tuple(c['square_start']), tuple(c['circle']), tuple(c['move']))

def stratum(sample):
    return sample['label']

def select_items(samples, n, seed):
    """Picks up to n items with distinct configurations, in the test split's own class mix.
    Renders of one configuration share a prompt, so they are not independent: keep one."""
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    pools, seen = defaultdict(list), set()
    for i in order:
        key = config_key(samples[i])
        if key not in seen:
            seen.add(key)
            pools[stratum(samples[i])].append(i)

    shares = {g: c / len(samples) for g, c in Counter(stratum(s) for s in samples).items()}
    cap = min(int(len(pools[g]) / share) for g, share in shares.items())
    if n > cap:
        print(f"Note: the test split has enough distinct configurations for only {cap} items in its class mix; using {cap}.")
        n = cap
    return [i for g, share in sorted(shares.items()) for i in pools[g][:round(n * share)]]

def evaluate():
    parser = argparse.ArgumentParser(description="Forced-choice evaluation on the held-out test split.")
    parser.add_argument("--data", default=DATA_DIR, help="directory with test.pt from preprocess.py")
    parser.add_argument("--image-ckpt", required=True, help="checkpoint of the model trained with image tokens")
    parser.add_argument("--n", type=int, default=NUM_SAMPLES, help="number of distinct test items to score")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    test_path = os.path.join(args.data, "test.pt")
    print(f"Loading {args.image_ckpt} and {test_path}...")
    data = torch.load(test_path, weights_only=False)
    samples = data['samples']
    meta = data['meta']
    
    config = GPTConfig(vocab_size=meta['vocab_size_total'], block_size=BLOCK_SIZE)
    model = GPT(config)
    model.load_state_dict(torch.load(args.image_ckpt, map_location=DEVICE, weights_only=False))
    model.eval().to(DEVICE)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    yes_ids, no_ids = tokenizer.encode(" Yes"), tokenizer.encode(" No")
    assert len(yes_ids)==1 and len(no_ids)==1, "Error: Yes/No tokens ambiguous."
    yes_id, no_id = yes_ids[0], no_ids[0]
    
    indices = select_items(samples, args.n, args.seed)
    
    results = {
        "teacher": 0, "blindfold": 0, "text_only": 0, 
        "imagined_greedy": 0, "imagined_sampled": 0,
        "total_valid": 0, "gt_yes_count": 0
    }
    
    print(f"\nRunning rigorous evaluation on {len(indices)} test items...")
    vocab_offset = meta['vocab_offset']

    for i, idx in enumerate(tqdm(indices)):
        s = samples[idx]
        seq = s['tokens'].to(DEVICE).unsqueeze(0)
        
        # Parse Sequence with the stored index: [prompt <IMG_START>] [256 image tokens] [<IMG_END>] [answer]
        ls = s['loss_start']
        prompt_part = seq[:, :ls] 
        real_image_part = seq[:, ls : ls + IMG_TOKENS]
        end_part = seq[:, ls + IMG_TOKENS : ls + IMG_TOKENS + 1]
        assert end_part.item() == meta['img_end_id'], "Unexpected sequence layout"
        
        # Ground truth comes from the stored label
        true_token = yes_id if s['label'] == "Yes" else no_id
            
        results["total_valid"] += 1
        if true_token == yes_id: results["gt_yes_count"] += 1
        
        # --- TEST 1: TEACHER FORCED (Ceiling) ---
        teacher_input = torch.cat((prompt_part, real_image_part, end_part), dim=1)
        if get_answer_probs_stable(model, teacher_input, yes_id, no_id) == true_token:
            results["teacher"] += 1
            
        # --- TEST 2: BLINDFOLD (Noise Baseline) ---
        noise = torch.randint(vocab_offset, meta['vocab_size_total'], (1, IMG_TOKENS)).to(DEVICE)
        blind_input = torch.cat((prompt_part, noise, end_part), dim=1)
        if get_answer_probs_stable(model, blind_input, yes_id, no_id) == true_token:
            results["blindfold"] += 1
            
        # --- TEST 3: TEXT-ONLY (Just Prompt + Start/End) ---
        text_only_input = torch.cat((prompt_part, end_part), dim=1)
        if get_answer_probs_stable(model, text_only_input, yes_id, no_id) == true_token:
            results["text_only"] += 1

        # --- TEST 4: IMAGINED (Greedy) ---
        curr_input = prompt_part
        with torch.no_grad():
            for _ in range(IMG_TOKENS):
                curr_input = crop_to_block(curr_input)
                logits, _ = model(curr_input)
                logits[:, :, :vocab_offset] = -1e10
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                curr_input = torch.cat((curr_input, next_token), dim=1)
        
        dream_greedy = torch.cat((curr_input, end_part), dim=1)
        if get_answer_probs_stable(model, dream_greedy, yes_id, no_id) == true_token:
            results["imagined_greedy"] += 1

        # --- TEST 4: IMAGINED (Sampled - Creative) ---
        curr_input_samp = prompt_part
        with torch.no_grad():
            for _ in range(IMG_TOKENS):
                curr_input_samp = crop_to_block(curr_input_samp)
                logits, _ = model(curr_input_samp)
                logits = logits[:, -1, :] / TEMPERATURE
                logits[:, :vocab_offset] = -1e10
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                curr_input_samp = torch.cat((curr_input_samp, next_token), dim=1) # unsqueeze fix
        
        dream_sampled = torch.cat((curr_input_samp, end_part), dim=1)
        if get_answer_probs_stable(model, dream_sampled, yes_id, no_id) == true_token:
            results["imagined_sampled"] += 1

    # --- FINAL REPORT ---
    total = results['total_valid']
    if total == 0: return print("Error: No valid samples.")
    
    # Class Imbalance Check: the scored items, and the whole test split
    yes_rate = results['gt_yes_count']/total * 100
    majority_baseline = max(yes_rate, 100 - yes_rate)
    split_yes_rate = sum(s['label'] == "Yes" for s in samples) / len(samples) * 100

    def report(name, key):
        """Prints accuracy with its 95% Wilson interval and returns the accuracy in %."""
        lo, hi = wilson_ci(results[key], total)
        acc = results[key]/total*100
        print(f"{name:<34}{acc:6.1f}%   95% CI [{lo*100:5.1f}, {hi*100:5.1f}]")
        return acc

    print("\n" + "="*66)
    print(f"SCIENTIFIC RESULTS (N={total} distinct test items)")
    print("="*66)
    print(f"Scored Items Balance (Yes %): {yes_rate:.1f}%  -> Majority Baseline {majority_baseline:.1f}% (Beat this to be useful)")
    print(f"Full Test Split (Yes %):      {split_yes_rate:.1f}%  -> Majority Baseline "
          f"{max(split_yes_rate, 100 - split_yes_rate):.1f}% ({len(samples)} samples)")
    print("-" * 66)
    acc_teach = report("Teacher Forced (Ceiling):", "teacher")
    acc_blind = report("Blindfold (Noise):", "blindfold")
    acc_text = report("Text-Only (No Image):", "text_only")
    print("-" * 66)
    acc_greedy = report("Imagined (Greedy):", "imagined_greedy")
    acc_sampled = report(f"Imagined (Sampled T={TEMPERATURE}):", "imagined_sampled")
    print("="*66)
    
    # VERDICT LOGIC
    print("\nVERDICT:")
    # 1. Did it learn to read images?
    if acc_teach > acc_blind + 10:
        print(" PASS: Model relies on visual data (Teacher >> Blindfold).")
    else:
        print(" FAIL: Model ignores images.")

    # 2. Does imagination work?
    best_imagination = max(acc_greedy, acc_sampled)
    if best_imagination > acc_blind + 5:
        print(f" PASS: Internal imagination works ({best_imagination:.1f}% vs {acc_blind:.1f}%).")
    elif best_imagination > majority_baseline:
        print(" WEAK PASS: Better than guessing, but not much better than noise.")
    else:
        print(" FAIL: Imagination is useless.")

if __name__ == "__main__":
    evaluate()
