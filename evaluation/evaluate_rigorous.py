import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import sys
import os
import math
import argparse
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_imaginer import load_checkpoint, config_id, scratchpad_text
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

def stratum(sample):
    # Yes / near-miss No / far No, so the scored items keep the split's mix of hard negatives
    return (sample['label'], sample['near_miss'])

def select_items(samples, n, seed):
    """Picks up to n items with distinct configurations, in the test split's own class mix.
    Renders of one configuration share a prompt, so they are not independent: keep one."""
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    pools, seen = defaultdict(list), set()
    for i in order:
        key = config_id(samples[i]['config'])
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
    parser.add_argument("--image-ckpt", help="model trained on prompt, image tokens, answer")
    parser.add_argument("--text-ckpt", help="model trained on prompt, answer")
    parser.add_argument("--filler-ckpt", help="model trained on prompt, 256 <THINK> tokens, answer")
    parser.add_argument("--scratchpad-ckpt", help="model trained on prompt, the square's final grid cell as text, answer")
    parser.add_argument("--n", type=int, default=NUM_SAMPLES, help="number of distinct test items to score")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ckpts = {"image": args.image_ckpt, "text": args.text_ckpt, "filler": args.filler_ckpt,
             "scratchpad": args.scratchpad_ckpt}
    ckpts = {fmt: path for fmt, path in ckpts.items() if path}
    if not ckpts:
        parser.error("pass at least one of --image-ckpt, --text-ckpt, --filler-ckpt, --scratchpad-ckpt")

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    test_path = os.path.join(args.data, "test.pt")
    print(f"Loading {test_path}...")
    data = torch.load(test_path, weights_only=False)
    samples = data['samples']
    meta = data['meta']
    
    # Each checkpoint must be a v2 checkpoint of the format its flag names, built on this vocabulary
    test_configs = {config_id(s['config']) for s in samples}
    models = {}
    for fmt, path in ckpts.items():
        print(f"Loading {fmt} model from {path}...")
        models[fmt], ckpt = load_checkpoint(path, DEVICE, expected_format=fmt)
        assert ckpt['meta'] == meta, f"{path} was trained on a different vocabulary than {test_path}"
        # HELD-OUT CHECK: meta is the same for every dataset, so compare the configurations themselves.
        # A checkpoint trained on another split (e.g. data generated with a different --seed) would fail here.
        seen = test_configs & set(ckpt['train_configs'])
        if seen:
            raise SystemExit(f"{path} trained on {len(seen)} of the {len(test_configs)} test configurations in "
                             f"{test_path}; evaluate it on the test split of the dataset it was trained on.")
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    yes_ids, no_ids = tokenizer.encode(" Yes"), tokenizer.encode(" No")
    assert len(yes_ids)==1 and len(no_ids)==1, "Error: Yes/No tokens ambiguous."
    yes_id, no_id = yes_ids[0], no_ids[0]
    
    indices = select_items(samples, args.n, args.seed)
    
    results = {
        "teacher": 0, "blindfold": 0, "picture_removed": 0, 
        "imagined_greedy": 0, "imagined_sampled": 0,
        "text_model": 0, "filler_model": 0,
        "scratchpad_teacher": 0, "scratchpad_own": 0, "scratchpad_correct": 0,
        "total_valid": 0, "gt_yes_count": 0
    }
    
    print(f"\nRunning rigorous evaluation on {len(indices)} test items...")
    vocab_offset = meta['vocab_offset']

    for i, idx in enumerate(tqdm(indices)):
        s = samples[idx]
        
        # Ground truth comes from the stored label
        true_token = yes_id if s['label'] == "Yes" else no_id
            
        results["total_valid"] += 1
        if true_token == yes_id: results["gt_yes_count"] += 1

        # --- BASELINES: models trained without pictures, on the same items ---
        # Each input is its format's tokens up to the stored loss-start index (where the answer comes next)
        for fmt, key in (("text", "text_model"), ("filler", "filler_model")):
            if fmt in models:
                baseline_input = s['tokens'][fmt][:s['loss_start'][fmt]].to(DEVICE).unsqueeze(0)
                if get_answer_probs_stable(models[fmt], baseline_input, yes_id, no_id) == true_token:
                    results[key] += 1

        # --- SCRATCHPAD: writes the square's final grid cell as text, then answers ---
        # Greedy, so it uses no randomness and leaves the image model's noise and sampling below unchanged
        if "scratchpad" in models:
            scratch_model = models["scratchpad"]
            prompt_ids = s['tokens']['text'][:s['loss_start']['text']].to(DEVICE).unsqueeze(0)
            true_pad = torch.tensor(tokenizer.encode(scratchpad_text(s['config'])), device=DEVICE).unsqueeze(0)
            # Ceiling: the true final cell is given (teacher forced)
            if get_answer_probs_stable(scratch_model, torch.cat((prompt_ids, true_pad), dim=1), yes_id, no_id) == true_token:
                results["scratchpad_teacher"] += 1
            # Own scratchpad: the model writes as many text tokens as the true scratchpad has, then answers
            written = prompt_ids
            with torch.no_grad():
                for _ in range(true_pad.size(1)):
                    logits, _ = scratch_model(written)
                    logits = logits[:, -1, :].clone()
                    logits[:, meta['img_start_id']:] = -1e10  # special and image tokens: text only
                    written = torch.cat((written, torch.argmax(logits, dim=-1, keepdim=True)), dim=1)
            results["scratchpad_correct"] += int(torch.equal(written[:, prompt_ids.size(1):], true_pad))
            if get_answer_probs_stable(scratch_model, written, yes_id, no_id) == true_token:
                results["scratchpad_own"] += 1

        if "image" not in models: continue
        model = models["image"]
        seq = s['tokens']['image'].to(DEVICE).unsqueeze(0)
        
        # Parse Sequence with the stored index: [prompt <IMG_START>] [256 image tokens] [<IMG_END>] [answer]
        ls = s['loss_start']['image']
        prompt_part = seq[:, :ls] 
        real_image_part = seq[:, ls : ls + IMG_TOKENS]
        end_part = seq[:, ls + IMG_TOKENS : ls + IMG_TOKENS + 1]
        assert end_part.item() == meta['img_end_id'], "Unexpected sequence layout"
        
        # --- TEST 1: TEACHER FORCED (Ceiling) ---
        teacher_input = torch.cat((prompt_part, real_image_part, end_part), dim=1)
        if get_answer_probs_stable(model, teacher_input, yes_id, no_id) == true_token:
            results["teacher"] += 1
            
        # --- TEST 2: BLINDFOLD (Noise Baseline) ---
        noise = torch.randint(vocab_offset, meta['vocab_size_total'], (1, IMG_TOKENS)).to(DEVICE)
        blind_input = torch.cat((prompt_part, noise, end_part), dim=1)
        if get_answer_probs_stable(model, blind_input, yes_id, no_id) == true_token:
            results["blindfold"] += 1
            
        # --- TEST 3: PICTURE REMOVED (Just Prompt + Start/End) ---
        # Out of distribution: the image model never trained without a picture. Not a text-only baseline.
        picture_removed_input = torch.cat((prompt_part, end_part), dim=1)
        if get_answer_probs_stable(model, picture_removed_input, yes_id, no_id) == true_token:
            results["picture_removed"] += 1

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
        print(f"{name:<54}{acc:6.1f}%   95% CI [{lo*100:5.1f}, {hi*100:5.1f}]")
        return acc

    print("\n" + "="*86)
    print(f"SCIENTIFIC RESULTS (N={total} distinct test items, same items for every model)")
    print("="*86)
    print(f"Scored Items Balance (Yes %): {yes_rate:.1f}%  -> Majority Baseline {majority_baseline:.1f}% (Beat this to be useful)")
    print(f"Full Test Split (Yes %):      {split_yes_rate:.1f}%  -> Majority Baseline "
          f"{max(split_yes_rate, 100 - split_yes_rate):.1f}% ({len(samples)} samples)")
    n_near = sum(samples[i]['near_miss'] for i in indices)
    print(f"Scored Items: {results['gt_yes_count']} Yes, {n_near} No near misses, "
          f"{total - results['gt_yes_count'] - n_near} No far apart")
    if "image" in models:
        print("-" * 86)
        acc_teach = report("Image model, teacher forced (ceiling):", "teacher")
        acc_blind = report("Image model, blindfold (VQ noise):", "blindfold")
        report("Image model, picture removed (out of distribution):", "picture_removed")
        print("-" * 86)
        acc_greedy = report("Image model, imagined (greedy):", "imagined_greedy")
        acc_sampled = report(f"Image model, imagined (sampled T={TEMPERATURE}):", "imagined_sampled")
    if "text" in models or "filler" in models:
        print("-" * 86)
    if "text" in models:
        report("Text model (prompt, answer):", "text_model")
    if "filler" in models:
        report("Filler model (prompt, 256 <THINK> tokens, answer):", "filler_model")
    if "scratchpad" in models:
        print("-" * 86)
        report("Scratchpad model, true final cell given (ceiling):", "scratchpad_teacher")
        report("Scratchpad model, own scratchpad (greedy):", "scratchpad_own")
        report("  ...scratchpad written exactly right:", "scratchpad_correct")
    print("="*86)
    if "image" not in models: return
    
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
