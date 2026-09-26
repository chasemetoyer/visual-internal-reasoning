import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_imaginer import GPT, GPTConfig 
import random
from tqdm import tqdm

# --- CONFIG ---
MODEL_PATH = "imaginer_ckpt_20000.pth"
DATA_PATH = "train_data_final.pt"
NUM_SAMPLES = 200  # Increased for statistical significance
IMG_TOKENS = 256
BLOCK_SIZE = 512
TEMPERATURE = 0.8  # For "Creative" Imagination

# Reproducibility
random.seed(42)
torch.manual_seed(42)

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

def evaluate():
    print(f"Loading {MODEL_PATH}...")
    data = torch.load(DATA_PATH, weights_only=False)
    samples = data['samples']
    meta = data['meta']
    
    config = GPTConfig(vocab_size=meta['vocab_size_total'], block_size=BLOCK_SIZE)
    model = GPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval().to(DEVICE)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    yes_ids, no_ids = tokenizer.encode(" Yes"), tokenizer.encode(" No")
    assert len(yes_ids)==1 and len(no_ids)==1, "Error: Yes/No tokens ambiguous."
    yes_id, no_id = yes_ids[0], no_ids[0]
    
    indices = random.sample(range(len(samples)), NUM_SAMPLES)
    
    results = {
        "teacher": 0, "blindfold": 0, "text_only": 0, 
        "imagined_greedy": 0, "imagined_sampled": 0,
        "total_valid": 0, "gt_yes_count": 0
    }
    
    print(f"\nRunning rigorous evaluation on {NUM_SAMPLES} samples...")
    vocab_offset = meta['vocab_offset']

    for i, idx in enumerate(tqdm(indices)):
        seq = samples[idx].to(DEVICE).unsqueeze(0)
        
        # Parse Sequence
        try:
            start_pos = (seq[0] == meta['img_start_id']).nonzero(as_tuple=True)[0].item()
            end_pos = (seq[0] == meta['img_end_id']).nonzero(as_tuple=True)[0].item()
        except: continue
            
        prompt_part = seq[:, :start_pos+1] 
        real_image_part = seq[:, start_pos+1 : end_pos]
        end_part = seq[:, end_pos : end_pos+1]
        
        # Robust GT Parsing
        answer_seq = seq[0, end_pos+1 : end_pos+6].tolist()
        answer_text = tokenizer.decode(answer_seq).strip().lower()
        if answer_text.startswith("yes"): true_token = yes_id
        elif answer_text.startswith("no"): true_token = no_id
        else: continue
            
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
    
    # Calculate Percentages
    acc_teach = results['teacher']/total*100
    acc_blind = results['blindfold']/total*100
    acc_text = results['text_only']/total*100
    acc_greedy = results['imagined_greedy']/total*100
    acc_sampled = results['imagined_sampled']/total*100
    
    # Class Imbalance Check
    yes_rate = results['gt_yes_count']/total * 100
    majority_baseline = max(yes_rate, 100 - yes_rate)

    print("\n" + "="*50)
    print(f"SCIENTIFIC RESULTS (N={total})")
    print("="*50)
    print(f"Dataset Balance (Yes %): {yes_rate:.1f}%")
    print(f"Majority Baseline:       {majority_baseline:.1f}% (Beat this to be useful)")
    print("-" * 50)
    print(f"Teacher Forced (Ceiling): {acc_teach:.1f}%")
    print(f"Blindfold (Noise):        {acc_blind:.1f}%")
    print(f"Text-Only (No Image):     {acc_text:.1f}%")
    print("-" * 50)
    print(f"Imagined (Greedy):        {acc_greedy:.1f}%")
    print(f"Imagined (Sampled T={TEMPERATURE}): {acc_sampled:.1f}%")
    print("="*50)
    
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
