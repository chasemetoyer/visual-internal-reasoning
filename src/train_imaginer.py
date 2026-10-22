import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import time
import os
import argparse

# --- CONFIGURATION ---
DATA_DIR = "processed_v2" # Default --data: train.pt / val.pt written by preprocess.py
OUT_DIR = "checkpoints" # Default --out-dir
BATCH_SIZE = 8 # Keep this low for Mac/MPS stability
BLOCK_SIZE = 512 # Context window size
LEARNING_RATE = 3e-4 # Learning rate
MAX_ITERS = 30000 # SCALED UP: 2.5k -> 30k (This is roughly 2.5 epochs)
VAL_INTERVAL = 1000 # Validate every 1000 iterations
SAVE_INTERVAL = 5000 # Save every 5000 iterations
FORMATS = ("image", "text", "filler") # Built by preprocess.py; identical training for all three
CKPT_VERSION = "v2" # In checkpoint names and contents. v2 = pre-LN residual fix; untagged checkpoints are invalid

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'
print(f"Using Device: {DEVICE}")
# On NVIDIA GPUs with tensor cores (A100, H100, L4), run float32 matmuls as TF32: several times faster,
# slightly less precise. No effect on MPS, the CPU, or GPUs without TF32 such as the T4.
torch.backends.cuda.matmul.allow_tf32 = True

# --- 1. DATASET WITH METADATA LOADING ---
class ImaginationDataset(Dataset):
    def __init__(self, pt_path, fmt):
        data = torch.load(pt_path, weights_only=False)
        self.samples = data['samples']
        self.meta = data['meta'] # Load the Truth
        self.fmt = fmt
        assert fmt in self.meta['formats'], f"{pt_path} has no '{fmt}' format"
        
        self.vocab_size = self.meta['vocab_size_total']
        self.img_start_id = self.meta['img_start_id']
        self.img_end_id = self.meta['img_end_id']
        
        print(f"Loaded {len(self.samples)} samples from {pt_path} ({fmt} format). Vocab: {self.vocab_size}, START: {self.img_start_id}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq, loss_start, answer_start = s["tokens"][self.fmt], s["loss_start"][self.fmt], s["answer_start"][self.fmt]
        if len(seq) > BLOCK_SIZE:
            cut = len(seq) - BLOCK_SIZE
            seq = seq[cut:]
            loss_start, answer_start = max(loss_start - cut, 1), max(answer_start - cut, 1)

        x = seq[:-1].clone()
        y = seq[1:].clone()

        # Mask Prompt using the STORED loss-start index (y[j] is the target for seq[j + 1])
        y[:loss_start - 1] = -100 # IGNORE_INDEX
        # Answer-only targets: the one loss that is comparable across sequence formats
        a = y.clone()
        a[:answer_start - 1] = -100
        return x, y, a

def pad_batch(batch):
    """Pads a batch to its longest sequence instead of BLOCK_SIZE. Attention is causal, so padding never
    affects a real position, and its targets are ignored: the loss is unchanged, the compute is not wasted."""
    xs, ys, answers = zip(*batch)
    return (pad_sequence(xs, batch_first=True, padding_value=0),
            pad_sequence(ys, batch_first=True, padding_value=-100),
            pad_sequence(answers, batch_first=True, padding_value=-100))

# --- 2. MODEL (Standard) ---
class GPT(nn.Module):
    # ... (Same GPT class as before, just pasting the essentials to save space) ...
    # ... Assume standard nanoGPT implementation here ...
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[
            Block(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.token_embedding_table(idx) + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

# Helper classes for GPT (Standard)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # No longer used for masking (scaled_dot_product_attention applies the causal mask itself);
        # kept so checkpoints saved before that change still load.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # PyTorch's fused causal attention (FlashAttention on recent NVIDIA GPUs; also runs on MPS and CPU):
        # softmax(q @ k^T / sqrt(head_size) with future positions masked) @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        )
    def forward(self, x):
        # Standard pre-LN block: each sub-layer's output is added to the residual stream
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer=6, n_head=6, n_embd=384):
        self.vocab_size = vocab_size; self.block_size = block_size
        self.n_layer = n_layer; self.n_head = n_head; self.n_embd = n_embd

# --- CHECKPOINTS ---
def config_id(config):
    """Hashable ID of a configuration (square start, circle, move): the unit the splits are made of."""
    return (*config["square_start"], *config["circle"], *config["move"])

def save_checkpoint(model, config, meta, fmt, train_configs, path):
    """Saves the weights with everything needed to rebuild and check them (see load_checkpoint).
    train_configs records which configurations the model trained on, so evaluation can verify that
    its test items were held out: meta only describes the vocabulary and is the same for every dataset."""
    torch.save({"version": CKPT_VERSION, "format": fmt, "config": vars(config), "meta": meta,
                "train_configs": train_configs, "state_dict": model.state_dict()}, path)

def load_checkpoint(path, device, expected_format=None):
    """Loads a checkpoint written by save_checkpoint and returns (model in eval mode, checkpoint).
    Refuses anything else, including every checkpoint saved before the residual-stream fix."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and ckpt.get("version") == CKPT_VERSION):
        raise SystemExit(f"{path} is not a {CKPT_VERSION} checkpoint. Checkpoints saved before the "
                         f"residual-stream fix are invalid; retrain with src/train_imaginer.py.")
    if expected_format is not None and ckpt["format"] != expected_format:
        raise SystemExit(f"{path} was trained on the '{ckpt['format']}' format, not '{expected_format}'.")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device), ckpt

# --- 3. TRAINING LOOP ---
def check_loss_mask(dataset):
    """CRITICAL CHECK: the trained targets of sample 0 are exactly what its format should predict."""
    _, y, a = dataset[0]
    targets, answer = y[y != -100], a[a != -100]
    n_img = dataset.meta['block_image_tokens']
    assert len(answer) >= 1 and torch.equal(targets[-len(answer):], answer)
    if dataset.fmt == "image":
        # The image tokens, <IMG_END>, then the answer
        assert len(targets) == n_img + 1 + len(answer), f"image: {len(targets)} trained targets"
        assert (targets[:n_img] >= dataset.meta['vocab_offset']).all() and targets[n_img] == dataset.img_end_id
    else:
        # text and filler: the answer only (the filler's <THINK> tokens are input, never targets)
        assert torch.equal(targets, answer), f"{dataset.fmt}: {len(targets)} trained targets, expected the answer only"
    print(f"Loss mask OK ({dataset.fmt}): {len(targets)} trained targets per sample, {len(answer)} of them the answer")

@torch.no_grad()
def estimate_val_loss(model, loader):
    """Token-averaged loss over the whole val split: (all trained tokens, answer tokens only)."""
    model.eval()
    sums, counts = [0.0, 0.0], [0, 0]
    for xb, yb, ab in loader:
        logits, _ = model(xb.to(DEVICE))
        logits = logits.view(-1, logits.size(-1))
        for k, targets in enumerate((yb, ab)):
            targets = targets.to(DEVICE).view(-1)
            sums[k] += F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum").item()
            counts[k] += (targets != -100).sum().item()
    model.train()
    return sums[0] / counts[0], sums[1] / counts[1]

def main():
    parser = argparse.ArgumentParser(description="Train the imaginer on the train split.")
    parser.add_argument("--format", required=True, choices=FORMATS,
                        help="image: prompt, image tokens, answer | text: prompt, answer | "
                             "filler: prompt, 256 <THINK> tokens, answer")
    parser.add_argument("--data", default=DATA_DIR, help="directory with train.pt and val.pt")
    parser.add_argument("--out-dir", default=OUT_DIR, help="where checkpoints are written")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--val-interval", type=int, default=VAL_INTERVAL)
    parser.add_argument("--seed", type=int, default=0, help="seed for weight init and batch order")
    args = parser.parse_args()

    # Train on the train split only; val is for monitoring, test is never touched here
    dataset = ImaginationDataset(os.path.join(args.data, "train.pt"), args.format)
    val_dataset = ImaginationDataset(os.path.join(args.data, "val.pt"), args.format)
    assert val_dataset.meta == dataset.meta, "train.pt and val.pt were built with different vocabularies"
    check_loss_mask(dataset)
    train_configs = sorted({config_id(s["config"]) for s in dataset.samples})

    torch.manual_seed(args.seed)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch,
                            generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_batch)
    
    config = GPTConfig(vocab_size=dataset.vocab_size, block_size=BLOCK_SIZE)
    model = GPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    os.makedirs(args.out_dir, exist_ok=True)
    
    iter_num = 0
    model.train()
    start = time.time()
    
    print("Starting Training...")
    while iter_num < args.max_iters:
        for xb, yb, _ in dataloader:
            if iter_num >= args.max_iters: break
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            # CRITICAL CHECK: Max ID
            if iter_num == 0:
                print(f"DEBUG: Max Token ID in batch: {xb.max().item()} (Vocab: {dataset.vocab_size})")
                assert xb.max().item() < dataset.vocab_size, "Token ID overflow!"

            logits, loss = model(xb, yb)
            
            optimizer.zero_grad()
            loss.backward()
            
            # STABILITY FIX: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # --- PROGRESS SAVING ---
            if iter_num > 0 and iter_num % SAVE_INTERVAL == 0:
                checkpoint_name = os.path.join(args.out_dir, f"imaginer_{CKPT_VERSION}_{args.format}_ckpt_{iter_num}.pth")
                save_checkpoint(model, config, dataset.meta, args.format, train_configs, checkpoint_name)
                print(f"--> Saved backup: {checkpoint_name}")
            
            if iter_num % 100 == 0:
                print(f"Iter {iter_num}: Loss {loss.item():.4f}")

            # --- VALIDATION ---
            if iter_num > 0 and iter_num % args.val_interval == 0:
                val_loss, val_answer_loss = estimate_val_loss(model, val_loader)
                print(f"Iter {iter_num}: Val loss {val_loss:.4f} | Val answer loss {val_answer_loss:.4f}")
            iter_num += 1

    print(f"Done in {time.time()-start:.1f}s")
    val_loss, val_answer_loss = estimate_val_loss(model, val_loader)
    print(f"Final: Val loss {val_loss:.4f} | Val answer loss {val_answer_loss:.4f}")
    final_name = os.path.join(args.out_dir, f"imaginer_{CKPT_VERSION}_{args.format}_final.pth")
    save_checkpoint(model, config, dataset.meta, args.format, train_configs, final_name)
    print(f"Saved {final_name}")

if __name__ == "__main__":
    main()