import os
import random
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
DATASET_SIZE = 100000  # Default total samples (half Yes, half No); override with --size
OUTPUT_DIR = "dataset_imagination_v2"  # Default --out
IMG_SIZE = 256
GRID_SCALE = 10
OBJ_SIZE = 20
UNIT_PX = IMG_SIZE // GRID_SCALE  # Pixels per grid unit (approx 25)
SPLITS = {"train": 0.90, "val": 0.05, "test": 0.05}  # Fractions of unique configurations

# Setup Colors
COLORS = { "red": (255, 0, 0), "blue": (0, 0, 255) }

class Shape:
    def __init__(self, color_rgb, x, y, shape_type):
        self.color_rgb = color_rgb
        self.x = x
        self.y = y
        self.type = shape_type  # "circle" or "square"

    def get_bbox(self):
        return [self.x - OBJ_SIZE, self.y - OBJ_SIZE, 
                self.x + OBJ_SIZE, self.y + OBJ_SIZE]

def check_overlap_dist(x1, y1, x2, y2):
    """Returns True if objects at (x1,y1) and (x2,y2) overlap."""
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    # Threshold: slightly less than 2*radius to ensure solid overlap
    return dist < (OBJ_SIZE * 1.5)

def render_scene(anchor, mover, filename):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    
    # 1. Anti-Cheat Noise
    for _ in range(random.randint(5, 15)):
        x1, y1 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        x2, y2 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        fill = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
        draw.line([(x1, y1), (x2, y2)], fill=fill, width=1)
    
    # 2. Draw Objects
    for s in [anchor, mover]:
        bbox = s.get_bbox()
        if s.type == "circle":
            draw.ellipse(bbox, fill=s.color_rgb, outline="black")
        else:
            draw.rectangle(bbox, fill=s.color_rgb, outline="black")
            
    img.save(filename)

def get_valid_coordinate():
    """Returns a random valid pixel coordinate within grid bounds (with padding)."""
    margin = OBJ_SIZE + 5
    return random.randint(margin, IMG_SIZE - margin)

def generate_sample(sample_id, force_overlap, out_dir):
    """
    force_overlap (bool): If True, we engineer the start pos so they collide.
                          If False, we ensure they DO NOT collide.
    """
    
    max_retries = 100
    for _ in range(max_retries):
        
        # 1. Define the Anchor (Static Blue Circle)
        # We place it somewhat centrally to increase chance of valid moves
        anchor_x = random.randint(50, 200)
        anchor_y = random.randint(50, 200)
        
        # 2. Define the Move Vector (Grid Units)
        dx = random.choice([-3, -2, -1, 1, 2, 3])
        dy = random.choice([-3, -2, -1, 1, 2, 3])
        move_px_x = dx * UNIT_PX
        move_px_y = dy * UNIT_PX
        
        jitter = random.randint(-5, 5)

        if force_overlap:
            # === REVERSE ENGINEERING ===
            # We want the Final Position to be roughly where the Anchor is.
            # Final_Mover = Anchor + Small_Random_Offset (so it's not perfect center-on-center)
            offset_x = random.randint(-15, 15) 
            offset_y = random.randint(-15, 15)
            
            final_mover_x = anchor_x + offset_x
            final_mover_y = anchor_y + offset_y
            
            # Back-calculate Start Position
            # Start = Final - Move
            start_mover_x = final_mover_x - move_px_x + jitter
            start_mover_y = final_mover_y - move_px_y + jitter
            
        else:
            # === RANDOM GENERATION (likely no overlap) ===
            start_mover_x = get_valid_coordinate()
            start_mover_y = get_valid_coordinate()
            
            # Calculate Final
            final_mover_x = start_mover_x + move_px_x + jitter
            final_mover_y = start_mover_y + move_px_y + jitter
            
            # If we accidentally created an overlap, RETRY (we want a strict 'No')
            if check_overlap_dist(final_mover_x, final_mover_y, anchor_x, anchor_y):
                continue

        # 3. Validation: Is the Start Position on screen?
        margin = OBJ_SIZE
        if not (margin < start_mover_x < IMG_SIZE - margin and 
                margin < start_mover_y < IMG_SIZE - margin):
            continue # Retry with new coordinates
            
        # 4. Success! Create Objects
        anchor = Shape(COLORS["blue"], anchor_x, anchor_y, "circle")
        mover = Shape(COLORS["red"], start_mover_x, start_mover_y, "square")
        
        # Update Mover to Final State for Rendering
        mover_final = Shape(COLORS["red"], final_mover_x, final_mover_y, "square")
        
        # Render
        img_filename = os.path.join(out_dir, "images", f"img_{sample_id:06d}.png")
        render_scene(anchor, mover_final, img_filename)
        
        # Text Prompt
        # Convert pixel start positions back to approximate grid units for the prompt
        # (We round to nearest integer so the text looks like clean logic)
        grid_start_x = int(start_mover_x / UNIT_PX)
        grid_start_y = int(start_mover_y / UNIT_PX)
        grid_anchor_x = int(anchor_x / UNIT_PX)
        grid_anchor_y = int(anchor_y / UNIT_PX)
        
        prompt = (f"Imagine a red square at grid ({grid_start_x}, {grid_start_y}) "
                  f"and a blue circle at ({grid_anchor_x}, {grid_anchor_y}). "
                  f"Move red square {dx} right, {dy} down. Overlap?")
        
        return {
            "id": sample_id,
            "prompt": prompt,
            "image_path": os.path.relpath(img_filename, out_dir),  # relative to the dataset dir
            "answer": "Yes" if force_overlap else "No",
            # The configuration is everything the prompt states; splits are made over these
            "config": {"square_start": [grid_start_x, grid_start_y],
                       "circle": [grid_anchor_x, grid_anchor_y],
                       "move": [dx, dy]},
        }

    print("Warning: Could not generate valid sample after retries.")
    return None

def config_key(config):
    return (tuple(config["square_start"]), tuple(config["circle"]), tuple(config["move"]))

def split_configs(keys, seed):
    """Assigns each unique configuration to train/val/test (90/5/5), shuffled with a fixed seed."""
    keys = sorted(set(keys))
    random.Random(seed).shuffle(keys)
    n_val = round(len(keys) * SPLITS["val"])
    n_test = round(len(keys) * SPLITS["test"])
    return {k: "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for i, k in enumerate(keys)}

# --- MAIN LOOP ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the synthetic overlap dataset (train/val/test).")
    parser.add_argument("--size", type=int, default=DATASET_SIZE, help="total number of samples")
    parser.add_argument("--seed", type=int, default=0, help="seed for sampling, noise lines and the split")
    parser.add_argument("--out", default=OUTPUT_DIR, help="output directory (must be new or empty)")
    args = parser.parse_args()

    # Never clear an existing directory: --out is user-supplied
    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} exists and is not empty; delete it or pass a different --out.")
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    random.seed(args.seed)

    print(f"Generating {args.size} balanced samples...")

    data_list = []

    for i in range(args.size):
        # Force 50/50 Split
        want_overlap = (i % 2 == 0)

        sample = generate_sample(i, want_overlap, args.out)
        if sample:
            data_list.append(sample)

        if i % 100 == 0:
            print(f"Progress: {i}/{args.size}", end="\r")

    # Split by configuration, not by sample: one prompt can be rendered many times with
    # different noise lines, and every rendering of it must land in the same split.
    split_of = split_configs([config_key(s["config"]) for s in data_list], args.seed)
    splits = {name: [] for name in SPLITS}
    for s in data_list:
        s["split"] = split_of[config_key(s["config"])]
        splits[s["split"]].append(s)

    # LEAKAGE CHECK: no test prompt may appear in train or val (and no val prompt in train)
    prompts = {name: {s["prompt"] for s in rows} for name, rows in splits.items()}
    assert not prompts["test"] & (prompts["train"] | prompts["val"]), "Test prompt leaked into train/val!"
    assert not prompts["val"] & prompts["train"], "Val prompt leaked into train!"

    print()
    for name, rows in splits.items():
        # Shuffle so the model doesn't learn "Yes, No, Yes, No" pattern
        random.shuffle(rows)
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for entry in rows:
                f.write(json.dumps(entry) + "\n")
        n_yes = sum(s["answer"] == "Yes" for s in rows)
        print(f"{name}: {len(rows)} samples ({n_yes} Yes) from {len(prompts[name])} configurations")

    print(f"Completed! Saved to {args.out}")