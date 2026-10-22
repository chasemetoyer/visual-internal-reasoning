"""Synthetic dataset for "Does the red square overlap the blue circle after the move?".

Every sample satisfies these invariants. They are asserted while generating, and
src/audit_datafactory.py re-checks them from the written files.
- The prompt fully determines the answer. Shapes sit exactly at grid-cell centres
  (cell g is centred at GRID_ORIGIN_PX + g * UNIT_PX px) with no hidden jitter. Only
  the faint background lines are random.
- The label is computed from the shapes as drawn (overlap of their rendered masks)
  and cross-checked against the exact square-circle distance.
- Both shapes lie fully inside the frame and stay visible when they overlap: the
  shared region is drawn as a red/blue mix and both outlines are drawn last.
- No borderline cases. Every "Yes" reaches >= MIN_PENETRATION_PX into the circle and
  shares >= MIN_OVERLAP_PX pixels; every "No" leaves >= MIN_GAP_PX of background
  between the shapes. On this grid the closest cases are 12.9 px deep (164 shared
  pixels) and 10 px apart, so every configuration clears the margins.
- Classes are 50/50, and half of the "No" samples are near misses (gap < NEAR_MISS_PX).
"""
import os
import math
import random
import json
import argparse
import itertools
import numpy as np
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
DATASET_SIZE = 100000  # Default total samples (half Yes, half No); override with --size
OUTPUT_DIR = "dataset_imagination_v2"  # Default --out
IMG_SIZE = 256
UNIT_PX = 25  # Pixels per grid unit
GRID_ORIGIN_PX = 15  # Pixel centre of grid cell 0: cell g is centred at 15 + 25 * g
GRID_CELLS = range(1, 9)  # Cells 1..8 (centres 40..215 px) keep every shape >= 20 px inside the frame
OBJ_SIZE = 20  # Square half-width and circle radius (41 px across as drawn)
MOVES = [-3, -2, -1, 1, 2, 3]  # Grid units per axis
SPLITS = {"train": 0.90, "val": 0.05, "test": 0.05}  # Fractions of unique configurations
BUCKET_SHARES = {"yes": 0.50, "near_miss": 0.25, "far": 0.25}  # 50/50 classes; half of "No" are near misses

# Margins that rule out borderline cases
MIN_PENETRATION_PX = 8  # "Yes": circle radius minus the circle centre's distance to the square
MIN_OVERLAP_PX = 100  # "Yes": pixels shared by the rendered shapes
MIN_GAP_PX = 8  # "No": background between the shapes
NEAR_MISS_PX = UNIT_PX  # A "No" whose gap is under one grid unit is a near miss

# Setup Colors
COLORS = {"red": (255, 0, 0), "blue": (0, 0, 255), "overlap": (128, 0, 128)}  # overlap: 50/50 red-blue mix

def pixel_center(cell):
    """Exact pixel centre of a grid cell (no jitter)."""
    return (GRID_ORIGIN_PX + cell[0] * UNIT_PX, GRID_ORIGIN_PX + cell[1] * UNIT_PX)

def get_bbox(center):
    x, y = center
    return [x - OBJ_SIZE, y - OBJ_SIZE, x + OBJ_SIZE, y + OBJ_SIZE]

def clearance_px(square_center, circle_center):
    """Exact signed gap between the shapes: > 0 is background between them, < 0 is overlap depth."""
    dx = max(abs(square_center[0] - circle_center[0]) - OBJ_SIZE, 0)
    dy = max(abs(square_center[1] - circle_center[1]) - OBJ_SIZE, 0)
    return math.hypot(dx, dy) - OBJ_SIZE

def shape_masks(square_center, circle_center):
    """Pixel masks of the square and the circle, drawn with the same PIL calls as the image."""
    square, circle = Image.new("1", (IMG_SIZE, IMG_SIZE)), Image.new("1", (IMG_SIZE, IMG_SIZE))
    ImageDraw.Draw(square).rectangle(get_bbox(square_center), fill=1)
    ImageDraw.Draw(circle).ellipse(get_bbox(circle_center), fill=1)
    return np.array(square, dtype=bool), np.array(circle, dtype=bool)

def render_scene(square_center, circle_center, filename, rng):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)

    # 1. Anti-Cheat Noise (the only randomness in the picture)
    for _ in range(rng.randint(5, 15)):
        x1, y1 = rng.randint(0, IMG_SIZE), rng.randint(0, IMG_SIZE)
        x2, y2 = rng.randint(0, IMG_SIZE), rng.randint(0, IMG_SIZE)
        fill = (rng.randint(200, 240), rng.randint(200, 240), rng.randint(200, 240))
        draw.line([(x1, y1), (x2, y2)], fill=fill, width=1)

    # 2. Draw Objects: the shared region is a red/blue mix, so neither shape hides the other
    square, circle = shape_masks(square_center, circle_center)
    pixels = np.array(img)
    pixels[square] = COLORS["red"]
    pixels[circle] = COLORS["blue"]
    pixels[square & circle] = COLORS["overlap"]
    img = Image.fromarray(pixels)

    # 3. Outlines last, so both boundaries stay visible
    draw = ImageDraw.Draw(img)
    draw.rectangle(get_bbox(square_center), outline="black")
    draw.ellipse(get_bbox(circle_center), outline="black")
    img.save(filename)
    return square, circle, np.array(img)

def check_invariants(square_center, circle_center, square, circle, pixels):
    """Asserts the module docstring's invariants for one rendered sample; returns (label, overlap px, gap px)."""
    for x0, y0, x1, y1 in (get_bbox(square_center), get_bbox(circle_center)):
        assert 0 <= x0 and 0 <= y0 and x1 < IMG_SIZE and y1 < IMG_SIZE, "Shape leaves the frame"

    # Label from the shapes as drawn, cross-checked against the exact geometry
    overlap_px = int((square & circle).sum())
    gap = clearance_px(square_center, circle_center)
    label = "Yes" if overlap_px > 0 else "No"
    assert (gap < 0) == (label == "Yes"), "Rendered overlap disagrees with the exact geometry"

    if label == "Yes":
        assert -gap >= MIN_PENETRATION_PX and overlap_px >= MIN_OVERLAP_PX, \
            f"Borderline overlap: {-gap:.1f} px deep, {overlap_px} px shared"
    else:
        assert gap >= MIN_GAP_PX, f"Borderline gap: {gap:.1f} px"

    # Visibility: no pixel of either shape shows the other shape's own colour
    assert not (pixels[circle] == COLORS["red"]).all(axis=-1).any(), "Square hides part of the circle"
    assert not (pixels[square] == COLORS["blue"]).all(axis=-1).any(), "Circle hides part of the square"
    return label, overlap_px, gap

def final_cell(config):
    (sx, sy), _, (dx, dy) = config
    return (sx + dx, sy + dy)

def all_configs():
    """Every (square start, circle, move) on the grid whose moved square is still on the grid."""
    cells = list(itertools.product(GRID_CELLS, repeat=2))
    starts_moves = [(start, move) for start in cells for move in itertools.product(MOVES, repeat=2)
                    if start[0] + move[0] in GRID_CELLS and start[1] + move[1] in GRID_CELLS]
    return [(start, circle, move) for start, move in starts_moves for circle in cells]

def bucket(config):
    """Sampling bucket from the exact geometry: yes, near_miss or far. Labels still come from the drawing."""
    gap = clearance_px(pixel_center(final_cell(config)), pixel_center(config[1]))
    return "yes" if gap < 0 else "near_miss" if gap < NEAR_MISS_PX else "far"

def split_configs(keys, seed):
    """Assigns each unique configuration to train/val/test (90/5/5), shuffled with a fixed seed."""
    keys = sorted(set(keys))
    random.Random(seed).shuffle(keys)
    n_val = round(len(keys) * SPLITS["val"])
    n_test = round(len(keys) * SPLITS["test"])
    return {k: "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for i, k in enumerate(keys)}

def sample_configs(pools, n, rng):
    """Draws n configurations in BUCKET_SHARES proportions. Each shuffled pool is cycled, so every
    configuration is used once before any repeats; repeats get new noise lines."""
    n_yes = round(n * BUCKET_SHARES["yes"])
    n_near = round((n - n_yes) * BUCKET_SHARES["near_miss"] / (1 - BUCKET_SHARES["yes"]))
    counts = {"yes": n_yes, "near_miss": n_near, "far": n - n_yes - n_near}
    chosen = []
    for name, k in counts.items():
        pool = pools[name][:]
        rng.shuffle(pool)
        chosen += [pool[i % len(pool)] for i in range(k)]
    return chosen

def generate_sample(sample_id, config, out_dir, rng):
    (sx, sy), (cx, cy), (dx, dy) = config
    square_center = pixel_center(final_cell(config))  # The image shows the square after the move
    circle_center = pixel_center((cx, cy))

    # Render
    img_filename = os.path.join(out_dir, "images", f"img_{sample_id:06d}.png")
    square, circle, pixels = render_scene(square_center, circle_center, img_filename, rng)
    label, overlap_px, gap = check_invariants(square_center, circle_center, square, circle, pixels)

    # Text Prompt: the same cells the shapes were drawn from
    prompt = (f"Imagine a red square at grid ({sx}, {sy}) "
              f"and a blue circle at ({cx}, {cy}). "
              f"Move red square {dx} right, {dy} down. Overlap?")

    return {
        "id": sample_id,
        "prompt": prompt,
        "image_path": os.path.relpath(img_filename, out_dir),  # relative to the dataset dir
        "answer": label,  # computed from the rendered shapes, never assigned by construction
        # The configuration is everything the prompt states; splits are made over these
        "config": {"square_start": [sx, sy], "circle": [cx, cy], "move": [dx, dy]},
        "near_miss": label == "No" and gap < NEAR_MISS_PX,
        "overlap_px": overlap_px,
        "gap_px": round(gap, 2),
    }

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
    rng = random.Random(args.seed)

    # Split every configuration before sampling (stratified by bucket), so the test
    # configurations depend only on --seed, not on --size.
    configs = all_configs()
    bucket_of = {c: bucket(c) for c in configs}
    split_of = {}
    for name in BUCKET_SHARES:
        split_of.update(split_configs([c for c in configs if bucket_of[c] == name], args.seed))

    n_val, n_test = round(args.size * SPLITS["val"]), round(args.size * SPLITS["test"])
    sizes = {"train": args.size - n_val - n_test, "val": n_val, "test": n_test}

    print(f"Generating {args.size} balanced samples from {len(configs)} configurations...")

    splits, sample_id = {}, 0
    for split in SPLITS:
        pools = {name: [c for c in configs if split_of[c] == split and bucket_of[c] == name]
                 for name in BUCKET_SHARES}
        rows = []
        for config in sample_configs(pools, sizes[split], rng):
            sample = generate_sample(sample_id, config, args.out, rng)
            # The bucket only steers sampling; the label comes from the drawing and must agree with it
            assert (sample["answer"] == "Yes") == (bucket_of[config] == "yes")
            assert sample["near_miss"] == (bucket_of[config] == "near_miss")
            sample["split"] = split
            rows.append(sample)
            sample_id += 1
            if sample_id % 100 == 0:
                print(f"Progress: {sample_id}/{args.size}", end="\r")
        splits[split] = rows

    # LEAKAGE CHECK: no test prompt may appear in train or val (and no val prompt in train)
    prompts = {name: {s["prompt"] for s in rows} for name, rows in splits.items()}
    assert not prompts["test"] & (prompts["train"] | prompts["val"]), "Test prompt leaked into train/val!"
    assert not prompts["val"] & prompts["train"], "Val prompt leaked into train!"

    # The picture depends only on (final square cell, circle cell). Different prompts can share one,
    # so count the test scenes that train also shows through a different start and move.
    def scene(s):
        c = s["config"]
        return (c["square_start"][0] + c["move"][0], c["square_start"][1] + c["move"][1], *c["circle"])
    scenes = {name: {scene(s) for s in rows} for name, rows in splits.items()}
    shared_scenes = len(scenes["test"] & scenes["train"])

    print()
    summary = {}
    for name, rows in splits.items():
        # Shuffle so the model doesn't learn "Yes, No, Yes, No" pattern
        rng.shuffle(rows)
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for entry in rows:
                f.write(json.dumps(entry) + "\n")
        n_yes = sum(s["answer"] == "Yes" for s in rows)
        n_near = sum(s["near_miss"] for s in rows)
        summary[name] = {"samples": len(rows), "yes": n_yes, "near_miss": n_near, "configurations": len(prompts[name])}
        print(f"{name}: {len(rows)} samples ({n_yes} Yes, {n_near} near-miss No) from {len(prompts[name])} configurations")
    print(f"Test scenes also shown in train (different prompt): {shared_scenes}/{len(scenes['test'])}")

    # The spec the audit checks the files against
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({
            "seed": args.seed, "size": args.size, "splits": summary,
            "img_size": IMG_SIZE, "unit_px": UNIT_PX, "grid_origin_px": GRID_ORIGIN_PX,
            "grid_cells": [GRID_CELLS.start, GRID_CELLS.stop - 1], "obj_size": OBJ_SIZE, "moves": MOVES,
            "colors": COLORS, "min_penetration_px": MIN_PENETRATION_PX, "min_overlap_px": MIN_OVERLAP_PX,
            "min_gap_px": MIN_GAP_PX, "near_miss_px": NEAR_MISS_PX,
            "test_scenes": len(scenes["test"]), "test_scenes_also_in_train": shared_scenes,
        }, f, indent=2)

    print(f"Completed! Saved to {args.out}")
