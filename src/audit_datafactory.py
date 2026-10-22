"""Audits generated datasets against the generator's invariants, using only their files.

For each dataset it reads the jsonl and PNGs, parses every prompt, finds the shapes
from pixel colours (it never calls the generator), and reports how often each
invariant fails. It reads both the old layout (dataset.jsonl, written by the old
generator) and the new one (train/val/test.jsonl + manifest.json), so the two
generators can be compared side by side:

    python src/audit_datafactory.py dataset_imagination_balanced dataset_imagination_v2 --limit 20000
"""
import os
import re
import json
import math
import random
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw

# The margins every sample must clear (documented in datafactory.py and the README)
from datafactory import MIN_PENETRATION_PX, MIN_OVERLAP_PX, MIN_GAP_PX, NEAR_MISS_PX

# The old generator wrote no manifest. Its prompts truncated pixel positions to 25 px grid
# cells, so the best reading of "grid (g)" is the centre of that cell: 25 * g + 12.5.
LEGACY_SPEC = {"img_size": 256, "unit_px": 25, "grid_origin_px": 12.5, "obj_size": 20,
               "colors": {"red": [255, 0, 0], "blue": [0, 0, 255]}}
PROMPT_RE = re.compile(r"red square at grid \((-?\d+), (-?\d+)\) and a blue circle at \((-?\d+), (-?\d+)\)\. "
                       r"Move red square (-?\d+) right, (-?\d+) down\.")
POSITION_TOLERANCE_PX = 1.0
# Checks marked * need both shapes located, so they count only images where both are visible
CHECKS = [
    ("prompt_label", "Label equals the answer implied by the prompt"),
    ("conflict", "Prompt never appears with the opposite label"),
    ("in_frame", "Both shapes fully inside the frame"),
    ("visible", "Neither shape hides any of the other"),
    ("position", "* Shapes drawn where the prompt puts them (+/-1 px)"),
    ("label_drawing", "* Label matches the shapes as drawn"),
    ("margin", f"* Clear margin (Yes: >={MIN_PENETRATION_PX} px deep, >={MIN_OVERLAP_PX} px shared; "
               f"No: >={MIN_GAP_PX} px gap)"),
]

def load_dataset(data_dir):
    """Returns (rows, spec). New layout: all splits plus the manifest. Old layout: dataset.jsonl."""
    manifest = os.path.join(data_dir, "manifest.json")
    if os.path.exists(manifest):
        spec = json.load(open(manifest))
        files = [f"{split}.jsonl" for split in ("train", "val", "test")]
    else:
        spec, files = LEGACY_SPEC, ["dataset.jsonl"]
    rows = []
    for name in files:
        with open(os.path.join(data_dir, name)) as f:
            rows += [json.loads(line) for line in f]
    return rows, spec

def resolve_image(data_dir, path):
    """New datasets store paths relative to the dataset dir; the old one stored them relative to the repo."""
    for candidate in (os.path.join(data_dir, path), path, os.path.join(data_dir, "images", os.path.basename(path))):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(path)

def bbox(center, obj):
    return [center[0] - obj, center[1] - obj, center[0] + obj, center[1] + obj]

def clearance(square_center, circle_center, obj):
    """Signed gap between an axis-aligned square and a circle of the same half-size."""
    dx = max(abs(square_center[0] - circle_center[0]) - obj, 0)
    dy = max(abs(square_center[1] - circle_center[1]) - obj, 0)
    return math.hypot(dx, dy) - obj

def fill_offsets(obj):
    """(dy, dx) offsets of a circle's fill pixels from its centre, as PIL draws it with an outline."""
    size = 2 * obj + 3
    canvas = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(canvas).ellipse([1, 1, 1 + 2 * obj, 1 + 2 * obj], fill=(0, 0, 255), outline="black")
    ys, xs = np.nonzero((np.array(canvas) == (0, 0, 255)).all(axis=-1))
    return ys - (1 + obj), xs - (1 + obj)

def find_square(square_fill, obj):
    """Square centre from its fill, which spans centre +/- (obj - 1) unless the frame clips it."""
    ys, xs = np.nonzero(square_fill)
    if len(xs) == 0:
        return None
    last = square_fill.shape[0] - 1
    def centre(lo, hi):
        if lo == 0 and hi < last: return hi - (obj - 1)
        if hi == last and lo > 0: return lo + (obj - 1)
        return (lo + hi) / 2
    return (centre(xs.min(), xs.max()), centre(ys.min(), ys.max()))

def find_circle(circle_fill, can_hide, offsets, obj):
    """Circle centre: the template position that covers every visible fill pixel and predicts fill
    only where something drawn later could hide it (or outside the frame). Ties go to the middle."""
    ys, xs = np.nonzero(circle_fill)
    if len(xs) == 0:
        return None
    h, w = circle_fill.shape
    reach = obj - 1
    best_cost, best = None, []
    for cy in range(ys.max() - reach, ys.min() + reach + 1):
        for cx in range(xs.max() - reach, xs.min() + reach + 1):
            py, px = offsets[0] + cy, offsets[1] + cx
            inside = (py >= 0) & (py < h) & (px >= 0) & (px < w)
            py, px = py[inside], px[inside]
            covered = circle_fill[py, px]
            cost = (len(xs) - covered.sum()) + (~covered & ~can_hide[py, px]).sum()
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, [(cx, cy)]
            elif cost == best_cost:
                best.append((cx, cy))
    mx, my = np.mean(best, axis=0)
    return min(best, key=lambda c: (c[0] - mx) ** 2 + (c[1] - my) ** 2)

def masks(square_center, circle_center, obj, size):
    sq, ci = Image.new("1", (size, size)), Image.new("1", (size, size))
    ImageDraw.Draw(sq).rectangle(bbox(square_center, obj), fill=1)
    ImageDraw.Draw(ci).ellipse(bbox(circle_center, obj), fill=1)
    return np.array(sq, dtype=bool), np.array(ci, dtype=bool)

def audit_image(pixels, spec, offsets):
    """Locates both shapes in one image and measures them. A centre is None if no pixel of it shows:
    nothing is ever drawn over the square, so a missing square lies entirely outside the frame, and a
    missing circle is completely hidden."""
    obj, size = spec["obj_size"], spec["img_size"]
    is_color = lambda rgb: (pixels == np.array(rgb, dtype=np.uint8)).all(axis=-1)
    red, blue = is_color(spec["colors"]["red"]), is_color(spec["colors"]["blue"])
    mix = is_color(spec["colors"]["overlap"]) if "overlap" in spec["colors"] else np.zeros_like(red)
    black = is_color([0, 0, 0])
    drawn = {"square": find_square(red | mix, obj), "circle": find_circle(blue | mix, red | black, offsets, obj)}
    located = [c for c in drawn.values() if c is not None]
    drawn["in_frame"] = drawn["square"] is not None and all(
        0 <= c[k] - obj and c[k] + obj < size for c in located for k in (0, 1))
    if len(located) < 2:
        drawn["hidden_px"] = None if drawn["circle"] is not None else 1  # a vanished circle is hidden
        return drawn
    sq, ci = masks(drawn["square"], drawn["circle"], obj, size)
    drawn.update({
        "overlap_px": int((sq & ci).sum()),
        "gap": clearance(drawn["square"], drawn["circle"], obj),
        # A shape is hidden where its region shows the other shape's own colour
        "hidden_px": int((ci & red).sum() + (sq & blue).sum()),
    })
    return drawn

def audit(data_dir, limit, seed):
    rows, spec = load_dataset(data_dir)
    obj, unit, origin = spec["obj_size"], spec["unit_px"], spec["grid_origin_px"]
    offsets = fill_offsets(obj)

    # Dataset-wide checks (all rows)
    labels_by_prompt = defaultdict(set)
    for r in rows:
        labels_by_prompt[r["prompt"]].add(r["answer"])
    yes_share = sum(r["answer"] == "Yes" for r in rows) / len(rows)

    # Per-image checks (a seeded sample of rows)
    sample = rows if limit is None or limit >= len(rows) else random.Random(seed).sample(rows, limit)
    fails, counted = defaultdict(int), defaultdict(int)
    def record(key, failed):
        counted[key] += 1
        fails[key] += bool(failed)
    n_no = n_near = 0
    for r in sample:
        sx, sy, cx, cy, dx, dy = map(int, PROMPT_RE.search(r["prompt"]).groups())
        implied_square = (origin + (sx + dx) * unit, origin + (sy + dy) * unit)
        implied_circle = (origin + cx * unit, origin + cy * unit)
        implied_label = "Yes" if clearance(implied_square, implied_circle, obj) < 0 else "No"
        record("prompt_label", implied_label != r["answer"])
        record("conflict", len(labels_by_prompt[r["prompt"]]) > 1)

        pixels = np.array(Image.open(resolve_image(data_dir, r["image_path"])).convert("RGB"))
        drawn = audit_image(pixels, spec, offsets)
        record("in_frame", not drawn["in_frame"])
        if drawn["hidden_px"] is not None:
            record("visible", drawn["hidden_px"] > 0)
        if "gap" not in drawn:
            continue
        error = max(abs(a - b) for a, b in zip(drawn["square"] + drawn["circle"], implied_square + implied_circle))
        record("position", error > POSITION_TOLERANCE_PX)
        drawn_label = "Yes" if drawn["overlap_px"] > 0 else "No"
        record("label_drawing", drawn_label != r["answer"])
        if r["answer"] == "Yes":
            clear = -drawn["gap"] >= MIN_PENETRATION_PX and drawn["overlap_px"] >= MIN_OVERLAP_PX
        else:
            clear = drawn["gap"] >= MIN_GAP_PX
            n_no += 1
            n_near += 0 < drawn["gap"] < NEAR_MISS_PX
        record("margin", not clear)

    summary = {"rows": len(rows), "audited": len(sample), "fails": dict(fails), "counted": dict(counted),
               "yes_share": yes_share, "near_miss_share_of_no": n_near / n_no if n_no else float("nan")}

    # Split checks (new layout only)
    if "split" in rows[0]:
        by_split = defaultdict(list)
        for r in rows:
            by_split[r["split"]].append(r)
        prompts = {k: {r["prompt"] for r in v} for k, v in by_split.items()}
        summary["test_prompts_in_train_val"] = len(prompts["test"] & (prompts["train"] | prompts["val"]))
        def scene(r):
            sx, sy, cx, cy, dx, dy = map(int, PROMPT_RE.search(r["prompt"]).groups())
            return (sx + dx, sy + dy, cx, cy)
        scenes = {k: {scene(r) for r in v} for k, v in by_split.items()}
        summary["test_scenes_in_train"] = (len(scenes["test"] & scenes["train"]), len(scenes["test"]))
    return summary

def main():
    parser = argparse.ArgumentParser(description="Check generated datasets against the generator's invariants.")
    parser.add_argument("data_dirs", nargs="+", help="dataset directories (old or new layout)")
    parser.add_argument("--limit", type=int, default=None, help="images to check per dataset (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="seed for choosing the --limit sample")
    args = parser.parse_args()

    results = {}
    for d in args.data_dirs:
        print(f"Auditing {d}...")
        results[d] = audit(d, args.limit, args.seed)

    width, col = 84, 32
    names = [os.path.basename(os.path.normpath(d)) for d in args.data_dirs]
    rule = width + col * len(names)
    print("\n" + "=" * rule)
    print(f"{'Failure rate per invariant':<{width}}" + "".join(f"{n[:col - 2]:>{col}}" for n in names))
    print("-" * rule)
    for key, label in CHECKS:
        cells = []
        for d in args.data_dirs:
            s = results[d]
            k, n = s["fails"].get(key, 0), s["counted"].get(key, 0)
            cells.append(f"{k / n * 100:.2f}% ({k}/{n})" if n else "n/a")
        print(f"{label:<{width}}" + "".join(f"{c:>{col}}" for c in cells))
    print("-" * rule)
    rows = [
        ("Yes share (target 50%)", lambda s: f"{s['yes_share'] * 100:.1f}%"),
        (f"Near misses among No (gap < {NEAR_MISS_PX} px; target about half)",
         lambda s: f"{s['near_miss_share_of_no'] * 100:.1f}%"),
        ("Test prompts also in train or val", lambda s: str(s.get("test_prompts_in_train_val", "n/a (no split)"))),
        ("Test scenes also in train (same picture, different prompt)",
         lambda s: "{}/{}".format(*s["test_scenes_in_train"]) if "test_scenes_in_train" in s else "n/a (no split)"),
    ]
    for label, fmt in rows:
        print(f"{label:<{width}}" + "".join(f"{fmt(results[d]):>{col}}" for d in args.data_dirs))
    print("=" * rule)
    print("* counted only where both shapes are visible somewhere in the image")
    for d in args.data_dirs:
        s = results[d]
        print(f"{d}: {s['rows']} rows, {s['audited']} images checked")

if __name__ == "__main__":
    main()
