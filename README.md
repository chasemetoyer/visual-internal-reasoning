# Visual Internal Reasoning: Causal Dependency on Latent Image Tokens

## **Abstract**

Large language models (LLMs) exhibit strong performance on linguistic reasoning tasks but continue to struggle with problems requiring spatial and visual reasoning, often relying on shallow textual heuristics rather than grounded internal representations. In this work, we investigate whether equipping a transformer-based language model with an explicit internal visual latent representation improves performance on visually grounded reasoning tasks.

We propose a unified decoder-only architecture in which discrete image tokens, obtained from a pretrained vector-quantized autoencoder (VQGAN), are incorporated directly into the model’s vocabulary alongside standard text tokens. The model is trained autoregressively to generate an intermediate sequence of image tokens—interpreted as an internal “imagined” visual state—prior to producing a textual answer.

In a first run on 1,000 held-out test questions (data seed 0, training seed 0), the model that imagines the picture before answering is correct 96.5% of the time with greedy decoding, and 98.5% when it samples the picture. An identically trained text-only model gets 61.3%, and a model given 256 filler tokens in place of the picture gets 53.6%. Replacing or removing the imagined picture drops the image model to chance (50.0% and 48.3%), so its answers depend on the visual representation.

> These are single-run results with open caveats; see [Results](#results-single-run). The original prototype's numbers (90.5% imagined vs. 57.0% blindfold) were not valid and are kept, marked superseded, under [Superseded: Original Prototype Results](#superseded-original-prototype-results-n200).

---

## **Results (Single Run)**

Forced-choice evaluation on 1,000 held-out test items with distinct configurations: 500 Yes, 250 near-miss No and 250 far No, so the majority baseline is 50.0%. One run on the 100,000-sample dataset: data seed 0, training seed 0, 30,000 training steps per model. Every model is scored on the same items; brackets are 95% Wilson confidence intervals.

| Model | Condition | Accuracy | 95% CI |
| --- | --- | --- | --- |
| Image | Teacher forced: ground-truth picture tokens (ceiling) | 100.0% | [99.6, 100.0] |
| Image | Imagined, greedy: the model generates its own picture | **96.5%** | [95.2, 97.5] |
| Image | Imagined, sampled (T = 0.8) | **98.5%** | [97.5, 99.1] |
| Image | Blindfold: picture replaced with random VQ tokens | 50.0% | [46.9, 53.1] |
| Image | Picture removed (out of distribution) | 48.3% | [45.2, 51.4] |
| Text | Prompt, then answer | 61.3% | [58.2, 64.3] |
| Filler | Prompt, 256 `<THINK>` tokens, then answer | 53.6% | [50.5, 56.7] |

**What this shows.** On questions it never trained on, the model that imagines the picture before answering is far more accurate than identically trained models that answer directly or after 256 filler tokens. The gaps are far wider than the confidence intervals. With its picture replaced or removed, the image model falls to chance, so its answers depend on the picture rather than on a shortcut through the prompt text. The filler model shows that extra sequence length and compute alone barely help.

**Caveats.**
- **One run.** The intervals cover which test items were drawn, not training randomness. The text model was still escaping a plateau: it was at chance at step 12,000 and reached 61.3% by step 30,000. Its score is therefore sensitive to the training seed and step budget, and the comparison holds at a fixed budget of 30,000 steps. Replicate with other training and data seeds before citing.
- **Picture or supervision?** The image model is trained on 256 picture tokens per example, while the text and filler models are trained on the answer token alone. The filler model controls for compute but not for that extra supervision, so these results do not yet show that the *visual* form of the intermediate is what helps. A text-scratchpad baseline that writes the square's final grid cell before answering would separate the two.
- **Familiar pictures.** No test prompt is trained on, but every test picture (the square's final cell plus the circle's cell) also appears in training under a different prompt. The result shows generalization to new questions, not to unseen scenes.
- **Greedy vs. sampled.** The intervals for the two imagined rows touch, so the difference between them is not established.

Reproduce it with [`colab_pipeline.ipynb`](colab_pipeline.ipynb) (`QUICK_TEST = False`) or the commands in [Getting Started](#getting-started).

---

## **Superseded: Original Prototype Results (N=200)**

> **Superseded — do not cite.** This table is kept for the record only. It is not valid because:
> - It was measured on training questions: the evaluation sampled its items from `train_data_final.pt`, the same file the model was trained on.
> - The "Text-Only" row was not a text-only baseline: it fed the image model `[prompt][IMG_START][IMG_END]`, an input it never saw in training.
> - The old generator's labels and pictures disagreed. `src/audit_datafactory.py` over all 100k old samples found that:
>   - 2.9% of labels contradict the shapes as drawn (2,699 of the 93,366 images where both shapes are visible), mostly "No" pairs drawn overlapping.
>   - 20.4% of images have a shape partly or wholly outside the frame.
>   - 6.6% of images show only one shape: in 6,145 the square is entirely outside the frame, and in 489 the circle is completely hidden.
>   - 7.4% of samples share their exact prompt with a sample of the opposite label, and 2.3% of labels differ from the answer their prompt implies.
>   - In 52% of all images (52,422 of 100,000) the square hides part of the circle.
>   - The prompts truncate positions to 25 px cells. The drawn square is on average 10.3 px (at most 24.7 px) from where its prompt puts it, and no sample has both shapes within 1 px of their prompt positions.
> - The model had a residual-stream bug: `Block.forward` returned `x + mlp(ln2(x + attn(ln1(x))))`, so the attention output never entered the residual stream directly. Every checkpoint trained with it (all untagged `imaginer_*.pth` files) is invalid.
>
> Valid numbers from the fixed pipeline are in [Results](#results-single-run).

We evaluated the model on a "Forced Choice" spatial reasoning task (*"Does the Red Square overlap the Blue Circle?"*). The results demonstrate a clear causal dependency on the internal visual state.

| Condition | Description | Accuracy | Interpretation |
| --- | --- | --- | --- |
| **Teacher Forced** | Ground truth visual tokens provided. | **100.0%** | **Upper Bound:** The model can perfectly interpret clear visual data. |
| **Imagined (Greedy)** | Model generates its own visual latents. | **90.5%** | **Method:** Internal simulation provides a +33.5% gain over baseline. |
| **Text-Only** | Visual tokens omitted entirely. | 59.0% | **Baseline:** Text priors alone are insufficient for this task. |
| **Blindfold** | Visual tokens replaced with VQ-noise. | 57.0% | **Control:** Performance collapses to random chance (57% majority class) without visual structure. |

> **Verdict:** The degradation from **90.5%** (Imagined) to **57.0%** (Blindfold) rejects the null hypothesis that the model relies solely on text shortcuts.
<img width="700" height="700" alt="BarGraph" src="https://github.com/user-attachments/assets/0f88db07-0829-4556-9808-f5a8a8692f43" />

---

## **Hypothesis**

This research investigates three core claims regarding multimodal reasoning in transformers:

1. **H1 (Inductive Bias):** Explicit internal visual representations provide a beneficial inductive bias for reasoning over spatial relations and object interactions.
2. **H2 (Necessity):** The generated visual tokens are necessary intermediates for correct reasoning, not merely auxiliary outputs.
3. **H3 (Generalization):** Models with internal visual intermediates exhibit improved robustness to linguistic paraphrasing and prompt perturbations compared to text-only models.

---

## **Repository Structure**

```plaintext
.
├── dataset_imagination_v2/       # Synthetic data: images + train/val/test.jsonl
├── processed_v2/                 # Preprocessed splits: train.pt, val.pt, test.pt (tokens + VQ indices + metadata)
├── checkpoints/                  # Trained weights: imaginer_v2_<format>_*.pth
├── taming/                       # VQGAN dependencies (Taming Transformers)
├── vqgan_imagenet_f16_16384.* # VQGAN Checkpoint & Config
│
├── src/
│   ├── train_imaginer.py         # Main autoregressive training loop
│   ├── preprocess.py             # Data pipeline: Images -> VQ Indices -> .pt
│   ├── datafactory.py            # Synthetic dataset generation engine
│   └── audit_datafactory.py      # Re-checks the dataset invariants from the written files
│
├── evaluation/
│   ├── evaluate_rigorous.py      # Statistical evaluation (Teacher vs Blindfold vs Imagined)
│   ├── verify_logic.py           # Sighted inference check
│   └── verify_blindfold.py       # Causal intervention check (Noise Injection)
│
├── visualization/
│   ├── visualize.py              # Generate and decode internal "dreams"
│   └── diagnostic_dream.png      # Sample output of internal state
│
├── tests/
│   └── test_block.py             # Residual-stream regression test for the transformer block
│
├── colab_pipeline.ipynb          # Runs the whole pipeline step by step on Colab (or locally)
└── requirements.txt              # Project dependencies
```

---

## **Methodology**

### **1. Architecture**

We utilize a standard GPT-2 style decoder-only transformer with pre-LN blocks (`x = x + attn(ln1(x))`, then `x = x + mlp(ln2(x))`). The vocabulary is expanded to include discrete codebook indices from a VQGAN trained on ImageNet.


<img width="396" height="41" alt="Screenshot 2025-12-29 at 3 08 50 AM" src="https://github.com/user-attachments/assets/bc6123dc-0cd8-436c-9b8a-110f14f5b116" />

### **2. Training Objective**

The model is trained to minimize the negative log-likelihood over the joint sequence of Text Prompt ($x_{prompt}$), Visual Latents ($v$), and Text Answer ($y_{answer}$):


<img width="188" height="58" alt="Screenshot 2025-12-29 at 3 09 02 AM" src="https://github.com/user-attachments/assets/72219fdb-3297-493f-bfdc-6317c95e05a2" />



where $x = [x_{prompt}, v, y_{answer}]$.
During training, we apply **loss masking** to the prompt tokens and padding, ensuring the model focuses capacity solely on visual generation and answer reasoning.

### **3. Causal Intervention (Blindfold Test)**

To verify H2 (Necessity), we perform an intervention at inference time:

1. Let the model generate the visual sequence $v$.
2. Replace $v$ with noise $v_{noise}$.
3. Force the model to predict $y_{answer}$ conditioned on $v_{noise}$.
Collapse in performance confirms the answer $y_{answer}$ is downstream of the visual state $v$.

### **4. Baselines**

Three models are trained identically (same architecture, hyperparameters, training split and number of steps) and share one vocabulary. Only the sequence format differs:

| Format | Sequence | Loss on |
| --- | --- | --- |
| `image` | prompt, `<IMG_START>`, 256 image tokens, `<IMG_END>`, answer | image tokens, `<IMG_END>`, answer |
| `text` | prompt, answer | answer |
| `filler` | prompt, `<IMG_START>`, 256 × `<THINK>`, `<IMG_END>`, answer | answer |

The filler model has the image model's sequence length and compute but no visual content, so it separates the effect of the picture from the effect of extra tokens. All three models are scored on the same test items. The image model with its picture removed (`[prompt][IMG_START][IMG_END]`) is still reported, but that input is out of distribution, so it is not a text-only baseline.

### **5. Dataset Invariants**

`src/datafactory.py` draws each shape exactly at a grid-cell centre: cell *g* sits at 15 + 25*g* px, for cells 1–8. The prompt therefore fully determines the picture and the answer; the only randomness is the faint background lines. Every sample is asserted to satisfy:

| Invariant | Rule |
| --- | --- |
| Prompt determines the answer | No position jitter: the cells in the prompt are the cells drawn. |
| Label comes from the drawing | `Yes` exactly when the rendered square and circle share a pixel, cross-checked against the exact square–circle distance. |
| Both shapes inside the frame | Both 41 px shapes stay at least 20 px from every edge. |
| Both shapes visible | The shared region is drawn purple (a red/blue mix), and both outlines are drawn last. |
| Clear overlap | Every `Yes` reaches at least 8 px into the circle and shares at least 100 pixels. The closest `Yes` on the grid is 12.9 px deep and shares 164 pixels. |
| Clear gap | Every `No` leaves at least 8 px of background between the shapes. The closest `No` on the grid is 10 px apart. |
| Balance | 50% `Yes`. Half of the `No` samples are near misses, with a gap under one grid unit (25 px). |

`src/audit_datafactory.py` re-checks these invariants from the files alone. It parses each prompt and locates both shapes from pixel colours, without calling the generator.

The split is over configurations (square start, circle position, move), so no test prompt is ever trained on. The picture, however, depends only on the square's final cell and the circle's cell. There are 4,096 such scenes, each reachable by about 20 start/move combinations. With `--size 100000`, every test scene also appears in training under a different prompt (`datafactory.py` prints the count). The test therefore measures new questions about familiar pictures, not unseen pictures.

---

## **Getting Started**

### **Prerequisites**

* Python 3.9+
* PyTorch (CUDA or MPS recommended)

```bash
pip install -r requirements.txt
```

### **VQGAN Weights**

The pretrained VQGAN (ImageNet, f16, 16384 codes) from [Taming Transformers](https://github.com/CompVis/taming-transformers) is not included in this repo. Download the config and checkpoint (~1 GB) into the repository root:

```bash
curl -L -o vqgan_imagenet_f16_16384.yaml "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"
curl -L -o vqgan_imagenet_f16_16384.ckpt "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1"
```

If the `.ckpt` is only a few KB, the download failed; retry. Run all scripts from the repository root so these paths resolve.

### **1. Data Generation**

Generate the synthetic spatial reasoning dataset (100k samples). Each unique configuration (square start, circle position, move) is assigned to train, val or test (90/5/5, fixed seed), so a prompt rendered several times with different noise lines always stays in one split and no test prompt appears in train or val. The output directory must be new or empty.

```bash
python src/datafactory.py --size 100000 --seed 0 --out dataset_imagination_v2
```

Audit the written files against the [dataset invariants](#5-dataset-invariants). Pass several directories to compare them side by side, including the old generator's `dataset_imagination_balanced/` if you still have it. Use `--limit N` to check a random sample of images instead of all of them.

```bash
python src/audit_datafactory.py dataset_imagination_v2
```

### **2. Preprocessing**

Tokenize text and encode images into discrete VQGAN indices. Writes one file per split (`train.pt`, `val.pt`, `test.pt`), storing each sample's configuration, label and loss-start index alongside its tokens.
*Note: Requires the VQGAN weights (see [VQGAN Weights](#vqgan-weights)).*

```bash
python src/preprocess.py --data dataset_imagination_v2 --out processed_v2
```

### **3. Training**

Train one model per format (see [Baselines](#4-baselines)) on the train split, with identical settings. Every 1,000 iterations each run logs validation loss over all trained tokens and over the answer token alone. Use the answer loss to compare the text and filler models only. The image model predicts its validation answer with the true picture already in its input (teacher forcing), so its answer loss is near zero whether or not it can imagine the picture. Compare it with the other two through the evaluation's imagined rows.

```bash
python src/train_imaginer.py --format image  --data processed_v2 --out-dir checkpoints
python src/train_imaginer.py --format text   --data processed_v2 --out-dir checkpoints
python src/train_imaginer.py --format filler --data processed_v2 --out-dir checkpoints
```

Checkpoints are written as `checkpoints/imaginer_v2_<format>_ckpt_<iter>.pth` and `checkpoints/imaginer_v2_<format>_final.pth`. Each one stores its format and vocabulary alongside the weights. `v2` marks the residual-stream fix: every script refuses checkpoints without the tag, because those were trained with the bug.

Training runs on CUDA, Apple MPS or the CPU. Each batch is padded only to its longest sequence, and attention uses PyTorch's fused `scaled_dot_product_attention`. On NVIDIA GPUs with tensor cores (A100, H100, L4), matrix multiplies run in TF32, which is faster but rounds slightly differently from other devices. Train all three formats on the same type of GPU so the comparison is not affected by hardware.

### **4. Evaluation**

Run the forced-choice evaluation on the held-out test split. It scores `--n` items with distinct configurations (default 1,000; repeated renders of one prompt are not independent) in the test split's class mix, runs every model on the same items, and reports each accuracy with a 95% Wilson confidence interval, plus the majority-class baseline. Any of the three checkpoints can be left out. Each checkpoint records the configurations it trained on, and the evaluation refuses one that trained on any configuration in this test split (for example, a model trained on data generated with a different `--seed`).

```bash
python evaluation/evaluate_rigorous.py --data processed_v2 --n 1000 \
    --image-ckpt checkpoints/imaginer_v2_image_final.pth \
    --text-ckpt checkpoints/imaginer_v2_text_final.pth \
    --filler-ckpt checkpoints/imaginer_v2_filler_final.pth
```

### **5. Tests**

Check that the transformer block adds the attention output to the residual stream (this test fails on the old wiring):

```bash
python tests/test_block.py
```

---

## **Visualizing Internal States**

To inspect the "imagination" of the model, run the visualization script on an image-format checkpoint. This decodes the generated token sequence back into pixel space using the VQGAN decoder. `visualization/visualize_diagnostic.py`, `evaluation/verify_logic.py` and `evaluation/verify_blindfold.py` take the same `--ckpt` flag.

```bash
python visualization/visualize.py --ckpt checkpoints/imaginer_v2_image_final.pth
```

**Sample Output:**

> *Prompt: "Imagine a red square at grid (2, 2) and a blue circle at (6, 6)..."*

| Diagnostic Dream | Interpretation |
| --- | --- |
| [Diagnostic Dream] <img width="256" height="256" alt="diagnostic_dream" src="https://github.com/user-attachments/assets/0522bb6f-f83b-48eb-83c6-5e72152552d1"> | The model successfully disentangles the two objects and places them in distinct coordinate spaces, allowing it to correctly deduce that they do not overlap. |

---

## **Citation**

If you use this code or methodology, please cite:

```bibtex
@misc{metoyer2026visualreasoning,
  title={Visual Internal Reasoning: Causal Dependency on Latent Image Tokens},
  author={Metoyer, Chase},
  year={2026},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/chasemetoyer/visual-internal-reasoning}}
}
```
