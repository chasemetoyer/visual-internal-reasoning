# Visual Internal Reasoning: Causal Dependency on Latent Image Tokens

## **Abstract**

Large language models (LLMs) exhibit strong performance on linguistic reasoning tasks but continue to struggle with problems requiring spatial and visual reasoning, often relying on shallow textual heuristics rather than grounded internal representations. In this work, we investigate whether equipping a transformer-based language model with an explicit internal visual latent representation improves performance on visually grounded reasoning tasks.

We propose a unified decoder-only architecture in which discrete image tokens, obtained from a pretrained vector-quantized autoencoder (VQGAN), are incorporated directly into the model’s vocabulary alongside standard text tokens. The model is trained autoregressively to generate an intermediate sequence of image tokens—interpreted as an internal “imagined” visual state—prior to producing a textual answer.

Our results demonstrate that models trained with forced internal visual intermediates outperform text-only baselines on spatial reasoning tasks and exhibit significant performance degradation when the imagined visual tokens are disrupted (Blindfold Accuracy: 57.0% vs. Imagined Accuracy: 90.5%), indicating that the visual representation is causally utilized.

---

## **Key Findings (N=200)**

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
├── dataset_imagination_balanced/ # Synthetic data generation output
├── taming/                       # VQGAN dependencies (Taming Transformers)
├── train_data_final.pt           # Preprocessed dataset (Tokenized Text + VQ Indices)
├── imaginer_final.pth            # Trained Model Weights (30k iters)
├── vqgan_imagenet_f16_16384.* # VQGAN Checkpoint & Config
│
├── src/
│   ├── train_imaginer.py         # Main autoregressive training loop
│   ├── preprocess.py             # Data pipeline: Images -> VQ Indices -> .pt
│   └── datafactory.py            # Synthetic dataset generation engine
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
└── requirements.txt              # Project dependencies
```

---

## **Methodology**

### **1. Architecture**

We utilize a standard GPT-2 style decoder-only transformer. The vocabulary is expanded to include discrete codebook indices from a VQGAN trained on ImageNet.


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

Generate the synthetic spatial reasoning dataset (100k samples).

```bash
python src/datafactory.py --size 100000
```

### **2. Preprocessing**

Tokenize text and encode images into discrete VQGAN indices.
*Note: Requires the VQGAN weights (see [VQGAN Weights](#vqgan-weights)).*

```bash
python src/preprocess.py
```

### **3. Training**

Train the autoregressive transformer.

```bash
python src/train_imaginer.py
```

### **4. Evaluation**

Run the rigorous forced-choice evaluation suite.

```bash
python evaluation/evaluate_rigorous.py
```

---

## **Visualizing Internal States**

To inspect the "imagination" of the model, run the visualization script. This decodes the generated token sequence back into pixel space using the VQGAN decoder.

```bash
python visualization/visualize.py
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
@misc{metoyer2024visualreasoning,
  title={Visual Internal Reasoning: Causal Dependency on Latent Image Tokens},
  author={Metoyer, Chase},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/chasemetoyer/visual-internal-reasoning}}
}
```
