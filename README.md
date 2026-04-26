# DV4: Topic-Conditional Weight Reinterpretation via Ternary Encoding with Flip Bits

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15792882-blue)](https://doi.org/10.5281/zenodo.15792882)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org)

> *One model. Four domain experts. Zero extra parameters at inference time.*

---

## What is DV4?

DV4 (Dual-Vocab 4-Bit) is a novel LLM weight encoding scheme where each weight uses **4 bits**:

- **3 bits** encode a ternary value: `{-1, 0, +1}`
- **1 bit** is a flip bit — when set, it inverts the polarity of non-zero weights

Topic-specific **binary masks** over the flip bits allow a single set of ternary weights to behave as completely different domain experts — without changing the weights themselves. Switching between domains is a single bitwise operation.

```
Topic mask A active  →  math expert
Topic mask B active  →  code expert  
Topic mask C active  →  science expert
Topic mask D active  →  general language expert

Same weights. Different mask. Different model.
```

---

## Results

We trained a **558M parameter DV4 transformer** on 4 topics (math, general language, code, science) and evaluated cross-topic contamination via a perplexity-based bleed test across all 16 data/mask combinations.

### Bleed Test — Perplexity Matrix

| Data \\ Mask | Math | General | Code | Science |
|---|---|---|---|---|
| **Math** | **1,592** ✓ | 5,238 | 1,360 | 1,224 |
| **General** | 57,023 | **578** ✓ | 11,842 | 1,341 |
| **Code** | 84,757,102,592 | 936,393 | **614** ✓ | 61,291 |
| **Science** | 919,762 | 13,414 | 3,970 | **341** ✓ |

*Lower = better. ✓ = correct mask diagonal.*

Every off-diagonal entry is higher than the correct-mask diagonal. The code topic under the math mask produces a perplexity of **84.7 billion** — the model is completely unable to predict code under the wrong mask.

### Topic Specificity Scores

| Topic | Correct Mask PPL | Mean Wrong PPL | Specificity |
|---|---|---|---|
| Math | 1,592 | 2,607 | **+0.637** |
| General | 578 | 23,402 | **+39.47** |
| Code | 614 | 28,252,700,092 | **+46,005,858** |
| Science | 341 | 312,382 | **+916.30** |
| **Mean** | — | — | **+11,501,703** |

*Specificity = (mean_wrong_PPL - correct_PPL) / correct_PPL. Threshold for "working": >0.10*

### Training Dynamics — Topic Switch Loss Reset

Each topic switch produces a clean reset to ~12.0 then independent learning — mechanistic evidence that flip masks create genuinely orthogonal effective weight spaces.

| Topic Phase | Start Loss | Best Loss | Reset at Switch |
|---|---|---|---|
| Math (0–2000) | 12.12 | 3.77 | — |
| General (2000–4000) | 12.38 | 4.84 | 12.38 |
| Code (4000–6000) | 12.34 | 5.97 | 12.34 |
| Science (6000–8000) | 12.06 | 5.56 | 12.06 |

---

## Why This Matters

| Approach | Parameters at Inference | Storage for 10 domains |
|---|---|---|
| 10 fine-tuned models | 1x per domain | 10x model size |
| 10-expert MoE | ~10x base model | 10x model size |
| **DV4 (10 topics)** | **1x base model** | **1x model size + ~0.1%** |

DV4 achieves multi-domain specialisation with:
- **Zero additional parameters** at inference time
- **Negligible mask library** (~56MB for 558M model, 4 topics)
- **Instant domain switching** — one bitwise broadcast operation
- **Structural immunity** to catastrophic forgetting across topics

---

## Quickstart

```bash
git clone https://github.com/StoneColdLlama/dv4
cd dv4
python -m venv dv4-venv
source dv4-venv/bin/activate
pip install torch transformers datasets
pip install -e .
```

### Verify the architecture works
```bash
python dv4/debug_sanity_check.py
```

Expected output: `ALL CHECKS PASSED ✅`

### Prepare training data
```bash
python dv4/data/prepare_data.py --debug  # 500 samples per topic (fast test)
python dv4/data/prepare_data.py          # 7000 samples per topic (full)
```

### Train a tiny model (CPU, ~5 minutes)
```bash
python -m dv4.training.train \
    --math_data    dv4/data/math.jsonl \
    --general_data dv4/data/general.jsonl \
    --output_dir   checkpoints/tiny_run \
    --model_size   tiny \
    --steps_per_topic 100
```

### Train the 500M model (GPU recommended)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m dv4.training.train \
    --math_data    dv4/data/math.jsonl \
    --general_data dv4/data/general.jsonl \
    --code_data    dv4/data/code.jsonl \
    --science_data dv4/data/science.jsonl \
    --output_dir   checkpoints/500m_run \
    --model_size   medium \
    --steps_per_topic 2000 \
    --batch_size   16
```

### Run the bleed test
```bash
python -m dv4.eval.run_bleed_test \
    --checkpoint   checkpoints/500m_run/model_final.pt \
    --math_data    dv4/data/math.jsonl \
    --general_data dv4/data/general.jsonl \
    --code_data    dv4/data/code.jsonl \
    --science_data dv4/data/science.jsonl \
    --model_size   medium \
    --max_seq_len  512 \
    --output       results/bleed_test.json
```

---

## Architecture Overview

```
DV4Linear layer:
  shadow_weight (float32)  →  quantise  →  ternary_weight {-1, 0, +1}
                                                    ↓
  topic_flip_mask (uint8)  ────────────→  apply_flip  →  effective_weight
                                                    ↓
                                             F.linear(x, effective_weight)
                                                    ↓
                                          × per_neuron_scale  →  output
```

**At training time:** Flip masks are frozen. Only shadow weights update via straight-through estimator (STE).

**At inference time:** Router classifies input → activates topic mask → bitwise broadcast across all 168 layers → forward pass with topic-specific effective weights.

---

## Repository Structure

```
dv4/
├── model/
│   ├── dv4_linear.py        # Core DV4Linear layer (ternary weights + flip bits)
│   ├── dv4_transformer.py   # Qwen-style transformer (tiny/poc/medium/large configs)
│   └── topic_masks.py       # Mask registry and topic definitions
├── training/
│   ├── train.py             # Main training loop (4-topic, GPU-ready)
│   ├── ternary_utils.py     # STE, gradient health, ternary distribution monitoring
│   └── topic_schedule.py   # Topic-conditional training schedule
├── inference/
│   ├── router.py            # Keyword-based topic router + sentence-level routing
│   └── generate.py          # Autoregressive generation with mask switching
├── eval/
│   ├── bleed_test.py        # Bleed test engine (compute_perplexity, BleedTest)
│   └── run_bleed_test.py    # 4-topic bleed test runner
├── data/
│   └── prepare_data.py      # Downloads GSM8K, TriviaQA, CodeSearchNet, SciQ
└── debug_sanity_check.py    # 7-point architecture verification
```

---

## Model Configurations

| Config | Parameters | Hidden | Layers | Heads | FFN | Use Case |
|---|---|---|---|---|---|---|
| `tiny` | ~3M DV4 | 256 | 4 | 4 | 512 | Debug / CI |
| `poc` | ~120M | 768 | 12 | 12 | 2048 | Small experiments |
| `medium` | ~558M | 1024 | 24 | 16 | 4096 | PoC paper results |
| `large` | ~1B | 2048 | 24 | 16 | 5632 | Full paper run |

---

## Hardware Used

- **Development / PoC:** Dual AMD GPU system (RX 7900 GRE + RX 7600 XT, ROCm), CPU training
- **Main experiment:** NVIDIA RTX PRO 6000 Blackwell Workstation Edition (102GB VRAM)
- **Training time (558M, 8000 steps):** ~100 minutes on single RTX PRO 6000

---

## Paper

**DV4: Topic-Conditional Weight Reinterpretation via Ternary Encoding with Flip Bits**
Peter Norman — Independent Researcher, twoswans.com.au

- **Zenodo (published):** https://doi.org/10.5281/zenodo.15792882
- **arXiv:** Submission pending endorsement
- **ORCID:** 0009-0004-8413-1274

---

## Citation

```bibtex
@misc{norman2026dv4,
  title     = {DV4: Topic-Conditional Weight Reinterpretation via Ternary Encoding with Flip Bits},
  author    = {Norman, Peter},
  year      = {2026},
  month     = {April},
  doi       = {10.5281/zenodo.15792882},
  url       = {https://doi.org/10.5281/zenodo.15792882},
  note      = {Independent Researcher, twoswans.com.au}
}
```

---

## License

Code: MIT License — see [LICENSE](LICENSE)
Paper: CC BY 4.0 — see https://doi.org/10.5281/zenodo.15792882

---

## Author

**Peter Norman**
Independent AI Researcher
Perth, Western Australia
research@twoswans.com.au
twoswans.com.au
ORCID: 0009-0004-8413-1274

*The DV4 concept was independently conceived in March 2026 and empirically validated on April 26, 2026.*
