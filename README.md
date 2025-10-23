# Language Confusion Gate (LCG)

Paper: [**Language Confusion Gate: Language‑Aware Decoding Through Model Self‑Distillation**](https://arxiv.org/abs/2510.17555)

*A lightweight, plug‑in gate that reduces unintended language mixing during LLM decoding without retraining the base model.*



<p align="center">
<img width="657" height="498" alt="Screenshot 2025-10-22 at 5 01 49 PM" src="https://github.com/user-attachments/assets/dc93bf70-66b9-4a27-86e8-fefd386671e4" />
</p>

## Highlights
* **Drop‑in at decode time.** Filters tokens by language family *only when needed*, leaving normal decoding untouched otherwise.
* **Single gate for all languages** No configurations is required for different languages, LCG automatically determines when and which language to mask
* **Norm-adjusted Self-distillation.** The gate learns to predict allowed next‑token language families by distilling the model's own logits.
* **Compatible with models across different architectures.** Evaluated on families like Qwen3, GPT‑OSS, Gemma3, and Llama 3.1.
* **Minimal overhead.** Very low overhead, works together with speculative decoding.
* **Don't harm normal performance.** Intervention is rare and doesn’t harm task quality.

---

## What is “language confusion”?

LLMs sometimes mix wrong language in its output (e.g., mixing Chinese characters into an Arabic reply). LCG mitigates this by predicting permissible language **families** for the next token and masking disallowed families only when the base sampler would otherwise pick them.

---

## Repository structure

```
language_confusion_gate/
├── lcg_plugin/           # vLLM plug‑in for inference-time gating (pip‑installable)
├── train/                # data prep + gate training 
├── eval/                 # evaluation scripts + scoring utilities and evaluation datasets helpers
├── README.md
└── .gitignore
```

---

## Installation

### 1) Create an environment (example)

```bash
# conda (recommended)
conda create -n lcg python=3.10 -y
conda activate lcg

# or: python -m venv .venv && source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install torch transformers datasets accelerate deepspeed
# for inference with the plug‑in
pip install vllm
```

### 3) Install the LCG plug‑in (editable)

```bash
cd lcg_plugin
pip install -e .
cd ..
```

> **Note:** Some vLLM releases may require a tiny patch for custom plug‑ins. If you hit an import or registry error, check the *Issues* for a one‑line fix and your vLLM version.

---

## Quickstart

### A. Prepare training data

```bash
cd train
python prepare_train_dataset.py
```

This builds the supervision used for self‑distillation.

### B. Train the gate

```bash
cd train
python gate_train.py Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-gate-norm
```

Arguments are:

* **BASE_MODEL** (e.g., `Qwen/Qwen3-4B-Instruct-2507`)
* **OUTPUT_DIR** (e.g., `qwen3-4b-gate-norm`)

### C. Evaluate

Install the plug‑in (above), then:

```bash
# from repo root
cd train
python prepare_eval_dataset.py

# Example: evaluate on FLORES (non‑Latin subset)
VLLM_USE_V1=0 \
TOK_PATH='Qwen/Qwen3-8B' \
python eval_gate.py flores-no-latin [MODEL_PATH]

# Score results
python score_result flores [OUTPUT_PATH]
```

Where **MODEL_PATH** is your serving checkpoint (e.g., a local path or HF repo). **OUTPUT_PATH** is the directory produced by the evaluator.

---

## How it works (one‑paragraph intuition)

LCG classifies each vocabulary token into one of a few **language families** (e.g., Chinese/Japanese, Latin, symbols, low‑resource scripts). At each decode step it predicts which families are allowed given the prompt + history. If the sampler’s candidate set contains a token from a disallowed family, LCG masks it; otherwise it stays out of the way. Because true confusion is rare and desired tokens are usually already high‑ranked, intervention is minimal.

---

## Reproducing results

* **Models:** Qwen, GPT‑OSS, Gemma, Llama 3.1 (various sizes)
* **Benchmarks:** FLORES (including non‑Latin subset) and internal multilingual evaluations
* **Metrics:** line‑level / word‑level pass rates (lower confusion is better)

> Scripts in `train/` and `eval/` replicate the dataset preparation, evaluation runs, and scoring used in the paper. We’ll release trained gate weights to make plug‑and‑play testing even easier.

---

## Roadmap

* [ ] Release pre‑trained gate weights for major base models
* [ ] Improve code and Instructions

---

## Citation

```bibtex
@misc{zhang2025languageconfusiongatelanguageaware,
      title={Language Confusion Gate: Language-Aware Decoding Through Model Self-Distillation}, 
      author={Collin Zhang and Fei Huang and Chenhan Yuan and Junyang Lin},
      year={2025},
      eprint={2510.17555},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.17555}, 
}
```

---

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

---

## Maintainers & contact

* Collin Zhang rz454@cornell.edu (GitHub: `@collinzrj`)

For questions, please open a GitHub issue.
