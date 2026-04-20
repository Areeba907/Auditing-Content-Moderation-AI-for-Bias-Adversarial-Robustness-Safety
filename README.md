# Responsible AI Assignment – Toxicity Moderation System

This repository contains a complete end-to-end implementation of a **toxicity classification and moderation system**, including:

- Baseline model training (DistilBERT)
- Bias auditing across demographic groups
- Adversarial robustness evaluation
- Fairness mitigation techniques
- Production-style moderation pipeline

The work is structured into 5 parts as required by the assignment.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `part1.ipynb` | Baseline toxicity classifier (DistilBERT) |
| `part2.ipynb` | Bias audit across identity groups |
| `part3.ipynb` | Adversarial attacks (evasion + poisoning) |
| `part4.ipynb` | Fairness mitigation techniques |
| `part5.ipynb` | Guardrail pipeline + calibration |
| `pipeline.py` | Final ModerationPipeline class |
| `requirements.txt` | All dependencies |
| `README.md` | This file |

---

## ⚙️ Environment Setup

**Python Version:** 3.10+  
**Hardware:** CPU only (no GPU used) — experiments adjusted with a smaller subset accordingly.

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

**Jigsaw Unintended Bias in Toxicity Classification**

> ⚠️ Dataset not included in repo (as per instructions). Place files manually:

```
dataset/jigsaw-unintended-bias-train.csv   # Parts 1–4
dataset/validation.csv                      # Part 5
```

---

## ▶️ How to Reproduce

Run notebooks **in order**:

### Part 1 — Baseline Model (`part1.ipynb`)

Trains a DistilBERT toxicity classifier.

**Outputs:** trained model, evaluation metrics, saved probabilities

---

### Part 2 — Bias Audit (`part2.ipynb`)

Audits model fairness across demographic identity groups using outputs from Part 1.

---

### Part 3 — Adversarial Attacks (`part3.ipynb`)

Evaluates robustness via:
- **Evasion attack** — text perturbation
- **Poisoning attack** — label flipping

---

### Part 4 — Fairness Mitigation (`part4.ipynb`)

Applies three mitigation techniques:
- Reweighing (AIF360)
- Threshold Optimization (Fairlearn)
- Oversampling

---

### Part 5 — Guardrail Pipeline (`part5.ipynb`)

Builds a full production-style moderation pipeline with calibration (Isotonic Regression) and a human-review tier.

---

## 🧠 Using the Final Pipeline

```python
from pipeline import ModerationPipeline

pipeline = ModerationPipeline()
result = pipeline.predict("You are an idiot")
print(result)
```

**Example output:**

```json
{
  "decision": "block",
  "layer": "model",
  "confidence": 0.82
}
```

---

## 📈 Key Results

### Part 1 — Baseline Model

| Metric | Value |
|--------|-------|
| Accuracy | 0.9425 |
| Macro F1 | 0.7882 |
| AUC-ROC | 0.9379 |
| Best Threshold | 0.3 |

### Part 2 — Bias Audit

- Higher FPR observed for high-black cohort
- TPR disparity detected across groups
- Small sample size — results are indicative, not definitive

### Part 3 — Adversarial Robustness

| Metric | Value |
|--------|-------|
| Attack Success Rate | 92.16% |
| Confidence (before attack) | 0.92 |
| Confidence (after attack) | 0.08 |

Model shows strong vulnerability to text perturbation.

### Part 4 — Mitigation Comparison

| Method | F1 | Fairness |
|--------|----|----------|
| Baseline | 0.788 | Moderate disparity |
| Reweighing | ↓ | No improvement |
| Threshold Optimization | ↓↓↓ | Fairness shift |
| Oversampling | ~ | Worse disparity |

> No perfect solution — a performance/fairness trade-off exists.

### Part 5 — Pipeline Decisions

| Decision | Count |
|----------|-------|
| Allow | 893 |
| Review | 87 |
| Block | 20 |

**Auto-action metrics:** F1: 0.765 · Precision: 0.75 · Recall: 0.43

---

## ⚠️ Important Notes

- Dataset and model weights are **not included** (per submission instructions)
- All notebooks are **fully executed with outputs visible**
- Reduced dataset size used due to CPU-only constraints — results remain valid for analysis

---

## 🧩 Key Learnings

- High accuracy ≠ fairness
- NLP models are highly fragile to adversarial text perturbations
- Fairness mitigation always introduces performance trade-offs
- Production moderation systems require calibration, threshold tuning, and human-in-the-loop review

---

