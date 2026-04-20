# Auditing-Content-Moderation-AI-for-Bias-Adversarial-Robustness-Safety

This repository contains a complete end-to-end implementation of a **toxicity classification and moderation system**, including:

- Baseline model training (DistilBERT)
- Bias auditing across demographic groups
- Adversarial robustness evaluation
- Fairness mitigation techniques
- Production-style moderation pipeline

The work is structured into 5 parts as required by the assignment.

---

# 📁 Repository Structure

| File | Description |
|------|------------|
| `part1.ipynb` | Baseline toxicity classifier (DistilBERT) |
| `part2.ipynb` | Bias audit across identity groups |
| `part3.ipynb` | Adversarial attacks (evasion + poisoning) |
| `part4.ipynb` | Fairness mitigation techniques |
| `part5.ipynb` | Guardrail pipeline + calibration |
| `pipeline.py` | Final ModerationPipeline class |
| `requirements.txt` | All dependencies |
| `README.md` | This file |

---

# ⚙️ Environment Setup

### Python Version
- Python **3.10+**

### Hardware Used
- **CPU only** (no GPU used)
- Experiments adjusted accordingly (smaller subset)

---

## 🔧 Install Dependencies

```bash
pip install -r requirements.txt
