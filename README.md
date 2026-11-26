# LLM-Safety-Evaluator

[![Validation](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/research.yml/badge.svg?branch=main&event=push)](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/research.yml)

Local toxicity classification, probability calibration and operating-point evaluation using Civil Comments human annotations. The benchmark evaluates text moderation; it does not establish comprehensive LLM safety.

[**Interactive dashboard**](https://rohanst14.github.io/LLM-Safety-Evaluator/) · [Experimental protocol](docs/research-protocol.md) · [Measured JSON](docs/results/moderation.json)

## Method

- Pretrained Toxic-BERT inference with a pinned model revision.
- Supervised word/character TF-IDF logistic regression fitted on the official training partition.
- Logistic recalibration and hybrid stacking fitted on a separate calibration subset.
- F1-selected thresholds and recall-constrained operating points selected on a disjoint policy subset.
- Common held-out test sample; comprehensive classification, discrimination and calibration metrics.

The dashboard reads the measured report and displays the actual comparison, including the validation-selected model and test trade-offs. The prior single-model experiment remains available in Git history. The expanded report is published only after execution and validation.

## Run locally

```bash
git clone https://github.com/rohanst14/LLM-Safety-Evaluator.git
cd LLM-Safety-Evaluator
python -m venv .venv
# Activate the environment for your operating system.
pip install -e . -r requirements-research.txt
python experiments/evaluate_moderation.py
python -m unittest discover -s tests -v
python tests/check_report.py results/moderation.json
```

## Repository structure

| Path | Purpose |
|---|---|
| src/llm_safety_evaluator/ | Model adapter, threshold selection and evaluation |
| experiments/ | Reproducible Civil Comments benchmark |
| tests/ | Unit tests, report checks and browser tests |
| docs/ | Protocol, dashboard and measured results |

The separate synthetic canary example in `examples/demo.py` tests string-matching behavior only, not real-model safety. See [methodology](docs/methodology.md).
