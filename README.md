# LLM-Safety-Evaluator

[![Validation](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/research.yml/badge.svg?branch=main&event=push)](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/research.yml)

Evaluation of language-model safety and utility through complementary moderation and response-behavior benchmarks. The suite separates toxicity detection, refusal behavior, instruction integrity, synthetic confidentiality and evidence-grounded abstention; no single aggregate score is treated as a safety guarantee.

[**Interactive dashboard**](https://rohanst14.github.io/LLM-Safety-Evaluator/) · [Experimental protocol](docs/research-protocol.md) · [Measured JSON](docs/results/moderation.json)

## Evaluation scope

| Dimension | Benchmark | Outcome |
|---|---|---|
| Toxicity moderation | Civil Comments | Discrimination, calibration and operating-point risk |
| Refusal behavior | XSTest, full 450-prompt set | Unsafe and safe refusal-marker rates (lexical proxy) |
| Instruction integrity | Controlled fictional records | Exact public-field response under untrusted quoted instructions |
| Confidentiality | Synthetic markers | Literal leakage rate |
| Grounded response | Missing-field probes | Exact abstention rate |

[Response-behavior dashboard](https://rohanst14.github.io/LLM-Safety-Evaluator/behavior.html) · [Response-level protocol](docs/behavior-protocol.md)

## Moderation method

- Pretrained Toxic-BERT inference with a pinned model revision.
- Supervised word/character TF-IDF logistic regression fitted on the official training partition.
- Supervised log-ratio feature weighting and regularized two- and three-score hybrid stacking.
- Logistic recalibration and hybrid stacking fitted on a separate calibration subset.
- F1-selected thresholds and recall-constrained operating points selected on a disjoint policy subset.
- Common held-out test sample; comprehensive classification, discrimination and calibration metrics.
- Paired stratified bootstrap intervals and descriptive error slices, with minimum class support for rate reporting.

The dashboard reads the measured report and displays the actual comparison, including the validation-selected model and test trade-offs. The prior single-model experiment remains available in Git history. The expanded report is published only after execution and validation.

## Run locally

```bash
git clone https://github.com/rohanst14/LLM-Safety-Evaluator.git
cd LLM-Safety-Evaluator
python -m venv .venv
# Activate the environment for your operating system.
pip install -e . -r requirements-research.txt
python experiments/evaluate_moderation.py
python experiments/evaluate_behavior.py
python -m unittest discover -s tests -v
python tests/check_report.py results/moderation.json
python tests/check_behavior_report.py results/behavior.json
```

## Repository structure

| Path | Purpose |
|---|---|
| src/llm_safety_evaluator/ | Model adapter, threshold selection and evaluation |
| experiments/ | Civil Comments and response-behavior benchmarks |
| tests/ | Unit tests, report checks and browser tests |
| docs/ | Protocol, dashboard and measured results |

The separate synthetic canary example in `examples/demo.py` tests string-matching behavior only, not real-model safety. See [methodology](docs/methodology.md).
