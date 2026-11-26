# LLM-Safety-Evaluator

[![Moderation validation](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/research.yml/badge.svg?branch=main&event=push)](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/research.yml) [![Response-level validation](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/behavior.yml/badge.svg?branch=main)](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/behavior.yml)

A research framework for evaluating language-model safety, reliability and utility through supervised toxicity classification, calibrated transformer ensembles and controlled response-level experiments.

[**Interactive evaluation dashboard**](https://rohanst14.github.io/LLM-Safety-Evaluator/) · [Moderation report](docs/experimental-report.md) · [Response-level report](docs/behavior-report.md)

## Research objectives

- Develop and compare contextual, lexical and hybrid classifiers for imbalanced toxicity detection.
- Assess probability reliability and risk-aware operating policies through independent calibration and threshold selection.
- Evaluate response behavior across refusal, instruction integrity, confidentiality, grounded abstention and benign-task utility.

## Measured moderation benchmark

Civil Comments: 30,000 training comments, 2,000 validation comments and 2,000 held-out test comments. The expanded experiment compares 13 classifier configurations and a majority-class reference.

| Metric | Toxic-BERT | Toxicity RoBERTa | Multiview ensemble |
|---|---:|---:|---:|
| Accuracy | 92.20% | 94.40% | 95.20% |
| Toxic-class precision | 49.46% | 60.19% | 75.44% |
| Toxic-class recall | 59.09% | 80.52% | 55.84% |
| Toxic-class F1 | 53.85% | 68.89% | 64.18% |
| Macro-F1 | 74.79% | 82.91% | 80.80% |
| Weighted F1 | 92.51% | 94.76% | 94.87% |
| ROC-AUC | 91.71% | 96.70% | 96.82% |
| Average precision | 56.72% | 75.17% | 76.36% |
| Calibration error (lower is better) | 2.80% | 3.31% | 1.64% |

The validation-selected multiview ensemble achieves 95.20% accuracy and 76.36% average precision. The RoBERTa comparator achieves 80.52% toxicity recall and 68.89% positive-class F1. Full model comparisons, checkpoint provenance, operating-point analysis and paired uncertainty intervals are provided in the [experimental report](docs/experimental-report.md) and [measured JSON](docs/results/moderation.json).

## Methodology

- Contextual inference with pretrained Toxic-BERT and task-specialized RoBERTa.
- Supervised word/character TF-IDF, log-ratio feature weighting, MiniLM embeddings, nonlinear MLP and sparse/dense feature fusion.
- Regularized probability stacking with independent calibration and validation-selected decision thresholds.
- Shared held-out evaluation, class-sensitive metrics, calibration diagnostics and paired bootstrap analysis.

[Moderation protocol](docs/research-protocol.md) · [Completed benchmark workflow](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/runs/45758913740)

## Response-level evaluation

Local Qwen2.5-0.5B-Instruct generation on all 450 XSTest prompts and 96 controlled probes per policy. The experiment measures refusal markers, public-field retrieval, instruction-boundary exact match, literal confidentiality leakage and grounded abstention.

[Response-level dashboard](https://rohanst14.github.io/LLM-Safety-Evaluator/behavior.html) · [Protocol](docs/behavior-protocol.md) · [Measured results](docs/behavior-report.md)

## Transformer fine-tuning

Full BERT-Mini encoder adaptation on CPU with a controlled BCE, weighted BCE and focal-loss ablation. Checkpoint selection uses internal development average precision; calibration and policy selection use separate validation subsets. Matched reference classifiers and a previously unreported confirmation sample support comparative analysis.

[Fine-tuning protocol](docs/fine-tuning-protocol.md) · [Fine-tuning dashboard](https://rohanst14.github.io/LLM-Safety-Evaluator/fine-tuning.html)

## Reproduce

```bash
git clone https://github.com/rohanst14/LLM-Safety-Evaluator.git
cd LLM-Safety-Evaluator
python -m venv .venv
# Activate the environment for your operating system.
pip install -e . -r requirements-research.txt
python experiments/evaluate_moderation.py
python experiments/evaluate_behavior.py
python experiments/fine_tune.py
python -m unittest discover -s tests -v
python tests/check_report.py results/moderation.json
python tests/check_behavior_report.py results/behavior.json
python tests/check_fine_tuning_report.py results/fine-tuning.json
```

## Repository structure

| Path | Purpose |
|---|---|
| `src/llm_safety_evaluator/` | Classifiers, calibration, evaluation metrics and model interfaces |
| `experiments/` | Reproducible moderation, response-level and fine-tuning experiments |
| `tests/` | Regression tests, numerical report checks and browser validation |
| `docs/` | Experimental protocols, measured reports and dashboards |
| `.github/workflows/` | Automated research validation and publication |
