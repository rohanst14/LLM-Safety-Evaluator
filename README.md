# LLM-Safety-Evaluator

[![Validation](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/research.yml/badge.svg?branch=main&event=push)](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/research.yml)

Evaluation of language-model safety and utility through complementary moderation and response-behavior benchmarks. The suite separates toxicity detection, refusal behavior, instruction integrity, synthetic confidentiality and evidence-grounded abstention; no single aggregate score is treated as a safety guarantee.

[**Interactive dashboard**](https://rohanst14.github.io/LLM-Safety-Evaluator/) · [Experimental protocol](docs/research-protocol.md) · [Measured JSON](docs/results/moderation.json) · [Moderation report](docs/experimental-report.md)

## Evaluation scope

| Dimension | Benchmark | Outcome |
|---|---|---|
| Toxicity moderation | Civil Comments | Discrimination, calibration and operating-point risk |
| Refusal behavior | XSTest, full 450-prompt set | Unsafe and safe refusal-marker rates (lexical proxy) |
| Instruction integrity | Controlled fictional records | Exact public-field response under untrusted quoted instructions |
| Confidentiality | Synthetic markers | Literal leakage rate |
| Grounded response | Missing-field probes | Exact abstention rate |

[Response-behavior dashboard](https://rohanst14.github.io/LLM-Safety-Evaluator/behavior.html) · [Response-level protocol](docs/behavior-protocol.md) · [Measured response-level report](docs/behavior-report.md)

## Moderation method

- Pretrained Toxic-BERT inference with a pinned model revision.
- Supervised word/character TF-IDF logistic regression fitted on the official training partition.
- Supervised log-ratio feature weighting, frozen MiniLM contextual embeddings, nonlinear MLP and sparse/dense feature fusion.
- Task-specialized RoBERTa reference, logistic recalibration, and dual-transformer/multiview stacking.
- Regularized two-, three-, five- and six-score hybrid stacking.
- Logistic recalibration and hybrid stacking fitted on a separate calibration subset.
- F1-selected thresholds and recall-constrained operating points selected on a disjoint policy subset.
- Common held-out test sample; comprehensive classification, discrimination and calibration metrics.
- Paired stratified bootstrap intervals and descriptive error slices, with minimum class support for rate reporting.

The RoBERTa checkpoint was trained on the related Jigsaw Civil Comments task; exact held-out overlap is not established, so its results are an in-domain comparison rather than independent external validation.

The dashboard reads the measured report and displays the actual comparison, including the validation-selected model and test trade-offs. The prior single-model experiment remains available in Git history. The dashboard reports the latest successfully published results; newer configurations are experimental until their measured reports are validated.

## Verified comparative benchmark (schema 3)

The [completed six-classifier experiment](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/runs/45758172050) used 30,000 training, 2,000 validation and 2,000 test comments (154 test positives). The validation-selected three-score hybrid improved F1 from 0.5385 to 0.6370 and average precision from 0.5672 to 0.6824 compared with tuned Toxic-BERT. Paired 95% bootstrap intervals for F1 difference: [0.0435, 0.1487]; for recall difference: [-0.0519, 0.0714]. The recall-constrained ratio-weighted lexical model achieved test recall 0.8312, with precision 0.3307. This is a safety–utility trade-off, not a universal safety guarantee. These are results of the earlier schema-3 configuration, not of the newer contextual or RoBERTa models.

[Measured artifact](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/runs/45758172050/artifacts/15089500134)

## Verified response-level benchmark

The [completed Qwen2.5-0.5B-Instruct experiment](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/runs/45758724507) evaluated all 450 XSTest prompts and 96 controlled probes per policy.

| Endpoint | Reference | Boundary-aware |
|---|---:|---:|
| Safe-prompt refusal marker | 27.60% | 15.20% |
| Unsafe-prompt refusal marker | 68.00% | 58.50% |
| Missing-field exact abstention | 100.00% | 0.00% |
| Instruction-boundary exact match | 100.00% | 0.00% |
| Confidentiality literal leakage | 0/24 | 0/24 |

The boundary-aware policy reduces safe-prompt refusal markers but is not a uniform improvement. Refusal-marker rates are lexical proxies; high token-limit rates and exact-match sensitivity limit interpretation. See the [measured response report](docs/behavior-report.md).

## CPU fine-tuning experiment

A separate [fine-tuning protocol](docs/fine-tuning-protocol.md) compares full BERT-Mini encoder adaptation with BCE, weighted BCE and focal loss against matched reference classifiers. The confirmation sample excludes the previously examined test comments. Results are published only after successful execution.

[Fine-tuning dashboard](https://rohanst14.github.io/LLM-Safety-Evaluator/fine-tuning.html)

## Run locally

```bash
git clone https://github.com/rohanst14/LLM-Safety-Evaluator.git
cd LLM-Safety-Evaluator
python -m venv .venv
# Activate the environment for your operating system.
pip install -e . -r requirements-research.txt
python experiments/evaluate_moderation.py
python experiments/evaluate_behavior.py
# Optional CPU training experiment:
python experiments/fine_tune.py
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
