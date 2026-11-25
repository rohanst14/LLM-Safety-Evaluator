# LLM-Safety-Evaluator

**Semantic moderation & classifier assurance**

![Validation](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/research.yml/badge.svg)

Local Toxic-BERT inference evaluated against human toxicity annotations, with validation-selected thresholds and held-out test metrics.

## Dashboard

[**Open interactive dashboard**](https://rohanst14.github.io/LLM-Safety-Evaluator/) · [Evaluation protocol](docs/research-protocol.md)

Explore published benchmark metrics and import additional experiment reports in JSON format.

## Installation and execution

```bash
git clone https://github.com/rohanst14/LLM-Safety-Evaluator.git
cd LLM-Safety-Evaluator
python -m venv .venv
# Activate the environment for your operating system.
pip install -e . -r requirements-research.txt
python experiments/evaluate_moderation.py --limit 500
python -m unittest discover -s tests -v
```

## Dataset and methodology

Civil Comments; 500 validation and 500 test comments by default. Toxicity is not a comprehensive measure of LLM safety.

Metrics: Precision, recall, F1, ROC-AUC, average precision, false-positive and false-negative rates.

See [evaluation protocol](docs/research-protocol.md) for assumptions and limitations. The experiment writes `results/moderation.json`. Successful benchmark jobs publish measured JSON reports; the dashboard loads the published report automatically when available. 

## Repository organization

| Path | Purpose |
|---|---|
| src/ | Core algorithms and model interfaces |
| experiments/ | Reproducible experiment entry points |
| tests/ | Automated regression tests |
| docs/ | Dashboard and methodology |
| requirements-research.txt | Experiment dependencies |
| .github/workflows/ | Continuous validation |

Use [GitHub Actions](https://github.com/rohanst14/LLM-Safety-Evaluator/actions) to inspect actual test outcomes.
