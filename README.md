# LLM-Safety-Evaluator

### LLM output safety evaluation

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![Tests](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/workflows/tests.yml/badge.svg) ![Stage](https://img.shields.io/badge/stage-research%20prototype-64748b)

Exercise a model adapter against explicit synthetic output expectations and canary-leakage checks.

## Research extension: semantic safety judge evaluation

Distinct from RedTeam-Lab's adversarial regression suite, this study measures the performance of a local toxicity classifier against human annotations from Civil Comments. It uses Toxic-BERT through Hugging Face Transformers, threshold selection on validation data, and held-out test evaluation.

```bash
pip install -e . -r requirements-research.txt
python experiments/evaluate_moderation.py --limit 500
python -m unittest discover -s tests -v
```

See [research protocol](docs/research-protocol.md). The default run targets 500 validation and 500 test comments. The full inference workflow is opt-in via Actions > Semantic moderation validation > Run workflow. No empirical classifier performance is claimed until execution. The original substring-based evaluator remains for compatibility.

## Design (original baseline)

Create cases → invoke predictor → case-fold responses → check forbidden and expected strings → aggregate.

## Quick start

```bash
git clone https://github.com/rohanst14/LLM-Safety-Evaluator.git
cd LLM-Safety-Evaluator
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -e .
python examples/demo.py
python -m unittest discover -s tests -v
```

## Usage

```python
from llm_safety_evaluator import evaluate, synthetic_cases
report = evaluate(synthetic_cases(), lambda p: 'hello' if 'hello' in p else 'Cannot disclose')
print(report)
```

## API contract

| Field | Description |
|---|---|
| Entry point | `evaluate(cases, predict)` |
| Inputs | Case objects with name, prompt, forbidden and expected strings; predictor callable accepting text and returning text. |
| Outputs | Per-case leakage and missing-expectation lists, pass flag, and aggregate pass rate (None for empty suite). |

## Repository map

| Path | Role |
|---|---|
| [src/llm_safety_evaluator/](src/llm_safety_evaluator/) | Python implementation |
| [examples/demo.py](examples/demo.py) | Reproducible synthetic demo |
| [tests/test_core.py](tests/test_core.py) | Regression checks |
| [docs/methodology.md](docs/methodology.md) | Evaluation design and limitations |
| [ROADMAP.md](ROADMAP.md) | Milestones and planned work |
| [CI workflow](.github/workflows/tests.yml) | Automated tests and demo |

## Evaluation and limitations

Use synthetic canaries only; compare models on a held-out, versioned test suite with human adjudication.

Substring checks miss paraphrases and context-sensitive failures. Mock demo does not establish real-model safety. The current demonstration is synthetic; no real-world performance claims are made.

## Project status

AI-assisted research prototype. See [roadmap](ROADMAP.md) for planned extensions.
