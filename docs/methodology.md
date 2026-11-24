# Methodology

## Objective
Exercise a model adapter against explicit synthetic output expectations and canary-leakage checks.

## Algorithm
Create cases → invoke predictor → case-fold responses → check forbidden and expected strings → aggregate.

## Interface
evaluate(cases, predict)

Inputs: Case objects with name, prompt, forbidden and expected strings; predictor callable accepting text and returning text.

Outputs: Per-case leakage and missing-expectation lists, pass flag, and aggregate pass rate (None for empty suite).

## Evaluation protocol
Use synthetic canaries only; compare models on a held-out, versioned test suite with human adjudication.

Report test corpus provenance, exact revision, configuration, failures and uncertainty. The included regression tests are not a benchmark.

## Threats to validity
Substring checks miss paraphrases and context-sensitive failures. Mock demo does not establish real-model safety.
