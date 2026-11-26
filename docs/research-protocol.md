# Experimental protocol

## 1. Dataset and task
- Civil Comments: human toxicity annotation fractions; binary label is toxicity ≥ 0.5.
- Official training, validation and test partitions; sample 30,000 training comments and 2,000 comments per validation/test split by default.
- Streaming shuffle buffer of 10,000 with seed 42; not a uniform sample of the complete corpus.
- Deduplicate normalized text across the sampled partitions in train → validation → test order; record duplicate counts, SHA-256 hashes, resolved revisions and package versions.
- Raw comments are neither committed nor included in the dashboard.

## 2. Models
- Majority reference: constant training prevalence, threshold 0.5.
- Toxic-BERT: pretrained `unitary/toxic-bert`, local sigmoid of the toxicity logit; truncate at 512 tokens.
- Lexical baseline: word 1–2 grams and character 3–5 grams, TF-IDF, L2-regularized logistic regression (C=4), fit on training only.
- Calibrated BERT: logistic recalibration of BERT log-odds.
- Hybrid: regularized logistic stacking of BERT and lexical log-odds; no test labels are used for fitting.

## 3. Validation and operating points
- Stratify validation into equal calibration and policy partitions; fit recalibration and stacking only on the calibration partition.
- Select F1-maximizing threshold on the policy partition over 0–1 at 0.01 intervals (higher threshold breaks ties).
- Independently select a policy threshold maximizing precision subject to at least 80% validation recall using the observed score thresholds.
- Select model by policy F1; alphabetically descending model key breaks ties. Evaluate every model on the same test sample; test metrics never determine selection.
- Report fixed 0.5, F1-selected, and recall-constrained operating points. The validation recall target is not a guarantee on test.

## 4. Metrics
- Confusion counts, accuracy, balanced accuracy, precision, recall, specificity, F1, F2, MCC, Cohen's kappa, NPV, false-positive and false-negative rates.
- ROC-AUC and average precision are threshold-independent ranking measures. Average precision should be compared with the positive prevalence; raw accuracy is misleading under imbalance.
- Log loss, Brier score and 15-bin probability calibration error quantify probabilistic performance (lower is better).
- Record reliability bins, ROC and precision–recall curves, paired correctness counts and validation policy metrics. Undefined ratios are null rather than fabricated.
- Curves may be downsampled for display; scalar ranking metrics use all observations.

## 5. Limitations
- Human toxicity ratings are contextual and imperfect; this is text moderation, not comprehensive LLM safety or jailbreak resistance.
- Pretrained model training overlap has not been ruled out. The dataset and model cards describe distinct Jigsaw tasks but shared domains; do not claim independent external validation.
- No demographic fairness or subgroup assurance is claimed. The standard Civil Comments split and sampled evaluation do not establish subgroup performance.
- Dataset may contain offensive text. No raw examples are rendered in the public dashboard.

## 6. Reproducibility
- Run `python experiments/evaluate_moderation.py --limit 2000 --train-limit 30000` after dependency installation.
- For exact reruns, pass the dataset and model revisions recorded in the measured JSON.
- Validate the output with `python tests/check_report.py results/moderation.json`.
- Sources: [Civil Comments](https://huggingface.co/datasets/google/civil_comments), [Toxic-BERT model card](https://huggingface.co/unitary/toxic-bert).
