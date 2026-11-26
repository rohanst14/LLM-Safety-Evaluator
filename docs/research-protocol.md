# Experimental protocol: toxicity moderation

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
- Ratio-weighted lexical: smoothed class-conditional document-frequency log ratios multiply TF-IDF features, followed by L2 logistic regression. This is NB-inspired feature weighting, not NB-SVM.
- Three-score hybrid: logistic stacking of transformer, lexical and ratio-weighted lexical log-odds, fitted on calibration only.

- Contextual MLP: frozen all-MiniLM-L6-v2, attention-masked mean pooling and L2-normalized 384-dimensional embeddings; 128/64-unit MLP with early stopping on a 15% internal training subset.
- Feature fusion: concatenate sparse word/character TF-IDF and contextual embeddings scaled by 3; fit L2 logistic regression (C=8) using official training labels.
- Contextual ensemble: calibrate a five-score stack of Toxic-BERT, lexical, ratio-weighted lexical, feature fusion and contextual MLP on the independent calibration subset.
- The encoder is frozen, with 256-token truncation and a resolved model revision. No toxicity labels from validation or test are used to train its supervised heads. General pretraining overlap cannot be ruled out.
- Source: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

- Task-specialized RoBERTa: `unitary/unbiased-toxic-roberta`, resolved revision, toxicity sigmoid with 512-token truncation. This checkpoint was trained for the Jigsaw unintended-bias task on Civil Comments; exact evaluation-split overlap is not established. Treat it as a potentially in-domain pretrained comparator, not independent evidence of generalization.
- Calibrated RoBERTa, dual-transformer logit stack, and six-score multiview ensemble combining both pretrained toxicity models, lexical and ratio-weighted lexical heads, feature fusion and contextual MLP; fit all stacks on the calibration subset only.
- Source: [RoBERTa model card](https://huggingface.co/unitary/unbiased-toxic-roberta).

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

## 5. Uncertainty and error analysis
- Use 1,000 paired, class-stratified bootstrap replicates (seed 1729) on fixed held-out predictions and thresholds. Report percentile 95% intervals for F1, recall, balanced accuracy, average precision, ROC-AUC and Brier score, including paired differences versus tuned Toxic-BERT.
- Intervals condition on the fitted models and sample prevalence; they do not incorporate training variability, topic clustering, annotation noise or multiple-comparison adjustment. Treat them as descriptive uncertainty, not proof of statistical superiority.
- Examine short, medium and long comments and annotation scores near the binary decision boundary. Suppress recall or FPR if the corresponding class has fewer than 20 observations. These slices are not demographic fairness metrics.
- Reference: [Wang and Manning (2012), Baselines and Bigrams](https://aclanthology.org/P12-2018/) for NB-inspired weighting; the implementation uses TF-IDF and logistic regression rather than their NB-SVM.

## 6. Limitations
- Human toxicity ratings are contextual and imperfect; this track evaluates text moderation. See the separate [response-level behavior protocol](behavior-protocol.md) for complementary safety and utility probes.
- Pretrained model training overlap has not been ruled out. The dataset and model cards describe distinct Jigsaw tasks but shared domains; do not claim independent external validation.
- No demographic fairness or subgroup assurance is claimed. The standard Civil Comments split and sampled evaluation do not establish subgroup performance.
- Dataset may contain offensive text. No raw examples are rendered in the public dashboard.

## 7. Reproducibility
- Run `python experiments/evaluate_moderation.py --limit 2000 --train-limit 30000` after dependency installation.
- For exact reruns, pass the dataset, BERT, RoBERTa and encoder revisions recorded in the measured JSON.
- Validate the output with `python tests/check_report.py results/moderation.json`.
- Sources: [Civil Comments](https://huggingface.co/datasets/google/civil_comments), [Toxic-BERT model card](https://huggingface.co/unitary/toxic-bert).
