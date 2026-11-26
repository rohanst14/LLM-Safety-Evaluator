# CPU fine-tuning experimental protocol

## Research question
- Compare end-to-end fine-tuning of a compact, general-pretrained BERT with existing lexical and toxicity-classifier references under controlled computational constraints.
- Test whether class weighting or focal loss improves positive-class discrimination over ordinary BCE. No 90% F1 guarantee is assumed.

## Dataset and partitions
- Use the previously recorded Civil Comments revision and verify the exact 30,000 training, 2,000 validation and 2,000 previously examined test sample checksums.
- Stratified sample of 9,000 from the training pool; 8,000 supervised fit and 1,000 internal development examples (seed 42). Select checkpoints and loss on development average precision only.
- Reuse the disjoint 1,000 calibration and 1,000 policy samples from official validation.
- Confirmation: the next 2,000 distinct comments in the same pinned official test stream, excluding all previously examined 2,000 test comments and normalized duplicates with training or validation. This is a previously unreported partition within the same corpus, not external validation. Record its checksum.
- This protocol is defined before confirmation results are observed. Avoid iterative tuning against that set.

## Model and loss ablation
- General-pretrained Google BERT-Mini: four transformer layers, hidden width 256, four attention heads; one newly initialized binary logit output. Update all encoder and classification parameters.
- Identical initialization seed, fit/development examples, batch order, token limit, learning rate, optimizer and epoch budget for BCE, positive-class-weighted BCE and focal loss (gamma=2, no alpha weighting).
- Weighted BCE uses negative/positive fit count. Standard BCE is the control. Focal loss scales each binary cross-entropy term by (1-p_t)^2.
- AdamW, learning rate 5e-5, weight decay 0.01, gradient norm clipping 1, linear schedule with 10% warm-up, batch size 16, dynamic padding, maximum 128 tokens, two epochs; CPU with four threads by default.
- Choose each checkpoint by maximum internal development AP, select loss by the same development AP, ties resolved by model key. Record epoch histories, trainable parameter count and before/after hash of a pretrained attention parameter to verify genuine encoder adaptation.
- Compare against TF-IDF, ratio-weighted lexical regression and a three-score Toxic-BERT/lexical hybrid trained on the same fit subset, plus fixed pretrained Toxic-BERT. The prior 30,000-example hybrid results are historical context, not the matched comparison.

## Calibration and evaluation
- Fit a logistic probability calibrator for each model on calibration only; choose F1-maximizing and 80%-recall-constrained thresholds using policy only.
- Report raw fixed-0.5, calibrated fixed-0.5, F1-selected and recall-constrained metrics on the identical confirmation sample. Include AP, ROC-AUC, positive-class F1, recall, precision, balanced accuracy, MCC, Brier, NLL and ECE.
- Paired 1,000-replicate bootstrap compares the development-selected loss with the matched hybrid reference; conditional on one training seed. No significance guarantee or training-variance claim.
- Record CPU training elapsed time (includes checkpoint evaluation and inference), package versions and resolved pretrained checkpoint.
- Limitations: small fit subset, short truncation, one seed, possible pretraining overlap and no external corpus. The benchmark concerns toxicity detection, not universal LLM safety.

## Reproduction
- `python experiments/fine_tune.py` (CPU default); optional `--device cuda` if available.
- `python tests/check_fine_tuning_report.py results/fine-tuning.json`
- [Google compact BERT model card](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) · [Compact model pretraining study](https://arxiv.org/abs/1908.08962).
