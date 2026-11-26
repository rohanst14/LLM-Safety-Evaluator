# Research methodology

## Moderation and calibration
- Compare pretrained toxicity classifiers, supervised lexical heads, contextual feature fusion and calibrated probability stacks using shared Civil Comments partitions.
- Fit supervised models on training data, calibrate on one validation subset and select operating thresholds on the other.
- Select the proposed multiview configuration by validation toxic-class F1; compare all fixed test results without using test labels for threshold selection.
- Report class-sensitive discrimination, probability quality, confusion counts, recall-constrained operating points and paired bootstrap intervals.
- [Moderation protocol](research-protocol.md) · [Measured report](experimental-report.md).

## Response-level safety and utility
- Generate responses locally with a pinned instruction-tuned model under two controlled system policies.
- Evaluate the full XSTest prompt set and fictional-record tasks for public-field retrieval, grounded abstention, instruction integrity and literal confidentiality leakage.
- Report refusal markers and exact-match task outcomes separately, with denominators and Wilson intervals.
- [Response protocol](behavior-protocol.md) · [Measured report](behavior-report.md).

## CPU transformer adaptation
- Train all BERT-Mini encoder and classification weights under BCE, weighted BCE and focal loss using the same training and development partitions.
- Select checkpoint and loss by development average precision; calibrate and select decision thresholds on disjoint validation subsets.
- Compare matched reference classifiers on a previously unreported confirmation partition, separate from the original test sample.
- [Fine-tuning protocol](fine-tuning-protocol.md) · [Fine-tuning dashboard](https://rohanst14.github.io/LLM-Safety-Evaluator/fine-tuning.html).

The lightweight callable adapter remains a separate regression-test utility; research benchmarks use real model inference.
