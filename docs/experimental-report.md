# Measured moderation benchmark

- Dataset revision: `f2970eb3a55777454c94069077cc8d9b5866312d`
- Toxic-BERT revision: `4d6c22e74ba2fdd26bc4f7238f50766b045a0d94`
- RoBERTa revision: `36295dd80b422dc49f40052021430dae76241adc`
- RoBERTa pretraining includes the related Civil Comments/Jigsaw task; exact overlap is unknown.
- Validation-selected model: Multiview ensemble
- Held-out comments: 2000; toxic comments: 154.
- [Experimental protocol](research-protocol.md) · [Raw measured JSON](results/moderation.json)

## Held-out comparison

| Metric | Majority | Toxic-BERT | TF-IDF | Calibrated BERT | Two-score hybrid | Ratio lexical | Three-score hybrid | Contextual MLP | Feature fusion | Contextual ensemble | Toxicity RoBERTa | Calibrated RoBERTa | Dual-transformer stack | Multiview ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| accuracy | 0.9230 | 0.9220 | 0.9115 | 0.9275 | 0.9145 | 0.9290 | 0.9315 | 0.9210 | 0.9235 | 0.9475 | 0.9440 | 0.9430 | 0.9450 | 0.9520 |
| balanced accuracy | 0.5000 | 0.7703 | 0.7973 | 0.7732 | 0.8287 | 0.7979 | 0.8260 | 0.6566 | 0.7770 | 0.7781 | 0.8804 | 0.8828 | 0.8601 | 0.7716 |
| precision | N/A | 0.4946 | 0.4493 | 0.5260 | 0.4647 | 0.5323 | 0.5427 | 0.4818 | 0.5027 | 0.6899 | 0.6019 | 0.5952 | 0.6158 | 0.7544 |
| recall | 0.0000 | 0.5909 | 0.6623 | 0.5909 | 0.7273 | 0.6429 | 0.7013 | 0.3442 | 0.6039 | 0.5779 | 0.8052 | 0.8117 | 0.7597 | 0.5584 |
| specificity | 1.0000 | 0.9496 | 0.9323 | 0.9556 | 0.9301 | 0.9529 | 0.9507 | 0.9691 | 0.9502 | 0.9783 | 0.9556 | 0.9540 | 0.9605 | 0.9848 |
| f1 | 0.0000 | 0.5385 | 0.5354 | 0.5566 | 0.5671 | 0.5824 | 0.6119 | 0.4015 | 0.5487 | 0.6290 | 0.6889 | 0.6868 | 0.6802 | 0.6418 |
| f2 | 0.0000 | 0.5687 | 0.6050 | 0.5767 | 0.6534 | 0.6172 | 0.6626 | 0.3650 | 0.5805 | 0.5973 | 0.7543 | 0.7567 | 0.7258 | 0.5890 |
| mcc | 0.0000 | 0.4986 | 0.4997 | 0.5183 | 0.5383 | 0.5468 | 0.5807 | 0.3663 | 0.5098 | 0.6037 | 0.6672 | 0.6658 | 0.6548 | 0.6247 |
| kappa | 0.0000 | 0.4962 | 0.4885 | 0.5172 | 0.5222 | 0.5439 | 0.5750 | 0.3605 | 0.5073 | 0.6010 | 0.6588 | 0.6563 | 0.6505 | 0.6167 |
| negative predictive value | 0.9230 | 0.9653 | 0.9707 | 0.9655 | 0.9761 | 0.9697 | 0.9745 | 0.9466 | 0.9664 | 0.9653 | 0.9833 | 0.9838 | 0.9796 | 0.9639 |
| false positive rate | 0.0000 | 0.0504 | 0.0677 | 0.0444 | 0.0699 | 0.0471 | 0.0493 | 0.0309 | 0.0498 | 0.0217 | 0.0444 | 0.0460 | 0.0395 | 0.0152 |
| false negative rate | 1.0000 | 0.4091 | 0.3377 | 0.4091 | 0.2727 | 0.3571 | 0.2987 | 0.6558 | 0.3961 | 0.4221 | 0.1948 | 0.1883 | 0.2403 | 0.4416 |
| roc auc | 0.5000 | 0.9171 | 0.8928 | 0.9171 | 0.9340 | 0.9187 | 0.9414 | 0.8397 | 0.9047 | 0.9433 | 0.9670 | 0.9670 | 0.9673 | 0.9682 |
| average precision | 0.0770 | 0.5672 | 0.5881 | 0.5672 | 0.6656 | 0.6335 | 0.6818 | 0.4331 | 0.5930 | 0.6863 | 0.7517 | 0.7517 | 0.7563 | 0.7636 |
| brier | 0.0711 | 0.0494 | 0.0472 | 0.0479 | 0.0407 | 0.0440 | 0.0393 | 0.0561 | 0.0470 | 0.0392 | 0.0412 | 0.0349 | 0.0346 | 0.0343 |
| log loss | 0.2714 | 0.1800 | 0.1720 | 0.1663 | 0.1460 | 0.1623 | 0.1420 | 0.2111 | 0.1733 | 0.1406 | 0.1350 | 0.1154 | 0.1147 | 0.1132 |
| ece | 0.0006 | 0.0280 | 0.0137 | 0.0137 | 0.0145 | 0.0161 | 0.0165 | 0.0205 | 0.0213 | 0.0133 | 0.0331 | 0.0166 | 0.0176 | 0.0164 |

## Recall-constrained operating policy

| Model | Validation recall | Test recall | Test precision | Test FNR |
|---|---:|---:|---:|---:|
| Toxic-BERT | 0.8243 | 0.7338 | 0.3717 | 0.2662 |
| TF-IDF | 0.8108 | 0.7468 | 0.3186 | 0.2532 |
| Calibrated BERT | 0.8243 | 0.7338 | 0.3717 | 0.2662 |
| Two-score hybrid | 0.8243 | 0.6883 | 0.4753 | 0.3117 |
| Ratio lexical | 0.8108 | 0.8506 | 0.3025 | 0.1494 |
| Three-score hybrid | 0.8108 | 0.7143 | 0.5046 | 0.2857 |
| Contextual MLP | 0.8108 | 0.7078 | 0.2460 | 0.2922 |
| Feature fusion | 0.8243 | 0.7662 | 0.3343 | 0.2338 |
| Contextual ensemble | 0.8108 | 0.7208 | 0.5092 | 0.2792 |
| Toxicity RoBERTa | 0.8243 | 0.7987 | 0.6029 | 0.2013 |
| Calibrated RoBERTa | 0.8243 | 0.7987 | 0.6029 | 0.2013 |
| Dual-transformer stack | 0.8108 | 0.7727 | 0.6134 | 0.2273 |
| Multiview ensemble | 0.8243 | 0.7662 | 0.6114 | 0.2338 |

## Paired uncertainty versus Toxic-BERT

| Metric | Difference | 95% percentile interval |
|---|---:|---:|
| f1 | 0.1033 | 0.0431 to 0.1665 |
| recall | -0.0325 | -0.1039 to 0.0325 |
| balanced_accuracy | 0.0014 | -0.0327 to 0.0341 |
| average_precision | 0.1964 | 0.1408 to 0.2620 |
| roc_auc | 0.0511 | 0.0354 to 0.0703 |
| brier | -0.0151 | -0.0200 to -0.0106 |

- Thresholds and model selection use the policy partition only. Test results are descriptive and do not establish universal safety.
- Bootstrap intervals condition on fitted models and sampled prevalence; pretraining overlap and domain generalization remain limitations.
