# Measured moderation benchmark

- Dataset revision: `f2970eb3a55777454c94069077cc8d9b5866312d`
- Toxic-BERT revision: `4d6c22e74ba2fdd26bc4f7238f50766b045a0d94`
- Validation-selected model: Three-score hybrid
- Held-out comments: 2000; toxic comments: 154.
- [Experimental protocol](research-protocol.md) · [Raw measured JSON](results/moderation.json)

## Held-out comparison

| Metric | Majority | Toxic-BERT | TF-IDF | Calibrated BERT | Two-score hybrid | Ratio lexical | Three-score hybrid |
|---|---:|---:|---:|---:|---:|---:|---:|
| accuracy | 0.9230 | 0.9220 | 0.9165 | 0.9275 | 0.9140 | 0.9320 | 0.9470 |
| balanced accuracy | 0.5000 | 0.7703 | 0.7941 | 0.7732 | 0.8284 | 0.8054 | 0.7898 |
| precision | N/A | 0.4946 | 0.4695 | 0.5260 | 0.4628 | 0.5489 | 0.6739 |
| recall | 0.0000 | 0.5909 | 0.6494 | 0.5909 | 0.7273 | 0.6558 | 0.6039 |
| specificity | 1.0000 | 0.9496 | 0.9388 | 0.9556 | 0.9296 | 0.9550 | 0.9756 |
| f1 | 0.0000 | 0.5385 | 0.5450 | 0.5566 | 0.5657 | 0.5976 | 0.6370 |
| f2 | 0.0000 | 0.5687 | 0.6031 | 0.5767 | 0.6527 | 0.6312 | 0.6167 |
| mcc | 0.0000 | 0.4986 | 0.5083 | 0.5183 | 0.5369 | 0.5635 | 0.6096 |
| kappa | 0.0000 | 0.4962 | 0.5003 | 0.5172 | 0.5205 | 0.5608 | 0.6085 |
| negative predictive value | 0.9230 | 0.9653 | 0.9698 | 0.9655 | 0.9761 | 0.9708 | 0.9672 |
| false positive rate | 0.0000 | 0.0504 | 0.0612 | 0.0444 | 0.0704 | 0.0450 | 0.0244 |
| false negative rate | 1.0000 | 0.4091 | 0.3506 | 0.4091 | 0.2727 | 0.3442 | 0.3961 |
| roc auc | 0.5000 | 0.9171 | 0.8974 | 0.9171 | 0.9349 | 0.9191 | 0.9413 |
| average precision | 0.0770 | 0.5672 | 0.5931 | 0.5672 | 0.6643 | 0.6349 | 0.6824 |
| brier | 0.0711 | 0.0494 | 0.0472 | 0.0479 | 0.0407 | 0.0438 | 0.0393 |
| log loss | 0.2714 | 0.1800 | 0.1709 | 0.1663 | 0.1456 | 0.1619 | 0.1420 |
| ece | 0.0006 | 0.0280 | 0.0148 | 0.0137 | 0.0117 | 0.0181 | 0.0142 |

## Recall-constrained operating policy

| Model | Validation recall | Test recall | Test precision | Test FNR |
|---|---:|---:|---:|---:|
| Toxic-BERT | 0.8243 | 0.7338 | 0.3717 | 0.2662 |
| TF-IDF | 0.8108 | 0.7662 | 0.3130 | 0.2338 |
| Calibrated BERT | 0.8243 | 0.7338 | 0.3717 | 0.2662 |
| Two-score hybrid | 0.8108 | 0.7078 | 0.4844 | 0.2922 |
| Ratio lexical | 0.8108 | 0.8312 | 0.3307 | 0.1688 |
| Three-score hybrid | 0.8243 | 0.7143 | 0.4955 | 0.2857 |

## Paired uncertainty versus Toxic-BERT

| Metric | Difference | 95% percentile interval |
|---|---:|---:|
| f1 | 0.0985 | 0.0435 to 0.1487 |
| recall | 0.0130 | -0.0519 to 0.0714 |
| balanced_accuracy | 0.0195 | -0.0130 to 0.0479 |
| average_precision | 0.1152 | 0.0686 to 0.1629 |
| roc_auc | 0.0242 | 0.0147 to 0.0350 |
| brier | -0.0101 | -0.0142 to -0.0064 |

- Thresholds and model selection use the policy partition only. Test results are descriptive and do not establish universal safety.
- Bootstrap intervals condition on fitted models and sampled prevalence; pretraining overlap and domain generalization remain limitations.
