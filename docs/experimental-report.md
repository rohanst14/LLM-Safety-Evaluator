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
| accuracy | 0.9230 | 0.9220 | 0.9165 | 0.9275 | 0.9140 | 0.9320 | 0.9470 | 0.9080 | 0.9200 | 0.9460 | 0.9440 | 0.9430 | 0.9450 | 0.9530 |
| balanced accuracy | 0.5000 | 0.7703 | 0.7941 | 0.7732 | 0.8284 | 0.8054 | 0.7898 | 0.6794 | 0.7781 | 0.7892 | 0.8804 | 0.8828 | 0.8601 | 0.7752 |
| precision | N/A | 0.4946 | 0.4695 | 0.5260 | 0.4628 | 0.5489 | 0.6739 | 0.4038 | 0.4845 | 0.6643 | 0.6019 | 0.5952 | 0.6158 | 0.7632 |
| recall | 0.0000 | 0.5909 | 0.6494 | 0.5909 | 0.7273 | 0.6558 | 0.6039 | 0.4091 | 0.6104 | 0.6039 | 0.8052 | 0.8117 | 0.7597 | 0.5649 |
| specificity | 1.0000 | 0.9496 | 0.9388 | 0.9556 | 0.9296 | 0.9550 | 0.9756 | 0.9496 | 0.9458 | 0.9745 | 0.9556 | 0.9540 | 0.9605 | 0.9854 |
| f1 | 0.0000 | 0.5385 | 0.5450 | 0.5566 | 0.5657 | 0.5976 | 0.6370 | 0.4065 | 0.5402 | 0.6327 | 0.6889 | 0.6868 | 0.6802 | 0.6493 |
| f2 | 0.0000 | 0.5687 | 0.6031 | 0.5767 | 0.6527 | 0.6312 | 0.6167 | 0.4080 | 0.5802 | 0.6151 | 0.7543 | 0.7567 | 0.7258 | 0.5959 |
| mcc | 0.0000 | 0.4986 | 0.5083 | 0.5183 | 0.5369 | 0.5635 | 0.6096 | 0.3566 | 0.5010 | 0.6044 | 0.6672 | 0.6658 | 0.6548 | 0.6328 |
| kappa | 0.0000 | 0.4962 | 0.5003 | 0.5172 | 0.5205 | 0.5608 | 0.6085 | 0.3566 | 0.4971 | 0.6036 | 0.6588 | 0.6563 | 0.6505 | 0.6247 |
| negative predictive value | 0.9230 | 0.9653 | 0.9698 | 0.9655 | 0.9761 | 0.9708 | 0.9672 | 0.9507 | 0.9668 | 0.9672 | 0.9833 | 0.9838 | 0.9796 | 0.9645 |
| false positive rate | 0.0000 | 0.0504 | 0.0612 | 0.0444 | 0.0704 | 0.0450 | 0.0244 | 0.0504 | 0.0542 | 0.0255 | 0.0444 | 0.0460 | 0.0395 | 0.0146 |
| false negative rate | 1.0000 | 0.4091 | 0.3506 | 0.4091 | 0.2727 | 0.3442 | 0.3961 | 0.5909 | 0.3896 | 0.3961 | 0.1948 | 0.1883 | 0.2403 | 0.4351 |
| roc auc | 0.5000 | 0.9171 | 0.8974 | 0.9171 | 0.9349 | 0.9191 | 0.9413 | 0.8401 | 0.9030 | 0.9420 | 0.9670 | 0.9670 | 0.9673 | 0.9680 |
| average precision | 0.0770 | 0.5672 | 0.5931 | 0.5672 | 0.6643 | 0.6349 | 0.6824 | 0.4324 | 0.5913 | 0.6862 | 0.7517 | 0.7517 | 0.7563 | 0.7635 |
| brier | 0.0711 | 0.0494 | 0.0472 | 0.0479 | 0.0407 | 0.0438 | 0.0393 | 0.0562 | 0.0474 | 0.0392 | 0.0412 | 0.0349 | 0.0346 | 0.0344 |
| log loss | 0.2714 | 0.1800 | 0.1709 | 0.1663 | 0.1456 | 0.1619 | 0.1420 | 0.2113 | 0.1756 | 0.1413 | 0.1350 | 0.1154 | 0.1147 | 0.1136 |
| ece | 0.0006 | 0.0280 | 0.0148 | 0.0137 | 0.0117 | 0.0181 | 0.0142 | 0.0250 | 0.0234 | 0.0131 | 0.0331 | 0.0166 | 0.0176 | 0.0169 |

## Recall-constrained operating policy

| Model | Validation recall | Test recall | Test precision | Test FNR |
|---|---:|---:|---:|---:|
| Toxic-BERT | 0.8243 | 0.7338 | 0.3717 | 0.2662 |
| TF-IDF | 0.8108 | 0.7662 | 0.3130 | 0.2338 |
| Calibrated BERT | 0.8243 | 0.7338 | 0.3717 | 0.2662 |
| Two-score hybrid | 0.8108 | 0.7078 | 0.4844 | 0.2922 |
| Ratio lexical | 0.8108 | 0.8312 | 0.3307 | 0.1688 |
| Three-score hybrid | 0.8243 | 0.7143 | 0.4955 | 0.2857 |
| Contextual MLP | 0.8378 | 0.7468 | 0.2406 | 0.2532 |
| Feature fusion | 0.8108 | 0.7922 | 0.3144 | 0.2078 |
| Contextual ensemble | 0.8108 | 0.7143 | 0.5116 | 0.2857 |
| Toxicity RoBERTa | 0.8243 | 0.7987 | 0.6029 | 0.2013 |
| Calibrated RoBERTa | 0.8243 | 0.7987 | 0.6029 | 0.2013 |
| Dual-transformer stack | 0.8108 | 0.7727 | 0.6134 | 0.2273 |
| Multiview ensemble | 0.8108 | 0.7662 | 0.6082 | 0.2338 |

## Paired uncertainty versus Toxic-BERT

| Metric | Difference | 95% percentile interval |
|---|---:|---:|
| f1 | 0.1108 | 0.0526 to 0.1738 |
| recall | -0.0260 | -0.0909 to 0.0390 |
| balanced_accuracy | 0.0049 | -0.0284 to 0.0395 |
| average_precision | 0.1963 | 0.1405 to 0.2626 |
| roc_auc | 0.0508 | 0.0351 to 0.0699 |
| brier | -0.0150 | -0.0199 to -0.0106 |

- Thresholds and model selection use the policy partition only. Test results are descriptive and do not establish universal safety.
- Bootstrap intervals condition on fitted models and sampled prevalence; pretraining overlap and domain generalization remain limitations.
