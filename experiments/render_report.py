"""Generate a human-readable report from measured benchmark JSON, never estimated scores."""
import argparse,json
from pathlib import Path
NAMES={"toxic_bert":"Toxic-BERT","lexical":"TF-IDF","calibrated_bert":"Calibrated BERT","hybrid":"Two-score hybrid","ratio_lexical":"Ratio lexical","hybrid_all":"Three-score hybrid","contextual_mlp":"Contextual MLP","feature_fusion":"Feature fusion","contextual_ensemble":"Contextual ensemble","toxic_roberta":"Toxicity RoBERTa","calibrated_roberta":"Calibrated RoBERTa","dual_transformer":"Dual-transformer stack","multiview_ensemble":"Multiview ensemble"}
KEYS=("accuracy","balanced_accuracy","precision","recall","specificity","f1","f2","mcc","kappa","negative_predictive_value","false_positive_rate","false_negative_rate","roc_auc","average_precision","brier","log_loss","ece")
def fmt(v):return "N/A" if v is None else f"{v:.4f}"
def main():
    p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    r=json.loads(a.input.read_text())
    models={"Majority":r["majority_baseline"],**{NAMES[k]:m["tuned"] for k,m in r["models"].items()}}
    lines=["# Measured moderation benchmark","","- Dataset revision: `"+r["dataset_revision"]+"`","- Toxic-BERT revision: `"+r["model_revision"]+"`","- Validation-selected model: "+NAMES[r["selected_model"]],f'- Held-out comments: {r["test_count"]}; toxic comments: {r["test_positive_count"]}.',"- [Experimental protocol](research-protocol.md) · [Raw measured JSON](results/moderation.json)","","## Held-out comparison","","| Metric | "+" | ".join(models)+" |","|---|"+"---:|"*len(models)]
    if r.get("roberta"):
        lines[4:4]=["- RoBERTa revision: `"+r["roberta"]["revision"]+"`","- RoBERTa pretraining includes the related Civil Comments/Jigsaw task; exact overlap is unknown."]
    for k in KEYS:lines.append("| "+k.replace("_"," ")+" | "+" | ".join(fmt(m[k]) for m in models.values())+" |")
    lines+=["","## Recall-constrained operating policy","","| Model | Validation recall | Test recall | Test precision | Test FNR |","|---|---:|---:|---:|---:|"]
    for k,m in r["models"].items():lines.append("| "+NAMES[k]+" | "+" | ".join(fmt(x) for x in (m["validation_recall_policy"]["recall"],m["recall_policy"]["recall"],m["recall_policy"]["precision"],m["recall_policy"]["false_negative_rate"]))+" |")
    lines+=["","## Paired uncertainty versus Toxic-BERT","","| Metric | Difference | 95% percentile interval |","|---|---:|---:|"]
    for k,m in r["uncertainty"]["metrics"].items():lines.append("| "+k+" | "+fmt(m["delta"])+" | "+ " to ".join(map(fmt,m["delta_ci"]))+" |")
    lines+=["","- Thresholds and model selection use the policy partition only. Test results are descriptive and do not establish universal safety.","- Bootstrap intervals condition on fitted models and sampled prevalence; pretraining overlap and domain generalization remain limitations."]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__":main()
