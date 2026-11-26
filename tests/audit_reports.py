"""Compare published numerical reports and validate measured JSON."""
import json
from pathlib import Path
from check_report import check
from check_behavior_report import check as behavior_check

def main():
    p=Path("docs/results/moderation.json");check(p)
    r=json.loads(p.read_text());report=Path("docs/experimental-report.md").read_text()
    rows=[r["majority_baseline"],*[m["tuned"] for m in r["models"].values()]]
    keys=("accuracy","balanced_accuracy","precision","recall","specificity","f1","f2","mcc","kappa","negative_predictive_value","false_positive_rate","false_negative_rate","roc_auc","average_precision","brier","log_loss","ece")
    for k in keys:
        line=next(x for x in report.splitlines() if x.startswith("| "+k.replace("_"," ")+" |"))
        cells=[v.strip() for v in line.split("|")[2:-1]]
        assert cells==["N/A" if m[k] is None else f"{m[k]:.4f}" for m in rows],k
    behavior_check("docs/results/behavior.json")
    fine=Path("docs/results/fine-tuning.json")
    if fine.exists():
        from check_fine_tuning_report import check as fine_check
        fine_check(fine)
    print(f"Verified {len(keys)*len(rows)} moderation table cells and published response endpoints.")
if __name__=="__main__":main()
