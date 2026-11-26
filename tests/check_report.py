"""Check measured report structure and metric arithmetic."""
import json,sys,math
from pathlib import Path
def check(path):
    r=json.loads(Path(path).read_text())
    if r.get("schema_version")!=2:
        assert r["default"]["count"]==r["tuned"]["count"]==r["test_count"]
        return
    assert r["dataset_revision"] and r["model_revision"]
    n=r["test_count"];pos=r["test_positive_count"]
    assert 0<pos<n
    assert r["splits"]["calibration"]["count"]+r["splits"]["policy"]["count"]==r["validation_count"]
    def verify(m,total,positive):
        tp,tn,fp,fn=(m[k] for k in ("tp","tn","fp","fn"))
        assert tp+tn+fp+fn==total and tp+fn==positive
        assert math.isclose(m["accuracy"],(tp+tn)/total,abs_tol=1e-10)
        assert math.isclose(m["recall"],tp/positive,abs_tol=1e-10)
        assert math.isclose(m["f1"],2*tp/(2*tp+fp+fn),abs_tol=1e-10)
        assert math.isclose(m["balanced_accuracy"],(tp/positive+tn/(total-positive))/2,abs_tol=1e-10)
        assert sum(b["count"] for b in m["reliability_bins"])==total
        assert math.isclose(m["ece"],sum(b["count"]*abs(b["probability"]-b["observed_rate"]) for b in m["reliability_bins"])/total,abs_tol=1e-10)
    verify(r["majority_baseline"],n,pos)
    for m in r["models"].values():
        for k in ("default","tuned","recall_policy"): verify(m[k],n,pos)
        for k in ("validation_f1","validation_recall_policy"): verify(m[k],r["splits"]["policy"]["count"],r["splits"]["policy"]["positive_count"])
        assert m["validation_recall_policy"]["recall"]>=r["minimum_validation_recall"]-1e-10
        assert m["tuned"]["roc_auc"]==m["default"]["roc_auc"]
    assert r["selected_model"]==max(r["models"],key=lambda k:(r["models"][k]["validation_f1"]["f1"] or 0,k))
    print("Moderation report integrity checks passed")
if __name__=="__main__":check(sys.argv[1])
