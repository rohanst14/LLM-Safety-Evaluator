"""Check measured report structure and metric arithmetic."""
import json,sys,math
from pathlib import Path
def check(path):
    r=json.loads(Path(path).read_text())
    if r.get("schema_version") not in (2,3,4,5):
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
    if r["schema_version"]>=3:
        assert len(r["models"])==({3:6,4:9,5:13}[r["schema_version"]])
        if r["schema_version"]>=4:
            assert r["encoder"]["revision"] and r["encoder"]["frozen"] is True
        if r["schema_version"]==5:
            assert r["roberta"]["revision"] and r["roberta"]["id"]=="unitary/unbiased-toxic-roberta"
            assert {"toxic_roberta","calibrated_roberta","dual_transformer","multiview_ensemble"}<=set(r["models"])
        u=r["uncertainty"];assert u["replicates"]==1000
        assert set(u["metrics"])=={"f1","recall","balanced_accuracy","average_precision","roc_auc","brier"}
        for k,m in u["metrics"].items():
            a=r["models"][r["selected_model"]]["tuned"][k];b=r["models"]["toxic_bert"]["tuned"][k]
            assert math.isclose(m["delta"],a-b,abs_tol=1e-10)
            for key in ("reference_ci","candidate_ci","delta_ci"):
                lo,hi=m[key];assert math.isfinite(lo) and math.isfinite(hi) and lo<=hi
        for rows in r["error_slices"].values():
            assert sum(z["count"] for z in rows[:3])==n
            assert sum(z["count"] for z in rows[3:])==n
    print("Moderation report integrity checks passed")
if __name__=="__main__":check(sys.argv[1])
