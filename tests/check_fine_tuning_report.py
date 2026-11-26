"""Validate fine-tuning report arithmetic and evidence of encoder updates."""
import json,sys,math
from pathlib import Path
def check(path):
    r=json.loads(Path(path).read_text());assert r["schema_version"]==1
    arms={"bce","weighted_bce","focal"};assert set(r["training"])==arms
    assert set(r["models"])==arms|{"lexical","ratio_lexical","toxic_bert","hybrid_reference"}
    for k,a in r["training"].items():
        assert a["encoder_before_sha256"]!=a["encoder_after_sha256"] and a["trainable_parameters"]>1_000_000
        assert len(a["history"])==r["config"]["epochs"]
        assert a["development_ap"]==max(h["development_ap"] for h in a["history"])
    assert r["selected_loss"]==max(arms,key=lambda k:(r["training"][k]["development_ap"],k))
    n=r["splits"]["confirmation_count"];pos=r["splits"]["confirmation_positive_count"];assert 0<pos<n
    for m in r["models"].values():
        for key in ("uncalibrated","calibrated_fixed","tuned","recall_policy"):
            v=m[key];tp,tn,fp,fn=[v[k] for k in ("tp","tn","fp","fn")]
            assert tp+tn+fp+fn==n and tp+fn==pos
            assert math.isclose(v["f1"],2*tp/(2*tp+fp+fn),abs_tol=1e-10)
            assert math.isclose(v["recall"],tp/pos,abs_tol=1e-10)
        assert m["validation_recall_policy"]["recall"]>=.8-1e-10
    for k,v in r["paired_vs_hybrid_reference"]["metrics"].items():
        assert math.isclose(v["delta"],r["models"][r["selected_loss"]]["tuned"][k]-r["models"]["hybrid_reference"]["tuned"][k],abs_tol=1e-10)
    print("Fine-tuning report verified")
if __name__=="__main__":check(sys.argv[1])
