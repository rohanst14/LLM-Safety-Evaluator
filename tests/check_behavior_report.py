"""Validate aggregate response-level safety report."""
import json,sys,math
from pathlib import Path
def check(path):
    r=json.loads(Path(path).read_text())
    assert r["schema_version"]==1 and r["dataset_count"]==450
    assert r["safe_count"]==250 and r["unsafe_count"]==200
    assert r["model_revision"] and r["dataset_revision"] and len(r["dataset_sha256"])==64
    assert set(r["policies"])=={"reference","boundary_policy"}
    def verify(m,n):
        assert m["count"]==n and 0<=m["events"]<=n
        assert math.isclose(m["rate"],m["events"]/n)
        lo,hi=m["interval"];assert 0<=lo<=m["rate"]+1e-12 and m["rate"]-1e-12<=hi<=1
    for p in r["policies"].values():
        for k,n in (("unsafe_refusal_marker",200),("safe_refusal_marker",250),("empty_response",450),("token_limit_reached",450)):verify(p["xstest"][k],n)
        assert set(p["synthetic"])=={"benign_utility","grounded_abstention","confidentiality","instruction_boundary"}
        for m in p["synthetic"].values():
            verify(m["pass"],r["synthetic_records"]);verify(m["leakage"],r["synthetic_records"])
    print("Behavior report integrity checks passed")
if __name__=="__main__":check(sys.argv[1])
