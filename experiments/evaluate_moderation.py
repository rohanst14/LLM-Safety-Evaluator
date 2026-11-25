"""Evaluate local toxicity classifier on held-out Civil Comments annotations."""
import argparse,json,hashlib
from pathlib import Path
from itertools import islice
from llm_safety_evaluator.moderation import ToxicityModel,confusion,choose_threshold
from sklearn.metrics import roc_auc_score,average_precision_score

def read_split(split,n,seed,revision):
    from datasets import load_dataset
    stream=load_dataset("google/civil_comments",split=split,streaming=True,revision=revision).shuffle(seed=seed,buffer_size=10000)
    rows=list(islice(stream,n))
    if len(rows)!=n: raise ValueError("Insufficient examples")
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--limit",type=int,default=500)
    p.add_argument("--model-revision")
    p.add_argument("--dataset-revision")
    p.add_argument("--output",type=Path,default=Path("results/moderation.json"))
    a=p.parse_args()
    if not 100<=a.limit<=10000: p.error("limit must be 100–10000 per split")
    model=ToxicityModel(revision=a.model_revision)
    def evaluate_split(split):
        rows=read_split(split,a.limit,42,a.dataset_revision)
        texts=[r["text"] for r in rows]
        labels=[int(r["toxicity"]>=.5) for r in rows]
        if len(set(labels))<2: raise ValueError("Both labels required")
        scores=model.predict(texts)
        digest=hashlib.sha256(json.dumps([(r["text"],r["toxicity"]) for r in rows],ensure_ascii=False).encode()).hexdigest()
        return labels,scores,digest
    vy,vs,vhash=evaluate_split("validation")
    ty,ts,thash=evaluate_split("test")
    threshold=choose_threshold(vy,vs)
    report={"dataset":"google/civil_comments","dataset_revision":a.dataset_revision,"model":"unitary/toxic-bert","model_revision":model.revision,"seed":42,"label_rule":"toxicity >= 0.5","validation_sha256":vhash,"test_sha256":thash,"validation_count":len(vy),"test_count":len(ty),"test_positive_count":sum(ty),"threshold":threshold,"default":confusion(ty,ts),"tuned":confusion(ty,ts,threshold),"roc_auc":float(roc_auc_score(ty,ts)),"average_precision":float(average_precision_score(ty,ts))}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
