"""Evaluate semantic toxicity classifiers against human labels."""
import math
from collections import defaultdict

def confusion(labels,scores,threshold=.5):
    if len(labels)!=len(scores) or not len(labels): raise ValueError("Aligned nonempty data required")
    if not 0<=threshold<=1: raise ValueError("Invalid threshold")
    tp=tn=fp=fn=0
    for y,s in zip(labels,scores):
        if y not in (0,1) or not math.isfinite(s) or not 0<=s<=1: raise ValueError("Invalid observation")
        pred=s>=threshold
        if y and pred: tp+=1
        elif y: fn+=1
        elif pred: fp+=1
        else: tn+=1
    def div(a,b): return a/b if b else None
    return {"count":len(labels),"tp":tp,"tn":tn,"fp":fp,"fn":fn,
            "precision":div(tp,tp+fp),"recall":div(tp,tp+fn),"f1":div(2*tp,2*tp+fp+fn),
            "false_positive_rate":div(fp,fp+tn),"false_negative_rate":div(fn,fn+tp),
            "accuracy":(tp+tn)/len(labels),"brier":sum((s-y)**2 for s,y in zip(scores,labels))/len(labels)}

def choose_threshold(labels,scores):
    candidates=[(i/100,confusion(labels,scores,i/100)) for i in range(101)]
    return max(candidates,key=lambda x:(x[1]["f1"] if x[1]["f1"] is not None else -1,x[0]))[0]

def subgroup_report(labels,scores,groups,threshold):
    if not (len(labels)==len(scores)==len(groups)): raise ValueError("Misaligned groups")
    indices=defaultdict(list)
    for i,tags in enumerate(groups):
        for tag in set(tags): indices[tag].append(i)
    return {tag:confusion([labels[i] for i in ix],[scores[i] for i in ix],threshold) for tag,ix in sorted(indices.items())}

class ToxicityModel:
    """Local inference; no hosted inference API."""
    def __init__(self,model_id="unitary/toxic-bert",revision=None,offline=False):
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=offline)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_id,revision=revision,local_files_only=offline).eval()
        labels={str(v).lower():int(k) for k,v in self.model.config.id2label.items()}
        if "toxic" not in labels and "toxicity" not in labels: raise ValueError("Missing toxicity output label")
        self.index=labels.get("toxic",labels.get("toxicity"))
        self.revision=getattr(self.model.config,"_commit_hash",None)
    def predict(self,texts,batch_size=16):
        import torch
        if batch_size<1: raise ValueError("Invalid batch size")
        scores=[None]*len(texts)
        order=sorted(range(len(texts)),key=lambda i:len(texts[i]))
        with torch.inference_mode():
            for start in range(0,len(order),batch_size):
                ix=order[start:start+batch_size]
                encoded=self.tokenizer([texts[i] for i in ix],padding=True,truncation=True,max_length=512,return_tensors="pt")
                values=self.model(**encoded).logits.sigmoid()[:,self.index].tolist()
                for i,value in zip(ix,values): scores[i]=value
        return scores
