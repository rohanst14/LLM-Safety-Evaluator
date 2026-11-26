"""Fit and evaluate toxicity models on disjoint official Civil Comments partitions."""
import argparse,hashlib,json,platform,os
from pathlib import Path
from importlib.metadata import version
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from huggingface_hub import HfApi
from llm_safety_evaluator.moderation import ToxicityModel,choose_threshold
from llm_safety_evaluator.lexical import RatioWeightedLogistic
from llm_safety_evaluator.encoder import FrozenEncoder,ENCODER,fuse,fit_heads
from llm_safety_evaluator.uncertainty import paired_bootstrap,strata
from llm_safety_evaluator.evaluation import (metrics,calibrate,calibrated_predict,
    policy_threshold,ranking_curves,paired_correctness)

DATASET="google/civil_comments"
MODEL="unitary/toxic-bert"
MODEL_REVISION="4d6c22e74ba2fdd26bc4f7238f50766b045a0d94"

def read_split(split,n,seed,revision,seen):
    from datasets import load_dataset
    stream=load_dataset(DATASET,split=split,streaming=True,revision=revision).shuffle(seed=seed,buffer_size=10000)
    rows=[];duplicates=0
    for row in stream:
        key=hashlib.sha256(" ".join(row["text"].casefold().split()).encode()).digest()
        if key in seen:
            duplicates+=1;continue
        seen.add(key);rows.append(row)
        if len(rows)==n: break
    if len(rows)!=n: raise ValueError("Insufficient distinct examples")
    y=np.array([int(r["toxicity"]>=.5) for r in rows])
    if len(set(y))!=2: raise ValueError("Both labels required")
    digest=hashlib.sha256(json.dumps([(r["text"],r["toxicity"]) for r in rows],ensure_ascii=False).encode()).hexdigest()
    return [r["text"] for r in rows],y,{"count":n,"positive_count":int(y.sum()),"sha256":digest,"duplicates_skipped":duplicates},np.array([r["toxicity"] for r in rows])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--limit",type=int,default=2000)
    p.add_argument("--train-limit",type=int,default=30000)
    p.add_argument("--dataset-revision")
    p.add_argument("--model-revision",default=MODEL_REVISION)
    p.add_argument("--encoder-revision")
    p.add_argument("--output",type=Path,default=Path("results/moderation.json"))
    a=p.parse_args()
    if not 500<=a.limit<=10000 or not 1000<=a.train_limit<=100000: p.error("Invalid split size")
    revision=HfApi().dataset_info(DATASET,revision=a.dataset_revision).sha
    seen=set()
    train,train_y,train_info,_=read_split("train",a.train_limit,42,revision,seen)
    val,vy,val_info,_=read_split("validation",a.limit,42,revision,seen)
    test,ty,test_info,test_annotation=read_split("test",a.limit,42,revision,seen)
    cal,policy=train_test_split(np.arange(len(vy)),test_size=.5,random_state=42,stratify=vy)
    # All vocabulary and supervised lexical weights are fitted on official training only.
    vectorizer=FeatureUnion([("word",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,dtype=np.float32)),
        ("character",TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=3,max_features=80000,sublinear_tf=True,dtype=np.float32))])
    x=vectorizer.fit_transform(train)
    lexical=LogisticRegression(C=4.,max_iter=500,random_state=42).fit(x,train_y)
    xv=vectorizer.transform(val);xt=vectorizer.transform(test)
    lv=lexical.predict_proba(xv)[:,1];lt=lexical.predict_proba(xt)[:,1]
    ratio=RatioWeightedLogistic().fit(x,train_y)
    rv=ratio.predict_proba(xv)[:,1];rt=ratio.predict_proba(xt)[:,1]
    import torch
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    encoder_revision=HfApi().model_info(ENCODER,revision=a.encoder_revision).sha
    print("Encoding training, validation and test text with frozen MiniLM",flush=True)
    encoder=FrozenEncoder(encoder_revision)
    et=encoder.encode(train);ev=encoder.encode(val);es=encoder.encode(test)
    linear,nonlinear=fit_heads(x,et,train_y)
    fv=linear.predict_proba(fuse(xv,ev))[:,1];ft=linear.predict_proba(fuse(xt,es))[:,1]
    nv=nonlinear.predict_proba(10.*ev)[:,1];nt=nonlinear.predict_proba(10.*es)[:,1]
    del encoder,et,ev,es
    print("Running Toxic-BERT reference inference",flush=True)
    bert=ToxicityModel(model_id=MODEL,revision=a.model_revision)
    bv=np.asarray(bert.predict(val));bt=np.asarray(bert.predict(test))
    cb=calibrate(vy[cal],[bv[cal]])
    stack=calibrate(vy[cal],[bv[cal],lv[cal]])
    stack_all=calibrate(vy[cal],[bv[cal],lv[cal],rv[cal]])
    stack_context=calibrate(vy[cal],[bv[cal],lv[cal],rv[cal],fv[cal],nv[cal]])
    candidates={
        "toxic_bert":(bv,bt),
        "lexical":(lv,lt),
        "calibrated_bert":(calibrated_predict(cb,[bv]),calibrated_predict(cb,[bt])),
        "hybrid":(calibrated_predict(stack,[bv,lv]),calibrated_predict(stack,[bt,lt])),
        "ratio_lexical":(rv,rt),
        "hybrid_all":(calibrated_predict(stack_all,[bv,lv,rv]),calibrated_predict(stack_all,[bt,lt,rt])),
        "contextual_mlp":(nv,nt),"feature_fusion":(fv,ft),
        "contextual_ensemble":(calibrated_predict(stack_context,[bv,lv,rv,fv,nv]),calibrated_predict(stack_context,[bt,lt,rt,ft,nt]))}
    prevalence=float(train_y.mean())
    models={}
    for key,(v,t) in candidates.items():
        threshold=choose_threshold(vy[policy],v[policy])
        recall_threshold=policy_threshold(vy[policy],v[policy])
        models[key]={"default":metrics(ty,t),"tuned":metrics(ty,t,threshold),
            "recall_policy":metrics(ty,t,recall_threshold),
            "validation_f1":metrics(vy[policy],v[policy],threshold),
            "validation_recall_policy":metrics(vy[policy],v[policy],recall_threshold),
            "curves":ranking_curves(ty,t)}
    selected=max(models,key=lambda k:(models[k]["validation_f1"]["f1"] or 0,k))
    ref=models["toxic_bert"]["tuned"];cand=models[selected]["tuned"]
    paired=paired_correctness(ty,bt,ref["threshold"],candidates[selected][1],cand["threshold"])
    intervals=paired_bootstrap(ty,bt,ref["threshold"],candidates[selected][1],cand["threshold"])
    slices={k:strata(ty,t,models[k]["tuned"]["threshold"],[len(s.split()) for s in test],test_annotation) for k,(_,t) in candidates.items()}
    report={"schema_version":4,"encoder":{"id":ENCODER,"revision":encoder_revision,"max_tokens":256,"pooling":"attention-masked mean; L2 normalized","frozen":True},"uncertainty":intervals,"error_slices":slices,"dataset":DATASET,"dataset_revision":revision,"model":MODEL,"model_revision":bert.revision or a.model_revision,
        "seed":42,"label_rule":"toxicity >= 0.5","sampling":"bounded-buffer shuffle (10000), normalized-text duplicate removal in train/validation/test order",
        "splits":{"train":train_info,"validation":val_info,"test":test_info,"calibration":{"count":len(cal),"positive_count":int(vy[cal].sum())},"policy":{"count":len(policy),"positive_count":int(vy[policy].sum())}},
        "test_count":len(ty),"test_positive_count":int(ty.sum()),"validation_count":len(vy),
        "majority_baseline":metrics(ty,np.full(len(ty),prevalence)),
        "models":models,"selected_model":selected,"selection_rule":"maximum policy-partition F1; alphabetical descending key breaks ties",
        "paired_vs_tuned_bert":paired,"minimum_validation_recall":.8,
        "environment":{"python":platform.python_version(),**{n:version(n) for n in ("numpy","scikit-learn","datasets","transformers","torch")}}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"selected_model":selected,"test_count":len(ty),"test_positive_count":int(ty.sum()),"models":{k:{m:v["tuned"][m] for m in ("accuracy","f1","recall","precision","roc_auc","average_precision","ece")} for k,v in models.items()}},indent=2))
if __name__=="__main__": main()
