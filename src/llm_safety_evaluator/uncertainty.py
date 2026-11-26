"""Paired stratified bootstrap and threshold risk diagnostics."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

KEYS=("f1","recall","balanced_accuracy","average_precision","roc_auc","brier")

def statistics(y,p,t):
    pred=p>=t
    pos=y==1;neg=~pos
    tp=int((pred&pos).sum());fp=int((pred&neg).sum())
    fn=int((~pred&pos).sum());tn=int((~pred&neg).sum())
    return np.array([2*tp/(2*tp+fp+fn),tp/(tp+fn),.5*(tp/(tp+fn)+tn/(tn+fp)),
        average_precision_score(y,p),roc_auc_score(y,p),np.mean((p-y)**2)])

def paired_bootstrap(y,reference,ref_threshold,candidate,candidate_threshold,repeats=1000,seed=1729):
    """Resample identical class-stratified indices for both models; thresholds remain fixed."""
    y=np.asarray(y);reference=np.asarray(reference);candidate=np.asarray(candidate)
    if y.ndim!=1 or y.shape!=reference.shape or y.shape!=candidate.shape or set(y)!={0,1} or repeats<100:
        raise ValueError("Invalid paired binary observations or replicate count")
    rng=np.random.default_rng(seed);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    ref=[];cand=[]
    for _ in range(repeats):
        ix=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
        ref.append(statistics(y[ix],reference[ix],ref_threshold))
        cand.append(statistics(y[ix],candidate[ix],candidate_threshold))
    ref=np.asarray(ref);cand=np.asarray(cand)
    point=statistics(y,candidate,candidate_threshold)-statistics(y,reference,ref_threshold)
    def interval(x,i):return [float(v) for v in np.quantile(x[:,i],[.025,.975])]
    return {"replicates":repeats,"seed":seed,"metrics":{k:{"reference_ci":interval(ref,i),"candidate_ci":interval(cand,i),
        "delta":float(point[i]),"delta_ci":interval(cand-ref,i)} for i,k in enumerate(KEYS)}}

def strata(y,p,t,text_lengths,annotation_scores):
    """Report descriptive error slices, not demographic fairness estimates."""
    y=np.asarray(y);p=np.asarray(p);length=np.asarray(text_lengths);a=np.asarray(annotation_scores)
    groups={"short (<=20 words)":length<=20,"medium (21-80 words)":(length>20)&(length<=80),
        "long (>80 words)":length>80,"near label boundary (0.3-0.7)":(a>=.3)&(a<=.7),
        "outside label boundary":(a<.3)|(a>.7)}
    result=[]
    for name,ix in groups.items():
        n=int(ix.sum());pos=int(y[ix].sum());neg=n-pos
        pred=p[ix]>=t;tp=int((pred&(y[ix]==1)).sum());fp=int((pred&(y[ix]==0)).sum())
        result.append({"slice":name,"count":n,"positive":pos,"negative":neg,
            "recall":tp/pos if pos>=20 else None,"false_positive_rate":fp/neg if neg>=20 else None})
    return result
