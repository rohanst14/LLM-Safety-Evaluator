"""Binary moderation evaluation and validation-only calibration."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, precision_recall_curve, roc_curve)
from sklearn.linear_model import LogisticRegression
from .moderation import confusion

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p))

def calibrate(y, columns):
    """Fit a regularized logistic stack on an independent calibration partition."""
    x=np.column_stack([logit(p) for p in columns])
    return LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(x,y)

def calibrated_predict(model,columns):
    return model.predict_proba(np.column_stack([logit(p) for p in columns]))[:,1]

def metrics(y,p,t=.5):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    m=confusion(y,p,t)
    pred=(p>=t).astype(int)
    if len(np.unique(y))!=2: raise ValueError("Both labels required")
    tp,tn,fp,fn=(m[k] for k in ("tp","tn","fp","fn"))
    m.update({"specificity":tn/(tn+fp),"balanced_accuracy":.5*(tp/(tp+fn)+tn/(tn+fp)),
        "negative_predictive_value":tn/(tn+fn) if tn+fn else None,
        "mcc":float(matthews_corrcoef(y,pred)),"kappa":float(cohen_kappa_score(y,pred)),
        "roc_auc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "f2":5*tp/(5*tp+4*fn+fp),"threshold":float(t)})
    bins=np.minimum((p*15).astype(int),14)
    reliability=[]
    for i in range(15):
        ix=bins==i
        if ix.any():
            reliability.append({"bin":i,"count":int(ix.sum()),"probability":float(p[ix].mean()),"observed_rate":float(y[ix].mean())})
    m["ece"]=sum(b["count"]*abs(b["probability"]-b["observed_rate"]) for b in reliability)/len(y)
    m["reliability_bins"]=reliability
    return m

def policy_threshold(y,p,minimum_recall=.8):
    """Maximize precision among validation points meeting recall target; ties favor recall."""
    if not 0<minimum_recall<=1 or len(set(y))!=2: raise ValueError("Invalid policy labels or recall target")
    pr,rc,thresholds=precision_recall_curve(y,p)
    choices=[(float(pr[i]),float(rc[i]),float(t)) for i,t in enumerate(thresholds) if rc[i]>=minimum_recall]
    if not choices: raise ValueError("No feasible validation operating point")
    return max(choices)[2]

def ranking_curves(y,p):
    """Downsample curve points for display; scalar AUC/AP use all predictions."""
    fpr,tpr,_=roc_curve(y,p)
    precision,recall,_=precision_recall_curve(y,p)
    def sample(a,b):
        ix=np.unique(np.linspace(0,len(a)-1,min(250,len(a)),dtype=int))
        return [{"x":float(a[i]),"y":float(b[i])} for i in ix]
    return {"roc":sample(fpr,tpr),"pr":sample(recall[::-1],precision[::-1])}

def paired_correctness(y,p1,t1,p2,t2):
    y=np.asarray(y);a=(np.asarray(p1)>=t1)==y;b=(np.asarray(p2)>=t2)==y
    return {"both_correct":int((a&b).sum()),"reference_only_correct":int((a&~b).sum()),
        "candidate_only_correct":int((~a&b).sum()),"both_wrong":int((~a&~b).sum()),
        "accuracy_delta":float(b.mean()-a.mean())}
