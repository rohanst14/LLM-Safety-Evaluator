"""Train compact BERT with three losses, independent calibration and fresh confirmation."""
import argparse,copy,hashlib,json,os,platform,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
from huggingface_hub import HfApi
from evaluate_moderation import read_split,DATASET,MODEL,MODEL_REVISION
from llm_safety_evaluator.fine_tuning import binary_loss
from llm_safety_evaluator.moderation import ToxicityModel,choose_threshold
from llm_safety_evaluator.evaluation import metrics,calibrate,calibrated_predict,policy_threshold
from llm_safety_evaluator.uncertainty import paired_bootstrap
from llm_safety_evaluator.lexical import RatioWeightedLogistic

CHECKPOINT="google/bert_uncased_L-4_H-256_A-4"
REVISION="e202cd790ed869c2af754c3ce19de62ccb3d5f26"
DATA_REVISION="f2970eb3a55777454c94069077cc8d9b5866312d"
SEED=42
MODES=("bce","weighted_bce","focal")

class Encoded(Dataset):
    def __init__(self,tokenizer,texts,labels,max_tokens):
        self.items=tokenizer(list(texts),truncation=True,max_length=max_tokens,padding=False)
        self.labels=np.asarray(labels,dtype=np.float32)
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):
        return {k:v[i] for k,v in self.items.items()},float(self.labels[i])

def make_loader(data,tokenizer,batch_size,shuffle):
    def collate(batch):
        x,y=zip(*batch)
        return tokenizer.pad(list(x),padding=True,return_tensors="pt"),torch.tensor(y,dtype=torch.float32)
    return DataLoader(data,batch_size=batch_size,shuffle=shuffle,collate_fn=collate,
        generator=torch.Generator().manual_seed(SEED),num_workers=0)

def predict(model,data,tokenizer,batch,device):
    model.eval();out=[]
    with torch.inference_mode():
        for x,_ in make_loader(data,tokenizer,batch,False):
            logits=model(**{k:v.to(device) for k,v in x.items()}).logits[:,0]
            out.extend(logits.sigmoid().cpu().tolist())
    return np.asarray(out)

def encoder_digest(model):
    # Hash a genuine pretrained attention parameter, excluding the new classification head.
    w=model.bert.encoder.layer[0].attention.self.query.weight.detach().cpu().contiguous().numpy()
    return hashlib.sha256(w.tobytes()).hexdigest()

def train_arm(mode,tokenizer,train,dev,ydev,validation,confirmation,args,device):
    torch.manual_seed(SEED);np.random.seed(SEED)
    model=AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,revision=REVISION,num_labels=1).to(device)
    if not all(p.requires_grad for p in model.parameters()):raise RuntimeError("Expected full encoder fine-tuning")
    before=encoder_digest(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01)
    weight=float((train.labels==0).sum()/(train.labels==1).sum())
    loader=make_loader(train,tokenizer,args.batch_size,True)
    steps=args.epochs*len(loader)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(.1*steps),num_training_steps=steps)
    best=None;best_ap=-1.;history=[];start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();total=0.
        for x,y in loader:
            x={k:v.to(device) for k,v in x.items()};y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(**x).logits[:,0]
            loss=binary_loss(logits,y,mode,weight)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step();total+=float(loss.detach())*len(y)
        ap=float(average_precision_score(ydev,predict(model,dev,tokenizer,args.batch_size,device)))
        history.append({"epoch":epoch+1,"training_loss":total/len(train),"development_ap":ap})
        print(mode,history[-1],flush=True)
        if ap>best_ap:
            best_ap=ap;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
    model.load_state_dict(best)
    after=encoder_digest(model)
    if before==after:raise RuntimeError("Pretrained encoder weights did not change")
    return predict(model,validation,tokenizer,args.batch_size,device),predict(model,confirmation,tokenizer,args.batch_size,device),{
        "best_epoch":best_epoch,"development_ap":best_ap,"history":history,"training_seconds":time.perf_counter()-start,
        "encoder_before_sha256":before,"encoder_after_sha256":after,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "positive_weight":weight if mode=="weighted_bce" else 1.}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fit-count",type=int,default=8000);p.add_argument("--dev-count",type=int,default=1000)
    p.add_argument("--confirmation-count",type=int,default=2000);p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=16);p.add_argument("--max-tokens",type=int,default=128)
    p.add_argument("--learning-rate",type=float,default=5e-5);p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--output",type=Path,default=Path("results/fine-tuning.json"));a=p.parse_args()
    if not 1000<=a.fit_count or not 500<=a.dev_count or a.fit_count+a.dev_count>=30000 or not 1000<=a.confirmation_count<=8000 or not 1<=a.epochs<=5 or not 1<=a.batch_size<=64 or not 32<=a.max_tokens<=512 or a.learning_rate<=0 or a.threads<1:p.error("Invalid experiment configuration")
    if a.device=="cuda" and not torch.cuda.is_available():p.error("CUDA unavailable")
    torch.set_num_threads(min(a.threads,os.cpu_count() or 1));device=torch.device(a.device)
    seen=set()
    train,ty,ti,_=read_split("train",30000,SEED,DATA_REVISION,seen)
    val,vy,vi,_=read_split("validation",2000,SEED,DATA_REVISION,seen)
    # Repeat the previously published test sampling exactly before reserving the next distinct rows.
    combined,cy,_,_=read_split("test",2000+a.confirmation_count,SEED,DATA_REVISION,seen)
    # Verify ordered prefix using the recorded sample count and content/label checksum from a separate replay.
    prefix_seen=set();_,_,ti2,_=read_split("train",30000,SEED,DATA_REVISION,prefix_seen)
    _,_,vi2,_=read_split("validation",2000,SEED,DATA_REVISION,prefix_seen)
    old,oy,old_info,_=read_split("test",2000,SEED,DATA_REVISION,prefix_seen)
    expected=("23678e5547c1d8dc35ebf5b41836dcde4badab70059e527bc6fbd3ba015b9570","75d569aeba5182b3a4352ae3233370e475b559a1e46180504b71208256912cc7","22f3701f614a36d342ce281a344c34da296328c6910f5c4c9e6b5fc3669c28c6")
    if (ti["sha256"],vi["sha256"],old_info["sha256"])!=expected or ti!=ti2 or vi!=vi2 or combined[:2000]!=old or not np.array_equal(cy[:2000],oy):
        raise RuntimeError("Dataset partitions differ from the published benchmark")
    confirm=combined[2000:];test_y=cy[2000:]
    if len(set(test_y))!=2:raise ValueError("Confirmation sample must contain both classes")
    fitdev,_=train_test_split(np.arange(len(ty)),train_size=a.fit_count+a.dev_count,random_state=SEED,stratify=ty)
    fit,dev=train_test_split(fitdev,test_size=a.dev_count,random_state=SEED,stratify=ty[fitdev])
    cal,policy=train_test_split(np.arange(len(vy)),test_size=.5,random_state=SEED,stratify=vy)
    tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT,revision=REVISION)
    training=Encoded(tokenizer,[train[i] for i in fit],ty[fit],a.max_tokens)
    development=Encoded(tokenizer,[train[i] for i in dev],ty[dev],a.max_tokens)
    validation=Encoded(tokenizer,val,vy,a.max_tokens);confirmation=Encoded(tokenizer,confirm,test_y,a.max_tokens)
    predictions={};training_info={}
    for mode in MODES:
        v,t,info=train_arm(mode,tokenizer,training,development,ty[dev],validation,confirmation,a,device)
        predictions[mode]=(v,t);training_info[mode]=info
    selected=max(MODES,key=lambda k:(training_info[k]["development_ap"],k))
    # Same supervised fit sample for lexical comparators; pretrained BERT is inference only.
    vectorizer=FeatureUnion([("word",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,dtype=np.float32)),("char",TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=3,max_features=80000,sublinear_tf=True,dtype=np.float32))])
    x=vectorizer.fit_transform([train[i] for i in fit]);xv=vectorizer.transform(val);xt=vectorizer.transform(confirm)
    for name,clf in (("lexical",LogisticRegression(C=4.,max_iter=500,random_state=SEED)),("ratio_lexical",RatioWeightedLogistic())):
        clf.fit(x,ty[fit]);predictions[name]=(clf.predict_proba(xv)[:,1],clf.predict_proba(xt)[:,1])
    reference=ToxicityModel(model_id=MODEL,revision=MODEL_REVISION)
    bv=np.asarray(reference.predict(val));bt=np.asarray(reference.predict(confirm));predictions["toxic_bert"]=(bv,bt)
    stack=calibrate(vy[cal],[bv[cal],predictions["lexical"][0][cal],predictions["ratio_lexical"][0][cal]])
    predictions["hybrid_reference"]=(calibrated_predict(stack,[bv,predictions["lexical"][0],predictions["ratio_lexical"][0]]),calibrated_predict(stack,[bt,predictions["lexical"][1],predictions["ratio_lexical"][1]]))
    models={};calibrated={}
    for name,(v,t) in predictions.items():
        calibrator=calibrate(vy[cal],[v[cal]])
        vp=calibrated_predict(calibrator,[v]);tp=calibrated_predict(calibrator,[t])
        threshold=choose_threshold(vy[policy],vp[policy]);rt=policy_threshold(vy[policy],vp[policy])
        models[name]={"uncalibrated":metrics(test_y,t),"calibrated_fixed":metrics(test_y,tp),"tuned":metrics(test_y,tp,threshold),
            "recall_policy":metrics(test_y,tp,rt),"validation_tuned":metrics(vy[policy],vp[policy],threshold),
            "validation_recall_policy":metrics(vy[policy],vp[policy],rt)}
        calibrated[name]=tp
    # Select loss only on development AP; calibrate and tune on disjoint validation subsets.
    ref=models["hybrid_reference"]["tuned"];cand=models[selected]["tuned"]
    ci=paired_bootstrap(test_y,calibrated["hybrid_reference"],ref["threshold"],calibrated[selected],cand["threshold"])
    def digest_indices(ix):return hashlib.sha256(json.dumps(ix.tolist()).encode()).hexdigest()
    def digest_confirmation():return hashlib.sha256(json.dumps(list(zip(confirm,map(int,test_y))),ensure_ascii=False).encode()).hexdigest()
    report={"schema_version":1,"checkpoint":CHECKPOINT,"checkpoint_revision":REVISION,"dataset":DATASET,"dataset_revision":DATA_REVISION,
        "seed":SEED,"device":a.device,"config":{"fit_count":len(fit),"dev_count":len(dev),"epochs":a.epochs,"batch_size":a.batch_size,"max_tokens":a.max_tokens,"learning_rate":a.learning_rate,"threads":torch.get_num_threads()},
        "splits":{"training_pool":ti,"validation":vi,"excluded_previous_test":old_info,"fit_indices_sha256":digest_indices(fit),"development_indices_sha256":digest_indices(dev),"calibration_indices_sha256":digest_indices(cal),"policy_indices_sha256":digest_indices(policy),"confirmation_sha256":digest_confirmation(),"confirmation_count":len(test_y),"confirmation_positive_count":int(test_y.sum())},
        "training":training_info,"selected_loss":selected,"selection_rule":"maximum development average precision; lexical key breaks ties","models":models,"paired_vs_hybrid_reference":ci,
        "environment":{"python":platform.python_version(),**{n:version(n) for n in ("torch","transformers","datasets","numpy","scikit-learn")}}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"selected_loss":selected,"models":{k:{z:v["tuned"][z] for z in ("f1","recall","precision","average_precision","roc_auc","ece")} for k,v in models.items()}},indent=2),flush=True)
if __name__=="__main__":main()
