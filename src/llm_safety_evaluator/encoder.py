"""Frozen MiniLM representations and supervised feature fusion."""
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

ENCODER="sentence-transformers/all-MiniLM-L6-v2"

def masked_mean(hidden,mask):
    """Mean pooling excludes padding tokens."""
    import torch
    weights=mask.unsqueeze(-1).to(hidden.dtype)
    return torch.nn.functional.normalize((hidden*weights).sum(1)/weights.sum(1).clamp(min=1),p=2,dim=1)

class FrozenEncoder:
    def __init__(self,revision):
        from transformers import AutoTokenizer,AutoModel
        self.tokenizer=AutoTokenizer.from_pretrained(ENCODER,revision=revision)
        self.model=AutoModel.from_pretrained(ENCODER,revision=revision).eval()
    def encode(self,texts,batch_size=64):
        import torch
        order=sorted(range(len(texts)),key=lambda i:len(texts[i]))
        result=np.empty((len(texts),self.model.config.hidden_size),dtype=np.float32)
        with torch.inference_mode():
            for start in range(0,len(order),batch_size):
                ix=order[start:start+batch_size]
                tokens=self.tokenizer([texts[i] for i in ix],padding=True,truncation=True,max_length=256,return_tensors="pt")
                output=self.model(**tokens).last_hidden_state
                result[ix]=masked_mean(output,tokens["attention_mask"]).cpu().numpy()
        return result

def fuse(sparse,dense):
    if sparse.shape[0]!=len(dense):raise ValueError("Unaligned lexical and contextual features")
    return hstack([sparse,csr_matrix(3.*dense)],format="csr",dtype=np.float32)

def fit_heads(x,dense,y):
    """All supervised heads are fitted on training data only."""
    linear=LogisticRegression(C=8.,max_iter=500,random_state=42).fit(fuse(x,dense),y)
    nonlinear=MLPClassifier(hidden_layer_sizes=(128,64),alpha=.001,max_iter=100,early_stopping=True,
        validation_fraction=.15,n_iter_no_change=8,random_state=42).fit(10.*dense,y)
    return linear,nonlinear
