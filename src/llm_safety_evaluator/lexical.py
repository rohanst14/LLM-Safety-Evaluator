"""Supervised feature reweighting using smoothed class-conditional document frequencies."""
import numpy as np
from sklearn.linear_model import LogisticRegression

class RatioWeightedLogistic:
    """NB-inspired weighting followed by L2 logistic regression (not an SVM)."""
    def fit(self,x,y):
        y=np.asarray(y)
        if set(y)!={0,1}:raise ValueError("Both classes required")
        binary=x.copy();binary.data=np.ones_like(binary.data)
        pos=np.asarray(binary[y==1].sum(axis=0)).ravel()
        neg=np.asarray(binary[y==0].sum(axis=0)).ravel()
        self.ratio=np.log((pos+1)/((y==1).sum()+2))-np.log((neg+1)/((y==0).sum()+2))
        self.model=LogisticRegression(C=4.,max_iter=500,random_state=42).fit(x.multiply(self.ratio).tocsr(),y)
        return self
    def predict_proba(self,x):
        return self.model.predict_proba(x.multiply(self.ratio).tocsr())
