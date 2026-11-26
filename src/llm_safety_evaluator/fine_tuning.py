"""CPU-capable end-to-end binary transformer fine-tuning."""
import torch
import torch.nn.functional as F

def binary_loss(logits, labels, mode, positive_weight=1., gamma=2.):
    if logits.shape!=labels.shape: raise ValueError("Logits and labels must align")
    if mode=="bce": return F.binary_cross_entropy_with_logits(logits,labels)
    if mode=="weighted_bce":
        if positive_weight<=0: raise ValueError("Invalid positive weight")
        return F.binary_cross_entropy_with_logits(logits,labels,pos_weight=logits.new_tensor(positive_weight))
    if mode=="focal":
        if gamma<0: raise ValueError("Invalid gamma")
        ce=F.binary_cross_entropy_with_logits(logits,labels,reduction="none")
        return ((-torch.expm1(-ce)).pow(gamma)*ce).mean()
    raise ValueError("Unknown loss")
