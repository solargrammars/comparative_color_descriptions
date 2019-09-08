#import ipdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CompModel(nn.Module):
    def __init__(self):
        super(CompModel, self).__init__()
        
        self.l1 = nn.Linear(603, 30)
        self.l2 = nn.Linear(33,3)

    def forward(self, w, ref):     
             
        w_ref   = torch.cat((w, ref), 1)
        out     = F.relu(self.l1(w_ref))
        out_ref = torch.cat((out, ref), 1)
        out     = self.l2(out_ref)
        return out
    
class CompLoss(nn.Module):
    def __init__(self):
        super(CompLoss, self).__init__()

    def forward(self, wg, ref, tgt):
        
        tr          = tgt - ref
        output_loss = 1 -  F.cosine_similarity(wg, tr)
        
        pred      = ref + wg
        pred_loss = F.pairwise_distance(pred, tgt)
        
        return output_loss + pred_loss
      