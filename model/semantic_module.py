import torch
from torch import nn
import math
import torch.nn.functional as F

class Semantic(nn.Module):
    """
    Semantic Module, referred to AGSDD (Wang, C et al., 2025, https://doi.org/10.1007/978-3-032-06066-2_21.)
    """
    def __init__(self, c_s:int):
        """
        c_s: input hidden state channel dimension
        """
        super(Semantic, self).__init__()
        self.c_s = c_s
        self.kv_embedding = nn.Embedding(20, c_s)
        self.linear_q=nn.Linear(c_s, c_s)
        self.linear_k=nn.Linear(c_s, c_s)
        self.linear_v = nn.Linear(c_s, c_s)
        self.gate=nn.Linear(2*c_s, c_s)
        self.linear_out = nn.Sequential(nn.Linear(2*c_s, c_s), nn.LeakyReLU(), nn.Linear(c_s, c_s))

        nn.init.zeros_(self.linear_out[-1].weight)
        nn.init.zeros_(self.linear_out[-1].bias)

    def forward(self,s:torch.Tensor):
        """
        s: input hidden state, [*, N ,c_s]
        """
        q=self.linear_q(s) # [*, N, c_s]
        k=self.kv_embedding.weight # [20, c_s]
        k=self.linear_k(k) # [20, c_s]
        v=self.kv_embedding.weight # [20, c_s]
        v=self.linear_v(v) # [20, c_s]

        # =====compute attention score=====
        k=k.view((1,)*len(q.shape[:-2])+tuple(k.shape[-2:])) # [*, 20, c_s]
        att_logits = torch.matmul(q,torch.transpose(k,-1,-2))/math.sqrt(self.c_s) # [*, N, 20]
        att_probs = F.softmax(att_logits,dim=-1) # [*, N, 20]

        # =====compute outputs=====
        v = v.view((1,) * len(q.shape[:-2]) + tuple(v.shape[-2:])) # [*, 20, c_s]
        h = torch.matmul(att_probs,v) # [*, N, c_s]
        h_comb = torch.cat((h,s),dim=-1)  # [*, N, 2*c_s]
        gate = torch.sigmoid(self.gate(h_comb)) # [*, N, c_s]
        delta=self.linear_out(h_comb) # [*, N, c_s]

        s_out = s + gate * delta

        return s_out, att_logits







