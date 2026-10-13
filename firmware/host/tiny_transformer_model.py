"""Float training model co-designed for tiny_transformer_reference."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * self.weight

class Block(nn.Module):
    def __init__(self, d: int, heads: int, dff: int):
        super().__init__(); self.n1=RMSNorm(d); self.n2=RMSNorm(d); self.heads=heads
        self.qkv=nn.Linear(d,3*d,bias=False); self.out=nn.Linear(d,d,bias=False)
        self.ff1=nn.Linear(d,dff,bias=False); self.ff2=nn.Linear(dff,d,bias=False)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        b,t,d=x.shape; h=self.heads; dh=d//h; q,k,v=self.qkv(self.n1(x)).chunk(3,-1)
        q=q.view(b,t,h,dh).transpose(1,2); k=k.view(b,t,h,dh).transpose(1,2); v=v.view(b,t,h,dh).transpose(1,2)
        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0).transpose(1,2).reshape(b,t,d)
        x=x+self.out(a); return x+self.ff2(F.leaky_relu(self.ff1(self.n2(x)),negative_slope=0.25))

class TinyDecoder(nn.Module):
    def __init__(self, T:int,D:int,H:int,Dff:int,L:int,vocab:int):
        super().__init__(); self.T=T; self.token=nn.Embedding(vocab,D); self.pos=nn.Embedding(T,D)
        self.blocks=nn.ModuleList([Block(D,H,Dff) for _ in range(L)]); self.final=RMSNorm(D)
    def forward(self,tokens: torch.Tensor) -> torch.Tensor:
        pos=torch.arange(tokens.shape[1],device=tokens.device); x=self.token(tokens)+self.pos(pos)[None]
        for block in self.blocks: x=block(x)
        return F.linear(self.final(x),self.token.weight)
