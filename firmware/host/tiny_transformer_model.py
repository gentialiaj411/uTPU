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
        # The deployed score tensor is signed Q4.4.  Training the same bounded
        # score range prevents a float-only attention tail from being clipped
        # before the frozen integer softmax.
        scores=(q @ k.transpose(-2,-1))/4.0
        scores=scores.clamp(-8.0,127.0/16.0).masked_fill(~mask,float('-inf'))
        a=(torch.softmax(scores,dim=-1) @ v).transpose(1,2).reshape(b,t,d)
        x=x+self.out(a); return x+self.ff2(F.leaky_relu(self.ff1(self.n2(x)),negative_slope=0.25))

class TinyDecoder(nn.Module):
    def __init__(self, T:int,D:int,H:int,Dff:int,L:int,vocab:int):
        super().__init__(); self.T=T; self.token=nn.Embedding(vocab,D); self.pos=nn.Embedding(T,D)
        self.blocks=nn.ModuleList([Block(D,H,Dff) for _ in range(L)]); self.final=RMSNorm(D)
    def forward(self,tokens: torch.Tensor) -> torch.Tensor:
        pos=torch.arange(tokens.shape[1],device=tokens.device); x=self.token(tokens)+self.pos(pos)[None]
        for block in self.blocks: x=block(x)
        return F.linear(self.final(x),self.token.weight)

    @torch.no_grad()
    def forward_with_trace(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Float counterpart to the independent-reference stage trace."""
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        trace: dict[str, torch.Tensor] = {}
        x = self.token(tokens) + self.pos(pos)[None]; trace['pos_add'] = x
        for block in self.blocks:
            n1 = block.n1(x); trace['rmsnorm1'] = n1
            q, k, v = block.qkv(n1).chunk(3, -1); dh = x.shape[-1] // block.heads
            q = q.view(b,t,block.heads,dh).transpose(1,2); k = k.view(b,t,block.heads,dh).transpose(1,2); v = v.view(b,t,block.heads,dh).transpose(1,2)
            trace['q'], trace['k'], trace['v'] = q, k, v
            scores = ((q @ k.transpose(-2,-1)) / (dh ** 0.5)).clamp(-8.0, 127.0 / 16.0)
            scores = scores.masked_fill(~torch.ones(t,t,device=tokens.device,dtype=torch.bool).tril(), float('-inf')); trace['scores'] = scores
            probs = torch.softmax(scores, dim=-1); trace['softmax_probs'] = probs
            av = (probs @ v).transpose(1,2).reshape(b,t,x.shape[-1]); trace['av'] = av
            out = block.out(av); trace['out_proj'] = out; x = x + out; trace['residual1'] = x
            n2 = block.n2(x); trace['rmsnorm2'] = n2
            ffn1 = block.ff1(n2); trace['ffn1'] = ffn1
            leaky = F.leaky_relu(ffn1, negative_slope=0.25); trace['leaky_relu'] = leaky
            ffn2 = block.ff2(leaky); trace['ffn2'] = ffn2; x = x + ffn2; trace['residual2'] = x
        final = self.final(x); trace['final_norm'] = final
        logits = F.linear(final,self.token.weight); trace['logits'] = logits
        return logits, trace
