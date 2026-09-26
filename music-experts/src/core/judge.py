# Autor: Adam Skrodzki
"""Judge — klasyfikator melodii ABC na trunku GPT (verbatim reuse core/gpt.py).
Trunk = cały GPT (embedding -> bloki -> ln_f); nowa głowa = pooling po pozycjach
(mean albo attention-pool, pool="mean"|"attn") + liniowe głowy klasyfikacyjne
(meter/mode/type).
Uwaga przyczynowa: padding na KOŃCU sekwencji — tokeny prawdziwe nigdy go nie widzą,
maska potrzebna tylko do poprawnego mean-poolingu.
"""
import torch
import torch.nn as nn
from core.gpt import GPT, GPTConfig

HEADS = ("meter", "mode", "type")


class JudgeGPT(nn.Module):
    def __init__(self, cfg: GPTConfig, n_classes: dict, pool: str = "mean"):
        super().__init__()
        self.gpt = GPT(cfg)                       # trunk verbatim (LM-head nieużywany)
        self.pool = pool
        # ModuleList zamiast ModuleDict: klucz "type" koliduje z nn.Module.type()
        self.heads = nn.ModuleList([nn.Linear(cfg.n_embd, n_classes[h]) for h in HEADS])
        if pool == "attn":
            # uwaga pooling: wagi uczące się per głowa; przy zerowych wagach = mean-pool
            self.score = nn.Linear(cfg.n_embd, len(HEADS))

    def trunk(self, idx):
        g = self.gpt
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = g.drop(g.tok_emb(idx) + g.pos_emb(pos))
        for blk in g.blocks:
            x = blk(x)
        return g.ln_f(x)                          # (B, T, C)

    def forward(self, idx, mask):
        """idx: (B,T) | mask: (B,T) — 1 = prawdziwy znak, 0 = padding (tylko na końcu).
        Zwraca {head: logity (B, n_classes)}."""
        h = self.trunk(idx)
        if self.pool == "attn":
            s = self.score(h).masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))  # (B,T,H)
            w = torch.softmax(s, dim=1)                                            # wagi po T
            pooled = torch.einsum("bth,btc->bhc", w, h)                            # (B,H,C)
            return {head: self.heads[k](pooled[:, k]) for k, head in enumerate(HEADS)}
        m = mask.unsqueeze(-1).to(h.dtype)        # (B,T,1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)   # mean-pool po prawdziwych znakach
        return {head: self.heads[k](pooled) for k, head in enumerate(HEADS)}
