"""NPLM (Bengio i in., JMLR 2003) — neural n-gram: MLP nad embeddingami N OSTATNICH znaków,
BEZ atencji, stałe okno. To „most" między twardym zliczaniem n-grama a zmiennym oknem transformera.

Ten sam korpus / split (90/10) / słownik co jig GPT → perplexity BEZPOŚREDNIO porównywalne
(n-gram baseline -> NPLM -> GPT na jednym held-out).
Użycie: python src/train/nplm.py [N]    (N = okno kontekstu w znakach, domyślnie 8)
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(20260620)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8     # okno kontekstu (znaki)
DATA = "data/jigs.abc"
D_EMB, D_HID = 32, 128
BATCH, LR, ITERS, EVAL_INT, EVAL_ITERS = 64, 3e-3, 4000, 400, 200

text = open(DATA, encoding="utf-8").read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n90 = int(0.9 * len(data))
train_data, val_data = data[:n90], data[n90:]


class NPLM(nn.Module):
    """Embed N znaków -> konkatenacja -> MLP (tanh) -> rozkład nad słownikiem (Bengio 2003)."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D_EMB)
        self.fc1 = nn.Linear(N * D_EMB, D_HID)
        self.fc2 = nn.Linear(D_HID, V)

    def forward(self, x):                       # x: [B, N]
        e = self.emb(x).view(x.size(0), -1)     # [B, N*D_EMB]
        return self.fc2(torch.tanh(self.fc1(e)))


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - N - 1, (BATCH,))
    x = torch.stack([d[i:i + N] for i in ix])        # N znaków kontekstu
    y = torch.stack([d[i + N] for i in ix])          # następny znak
    return x.to(DEV), y.to(DEV)


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for sp in ("train", "val"):
        L = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(sp)
            L[k] = F.cross_entropy(model(x), y).item()
        out[sp] = L.mean().item()
    model.train()
    return out


model = NPLM().to(DEV)
nparams = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"NPLM | okno N={N} | params {nparams:,} | słownik {V} | dev {DEV}")
best = float("inf")
for it in range(ITERS + 1):
    if it % EVAL_INT == 0 or it == ITERS:
        Lv = evaluate()
        print(f"iter {it:4d} | train {Lv['train']:.3f} | val {Lv['val']:.3f} | ppl {math.exp(Lv['val']):.2f}")
        best = min(best, Lv["val"])
    if it == ITERS:
        break
    x, y = get_batch("train")
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
print(f"\nNPLM N={N}: best val ppl {math.exp(best):.2f}  ({nparams:,} param)")
