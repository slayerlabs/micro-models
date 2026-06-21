"""E0 — self-stitch (benchmark zerowy).
Wstaw liniowy mapper g (n_embd x n_embd) na styku PO bloku 2 TEGO SAMEGO modelu,
zamroź resztę. Waliduje MECHANIZM mappera (hydraulikę), NIE tezę.
  - sanity: g = identyczność -> dokładnie baseline.
  - E0:     g losowy -> trenuj tylko g -> powinien wrócić do baseline (Δppl≈0).
Użycie: python src/e0_stitch.py [ckpt] [dane]
"""
import sys, math
sys.path.insert(0, "src")
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt import GPT

sys.stdout.reconfigure(encoding="utf-8")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "data/gpt_ckpt.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/jigs.abc"
SEAM = 2          # g przed blokiem o indeksie 2 = styk po 2 blokach
TRAIN_STEPS = 500

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
stoi, itos, cfg = ck["stoi"], ck["itos"], ck["config"]
model = GPT(cfg); model.load_state_dict(ck["model"]); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
print(f"model: {model.num_params():,} param | val loss z ckpt: {ck['val_loss']:.4f} | bloków: {cfg.n_layer} | styk po: {SEAM}")

text = open(DATA, encoding="utf-8").read()
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data)); train, val = data[:n], data[n:]
block, bs = cfg.block_size, 32

def get_batch(split):
    d = train if split == "train" else val
    ix = torch.randint(len(d) - block, (bs,))
    x = torch.stack([d[i:i+block] for i in ix])
    y = torch.stack([d[i+1:i+1+block] for i in ix])
    return x, y

g = nn.Linear(cfg.n_embd, cfg.n_embd)

def fwd(idx, targets, use_g):
    B, T = idx.shape
    pos = torch.arange(T)
    x = model.drop(model.tok_emb(idx) + model.pos_emb(pos))
    for i, blk in enumerate(model.blocks):
        if use_g and i == SEAM:
            x = g(x)
        x = blk(x)
    logits = model.head(model.ln_f(x))
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

@torch.no_grad()
def eval_loss(use_g, iters=60, seed=1234):
    torch.manual_seed(seed)               # te same batche val dla każdej oceny
    return sum(fwd(*get_batch("val"), use_g).item() for _ in range(iters)) / iters

base = eval_loss(use_g=False)
with torch.no_grad():                      # sanity: identyczność
    g.weight.copy_(torch.eye(cfg.n_embd)); g.bias.zero_()
ident = eval_loss(use_g=True)
print(f"baseline (bez g):    val {base:.4f} | ppl {math.exp(base):.4f}")
print(f"g = identyczność:    val {ident:.4f} | ppl {math.exp(ident):.4f}   (sanity: == baseline)")

torch.manual_seed(1)                        # E0: g losowy -> trening tylko g
nn.init.normal_(g.weight, std=0.02); nn.init.zeros_(g.bias)
rand0 = eval_loss(use_g=True)
opt = torch.optim.AdamW(g.parameters(), lr=1e-3)
for _ in range(TRAIN_STEPS):
    loss = fwd(*get_batch("train"), True)
    opt.zero_grad(); loss.backward(); opt.step()
trained = eval_loss(use_g=True)
print(f"g losowy (przed):    val {rand0:.4f} | ppl {math.exp(rand0):.4f}")
print(f"g po treningu ({TRAIN_STEPS}): val {trained:.4f} | ppl {math.exp(trained):.4f}")

dppl = math.exp(trained) - math.exp(base)
print(f"\nΔppl (po treningu - baseline): {dppl:+.4f}")
print("WERDYKT E0:", "PASS ✅ — mapper bezstratny, mechanizm szwu działa (benchmark zerowy)"
      if abs(dppl) < 0.05 else "FAIL ❌ — bug w mechanizmie szwu, stop")
