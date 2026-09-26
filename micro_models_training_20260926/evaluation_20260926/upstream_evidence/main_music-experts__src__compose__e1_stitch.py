"""E1 — stitch DWÓCH różnych ekspertów (wspólny słownik): front walca × g × back reela.
Trenuje mapper g (reszta zamrożona) na połączonym korpusie walc+reel; porównuje perplexity:
walc-alone / reel-alone / ensemble / stitch — na wspólnym held-out. Generuje próbkę stitchu.
Pytanie (KMS1): czy stitch reprezentacji bije ensemble (spójne złączenie), czy mush.
"""
import sys, math, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.nn import functional as F
from core.gpt import GPT
from core.abc_to_midi import to_midi

sys.stdout.reconfigure(encoding="utf-8")
A_CKPT, B_CKPT = "data/models/waltz_ckpt.pt", "data/models/reel_sv_ckpt.pt"
SEAM, TRAIN_STEPS = 2, 800

def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = GPT(ck["config"]); m.load_state_dict(ck["model"]); m.eval()
    for q in m.parameters(): q.requires_grad_(False)
    return m, ck

A, ckA = load(A_CKPT); B, ckB = load(B_CKPT)
assert ckA["stoi"] == ckB["stoi"], "różny słownik — najpierw wspólny!"
stoi, itos = ckA["stoi"], ckA["itos"]
block, n_embd = ckA["config"].block_size, ckA["config"].n_embd
print(f"front=walc, back=reel | wspólny słownik {len(stoi)} | styk po bloku {SEAM}")

wtext = open("data/corpus/waltz.abc", encoding="utf-8").read()
rtext = open("data/corpus/reel.abc", encoding="utf-8").read()
L = min(len(wtext), len(rtext))                       # ZBALANSUJ: równo znaków z obu
wd = torch.tensor([stoi[c] for c in wtext[:L] if c in stoi], dtype=torch.long)
rd = torch.tensor([stoi[c] for c in rtext[:L] if c in stoi], dtype=torch.long)
nw, nr = int(0.9 * len(wd)), int(0.9 * len(rd))
train = torch.cat([wd[:nw], rd[:nr]])
val = torch.cat([wd[nw:], rd[nr:]])                   # val = 50% walc + 50% reel (uczciwy test fuzji)
print(f"zbalansowane: po {L:,} znaków z każdego | val walc+reel po równo")
bs = 32

def get_batch(split):
    d = train if split == "train" else val
    ix = torch.randint(len(d) - block, (bs,))
    return (torch.stack([d[i:i+block] for i in ix]),
            torch.stack([d[i+1:i+1+block] for i in ix]))

g = nn.Linear(n_embd, n_embd)
nn.init.normal_(g.weight, std=0.02); nn.init.zeros_(g.bias)

def emb(m, idx):
    return m.drop(m.tok_emb(idx) + m.pos_emb(torch.arange(idx.shape[1])))

def run_single(m, idx):
    x = emb(m, idx)
    for blk in m.blocks: x = blk(x)
    return m.head(m.ln_f(x))

def run_ensemble(idx, alpha=0.5):
    return alpha * run_single(A, idx) + (1 - alpha) * run_single(B, idx)

def run_stitch(idx):
    x = emb(A, idx)
    for i in range(SEAM): x = A.blocks[i](x)        # front: walc
    x = g(x)                                        # mapper na styku
    for i in range(SEAM, len(B.blocks)): x = B.blocks[i](x)  # back: reel
    return B.head(B.ln_f(x))

def ce(logits, y): return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

@torch.no_grad()
def ev(run, iters=60, seed=1234):
    torch.manual_seed(seed)
    return sum(ce(run(x), y).item() for x, y in (get_batch("val") for _ in range(iters))) / iters

pA = ev(lambda i: run_single(A, i)); pB = ev(lambda i: run_single(B, i)); pE = ev(run_ensemble)
print(f"walc-alone:  ppl {math.exp(pA):.2f}")
print(f"reel-alone:  ppl {math.exp(pB):.2f}")
print(f"ensemble:    ppl {math.exp(pE):.2f}   (baseline do pobicia)")

opt = torch.optim.AdamW(g.parameters(), lr=1e-3)
for _ in range(TRAIN_STEPS):
    x, y = get_batch("train")
    loss = ce(run_stitch(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
pS = ev(run_stitch)
print(f"STITCH:      ppl {math.exp(pS):.2f}   (po {TRAIN_STEPS} krokach treningu g)")
print(f"\nWERDYKT: stitch {math.exp(pS):.2f} vs ensemble {math.exp(pE):.2f} -> "
      + ("STITCH BIJE ENSEMBLE ✅" if pS < pE else "ensemble nie gorszy ❌"))

@torch.no_grad()
def gen_stitch(seed_str, n_new=400, temp=0.85, topk=18):
    idx = torch.tensor([[stoi[c] for c in seed_str]])
    for _ in range(n_new):
        logits = run_stitch(idx[:, -block:])[:, -1, :] / temp
        v, _ = torch.topk(logits, topk); logits[logits < v[:, [-1]]] = float("-inf")
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return "".join(itos[t] for t in idx[0].tolist())

def first_tune(raw):
    out = []
    for ln in raw.split("\n"):
        if ln.startswith("X:") and out: break
        out.append(ln)
    return "\n".join(out).strip()

os.makedirs("data/recordings/stitch", exist_ok=True)
for i, key in enumerate(["D", "G", "Emin"], 1):
    tune = first_tune(gen_stitch(f"X:1\nM:4/4\nK:{key}\n"))
    base = f"data/recordings/stitch/stitch_{i}_{key}"
    open(base + ".abc", "w", encoding="utf-8").write(tune + "\n")
    to_midi(tune, base + ".mid")
print("\npróbki stitchu -> data/recordings/stitch/ (render fortepian)")
torch.save({"A": A_CKPT, "B": B_CKPT, "g": g.state_dict(), "seam": SEAM}, "data/models/stitch_g.pt")
