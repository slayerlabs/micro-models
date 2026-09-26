"""Trening GPT od zera na korpusie ABC (L03/L08 w praktyce).
Batche -> strata cross-entropy -> backprop -> AdamW -> val loss -> checkpoint.
"""
import os, time, math, sys
from contextlib import nullcontext
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.gpt import GPT, GPTConfig

# --- hiperparametry ---
block_size  = 128
batch_size  = 32
n_layer     = int(os.environ.get("N_LAYER", 4))    # ENV: sweep skali (E_CKA)
n_head      = 4
n_embd      = int(os.environ.get("N_EMBD", 128))    # ENV: sweep skali (E_CKA)
dropout     = 0.1
lr          = 3e-4
max_iters   = 2000
eval_interval = 200
eval_iters  = 100
warmup      = 100

SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 20260620   # argv[5]: seed (niezależne modele do E_CKA/E0.5)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
sys.stdout.reconfigure(encoding="utf-8")
print(f"urządzenie: {device} | bf16: {use_bf16}")

# --- ścieżki z argumentów (domyślnie: jigi) ---
DATA    = sys.argv[1] if len(sys.argv) > 1 else "data/jigs.abc"
CKPT    = sys.argv[2] if len(sys.argv) > 2 else "data/models/jig_ckpt.pt"
LOSSLOG = sys.argv[3] if len(sys.argv) > 3 else "data/models/jig_loss_log.csv"
VOCAB_FROM = sys.argv[4] if len(sys.argv) > 4 else None   # wspólny słownik z innego ckpt (do stitchu)
print(f"dane: {DATA} -> checkpoint: {CKPT}")

# --- dane: char-level ---
text = open(DATA, encoding="utf-8").read()
if VOCAB_FROM:
    vck = torch.load(VOCAB_FROM, map_location="cpu", weights_only=False)
    stoi, itos = vck["stoi"], vck["itos"]
    chars = [itos[i] for i in range(len(itos))]
    print(f"wspólny słownik z {VOCAB_FROM}: {len(chars)} znaków")
else:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"słownik: {len(chars)} | tokeny: train {len(train_data):,} / val {len(val_data):,}")

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def lr_at(it):  # warmup + cosine decay (L08)
    if it < warmup:
        return lr * it / warmup
    r = (it - warmup) / (max_iters - warmup)
    return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * r))

cfg = GPTConfig(vocab_size=len(chars), block_size=block_size,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout)
model = GPT(cfg).to(device)
print(f"parametry modelu: {model.num_params():,}")
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.1)

best_val = float("inf")
log = [("iter", "train_loss", "val_loss")]
t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_interval == 0 or it == max_iters:
        L = estimate_loss()
        ppl = math.exp(L["val"])
        print(f"iter {it:4d} | train {L['train']:.3f} | val {L['val']:.3f} | ppl {ppl:.2f} | {time.time()-t0:.0f}s")
        log.append((it, round(L["train"], 4), round(L["val"], 4)))
        if L["val"] < best_val:           # zapis najlepszego (early-stop logic)
            best_val = L["val"]
            torch.save({"model": model.state_dict(), "config": cfg,
                        "stoi": stoi, "itos": itos, "val_loss": best_val},
                       CKPT)
    if it == max_iters:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    x, y = get_batch("train")
    with ctx:
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

with open(LOSSLOG, "w", encoding="utf-8") as f:
    f.write("\n".join(",".join(map(str, row)) for row in log))
print(f"\ngotowe. best val loss: {best_val:.3f} (ppl {math.exp(best_val):.2f})")
print(f"checkpoint -> {CKPT} | krzywa -> {LOSSLOG}")
