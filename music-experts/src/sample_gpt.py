"""Generacja melodii z wytrenowanego GPT (wczytuje checkpoint)."""
import sys
import torch
from gpt import GPT

sys.stdout.reconfigure(encoding="utf-8")
device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("data/gpt_ckpt.pt", map_location=device, weights_only=False)
stoi, itos, cfg = ck["stoi"], ck["itos"], ck["config"]

model = GPT(cfg).to(device)
model.load_state_dict(ck["model"])
model.eval()
print(f"wczytano GPT ({model.num_params():,} param) | val loss {ck['val_loss']:.3f}")

def gen(seed, n_new=400, temp=0.8, top_k=20):
    idx = torch.tensor([[stoi[c] for c in seed]], dtype=torch.long, device=device)
    out = model.generate(idx, n_new, temperature=temp, top_k=top_k)[0].tolist()
    return "".join(itos[i] for i in out)

seed = "X:1\nM:6/8\nK:D\n"
raw = gen(seed, n_new=500, temp=0.8, top_k=20)
# potnij na osobne melodie po pustej linii
tunes, cur = [], []
for line in raw.split("\n"):
    if line.startswith("X:") and cur:
        tunes.append("\n".join(cur)); cur = []
    cur.append(line)
if cur:
    tunes.append("\n".join(cur))
tunes = [t.strip() for t in tunes if "X:" in t][:5]

for i, t in enumerate(tunes, 1):
    print(f"\n===== GPT JIG #{i} =====\n{t}")
with open("data/generated_gpt.abc", "w", encoding="utf-8") as f:
    f.write("\n\n".join(tunes) + "\n")
print("\n[zapisano -> data/generated_gpt.abc]")
