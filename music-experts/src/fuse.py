"""Złączenie przez ensemble (logit-mix) DWÓCH modeli o WSPÓLNYM słowniku.
W każdym kroku: logits = alpha*A + (1-alpha)*B -> sampluj. Melodia wychodzi stylistycznie pomiędzy.
To jest baseline „płaskiego ważenia" (KMS5). Stitch reprezentacji = osobny eksperyment (E1).
Użycie:
  python src/fuse.py --a data/models/waltz_ckpt.pt --b data/models/reel_sv_ckpt.pt \
      --alpha 0.5 --meter 3/4 --keys D,G,Emin --inst piano --out data/recordings/fuzja
"""
import argparse, sys, os, math
sys.path.insert(0, "src")
import torch
from torch.nn import functional as F
from gpt import GPT
from abc_to_midi import to_midi
from music21 import instrument as M

INST = {"piano": M.Piano, "violin": M.Violin, "none": None}

def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = GPT(ck["config"]); m.load_state_dict(ck["model"]); m.eval()
    return m, ck

def first_tune(raw):
    out = []
    for ln in raw.split("\n"):
        if ln.startswith("X:") and out:
            break
        out.append(ln)
    return "\n".join(out).strip()

@torch.no_grad()
def gen_mix(A, B, idx, n, alpha, temp, topk, block):
    for _ in range(n):
        ic = idx[:, -block:]
        la, _ = A(ic); lb, _ = B(ic)
        logits = (alpha * la[:, -1, :] + (1 - alpha) * lb[:, -1, :]) / temp
        if topk:
            v, _ = torch.topk(logits, topk)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--meter", default="3/4"); ap.add_argument("--keys", default="D,G,Emin")
    ap.add_argument("--inst", default="piano", choices=list(INST))
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", type=int, default=420); ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--topk", type=int, default=18)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    A, ckA = load(a.a); B, ckB = load(a.b)
    assert ckA["stoi"] == ckB["stoi"], "Modele mają RÓŻNY słownik — najpierw wspólny słownik!"
    stoi, itos = ckA["stoi"], ckA["itos"]
    block = ckA["config"].block_size
    os.makedirs(a.out, exist_ok=True)
    inst_cls = INST[a.inst]
    print(f"FUZJA α={a.alpha} (A={a.a} : B={a.b}) | wspólny słownik {len(stoi)} | -> {a.out}")
    torch.manual_seed(20260621)
    ok = 0
    for i, key in enumerate(a.keys.split(","), 1):
        seed = f"X:1\nM:{a.meter}\nK:{key}\n"
        idx = torch.tensor([[stoi[c] for c in seed]])
        gen = gen_mix(A, B, idx, a.new, a.alpha, a.temp, a.topk, block)[0].tolist()
        tune = first_tune("".join(itos[t] for t in gen))
        base = f"{a.out}/fuzja_{i}_{key}"
        open(base + ".abc", "w", encoding="utf-8").write(tune + "\n")
        good = to_midi(tune, base + ".mid", inst=inst_cls() if inst_cls else None)
        ok += good
        print(f"  #{i} ({key}) [{'MIDI OK' if good else 'błąd'}] -> {base}.mid")
    print(f"\ngotowe: {ok}/{len(a.keys.split(','))} fuzji w {a.out}/")

if __name__ == "__main__":
    main()
