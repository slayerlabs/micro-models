"""Generuje N próbek z wytrenowanego modelu i renderuje do MIDI z wybranym instrumentem.
Po jednej melodii na podaną tonację. Zapisuje ABC + MIDI do osobnego katalogu.
Użycie:
  python src/gen_samples.py --ckpt data/models/waltz_ckpt.pt --meter 3/4 --keys D,G,C,Emin,Amin --inst piano  --out data/recordings/waltz
  python src/gen_samples.py --ckpt data/models/reel_ckpt.pt  --meter 4/4 --keys D,G,A,Emin,Bmin --inst violin --out data/recordings/reel
"""
import argparse, sys, os
sys.path.insert(0, "src")
import torch
from gpt import GPT
from abc_to_midi import to_midi
from music21 import instrument as M

INST = {"piano": M.Piano, "violin": M.Violin, "guitar": M.AcousticGuitar, "none": None}

def first_tune(raw):
    lines = []
    for ln in raw.split("\n"):
        if ln.startswith("X:") and lines:
            break
        lines.append(ln)
    return "\n".join(lines).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meter", default="4/4")
    ap.add_argument("--keys", default="D,G,A,Emin,Bmin")
    ap.add_argument("--inst", default="none", choices=list(INST))
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", type=int, default=420)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--topk", type=int, default=18)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    stoi, itos, cfg = ck["stoi"], ck["itos"], ck["config"]
    model = GPT(cfg); model.load_state_dict(ck["model"]); model.eval()
    os.makedirs(a.out, exist_ok=True)
    inst_cls = INST[a.inst]
    print(f"{a.ckpt} | val {ck['val_loss']:.3f} | instrument: {a.inst} -> {a.out}")
    torch.manual_seed(20260621)
    ok = 0
    for i, key in enumerate(a.keys.split(","), 1):
        seed = f"X:1\nM:{a.meter}\nK:{key}\n"
        if any(c not in stoi for c in seed):
            print(f"  #{i} ({key}): seed ma znak spoza słownika — pomijam"); continue
        idx = torch.tensor([[stoi[c] for c in seed]])
        gen = model.generate(idx, a.new, temperature=a.temp, top_k=a.topk)[0].tolist()
        tune = first_tune("".join(itos[t] for t in gen))
        base = f"{a.out}/sample_{i}_{key}"
        open(base + ".abc", "w", encoding="utf-8").write(tune + "\n")
        good = to_midi(tune, base + ".mid", inst=inst_cls() if inst_cls else None)
        ok += good
        print(f"  #{i} ({key}) [{'MIDI OK' if good else 'błąd'}] -> {base}.mid")
    print(f"\ngotowe: {ok}/{len(a.keys.split(','))} próbek w {a.out}/")

if __name__ == "__main__":
    main()
