"""DUET: walc (fortepian) + reel (skrzypce) grają RAZEM — dwie ścieżki naraz, ta sama tonacja.
To NIE fuzja w jedną linię (jak ensemble), tylko dwie niezależne linie złożone jednocześnie.
"""
import sys, os
sys.path.insert(0, "src")
import torch
from gpt import GPT
from abc_to_midi import sanitize
from music21 import converter, stream, instrument as M, tempo

def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
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
def gen(model, stoi, itos, seed, n=360, temp=0.85, topk=18):
    idx = torch.tensor([[stoi[c] for c in seed]])
    out = model.generate(idx, n, temperature=temp, top_k=topk)[0].tolist()
    return first_tune("".join(itos[t] for t in out))

def to_part(abc, inst):
    sc = converter.parse(sanitize(abc), format="abc")
    notes = sc.flatten().notesAndRests.stream()
    notes.insert(0, inst)
    return notes

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    waltz, ckW = load("data/models/waltz_ckpt.pt")
    reel, ckR = load("data/models/reel_sv_ckpt.pt")
    stoi, itos = ckW["stoi"], ckW["itos"]
    os.makedirs("data/recordings/duet", exist_ok=True)
    torch.manual_seed(20260621)
    ok = 0
    for key in ["D", "G", "Emin"]:
        wabc = gen(waltz, stoi, itos, f"X:1\nM:3/4\nK:{key}\n")   # fortepian, walc 3/4
        rabc = gen(reel,  stoi, itos, f"X:1\nM:4/4\nK:{key}\n")   # skrzypce, reel 4/4
        try:
            sc = stream.Score()
            sc.insert(0, tempo.MetronomeMark(number=96))
            sc.insert(0, to_part(wabc, M.Piano()))
            sc.insert(0, to_part(rabc, M.Violin()))
            path = f"data/recordings/duet/duet_{key}.mid"
            sc.write("midi", fp=path)
            print(f"  duet {key} [OK] -> {path}"); ok += 1
        except Exception as e:
            print(f"  duet {key} [błąd]: {type(e).__name__}: {e}")
    print(f"\ngotowe: {ok}/3 duetów w data/recordings/duet/ — fortepian + skrzypce RAZEM 🎻🎹")

if __name__ == "__main__":
    main()
