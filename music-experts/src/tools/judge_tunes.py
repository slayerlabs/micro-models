# Autor: Adam Skrodzki
"""Judge: klasyfikuje wygenerowane melodie ABC (meter/mode/type) z samego ciała.
Trunk GPT + mean-pool (core/judge.py). Ciało dzielone na okna 256 znaków,
softmax uśredniony po oknach (pełne pokrycie — jak w evaluate() train_judge.py).
Wejście: pliki .abc lub katalogi. Wynik: tabela + agregat; --json dla maszynowej postaci.
Użycie: python src/tools/judge_tunes.py out/ [--json] [--judge data/models/judge_v2.pt]
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from core.gpt import GPTConfig
from core.judge import JudgeGPT, HEADS
from core.abc_corpus import tune_body

def split_tunes(text):
    """Kolejne bloki melodii z pliku ABC (blok kończy się na następnej linii X:)."""
    tunes, cur = [], []
    for ln in text.split("\n"):
        if ln.startswith("X:") and cur:
            tunes.append("\n".join(cur)); cur = []
        cur.append(ln)
    if cur and any(l.startswith("X:") for l in cur):
        tunes.append("\n".join(cur))
    return tunes

def windows(body, block):
    return [body[s:s + block] for s in range(0, len(body), block)] or [" "]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="pliki .abc lub katalogi")
    ap.add_argument("--judge", default="data/models/judge_v2.pt")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ck = torch.load(a.judge, map_location="cpu", weights_only=False)
    cfg, chars, classes = ck["config"], ck["chars"], ck["classes"]
    stoi = {c: j + 1 for j, c in enumerate(chars)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JudgeGPT(cfg, {h: len(classes[h]) for h in HEADS}, pool=ck.get("pool", "mean"))
    model.load_state_dict(ck["model"]); model.eval().to(device)
    print(f"urządzenie: {device}")

    files = []
    for path in a.inputs:
        if os.path.isdir(path):
            files += sorted(glob.glob(os.path.join(path, "**", "*.abc"), recursive=True))
        else:
            files.append(path)
    if not files:
        sys.exit("brak plików .abc do oceny")

    tunes = []
    for path in files:
        for ti, tune in enumerate(split_tunes(open(path, encoding="utf-8").read())):
            body = tune_body(tune)
            if len(body) >= 40:
                tunes.append({"file": path, "tune": ti, "body": body})

    results = []
    with torch.no_grad():
        for t in tunes:
            wins = windows(t["body"], cfg.block_size)
            idx = torch.zeros(len(wins), cfg.block_size, dtype=torch.long, device=device)
            mask = torch.zeros(len(wins), cfg.block_size, device=device)
            for b, w in enumerate(wins):
                ids = [stoi[c] for c in w if c in stoi] or [stoi[" "]]
                idx[b, :len(ids)] = torch.tensor(ids, device=device)
                mask[b, :len(ids)] = 1.0
            logits = model(idx, mask)
            rec = {"file": t["file"], "tune": t["tune"], "windows": len(wins)}
            for h in HEADS:
                mean_p = F.softmax(logits[h], dim=-1).mean(dim=0)
                best = int(mean_p.argmax())
                rec[h] = classes[h][best]
                rec[f"{h}_conf"] = round(float(mean_p[best]), 3)
            results.append(rec)

    if a.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print(f"{'plik':40s} {'okna':4s} {'meter':7s} {'mode':7s} {'type':10s} {'pewność (m/m/t)'}")
    for r in results:
        conf = f"{r['meter_conf']:.2f}/{r['mode_conf']:.2f}/{r['type_conf']:.2f}"
        print(f"{r['file']:40s} {r['windows']:4d} {r['meter']:7s} {r['mode']:7s} {r['type']:10s} {conf}")
    accs = ck.get("accs", {})
    print(f"\noceniono: {len(results)} melodii z {len(files)} plików | judge: {a.judge} "
          + ("(val acc: " + ", ".join(f"{h} {v:.3f}" for h, v in accs.items()) + ")" if accs else ""))

if __name__ == "__main__":
    main()
