# Autor: Adam Skrodzki
"""OOD transfer matrix: ekspert × docelowa domena. Jak wypada model POZA swoją domeną?

Dla każdej pary (model, cel): pula = val melodie DOMENY CELU (type+meter jak prepare_data),
prompt z nagłówkami celu (scratch) lub z ćwiartką ciała melodii celu (continuation).
PRAWDA = to, o co prosimy (etykiety z melodii celu), nie domena modelu — model jest proszony,
więc nie ma wymówki. Porażka w OOD (śmieci, znaki spoza słownika) = 0, jak w benchmarku
domenowym; pokrycie raportowane osobno.

Metryki w komórce macierzy: score/ref — score = średnie P sędziego przy prawdziwych klasach,
ref = to samo dla PRAWDZIWYCH melodii celu (sufit). hb = home-bias: P sędziego przy KLASIE
DOMOWEJ modelu (meter dla scratch; średnia meter+type dla continuation) — rozróżnia sztywnego
modelu (wysoki hb, ignoruje prompt) od papki (niskie wszystko). * = komórka domenowa (diagonala).

Użycie:
  python src/tools/benchmark_ood.py                       # wszystkie ckpt z domenami
  python src/tools/benchmark_ood.py --models data/models/jig_ckpt.pt --samples 50
  python src/tools/benchmark_ood.py --json > data/benchmarks/ood.json
"""
import argparse, glob, json, sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.gpt import GPT
from core.judge import JudgeGPT, HEADS
from train_judge import load_rows, group_split, MIN_BODY
from benchmark_judge import mean_probs

def gen_body(model, stoi, itos, device, prompt, a):
    """Ciało wygenerowanej melodii (bez linii nagłówkowych) albo None, gdy prompt
    niekodowalny / ciało zbyt krótkie (porażka OOD = 0 punktu)."""
    if any(c not in stoi for c in prompt):
        return None, "vocab"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(idx, a.new, temperature=a.temp, top_k=a.topk)[0].tolist()
    body = "\n".join(l for l in (prompt + "".join(itos[t] for t in gen)).split("\n")
                     if not l.startswith(("X:", "T:", "C:", "M:", "K:", "N:", "%"))).strip()
    return (body, None) if len(body) >= MIN_BODY else (None, "short_body")

def run_cell(model, stoi, itos, judge, jcfg, jstoi, classes, device, rows,
             sample_ids, home, task, a):
    """Jedna komórka macierzy (model × cel × zadanie). Zwraca statystyki."""
    truth = {h: {c: i for i, c in enumerate(classes[h])} for h in HEADS}
    heads = HEADS if task == "continuation" else [h for h in HEADS if h != "type"]
    home_heads = [h for h in ("meter", "type") if h in heads and home.get(h) in truth[h]]
    p_sum = {h: 0.0 for h in heads}; hb_sum = {h: 0.0 for h in home_heads}
    score, invalid = 0.0, {"vocab": 0, "short_body": 0}
    for i in sample_ids:
        row = rows[i]
        prompt = f"X:1\nM:{row['meter']}\nK:{row['mode']}\n"
        if task == "continuation":
            prompt += row["body"][:max(40, len(row["body"]) // 4)]
        body, why = gen_body(model, stoi, itos, device, prompt, a)
        if body is None:
            invalid[why] += 1
            continue   # = 0 punktu
        probs = mean_probs(judge, jcfg, jstoi, device, body)
        for h in heads:
            t = truth[h][row[h]]
            score += float(probs[h][t])
            p_sum[h] += float(probs[h][t])
            if h in home_heads:
                hb_sum[h] += float(probs[h][truth[h][home[h]]])
    n = max(len(sample_ids), 1)
    nh = len(heads)
    return {"score": score / (n * nh),
            "p": {h: p_sum[h] / n for h in heads},
            "home_bias": {h: hb_sum[h] / n for h in home_heads},
            "invalid": invalid, "coverage": 1 - sum(invalid.values()) / n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="ckpty do macierzy (domyślnie wszystkie *_ckpt.pt z domeną)")
    ap.add_argument("--judge", default="data/models/judge_v2.pt")
    ap.add_argument("--csv", default="data/tunes.csv")
    ap.add_argument("--domains", default="data/models/domains.json")
    ap.add_argument("--samples", type=int, default=100, help="melodii na komórkę")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--new", type=int, default=420)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--topk", type=int, default=18)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    print(f"urządzenie: {device} | seed: {a.seed} | temp {a.temp} topk {a.topk} new {a.new}")

    with open(a.domains, encoding="utf-8") as f:
        domains = json.load(f)

    ck = torch.load(a.judge, map_location="cpu", weights_only=False)
    jcfg, chars, classes = ck["config"], ck["chars"], ck["classes"]
    jstoi = {c: j + 1 for j, c in enumerate(chars)}
    judge = JudgeGPT(jcfg, {h: len(classes[h]) for h in HEADS}, pool=ck.get("pool", "mean"))
    judge.load_state_dict(ck["model"]); judge.eval().to(device)

    # cele = wszystkie różne domeny (type, meter); diag = domena modelu
    # (_comment w domains.json to string — bierzemy tylko wpisy słownikowe)
    targets = sorted({(d["type"], d["meter"]) for n, d in domains.items()
                      if isinstance(d, dict) and d.get("type") and not n.startswith("judge")})
    print(f"cele: " + ", ".join(f"{t}:{m}" for t, m in targets))

    print(f"czytam {a.csv} ...")
    rows = load_rows(a.csv)
    _, va = group_split(rows, a.split_seed)

    # pula i próbki per CEL: niezależne od modeli (ten sam seed -> te same melodie
    # w kolumnie dla każdego modelu) + referencja = prawdziwe melodie celu
    pools, refs = {}, {}
    for ttype, tmeter in targets:
        key = f"{ttype}:{tmeter}"
        rng = random.Random(f"{a.seed}|{key}")   # deterministyczny per cel
        pool = [i for i in va
                if ttype in rows[i]["type"].lower()
                and rows[i]["meter"].strip() == tmeter
                and len(rows[i]["body"]) >= 4 * 40]
        ids = rng.sample(pool, min(a.samples, len(pool)))
        pools[key] = ids
        # referencja: sędzia na prawdziwych ciałach celu (nie zależy od modelu)
        s = {h: 0.0 for h in HEADS}
        for i in ids:
            probs = mean_probs(judge, jcfg, jstoi, device, rows[i]["body"])
            for h in HEADS:
                s[h] += float(probs[h][classes[h].index(rows[i][h])])
        refs[key] = sum(s.values()) / (len(ids) * len(HEADS))
        print(f"  cel {key}: pula {len(pool)}, referencja {refs[key]:.3f}")

    models = a.models or sorted(glob.glob("data/models/*_ckpt.pt"))
    cells = {}
    for mp in models:
        name = os.path.basename(mp)
        dom = domains.get(name)
        if dom is None or dom.get("type") is None:
            print(f"pomijam {name}: brak domeny w {a.domains}")
            continue
        home = {"type": dom["type"], "meter": dom["meter"]}
        mck = torch.load(mp, map_location=device, weights_only=False)
        stoi, itos, mcfg = mck["stoi"], mck["itos"], mck["config"]
        model = GPT(mcfg); model.load_state_dict(mck["model"]); model.eval().to(device)
        for ttype, tmeter in targets:
            key = f"{ttype}:{tmeter}"
            diag = (ttype == dom["type"] and tmeter == dom["meter"])
            for task in ("scratch", "continuation"):
                st = run_cell(model, stoi, itos, judge, jcfg, jstoi, classes, device,
                              rows, pools[key], home, task, a)
                cells[(name, key, task)] = st
                if not a.json:
                    print(f"  {name} x {key} [{task}]: score {st['score']:.3f} "
                          f"(hb {st['home_bias']}, pokrycie {st['coverage']:.0%})")
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    if a.json:
        print(json.dumps({"seed": a.seed, "targets": [f"{t}:{m}" for t, m in targets],
                          "refs": refs, "cells": {f"{n}|{k}|{t}": v
                                                  for (n, k, t), v in cells.items()}},
                         ensure_ascii=False, indent=2))
        return

    # macierze: wiersz = model, kolumna = cel; komórka score/ref hb.. c..% (* = diag)
    for task in ("scratch", "continuation"):
        w = 24
        print(f"\n=== {task} | komórka: score/ref h=home-bias c=pokrycie ===")
        print(f"{'model':22s}" + "".join(f"{k:>{w}}" for k in refs))
        for name in sorted({n for n, _, _ in cells}):
            row = f"{name:22s}"
            for key in refs:
                st = cells.get((name, key, task))
                if st is None:
                    row += "—".rjust(w)
                    continue
                diag = "*" if domains.get(name, {}).get("type") == key.split(":")[0] else ""
                hb = "/".join(f"{v:.2f}" for v in st["home_bias"].values()) or "—"
                cell = f"{st['score']:.2f}/{refs[key]:.2f}h{hb}c{st['coverage']:.0%}{diag}"
                row += cell[:w].rjust(w)
            print(row)
    print("\nref = sędzia na prawdziwych melodiach celu (sufit); hb wysokie = model ignoruje "
          "prompt i wraca do domeny; niskie score i niskie hb = papka. * = komórka domenowa.")

if __name__ == "__main__":
    main()
