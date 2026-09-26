# Autor: Adam Skrodzki
"""Benchmark eksperta GPT oceniany przez sędziego (judge). Dwa zadania:

  scratch      — tylko etykiety (meter/mode) w promptcie `X:1\nM:m\nK:k\n`; model generuje
                 od zera; sędzia klasyfikuje ciało.
  continuation — jak wyżej + w promptcie ćwiartka prawdziwego ciała melodii z val; model
                 kontynuuje; sędzia klasyfikuje CAŁOŚĆ (prefiks + kontynuacja) — pytanie
                 brzmi: czy wynikowy utwór nadal jest "w stylu".

Domena: pula melodii = val split ograniczony do DOMENY checkpointu (data/models/domains.json:
type substring + meter, jak w prepare_data.py). Melodie spoza domeny nie są błędem modelu —
nie trafiają do puli wcale (bach: type=null = poza benchmarkiem). W domenie liczą się WSZYSTKIE
próbki: porażka modelu (śmieci zamiast melodii) = 0 punktu.

Wynik: dla każdej próbki sumujemy prawdopodobieństwa, jakie sędzia przyznaje PRAWDZIWYM
klasom, uśredniamy -> score 0..1 (+ pokrycie). Dodatkowo accuracy (argmax == prawda) per
głowa. Pula = val sędziego (split po tune_id, ten sam seed co train_judge), więc sędzia
ocenia melodie, których nigdy nie widział.

Użycie:
  python src/tools/benchmark_judge.py --model data/models/jig_ckpt.pt
  python src/tools/benchmark_judge.py --model data/models/jig_ckpt.pt --task continuation
  python src/tools/benchmark_judge.py --model data/models/jig_ckpt.pt --json
"""
import argparse, collections, json, sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from core.gpt import GPT
from core.abc_corpus import parse_tune_row
from core.judge import JudgeGPT, HEADS
from train_judge import load_rows, group_split, MIN_BODY   # reuse: czyszczenie + split
from judge_tunes import windows                            # reuse: okna ciała

def mean_probs(judge, jcfg, jstoi, device, body, bs=64):
    """Średni softmax sędziego po wszystkich oknach ciała -> {head: wektor prawd.}."""
    wins = windows(body, jcfg.block_size)
    idx = torch.zeros(len(wins), jcfg.block_size, dtype=torch.long, device=device)
    mask = torch.zeros(len(wins), jcfg.block_size, device=device)
    for b, w in enumerate(wins):
        ids = [jstoi[c] for c in w if c in jstoi] or [jstoi[" "]]
        idx[b, :len(ids)] = torch.tensor(ids, device=device)
        mask[b, :len(ids)] = 1.0
    with torch.no_grad():
        logits = judge(idx, mask)
    return {h: F.softmax(logits[h], dim=-1).mean(dim=0) for h in HEADS}

def run_task(model, stoi, itos, judge, jcfg, jstoi, classes, device, task,
             rows, sample_ids, a, failed=None):
    """Jedno zadanie benchmarku. Zwraca (score 0..1, statystyki, liczba pominiętych).
    Scratch ocenia tylko meter i mode — type NIE jest w promptcie, model nie ma jak go
    zgadnąć (generuje styl swojego korpusu), więc ocenianie go byłoby mylące.
    Pominięte próbki (znaki spoza słownika, ciało < MIN_BODY) liczą się jako 0 — inaczej
    survivor bias zawyżałby wynik słabych modeli (score tylko nad udanymi generacjami).
    W continuation liczy też statystyki PRAWDZIWEJ melodii (całe ciało z val) — referencja;
    referencja NIE zależy od modelu, więc liczona dla WSZYSTKICH próbek."""
    truth = {h: {c: i for i, c in enumerate(classes[h])} for h in HEADS}
    heads = HEADS if task == "continuation" else [h for h in HEADS if h != "type"]
    p_sum = {h: 0.0 for h in heads}; acc = {h: 0 for h in heads}
    p_ref = {h: 0.0 for h in heads}; acc_ref = {h: 0 for h in heads}   # referencja = prawdziwe ciało
    score_sum, score_ref, skipped = 0.0, 0.0, 0
    skip_reasons = collections.Counter()

    def tally(row, probs):
        """Zbierz (sum prawd. prawdziwych klas / liczba ocenianych głów, per-head p/acc)."""
        ps = {h: 0.0 for h in heads}; ac = {h: 0 for h in heads}
        for h in heads:
            t = truth[h][row[h]]
            ps[h] = float(probs[h][t]); ac[h] = int(int(probs[h].argmax()) == t)
        return sum(ps.values()) / len(heads), ps, ac

    for s in sample_ids:
        row = rows[s]
        if task == "continuation":
            # referencja najpierw: nie zależy od modelu, liczona nawet gdy model zawiedzie
            s_true, ps, ac = tally(row, mean_probs(judge, jcfg, jstoi, device, row["body"]))
            score_ref += s_true; p_ref.update((h, p_ref[h] + ps[h]) for h in heads)
            acc_ref.update((h, acc_ref[h] + ac[h]) for h in heads)
        if task == "scratch":
            prompt = f"X:1\nM:{row['meter']}\nK:{row['mode']}\n"
        else:  # continuation: ćwiartka prawdziwego ciała jako kontekst stylu
            body_true = row["body"]
            prompt = f"X:1\nM:{row['meter']}\nK:{row['mode']}\n{body_true[:max(40, len(body_true) // 4)]}"
        if any(c not in stoi for c in prompt):
            skipped += 1   # = 0 punktu
            missing = sorted({c for c in prompt if c not in stoi})
            skip_reasons["vocab"] += 1
            if failed is not None:
                failed.append({"task": task, "reason": "vocab", "prompt": prompt,
                               "missing_chars": missing})
            continue
        idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.generate(idx, a.new, temperature=a.temp, top_k=a.topk)[0].tolist()
        raw = "".join(itos[t] for t in gen)
        body = "\n".join(l for l in (prompt + raw).split("\n")
                         if not l.startswith(("X:", "T:", "C:", "M:", "K:", "N:", "%"))).strip()
        if len(body) < MIN_BODY:
            skipped += 1   # = 0 punktu
            skip_reasons["short_body"] += 1
            if failed is not None:
                failed.append({"task": task, "reason": "short_body", "prompt": prompt,
                               "generated": raw, "body_len": len(body)})
            continue
        s_model, ps, ac = tally(row, mean_probs(judge, jcfg, jstoi, device, body))
        score_sum += s_model; p_sum.update((h, p_sum[h] + ps[h]) for h in heads)
        acc.update((h, acc[h] + ac[h]) for h in heads)
    n = max(len(sample_ids), 1)   # mianownik = WSZYSTKIE próbki, pominięte wliczone jako 0
    stats = {h: {"p_correct": p_sum[h] / n, "acc": acc[h] / n} for h in heads}
    if task == "continuation":
        stats["actual"] = {"score": score_ref / n,
                           **{h: {"p_correct": p_ref[h] / n, "acc": acc_ref[h] / n} for h in heads}}
    return score_sum / n, stats, skipped, dict(skip_reasons)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ckpt eksperta GPT (ten, którego oceniamy)")
    ap.add_argument("--judge", default="data/models/judge_v2.pt")
    ap.add_argument("--csv", default="data/tunes.csv")
    ap.add_argument("--domains", default="data/models/domains.json",
                    help="mapowanie checkpoint -> domena (type/meter); pula = val w domenie")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--task", default="both", choices=["scratch", "continuation", "both"])
    ap.add_argument("--seed", type=int, default=42,
                    help="seed losowania próbek i generacji; ten sam seed = IDENTYCZNE "
                         "melodie dla każdego modelu (porównywalne benchmarki)")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="seed splitu val — musi zgadzać się z seedem treningu sędziego")
    ap.add_argument("--new", type=int, default=420, help="ile znaków generować")
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--topk", type=int, default=18)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dump-failed", default=None, metavar="PLIK",
                    help="zapisz pominięte generacje (prompt + surowy output) do PLIKU (json)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    with open(a.domains, encoding="utf-8") as f:
        domains = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)   # deterministyczna generacja
    rng = random.Random(a.seed) # deterministyczny dobór próbek
    print(f"urządzenie: {device} | seed: {a.seed}")
    print(f"sędzia: {a.judge}")
    ck = torch.load(a.judge, map_location="cpu", weights_only=False)
    jcfg, chars, classes = ck["config"], ck["chars"], ck["classes"]
    jstoi = {c: j + 1 for j, c in enumerate(chars)}
    judge = JudgeGPT(jcfg, {h: len(classes[h]) for h in HEADS}, pool=ck.get("pool", "mean"))
    judge.load_state_dict(ck["model"]); judge.eval().to(device)
    judge_accs = ck.get("accs", {})

    print(f"model: {a.model}")
    mck = torch.load(a.model, map_location=device, weights_only=False)
    stoi, itos, mcfg = mck["stoi"], mck["itos"], mck["config"]
    model = GPT(mcfg); model.load_state_dict(mck["model"]); model.eval().to(device)

    print(f"czytam {a.csv} ...")
    rows = load_rows(a.csv)
    _, va = group_split(rows, a.split_seed)

    # Domena modelu (data/models/domains.json): benchmarkujemy TYLKO melodie z domeny
    # checkpointu. Melodia spoza domeny nie jest błędem modelu — nie może jej być ani
    # brnąć w nią; nie liczy się w ogóle. Błędy W domenie (śmieci, brak ciała) = 0.
    dom = domains.get(os.path.basename(a.model))
    if dom is None:
        sys.exit(f"brak domeny dla {os.path.basename(a.model)} w {a.domains} — dopisz wpis")
    if dom.get("type") is None:
        sys.exit(f"{os.path.basename(a.model)}: domena poza zakresem tunes.csv "
                 f"(type=null w {a.domains}) — benchmark nie ma czego mierzyć")
    dom_pool = [i for i in va
                if dom["type"] in rows[i]["type"].lower()
                and rows[i]["meter"].strip() == dom["meter"]]
    pool = [i for i in dom_pool if len(rows[i]["body"]) >= 4 * 40]  # ćwiartka prefiksu >= 40 znaków
    print(f"domena modelu: type~'{dom['type']}' + meter {dom['meter']}")
    print(f"pula melodii: val {len(va)} -> w domenie {len(dom_pool)} -> z ciałem {len(pool)}")
    if not pool:
        sys.exit("pusta pula domenowa — benchmark niemożliwy")
    print(f"val acc sędziego: " + " | ".join(f"{h} {v:.3f}" for h, v in judge_accs.items()))
    sample_ids = rng.sample(pool, min(a.samples, len(pool)))

    tasks = ["scratch", "continuation"] if a.task == "both" else [a.task]
    failed_all = []
    results = {}
    for task in tasks:
        score, stats, skipped, skip_reasons = run_task(model, stoi, itos, judge, jcfg, jstoi,
                                                       classes, device, task, rows, sample_ids,
                                                       a, failed=failed_all)
        results[task] = {"score": score, "skipped": skipped, "skip_reasons": skip_reasons, **stats}
        if a.json:
            continue
        ref = stats.get("actual")
        why = ", ".join(f"{k}: {v}" for k, v in skip_reasons.items()) or "-"
        cov = 100 * (len(sample_ids) - skipped) / max(len(sample_ids), 1)
        print(f"\n=== {task} (score {score:.3f} @ pokrycie {cov:.0f}%"
              + (f", prawdziwa melodia {ref['score']:.3f}" if ref else "")
              + f", pominięto {skipped}/{len(sample_ids)} [{why}]) ===")
        for h in [h for h in HEADS if h in stats]:
            st = stats[h]
            line = f"  {h:6s}: P(prawdziwa klasa) {st['p_correct']:.3f} | acc {st['acc']:.3f}"
            if ref:
                ra = ref[h]
                line += f"  [prawdziwa: P {ra['p_correct']:.3f} | acc {ra['acc']:.3f}]"
            print(line)

    if a.dump_failed and failed_all:
        os.makedirs(os.path.dirname(a.dump_failed) or ".", exist_ok=True)
        with open(a.dump_failed, "w", encoding="utf-8") as f:
            json.dump({"model": a.model, "reasons": dict(collections.Counter(
                          r["reason"] for r in failed_all)), "failed": failed_all},
                      f, ensure_ascii=False, indent=2)
        print(f"\npominięte generacje ({len(failed_all)}) -> {a.dump_failed}")

    if a.json:
        print(json.dumps({"model": a.model, "judge": a.judge, "samples": a.samples,
                          "judge_val_accs": judge_accs, "tasks": results},
                         ensure_ascii=False, indent=2))
        return

    print(f"\nbenchmark: {a.model} | {len(sample_ids)} melodii z val | temp {a.temp} topk {a.topk}")
    for task in tasks:
        r = results[task]
        cov = 100 * (len(sample_ids) - r["skipped"]) / max(len(sample_ids), 1)
        line = f"  {task:12s} score: {r['score']:.3f} @ pokrycie {cov:.0f}%"
        if "actual" in r:
            line += f"  (prawdziwe melodie: {r['actual']['score']:.3f})"
        print(line)
    print("score = średnie prawdopodobieństwo sędziego przy prawdziwych klasach (0..1); "
          "pominięte generacje liczą się jako 0. Score porównuj tylko przy zbliżonym pokryciu — "
          "niskie pokrycie = model nie jest w stanie nawet zakodować promptu (inny alfabet korpusu)")

if __name__ == "__main__":
    main()
