"""eval_obsidian.py — GŁÓWNY DOWÓD iteracji 1: bits/char na WSPÓLNYM raw held-out.

bits/char = (Σ nats po tokenach) / (liczba ZNAKÓW raw) / ln2 — tokenizer-niezależne, więc
n-gram (char) i dwa GPT (różne BPE) są porównywalne na tym samym surowym tekście held-out.

Tabela 3 wierszy:
  n-gram     (ngram_obsidian.json)         — podłoga pamięci
  GPT-raw    (model-obsidian-raw.pt)       — uczony na surowych notatkach
  GPT-decomp (model-obsidian.pt)           — uczony na notatkach zdekomponowanych (markery AST-T2)

Dwie falsyfikowalne bramki:
  GENERALIZACJA: bits/char(GPT-decomp) < bits/char(n-gram)  -> uczy gładką funkcję, nie look-up
  ABLACJA:       bits/char(GPT-decomp) < bits/char(GPT-raw) -> dekompozycja realnie pomaga
(ocena na surowym held-out jest KONSERWATYWNA dla modelu-decomp -> wygrana mimo to jest mocna)

Użycie: python eval_obsidian.py
"""
import os, json, math, torch
from gpt import GPT
from train_tokenizer import encode

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def bits_per_char(model, merges, raw_text, block, device):
    """Σ CE (nats) w nienakładających się oknach `block`, podzielone przez liczbę ZNAKÓW raw / ln2."""
    model.eval()
    ids = torch.tensor(encode(raw_text, merges), dtype=torch.long)
    total_nats = 0.0
    for i in range(0, len(ids) - block - 1, block):
        x = ids[i:i + block].unsqueeze(0).to(device)
        y = ids[i + 1:i + block + 1].unsqueeze(0).to(device)
        _, loss = model(x, y)                       # loss = średnia CE/token (nats) w oknie
        total_nats += loss.item() * block
    return total_nats / len(raw_text) / math.log(2), len(ids)


def load_gpt(path, device):
    ck = torch.load(os.path.join(HERE, path), map_location=device, weights_only=False)
    cfg, merges = ck["config"], ck["merges"]
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model, merges, cfg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        __import__("sys").stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    raw = open(os.path.join(HERE, "data/heldout-raw.txt"), encoding="utf-8").read()
    print(f"urządzenie: {device} | held-out raw: {len(raw):,} znaków\n")

    rows = []

    # n-gram (z Kroku 4)
    ngram_path = os.path.join(HERE, "ngram_obsidian.json")
    if os.path.exists(ngram_path):
        ng = json.load(open(ngram_path, encoding="utf-8"))
        rows.append(("n-gram", ng["bits_per_char"], f"M={ng['order']}"))
    else:
        print("UWAGA: brak ngram_obsidian.json — uruchom najpierw `python ngram_obsidian.py`.")

    # dwa GPT trenowane od zera
    for label, path in [("GPT-raw", "model-obsidian-raw.pt"), ("GPT-decomp", "model-obsidian.pt")]:
        model, merges, cfg = load_gpt(path, device)
        bpc, ntok = bits_per_char(model, merges, raw, cfg.block_size, device)
        rows.append((label, bpc, f"{model.num_params():,} param, {ntok:,} tok"))

    print(f"{'model':>11} | {'bits/char':>9} | szczegóły")
    print("-" * 60)
    for name, bpc, det in rows:
        print(f"{name:>11} | {bpc:>9.4f} | {det}")

    res = {name: bpc for name, bpc, _ in rows}

    print("\n=== BRAMKI (falsyfikowalne) ===")
    verdicts = {}
    if "n-gram" in res and "GPT-decomp" in res:
        gen = res["GPT-decomp"] < res["n-gram"]
        verdicts["generalizacja"] = gen
        mark = "✓" if gen else "✗"
        rel = "<" if gen else "≥"
        print(f"  {mark} GENERALIZACJA: GPT-decomp {res['GPT-decomp']:.4f} {rel} n-gram {res['n-gram']:.4f} "
              + ("— model generalizuje (gładka funkcja, nie tablica)."
                 if gen else "— przy tej skali to pamięć (wynik NEGATYWNY = raportujemy)."))
    if "GPT-raw" in res and "GPT-decomp" in res:
        abl = res["GPT-decomp"] < res["GPT-raw"]
        verdicts["ablacja_decomp_pomaga"] = abl
        mark = "✓" if abl else "✗"
        rel = "<" if abl else "≥"
        print(f"  {mark} ABLACJA:       GPT-decomp {res['GPT-decomp']:.4f} {rel} GPT-raw {res['GPT-raw']:.4f} "
              + ("— dekompozycja realnie pomaga (lepsze reprezentacje)."
                 if abl else "— dekompozycja nie pomaga przy tej skali."))

    out = {"held_out_chars": len(raw), "bits_per_char": res, "verdicts": verdicts}
    json.dump(out, open(os.path.join(HERE, "eval_obsidian.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nzapisano -> eval_obsidian.json")


if __name__ == "__main__":
    main()
