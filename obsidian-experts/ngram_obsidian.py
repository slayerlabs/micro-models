"""ngram_obsidian.py — baseline n-gram (char-level) na korpusie Obsidiana -> bits/char.

Adaptacja micro-models/music-experts/src/train/ngram_eval.py:
  - train = data/korpus-obsidian-raw.txt, val = data/heldout-raw.txt (osobne pliki, NIE wewn. split)
  - metryka = bits/char = (Σ −ln p)/n_chars/ln2 przy rzędzie M=ORDER (zamiast perplexity)

To PODŁOGA PAMIĘCI: interpolowany n-gram nie generalizuje poza widziane konteksty.
GPT, który ją bije na TYM SAMYM held-out, robi coś więcej niż look-up tablicowy.
Wynik -> ngram_obsidian.json (czyta go eval_obsidian.py do wspólnej tabeli 3 wierszy).

Użycie: python ngram_obsidian.py
"""
import os, sys, json, math, collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = 6
K = 0.01                                # add-k floor unigramowy

train = open(os.path.join(HERE, "data/korpus-obsidian-raw.txt"), encoding="utf-8").read()
val = open(os.path.join(HERE, "data/heldout-raw.txt"), encoding="utf-8").read()
V = len(set(train) | set(val))          # słownik = pełny alfabet (train ∪ held-out)
print(f"znaki: train {len(train):,} / val {len(val):,} | słownik {V}")

# zliczanie kontekstów rzędów 0..ORDER na train
counts = {o: collections.defaultdict(collections.Counter) for o in range(ORDER + 1)}
for j in range(len(train)):
    ch = train[j]
    for o in range(ORDER + 1):
        if j >= o:
            counts[o][train[j - o:j]][ch] += 1
uni = counts[0][""]
uni_total = sum(uni.values())


def p_unigram(ch):
    return (uni[ch] + K) / (uni_total + K * V)


def interp_prob(history, ch, M):
    """P(ch | history) interpolacją rzędów 0..M, renormalizowaną po aktywnych."""
    num, Z = 0.0, 0.0
    for o in range(M + 1):
        w = 2.0 ** o
        if o == 0:
            num += w * p_unigram(ch)
            Z += w
        else:
            ctx = history[-o:]
            c = counts[o].get(ctx)
            tot = sum(c.values()) if c else 0
            if tot > 0:
                num += w * (c[ch] / tot)
                Z += w
    return num / Z


print(f"\n{'rząd M':>7} | {'bits/char':>9} | {'val ppl':>9} | {'#kontekstów (rząd M)':>20}")
results = {}
for M in range(1, ORDER + 1):
    nll, n = 0.0, 0
    for i in range(1, len(val)):
        ctx = val[max(0, i - ORDER):i]
        p = interp_prob(ctx, val[i], M)
        nll += -math.log(p)
        n += 1
    bpc = nll / n / math.log(2)
    ppl = math.exp(nll / n)
    results[M] = bpc
    print(f"{M:>7} | {bpc:>9.4f} | {ppl:>9.2f} | {len(counts[M]):>20,}")

bpc_final = results[ORDER]
out = {"model": "n-gram", "order": ORDER, "bits_per_char": bpc_final,
       "val_chars": len(val), "vocab": V}
json.dump(out, open(os.path.join(HERE, "ngram_obsidian.json"), "w", encoding="utf-8"), indent=2)
print(f"\nbaseline n-gram (M={ORDER}): {bpc_final:.4f} bits/char -> ngram_obsidian.json")
