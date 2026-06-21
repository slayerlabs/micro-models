"""Baseline n-gram — PERPLEXITY na tym samym split co GPT/NPLM (do spektrum w paperze).

Interpolacja liniowa rzędów 0..M (wagi ∝ 2^o, renormalizowane po rzędach AKTYWNYCH),
z floorem unigramowym add-k → poprawny rozkład (Σ_ch P = 1, brak zer → skończone ppl).
Sweep M=1..6: ppl + liczba kontekstów (pamięć) → pokazuje skalowanie i eksplozję pamięci.

Korpus/split/słownik jak jig GPT (3,80) i NPLM. Użycie: python src/train/ngram_eval.py
"""
import sys, math, collections

sys.stdout.reconfigure(encoding="utf-8")
ORDER = 6
K = 0.01                       # add-k dla floora unigramowego

text = open("data/jigs.abc", encoding="utf-8").read()
chars = sorted(set(text))
V = len(chars)
n90 = int(0.9 * len(text))
train, val = text[:n90], text[n90:]
print(f"znaki: {len(text):,} | słownik {V} | train {len(train):,} / val {len(val):,}")

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


print(f"\n{'rząd M':>7} | {'val ppl':>9} | {'#kontekstów (rząd M)':>20} | {'#wszystkie ≤M':>13}")
for M in range(1, ORDER + 1):
    nll, n = 0.0, 0
    for i in range(1, len(val)):
        ctx = val[max(0, i - ORDER):i]          # tylko ostatnie ≤ORDER znaków (bez kopii O(n²))
        p = interp_prob(ctx, val[i], M)
        nll += -math.log(p)
        n += 1
    ppl = math.exp(nll / n)
    ctx_M = len(counts[M])
    ctx_all = sum(len(counts[o]) for o in range(M + 1))
    print(f"{M:>7} | {ppl:>9.2f} | {ctx_M:>20,} | {ctx_all:>13,}")

print("\nodniesienie: NPLM (okno 8) ppl ~4,41 | GPT (atencja, okno 128) ppl 3,80")
