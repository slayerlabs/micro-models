"""Metryki do raportu: rozkład znaków, nowość (kopiuje vs komponuje), długości."""
import sys, collections, json

sys.stdout.reconfigure(encoding="utf-8")
train = open("data/jigs.abc", encoding="utf-8").read()
gen = open("data/generated.abc", encoding="utf-8").read()

# ciało generacji bez nagłówków (do metryki nowości)
def bodies(text):
    return "".join(ln for ln in text.split("\n")
                    if ln and not ln[:2] in ("X:", "M:", "K:"))

gen_body = bodies(gen)

# 1) rozkład znaków (top 15) w treningu
freq = collections.Counter(c for c in train if c != "\n")
top = freq.most_common(15)

# 2) nowość: jaki % k-gramów generacji WYSTĘPUJE w treningu (kopiowanie)
novelty = {}
for k in (4, 6, 8):
    tr = set(train[i:i+k] for i in range(len(train)-k))
    gk = [gen_body[i:i+k] for i in range(len(gen_body)-k)]
    seen = sum(1 for g in gk if g in tr)
    novelty[k] = {"skopiowane_%": round(100*seen/len(gk), 1),
                  "nowe_%": round(100*(1-seen/len(gk)), 1)}
    del tr

# 3) długości wygenerowanych melodii
tunes = [t for t in gen.split("\n\n") if "X:" in t]
lens = [len(t) for t in tunes]

print(json.dumps({
    "korpus_znaki": len(train),
    "slownik": len(set(train)),
    "top_znaki": top,
    "nowosc": novelty,
    "wygenerowane_melodie": len(tunes),
    "srednia_dlugosc_znaki": round(sum(lens)/len(lens), 1),
}, ensure_ascii=False, indent=2))
