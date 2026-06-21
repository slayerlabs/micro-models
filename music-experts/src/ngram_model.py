"""Baseline: char-level n-gram z backoffem (czysty Python, bez bibliotek).
Uczy się 'następny znak' z korpusu ABC i generuje nowe jigi.
To samo zadanie co LLM (next-token), tu na znakach i bez sieci neuronowej.
"""
import random, sys, collections

ORDER = 6                      # ile znaków kontekstu (max)
CTX = list(range(ORDER, -1, -1))  # [6,5,4,3,2,1,0] — backoff od najdłuższego

def train(text):
    models = {c: collections.defaultdict(collections.Counter) for c in CTX}
    for i, ch in enumerate(text):
        for c in CTX:
            ctx = text[i - c:i] if c <= i else None
            if ctx is None and c != 0:
                continue
            models[c][text[i - c:i]][ch] += 1
    return models

def sample_next(models, history, temp):
    for c in CTX:                          # backoff: najdłuższy znaleziony kontekst
        ctx = history[-c:] if c > 0 else ""
        counter = models[c].get(ctx)
        if counter:
            chars = list(counter)
            weights = [counter[ch] ** (1.0 / temp) for ch in chars]
            return random.choices(chars, weights=weights)[0]
    return "\n"

def generate(models, seed, temp=0.8, maxlen=600, minbody=90):
    out = seed
    while len(out) < maxlen:
        ch = sample_next(models, out, temp)
        out += ch
        body = out[len(seed):]
        if out.endswith("\n\n") and len(body) >= minbody:  # koniec melodii
            break
    return out.strip()

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    text = open("data/jigs.abc", encoding="utf-8").read()
    print(f"trenuję n-gram (order={ORDER}) na {len(text):,} znakach…")
    models = train(text)
    print("gotowe. Konteksty na poziom:",
          {c: len(models[c]) for c in CTX})

    seed = "X:1\nM:6/8\nK:D\n"
    random.seed(20260620)
    outs = []
    for i in range(3):
        tune = generate(models, seed, temp=0.7)
        outs.append(tune)
        print(f"\n========== WYGENEROWANY JIG #{i+1} ==========")
        print(tune)

    with open("data/generated.abc", "w", encoding="utf-8") as f:
        f.write("\n\n".join(outs) + "\n")
    print("\n[zapisano do data/generated.abc]")

if __name__ == "__main__":
    main()
