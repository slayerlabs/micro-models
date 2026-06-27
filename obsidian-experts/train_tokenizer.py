"""train_tokenizer.py — byte-level BPE OD ZERA (L05).

Algorytm (L05): bajty → najczęstsza sąsiednia para → sklej w nowy token → powtórz ×N.
Tokenizer LICZY (statystyka, deterministycznie), NIE przewiduje — to robi model (softmax).
Trenowany RAZ, używany stale (lock-in, L02). Trenuj na SWOIM języku (PL) → polskie cząstki
(„nie", „ie", „województw") dostają własne tokeny → mniej tokenów na polski tekst.

Dowód (L05, „żadnej tezy bez dowodu"): na końcu mierzymy znaki/token nasz vs GPT-2 (tiktoken).

Użycie:
  python train_tokenizer.py --corpus data/korpus-pl.txt --merges 512 --out tokenizer.json
"""
import argparse, json, os, time
from collections import Counter


def get_stats(ids):
    """Policz wystąpienia każdej sąsiedniej pary (czysta statystyka — L05)."""
    counts = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def merge(ids, pair, new_id):
    """Zastąp każde wystąpienie pary nowym tokenem."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


def train(text, num_merges):
    ids = list(text.encode("utf-8"))          # start: bajty 0..255 (byte-level)
    merges = []                               # kolejność ma znaczenie
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        if stats[pair] < 2:                   # nie ma już co sklejać
            break
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges.append([pair[0], pair[1], new_id])
    return merges


def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for a, b, nid in merges:
        vocab[nid] = vocab[a] + vocab[b]
    return vocab


def encode(text, merges):
    ids = list(text.encode("utf-8"))
    for a, b, nid in merges:                  # stosuj scalenia w wyuczonej kolejności
        ids = merge(ids, (a, b), nid)
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/korpus-pl.txt")
    ap.add_argument("--merges", type=int, default=512)
    ap.add_argument("--out", default="tokenizer.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    corpus = os.path.join(here, args.corpus)
    text = open(corpus, encoding="utf-8").read()

    t0 = time.time()
    merges = train(text, args.merges)
    vocab_size = 256 + len(merges)
    json.dump({"merges": merges, "vocab_size": vocab_size},
              open(os.path.join(here, args.out), "w", encoding="utf-8"))
    print(f"BPE: {len(merges)} scaleń, vocab={vocab_size}, {time.time()-t0:.1f}s -> {args.out}")

    # --- DOWÓD (L05): znaki/token nasz vs GPT-2 ---
    ids = encode(text, merges)
    nasz = len(text) / len(ids)
    print(f"\nDOWÓD (L05) — gęstość na korpusie PL:")
    print(f"  nasz BPE:  {nasz:.2f} znaki/token  ({len(ids):,} tokenów)")
    try:
        import tiktoken
        gpt2 = tiktoken.get_encoding("gpt2")
        gids = gpt2.encode(text)
        g = len(text) / len(gids)
        print(f"  GPT-2 BPE: {g:.2f} znaki/token  ({len(gids):,} tokenów)")
        print(f"  → nasz tokenizer pakuje PL {nasz/g:.2f}× gęściej (mniej tokenów = taniej).")
    except ImportError:
        print("  (tiktoken niezainstalowany — `pip install tiktoken` by porównać z GPT-2)")

    # sanity: encode->decode odwracalne
    assert decode(ids, build_vocab(merges)) == text, "BPE nie jest odwracalne!"
    print("\n  ✓ encode→decode odwracalne (100%).")


if __name__ == "__main__":
    main()
