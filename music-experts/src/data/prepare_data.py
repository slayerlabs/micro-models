"""Przygotowanie korpusu ABC z thesession.org tunes.csv.
Filtr: typ + metrum z argumentów (domyślnie jigi 6/8). Buduje grające bloki ABC + normalizuje tonację.
--mixed: bez filtra typu/metrum (uniwersalista, nagłówek M: z wiersza).
Użycie: python src/data/prepare_data.py [typ] [metrum] [wyjście] [--mixed]
  np. python src/data/prepare_data.py waltz 3/4 data/corpus/waltz.abc
  python src/data/prepare_data.py --mixed data/mixed.abc
"""
import csv, sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.abc_corpus import ALLOWED, clean_abc, norm_key

csv.field_size_limit(10**7)

ap = argparse.ArgumentParser()
ap.add_argument("type_kw", nargs="?", default="jig")
ap.add_argument("meter", nargs="?", default="6/8")
ap.add_argument("out", nargs="?", default=None)
ap.add_argument("--mixed", action="store_true",
                help="cały korpus bez filtra typu/metrum (M: z wiersza)")
a = ap.parse_args()
if a.out is None:
    a.out = "data/mixed.abc" if a.mixed else "data/jigs.abc"

def main():
    rows_out, n_total, n_kept = [], 0, 0
    with open("data/tunes.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            n_total += 1
            if not a.mixed:
                if a.type_kw not in row["type"].lower():
                    continue
                if row["meter"].strip() != a.meter:
                    continue
            body = clean_abc(row["abc"])
            if not (40 <= len(body) <= 700):
                continue
            if any(ch not in ALLOWED for ch in body.replace("\n", "")):
                continue
            key = norm_key(row["mode"])
            meter = row["meter"].strip() if a.mixed else a.meter
            block = f"X:1\nM:{meter}\nK:{key}\n{body}\n"
            rows_out.append(block)
            n_kept += 1

    text = "\n".join(rows_out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)

    vocab = sorted(set(text))
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"melodii w pliku        : {n_total}")
    desc = "MIXED (bez filtra)" if a.mixed else f"{a.type_kw} ({a.meter})"
    print(f"{desc} zachowane   : {n_kept}")
    print(f"znaki łącznie          : {len(text):,}")
    print(f"słownik ({len(vocab)})        : {''.join(vocab)!r}")
    print("\n--- pierwszy blok ---")
    print(rows_out[0] if rows_out else "BRAK")

if __name__ == "__main__":
    main()
