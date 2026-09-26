"""Wspólne narzędzia czyszczenia ABC (używane przez prepare_data.py i judge).
Przeniesione 1:1 z prepare_data.py + parser ciała melodii dla judge'a.
"""
import re

MODE_TABLE = {
    "major": "", "ionian": "", "minor": "min", "aeolian": "min",
    "dorian": "dor", "mixolydian": "mix", "phrygian": "phr",
    "lydian": "lyd", "locrian": "loc", "": "",
}

ALLOWED = set("ABCDEFGabcdefg0123456789|:[]()<>/'^_=.,~- zZxX")

HEADER_PREFIXES = ("X:", "T:", "C:", "M:", "K:", "N:", "%")


def norm_key(mode: str) -> str:
    m = re.match(r"^([A-Ga-g][#b]?)(.*)$", mode.strip())
    if not m:
        return "C"
    root, word = m.group(1), m.group(2).lower()
    return root + MODE_TABLE.get(word, "")


def clean_abc(body: str) -> str:
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    body = re.sub(r'"[^"]*"', "", body)       # usuń symbole akordów / adnotacje "..."
    body = re.sub(r"[ \t]+", " ", body)        # scal podwójne spacje po usunięciu
    body = re.sub(r"\n+", "\n", body)
    return body


def tune_body(abc: str) -> str:
    """Ciało melodii bez linii nagłówkowych (X:/T:/C:/M:/K:/N:/%)."""
    body = "\n".join(ln for ln in abc.split("\n") if not ln.startswith(HEADER_PREFIXES))
    return clean_abc(body)


def parse_tune_row(row: dict) -> tuple[str, str, str, str]:
    """(body, meter, mode_label, type_label) z wiersza tunes.csv; mode normalizowane."""
    body = tune_body(row["abc"])
    return body, row["meter"].strip(), norm_key(row["mode"]), row["type"].strip().lower()
