"""Konwersja wygenerowanych melodii ABC -> MIDI (fortepian) przez music21.
Każda melodia osobno; błędy parsowania nie wywalają całości.
"""
import sys, re
from music21 import converter, instrument

def sanitize(abc: str) -> str:
    """Usuwa powtórzenia/volty z ciała, ALE ZOSTAWIA akordy [ace]. Nagłówki nietknięte."""
    out = []
    for ln in abc.split("\n"):
        if re.match(r"^[A-Za-z]:", ln):       # nagłówek
            out.append(ln); continue
        b = re.sub(r"\[[12]", "", ln)         # volty [1 [2  (akordy [ace] ZOSTAJĄ!)
        b = b.replace(":", "")                # dwukropki powtórzeń (|: :| ::)
        b = re.sub(r"\|\s*[12]", "|", b)      # numery volt po kresce |1 |2
        b = re.sub(r"\|+", "|", b)            # scal wielokrotne kreski
        b = re.sub(r"\^=|=\^|_=|=_", lambda m: m.group(0).replace("=", ""), b)  # zepsute akcydencje (^=)
        out.append(b)
    return "\n".join(out)

def to_midi(abc: str, out_path: str, inst=None) -> bool:
    try:
        score = converter.parse(sanitize(abc), format="abc")
        # spłaszcz do samych nut/pauz (omija konflikt TimeSignature przy budowie taktów)
        notes = score.flatten().notesAndRests.stream()
        if inst is not None:                  # ustaw instrument (np. Violin); domyślnie fortepian (GM 0)
            notes.insert(0, inst)
        notes.write("midi", fp=out_path)
        return True
    except Exception as e:
        print(f"  [pominięto] {out_path}: {type(e).__name__}: {e}")
        return False

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    infile = sys.argv[1] if len(sys.argv) > 1 else "data/generated.abc"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "data/jig"
    text = open(infile, encoding="utf-8").read()
    tunes = [t.strip() for t in text.split("\n\n") if "X:" in t]
    print(f"melodii do konwersji: {len(tunes)} (z {infile})")
    ok = 0
    for i, tune in enumerate(tunes, 1):
        path = f"{prefix}_{i}.mid"
        if to_midi(tune, path):
            print(f"  OK -> {path}")
            ok += 1
    print(f"\ngotowe: {ok}/{len(tunes)} plików MIDI (fortepian) w data/")

if __name__ == "__main__":
    main()
