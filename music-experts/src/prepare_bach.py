"""Monofoniczny korpus Bacha (ABC) z linii SOPRANU chorałów (music21, lokalnie).
Chorały są 4-głosowe (SATB) — bierzemy TYLKO sopran => monofonia, zgodna z kontraktem.
Własny enkoder mono->ABC (music21 nie umie pisać ABC); słownik zgodny z jigami.
"""
import sys, os
from fractions import Fraction
from music21 import corpus

def pitch_to_abc(p):
    step, octv = p.step, p.octave
    acc = ""
    if p.accidental is not None:
        a = p.accidental.alter
        acc = {1: "^", -1: "_", 0: "=", 2: "^^", -2: "__"}.get(int(a), "")
    if octv >= 5:
        return acc + step.lower() + "'" * (octv - 5)
    return acc + step.upper() + "," * (4 - octv)

def dur_to_abc(ql):
    units = Fraction(ql).limit_denominator(8) / Fraction(1, 2)  # L:1/8 => ósemka=1
    if units == 1: return ""
    if units.denominator == 1: return str(units.numerator)
    if units.numerator == 1: return "/" + str(units.denominator)
    return f"{units.numerator}/{units.denominator}"

def chorale_to_abc(score):
    sop = score.parts[0]
    # tonacja z sygnatury (szybciej niż analyze)
    kname = "C"
    ks = sop.recurse().getElementsByClass('KeySignature')
    if ks:
        try:
            k = ks[0].asKey()
            kname = k.tonic.name.replace('-', 'b') + ("" if k.mode == "major" else "m")
        except Exception:
            pass
    meter = "4/4"
    ts = sop.recurse().getElementsByClass('TimeSignature')
    if ts:
        meter = ts[0].ratioString
    toks = []
    for m in sop.getElementsByClass('Measure'):
        for el in m.notesAndRests:
            if el.isRest:
                toks.append("z" + dur_to_abc(el.quarterLength))
            elif el.isChord:
                top = max(el.notes, key=lambda x: x.pitch.midi)
                toks.append(pitch_to_abc(top.pitch) + dur_to_abc(el.quarterLength))
            else:
                toks.append(pitch_to_abc(el.pitch) + dur_to_abc(el.quarterLength))
        toks.append("|")
    return f"X:1\nM:{meter}\nL:1/8\nK:{kname}\n" + " ".join(toks)

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    paths = corpus.getComposer('bach')
    blocks, n = [], 0
    for p in paths:
        n += 1
        try:
            sc = corpus.parse(p)
            if len(sc.parts) < 1:
                continue
            abc = chorale_to_abc(sc)
            if 40 < len(abc) < 2500 and abc.count("|") >= 4:
                blocks.append(abc)
        except Exception:
            continue
        if len(blocks) >= 350:
            break
    text = "\n\n".join(blocks)
    os.makedirs("data/corpus", exist_ok=True)
    with open("data/corpus/bach.abc", "w", encoding="utf-8") as f:
        f.write(text)
    vocab = sorted(set(text))
    print(f"przetworzono plików: {n} | zachowanych bloków (sopran): {len(blocks)}")
    print(f"znaki: {len(text):,} | słownik: {len(vocab)}")
    print("słownik:", repr("".join(vocab)))
    print("\n--- pierwszy blok ---")
    print(blocks[0] if blocks else "BRAK")

if __name__ == "__main__":
    main()
