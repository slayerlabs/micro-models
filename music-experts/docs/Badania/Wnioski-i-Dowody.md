---
type: nauka-wnioski
title: "Wnioski i dowody — micro-models (muzyka + kompozycja)"
status: zywy-dokument
data: 2026-06-21
created_at: 2026-06-21
author: Arkadiusz Słota
---

# Wnioski i dowody

> Skonsolidowany rejestr: **twierdzenie → dowód (liczba/źródło) → werdykt.** Zasada: żadnej tezy bez dowodu. Żywy dokument.

## ✅ Udowodnione (z dowodem)
| Twierdzenie | Dowód | Werdykt |
|---|---|---|
| Char-LM uczy się struktury muzycznej (metrum, tonacje, kadencje) z samej predykcji następnego znaku | 4 modele generują poprawne ABC; respektują metrum i sygnatury (np. `^F` w G-dur) — bez teorii w kodzie | ✅ |
| Czyszczenie danych obniża perplexity | jig: ppl **3,88 → 3,80** po usunięciu szumu akordowego | ✅ zmierzone |
| `[ace]` musi zostać z nawiasami, by grało jako akord | test: 2 akordy w ABC → 2 akordy w MIDI | ✅ |
| Mechanizm stitchu jest bezstratny (E0) | g=identyczność == baseline co do cyfry; g losowy (ppl **122**) → trening 16K param → ppl **3,83** (Δ+0,03) | ✅ |
| **Ensemble ekspertów bije pojedyncze modele** (zadanie mieszane) | zbalansowany held-out: ensemble **5,15** < reel 5,87 < walc 6,20 | ✅ zmierzone |

## ❌ NIE udowodnione (jeszcze)
| Hipoteza | Co wyszło | Status |
|---|---|---|
| **Stitch reprezentacji bije ensemble** (KMS1) | E1: stitch **5,18 ≈ ensemble 5,15** (liniowy g, post-hoc, bez kontraktu) | ❌ niepotwierdzone tą drogą |
| Most n-gram→transformer (NPLM, kNN-LM jako ogniwa) | eksperymenty niezrobione | ⏳ planowane |
| Mała skala (Bach 79K, Slayer ~150 utw.) wystarcza na mocny model | za mało danych | ⏳ |

## 🔬 Lekcje metodologiczne
- **Pomiar trzeba sprawdzać:** pierwszy E1 dał „stitch bije ensemble" — **artefakt** (val = czysty reel ze sklejenia korpusów). Po zbalansowaniu: ensemble nie gorszy. *Setup ewaluacji = część dowodu.*
- **Negatywny wynik to wynik:** prosty baseline (ensemble) jest mocny; złożoność (stitch) musi **zasłużyć** na siebie.
- **Bach ppl 2,09 nieporównywalne** z jig 3,80 (mniejszy słownik + bardzo powtarzalne dane).
- **Pułapka metrum:** style w różnym metrum (jig 6/8, walc 3/4, reel 4/4) są rozłączne po nagłówku `M:` → router pokaże ~100% trafności ściągając z JEDNEGO tokena, nie z reprezentacji. Test routingu MUSI mieć domeny w **tym samym metrum** (walc vs mazur). *Dotyczy routingu, nie E1.*
- **Symetria E1:** stitch kierunkowy (głowa reela) vs ensemble symetryczny — porównanie lekko jabłka/gruszki; uczciwszy test = symetryczna fuzja reprezentacji do wspólnej głowy.
- **Prawo zachowania kotwicy:** E1 padł, bo nie było wspólnego podłoża; wspólna geometria **nie emerguje za darmo** z niezależnych modeli — ktoś musi raz zapłacić za szeroką kotwicę. ([[Emergencja-i-Wspolna-Reprezentacja]])

## 📚 Referencje zweryfikowane u źródeł (deep-research: 13/13 charakterystyk trafnych)
ZipIt! (ICLR'24) · Git Re-Basin (ICLR'23) · model stitching (Lenc-Vedaldi CVPR'15; Bansal NeurIPS'21) · SN-Net (CVPR'23) · VQ-VAE (NeurIPS'17) · SAE / Towards Monosemanticity (Bricken, Cunningham '23) · Rosetta Neurons (ICCV'23) · AoANet (ICCV'19) · deep ensembles (NeurIPS'17) · MEMIT (ICLR'23) + Hase i in. (NeurIPS'23) · NPLM (JMLR'03) · kNN-LM (ICLR'20, ppl 15,79) · Infini-gram (COLM'24).

## ➡️ Następne dowody do zdobycia (kolejność wg wartość/wysiłek)
1. **E_CKA (NOWY, ~godzina, decydujący)** — CKA/mutual-kNN między 4 modelami: czy niezależne maluchy mają wspólną geometrię (platońska konwergencja na mikro-skali). **Sweep po skali + baseline** (inaczej nieinterpretowalne). Rozstrzyga w obie strony, tłumaczy E1. → [[Emergencja-i-Wspolna-Reprezentacja]].
2. **Pre-check wariancja vs CE** (tani) — czy wariancja koreluje z błędem per token; bramka przed routingiem.
3. **Kontrakt shared-trunk (KMS2)** — wspólny zamrożony front+głowa, trenuj tylko tyły, na domenach w **tym samym metrum** (walc vs mazur — ⚠️ pułapka metrum). Rozstrzyga tezę kompozycji.
4. **Bogatszy mapper (KMT3)** — słownik/SAE / relative representations — dopiero jeśli (3) pokaże iskrę.
5. **Most n-gram→transformer (NPLM)** — równolegle, pewny rezultat → paper Kacpra ([[Research-NGram-vs-MiniTransformer]]).

> Oś prostopadła (osobny kierunek): [[Granie-Razem-Polifonia]]. Kotwica produktu: [[Cele-Globalne-i-Kotwica]].

## Powiązania
[[Kompozycja-Eksperymenty]] · [[Kompozycja-Malych-Modeli]] · [[Capstone-Muzyka-LOG-Realizacja]] · [[Research-NGram-vs-MiniTransformer]] · [[Badania-INDEX]]
