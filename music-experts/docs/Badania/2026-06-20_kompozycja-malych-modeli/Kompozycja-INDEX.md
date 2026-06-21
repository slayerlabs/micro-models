---
type: index
id: KM-INDEX
title: "Kompozycja małych modeli — warsztat dialektyczny"
status: aktywny
data: 2026-06-20
created_at: 2026-06-20
author: Arkadiusz Słota
---

# Kompozycja małych modeli — warsztat dialektyczny

> Czy wielu ekstremalnie małych, poprawnych ekspertów + wymuszony kontrakt (na reprezentacji) + mapper + wariancja = tania, składalna alternatywa dla dużego modelu? Sandbox: muzyka → IFC. Repo: github.com/slayerlabs/micro-models.

## Cel
- [[00-Cele/KMC1-Cel-Kompozycja-Malych-Modeli|KMC1]] — ramy + metoda (E0 → E0.5 → E1 Bach → E2)

## Nitki dialektyczne
| # | Teza | Antyteza | Synteza |
|---|------|----------|---------|
| 1 | [[10-Tezy/KMT1-Kompozycja-Malych\|KMT1]] kompozycja małych | [[15-Antytezy/KMAT1-Jeden-Wiekszy\|KMAT1]] jeden większy | [[25-Syntezy/KMS1-Kompozycja-Gdy-Kontrakt\|KMS1]] |
| 2 | [[10-Tezy/KMT2-Kontrakt-W-Treningu\|KMT2]] kontrakt w treningu | [[15-Antytezy/KMAT2-Wyrownanie-Posthoc\|KMAT2]] post-hoc | [[25-Syntezy/KMS2-Kontrakt-Vs-Posthoc\|KMS2]] |
| 3 | [[10-Tezy/KMT3-Slownik-Z-Aktywacji\|KMT3]] słownik z aktywacji | [[15-Antytezy/KMAT3-Liniowy-Mapper\|KMAT3]] liniowy mapper | [[25-Syntezy/KMS3-Liniowy-Najpierw\|KMS3]] |
| 4 | [[10-Tezy/KMT4-Routing-Przez-Wariancje\|KMT4]] routing przez wariancję | [[15-Antytezy/KMAT4-Nauczony-Router\|KMAT4]] nauczony router | [[25-Syntezy/KMS4-Wariancja-Plus-Router\|KMS4]] |
| 5 | [[10-Tezy/KMT5-Meta-Atencja\|KMT5]] meta-atencja | [[15-Antytezy/KMAT5-Plaskie-Wazenie\|KMAT5]] płaskie ważenie | [[25-Syntezy/KMS5-Meta-Gdy-Trudne-Obszary\|KMS5]] |
| 6 | [[10-Tezy/KMT6-Konwergencja-Umozliwia-Kompozycje\|KMT6]] konwergencja umożliwia kompozycję | [[15-Antytezy/KMAT6-Konwergencja-To-Redundancja\|KMAT6]] konwergencja = redundancja | [[25-Syntezy/KMS6-Konwergencja-Vs-Komplementarnosc\|KMS6]] (z E_CKA) |

Syntezy = **propozycje; rozstrzygane POMIAREM** (E0/E0.5/E1/E2), nie przekonaniem.

## Rozstrzygnięty widelec
**Kontrakt = styk na reprezentacji pośredniej, nie na logitach** (styk na logitach → ensembling, nasza nisza znika).

## Overview + referencje
- [[Kompozycja-Malych-Modeli]] — opis koncepcji + **referencje zweryfikowane u źródeł** (deep-research round 1 + dorzut A, 2026-06-20; wszystkie charakterystyki trafne).
- [[Emergencja-i-Wspolna-Reprezentacja]] — **rama teoretyczna** (NOWE 2026-06-21): warunki konieczne emergencji, czym zastąpić „jednocześnie", prawo zachowania kotwicy, świeży front (relative reps, Platonic, task arithmetic, CKA — cytaty zweryfikowane).
- Rejestr badań: [[Badania-INDEX]] · sąsiad: [[Research-NGram-vs-MiniTransformer]]

## Plan eksperymentów (kolejność wg wartość/wysiłek — zaktualizowano 2026-06-21)
- ✅ **E0** self-stitch — waliduje mechanizm mappera (nie tezę). PASS.
- ✅ **E1** stitch reprezentacji walc×reel — **stitch ≈ ensemble** (negatyw; post-hoc liniowy, bez kontraktu). → [[Kompozycja-Eksperymenty]].
- ⏭️ **E_CKA** (NOWY, tani ~h, decydujący) — CKA/mutual-kNN między niezależnymi modelami: czy mają wspólną geometrię (platońska konwergencja na mikro-skali). **Sweep po skali + baseline.** → [[Emergencja-i-Wspolna-Reprezentacja]].
- **Pre-check wariancja vs CE** (tani) — czy wariancja koreluje z błędem; bramka przed routingiem.
- **Kontrakt shared-trunk** ([[25-Syntezy/KMS2-Kontrakt-Vs-Posthoc|KMS2]]) — wspólny zamrożony front+głowa, trenuj tylko tyły, na domenach w **TYM SAMYM metrum** (walc vs mazur — ⚠️ pułapka metrum). Rozstrzyga tezę kompozycji.
- **Bogatszy mapper** ([[10-Tezy/KMT3-Slownik-Z-Aktywacji|KMT3]]) / słownik / relative reps — dopiero jeśli kontrakt pokaże iskrę.
- **E0.5** (niezależny seed, jig-v2) — aktywny test wspólnej geometrii przez złączenie (E_CKA mierzy to pasywnie).
- Oś prostopadła: [[Granie-Razem-Polifonia]] (polifonia, re-sync). Domena: muzyka → potem IFC ([[Cele-Globalne-i-Kotwica]]).
