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

Syntezy = **propozycje; rozstrzygane POMIAREM** (E0/E0.5/E1/E2), nie przekonaniem.

## Rozstrzygnięty widelec
**Kontrakt = styk na reprezentacji pośredniej, nie na logitach** (styk na logitach → ensembling, nasza nisza znika).

## Overview + referencje
- [[Kompozycja-Malych-Modeli]] — opis koncepcji + **referencje zweryfikowane u źródeł** (deep-research round 1 + dorzut A, 2026-06-20; wszystkie charakterystyki trafne).
- Rejestr badań: [[Badania-INDEX]] · sąsiad: [[Research-NGram-vs-MiniTransformer]]

## Plan eksperymentów
- **E0** self-stitch — waliduje mechanizm mappera (nie tezę); benchmark zerowy.
- **E0.5** zszycie z **niezależnym seedem** — czy kontrakt wymusza wspólną geometrię (bramka przed E1).
- **E1** drugi model, ten sam kontrakt, inny styl: **BACH** (BWV 1007–1012 + 1001–1006, monofonia) + **Slayer-style** (kontrast). Mierz perplexity/jakość vs osobno.
- **Pre-check** (przed E2): wariancja vs CE per token — czy wariancja koreluje z błędem.
- **E2** wariancja-routing — czy przewyższa naiwne mieszanie.
- Domena: muzyka (darmowe nuty) → potem IFC.
