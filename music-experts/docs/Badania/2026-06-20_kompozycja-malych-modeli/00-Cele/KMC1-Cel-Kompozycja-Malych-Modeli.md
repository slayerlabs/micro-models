---
type: cel
id: KMC1
title: "Zbadać, czy kompozycja małych modeli (wymuszony kontrakt + mapper + wariancja) daje tanią, składalną alternatywę dla jednego dużego modelu"
status: aktywny
realizacja: w-planowaniu
data: 2026-06-20
created_at: 2026-06-20
author: Arkadiusz Słota
children: ["KMT1", "KMT2", "KMT3", "KMT4", "KMT5"]
---

# KMC1 — Cel: kompozycja małych modeli

## Cel
Sprawdzić empirycznie, czy **wielu ekstremalnie małych, poprawnych ekspertów** spiętych **stałym interfejsem (kontraktem)** + lekkim mapperem **przewyższa lub dorównuje** jednemu większemu modelowi — taniej i w sposób składalny. Sygnałem trudnych obszarów (styku) ma być **wariancja** między ekspertami.

## Ramy (dane, nie do sporu)
- **Ekstremalnie małe modele** — jak nasz `slay-piano-gpt` (0,82M, generuje poprawne melodie). „Mały + poprawny" = podstawowy moduł kompozycji.
- **Styk = reprezentacja pośrednia** (rozstrzygnięty widelec logity/reprezentacja). Wspólna `lm_head` tylko dla porównywalności wyjścia; mapper/słownik/wariancja działają na reprezentacji. Styk na logitach zredukowałby program do ensemblingu.
- **Sandbox = muzyka** (nuty = otwarty, stale dostępny zbiór danych) → docelowo **IFC** (auto-cleaner, osobny temat).
- **Repo programu:** github.com/slayerlabs/micro-models.
- **Metoda:** E0 self-stitch (waliduje mechanizm, nie tezę) → **E0.5 zszycie z niezależnym seedem** (czy kontrakt wymusza wspólną geometrię) → **E1 = BACH** (inny styl, ten sam kontrakt; + Slayer-style jako kontrast) → E2 wariancja-routing (po pre-checku kalibracji).
- **Uczciwość:** każde twierdzenie oznaczone proven/spekulacja; referencje **zweryfikowane u źródeł** (deep-research, 2026-06-20).

## Osie dialektyczne
1. [[../10-Tezy/KMT1-Kompozycja-Malych|KMT1]] ↔ [[../15-Antytezy/KMAT1-Jeden-Wiekszy|KMAT1]] → [[../25-Syntezy/KMS1-Kompozycja-Gdy-Kontrakt|KMS1]]
2. [[../10-Tezy/KMT2-Kontrakt-W-Treningu|KMT2]] ↔ [[../15-Antytezy/KMAT2-Wyrownanie-Posthoc|KMAT2]] → [[../25-Syntezy/KMS2-Kontrakt-Vs-Posthoc|KMS2]]
3. [[../10-Tezy/KMT3-Slownik-Z-Aktywacji|KMT3]] ↔ [[../15-Antytezy/KMAT3-Liniowy-Mapper|KMAT3]] → [[../25-Syntezy/KMS3-Liniowy-Najpierw|KMS3]]
4. [[../10-Tezy/KMT4-Routing-Przez-Wariancje|KMT4]] ↔ [[../15-Antytezy/KMAT4-Nauczony-Router|KMAT4]] → [[../25-Syntezy/KMS4-Wariancja-Plus-Router|KMS4]]
5. [[../10-Tezy/KMT5-Meta-Atencja|KMT5]] ↔ [[../15-Antytezy/KMAT5-Plaskie-Wazenie|KMAT5]] → [[../25-Syntezy/KMS5-Meta-Gdy-Trudne-Obszary|KMS5]]

## Powiązania
Overview: [[Kompozycja-Malych-Modeli]] · rejestr: [[Badania-INDEX]] · sąsiad: [[Research-NGram-vs-MiniTransformer]]
