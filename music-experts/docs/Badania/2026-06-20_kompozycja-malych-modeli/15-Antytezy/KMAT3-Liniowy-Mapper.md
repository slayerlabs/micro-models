---
type: antyteza
id: KMAT3
title: "Prosty liniowy mapper (Procrustes) wystarczy — tańszy i stabilniejszy niż uczony słownik"
status: w-dyskusji
data: 2026-06-20
created_at: 2026-06-20
parents: ["KMT3"]
author: Arkadiusz Słota
---

# KMAT3 — Liniowy mapper (steelman)

## Antyteza
Po co uczyć słownik cech, skoro **liniowa transformacja** (regresja / Procrustes alignment) między przestrzeniami często wystarcza? Tańsze, stabilne, interpretowalne, mało parametrów, brak dodatkowego treningu słownika.

## Granica
Liniowy mapper obejmuje **zgrubne** struktury (domena, routing), ale nie subtelną, nieliniową relację cech — przy bogatszym styku może nie wystarczyć.

## Co rozstrzyga
Czy liniowy baseline **wystarcza** dla naszej jakości — mierzone. Słownik tylko, gdy liniowy zawodzi. → [[../25-Syntezy/KMS3-Liniowy-Najpierw|KMS3]]

## Powiązania
teza: [[../10-Tezy/KMT3-Slownik-Z-Aktywacji|KMT3]]
