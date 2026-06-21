---
type: synteza
id: KMS3
title: "PROPOZYCJA — najpierw liniowy mapper (baseline); uczony słownik (VQ/SAE) tylko gdy liniowy nie wystarcza"
status: propozycja
data: 2026-06-20
created_at: 2026-06-20
parents: ["KMT3", "KMAT3"]
author: Arkadiusz Słota
---

# KMS3 — Liniowy najpierw, słownik gdy trzeba

## Synteza
Zasada Ockhama (preferuj prostsze): **zacznij od liniowego mappera** (Procrustes/regresja) jako baseline — tani, stabilny, szybki do sprawdzenia. **Uczony słownik cech** (VQ/SAE, koherencja aktywacji) wprowadzamy **dopiero, gdy liniowy nie osiąga** wymaganej jakości. Nie odwrotnie. Słownik to nasza nisza badawcza, ale musi *zasłużyć* na złożoność wynikiem.

## Kryterium / decyzja
Liniowy baseline vs próg jakości. Jeśli liniowy wystarcza → koniec. Jeśli nie → słownik (i mierzymy przyrost vs koszt).

## Powiązania
[[../10-Tezy/KMT3-Slownik-Z-Aktywacji|KMT3]] ↔ [[../15-Antytezy/KMAT3-Liniowy-Mapper|KMAT3]]
