---
type: teza
id: KMT1
title: "Komponuj wielu małych, poprawnych ekspertów + lekki mapper — zamiast jednego dużego modelu"
status: w-dyskusji
data: 2026-06-20
created_at: 2026-06-20
parents: ["KMC1"]
author: Arkadiusz Słota
---

# KMT1 — Kompozycja małych ekspertów

## Teza
Małe, **poprawne** moduły (nasz pianino-gpt: 0,82M, generuje poprawne melodie bez dodatkowych usprawnień) + **lekki mapper** = tania, składalna alternatywa dla dużego modelu. Unikamy kosztownego, „każdy-przez-każdy" mieszania całych wag. Pojemność rośnie przez dokładanie ekspertów, nie przez rozrost jednego modelu.

## Antyteza
[[../15-Antytezy/KMAT1-Jeden-Wiekszy|KMAT1]] — jeden większy model jest prostszy i ma **wspólną reprezentację**; kompozycja traci spójność na styku.

## Powiązania
[[../00-Cele/KMC1-Cel-Kompozycja-Malych-Modeli|KMC1]] · synteza: [[../25-Syntezy/KMS1-Kompozycja-Gdy-Kontrakt|KMS1]]
