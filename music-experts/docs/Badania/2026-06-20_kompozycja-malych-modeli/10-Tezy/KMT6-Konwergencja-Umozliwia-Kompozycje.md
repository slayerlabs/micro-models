---
type: teza
id: KMT6
title: "Konwergencja reprezentacji umożliwia tanią kompozycję i interoperacyjność (relative reps, zero-shot)"
status: w-dyskusji
data: 2026-06-21
created_at: 2026-06-21
parents: ["KMC1"]
author: Arkadiusz Słota
---

# KMT6 — Konwergencja umożliwia kompozycję

## Teza
Niezależnie trenowane mikro-modele **dzielą geometrię reprezentacji** (E_CKA: CKA jig–jig-v2 **0,97**, jig–waltz **0,85** vs null **0,35**). Skoro geometria jest wspólna, da się je **składać i interoperować zero-shot** — bez wspólnej kotwicy, bez joint-treningu — przez reprezentacje względne (relative representations, Moschella i in., ICLR 2023). To czyni „dodaj eksperta, od razu gra z resztą" wykonalnym tanio.

## Dowód wstępny
[[Kompozycja-Eksperymenty]] (E_CKA, 1 punkt skali, ~0,8M; sweep po skali w toku). Rama: [[Emergencja-i-Wspolna-Reprezentacja]].

## Antyteza
[[../15-Antytezy/KMAT6-Konwergencja-To-Redundancja|KMAT6]] — wspólna geometria = redundancja → nie ma czego składać.

## Powiązania
[[../00-Cele/KMC1-Cel-Kompozycja-Malych-Modeli|KMC1]] · synteza: [[../25-Syntezy/KMS6-Konwergencja-Vs-Komplementarnosc|KMS6]] · [[Emergencja-i-Wspolna-Reprezentacja]]
