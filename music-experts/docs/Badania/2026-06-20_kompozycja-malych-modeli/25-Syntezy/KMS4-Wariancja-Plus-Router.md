---
type: synteza
id: KMS4
title: "PROPOZYCJA — wariancja jako tani sygnał diagnostyczny (gdzie trudno), nauczony router gdy trzeba twardej decyzji (hybryda)"
status: propozycja
data: 2026-06-20
created_at: 2026-06-20
parents: ["KMT4", "KMAT4"]
author: Arkadiusz Słota
---

# KMS4 — Wariancja + router (hybryda)

## Synteza
Nie „albo-albo": **wariancja/aktywacje** to świetny, tani **sygnał diagnostyczny** — mówi *gdzie* jest trudno (styk, niepewność) bez treningu. **Nauczony router** wchodzi tam, gdzie trzeba **twardej, generalizującej decyzji**. Czyli: wariancją lokalizujemy granicę (styk domen), routerem rozstrzygamy na tej granicy. Tani sygnał + mała jednostka decyzyjna dokładnie tam, gdzie potrzebna.

## Kryterium / decyzja
**Pre-check (zanim zbudujemy routing):** wariancja musi korelować z błędem — wykres **wariancja vs CE per token**. Przy malutkich ekspertach bywa zaszumiona i łapie „trudny input" zamiast granicy domen; brak korelacji → routing-przez-wariancję nie ma fundamentu. Dopiero potem: czy sama wariancja wystarcza (→ bez routera), czy dołożyć nauczony router na granicy. Mierzone.

## Powiązania
[[../10-Tezy/KMT4-Routing-Przez-Wariancje|KMT4]] ↔ [[../15-Antytezy/KMAT4-Nauczony-Router|KMAT4]]
