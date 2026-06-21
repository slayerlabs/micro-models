---
type: synteza
id: KMS2
title: "PROPOZYCJA — wymuszony kontrakt domyślnie (bo trenujemy ekspertów od zera); post-hoc gdy scalamy gotowce"
status: propozycja
data: 2026-06-20
created_at: 2026-06-20
parents: ["KMT2", "KMAT2"]
author: Arkadiusz Słota
---

# KMS2 — Kontrakt wymuszony vs post-hoc

## Synteza
Decyduje **czy kontrolujemy trening ekspertów**:
- **Nasz przypadek** (małe modele od zera, jeden warsztat) → **wymuszony kontrakt** (wspólna zamrożona głowica). Pewniejsze, czystsze złożenie, mniej tracimy. Domyślne.
- **Scalanie gotowców** (cudze/istniejące modele) → **post-hoc** (ZipIt!/Re-Basin), świadomie godząc się na stratność.

Skoro program to micro-models trenowane u nas — wybieramy **wymuszony kontrakt**, a post-hoc pozostawiamy jako opcję dla cudzych modeli.

## Kryterium / decyzja
Pochodzenie ekspertów (własne vs gotowe). Rekomendacja: wymuszony kontrakt. Zmierzyć stratność post-hoc dla porównania.

## Doostrzenie (2026-06-21) — najczystszy wariant kontraktu: shared-trunk
Po E1-negatywie (post-hoc liniowy ≈ ensemble) doostrzamy wymuszony kontrakt do wariantu, który **izoluje jedno pytanie**:
- **Wspólny ZAMROŻONY front** (bloki 0..SEAM) **ORAZ wspólna ZAMROŻONA głowa**; trenuj tylko **style-specyficzne tyły**.
- Wtedy szew jest **poprawny z konstrukcji** (zero problemu geometrii — front i głowa wspólne), więc testujemy czysto: *czy routing między tyłami na szwie bije ensembling ich wyjść?*
- Iskra → dopiero wtedy słownik/SAE ([[../10-Tezy/KMT3-Slownik-Z-Aktywacji|KMT3]]) ma co poprawiać. Remis → **mocny, publikowalny negatyw**: „przy tej skali kompozycja = ensemble".

## Prawo zachowania kotwicy (rama dla całego programu)
Każda droga do „taniej kompozycji" (zamrożony backbone+adaptery, wspólny codebook/interlingua [[../10-Tezy/KMT3-Slownik-Z-Aktywacji|KMT3]], co-trening naprzemienny, destylacja, orkiestrator MoE) wymaga **wspólnej kotwicy, którą KTOŚ raz nauczył szeroko**. Można uczynić **rozszerzanie** tanim i modularnym — ale wspólna geometria **nie emerguje za darmo** z niezależnych części. **E1 padł dokładnie dlatego, że kotwicy nie było** (post-hoc + liniowy mapper, zero wspólnego podłoża). To nie pech — to ta zasada. Pełny rozkład: [[Emergencja-i-Wspolna-Reprezentacja]].

## Powiązania
[[../10-Tezy/KMT2-Kontrakt-W-Treningu|KMT2]] ↔ [[../15-Antytezy/KMAT2-Wyrownanie-Posthoc|KMAT2]]
