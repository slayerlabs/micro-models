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

## Powiązania
[[../10-Tezy/KMT2-Kontrakt-W-Treningu|KMT2]] ↔ [[../15-Antytezy/KMAT2-Wyrownanie-Posthoc|KMAT2]]
