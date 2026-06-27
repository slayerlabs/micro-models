# Koncepcja — Obsidian → mikro-model PL (dekompozycja dialektyczna)

> Czy mikro-GPT trenowany OD ZERA na vaulcie (notatki rozbite AST-T2) **uczy się i GENERALIZUJE** —
> i czy **dekompozycja drzewami pomaga**? Rozstrzygamy pomiarem, nie przeczuciem. Ethos: żadnej tezy bez
> dowodu; negatywny wynik to wynik.

## Cel
1. Udowodnić, że vault → uczciwy dataset treningowy (dekompozytor strukturalny, podejście A — deterministyczne, NIE LLM).
2. Zmierzyć, czy mikro-model od zera **generalizuje**, czy **pamięta** (uczciwy przeciwnik = n-gram, czysty look-up).
3. Ablacją rozstrzygnąć, czy struktura AST-T2 (markery `### TYP/SEDNO/ARGUMENTACJA/...`) buduje **lepsze reprezentacje** niż surowy tekst.

## Dekompozycja (AST-T2, podejście A)
Każda notatka dialektyczna → rekord z markerami: `### TYP / TYTUL / SEDNO / ZALOZENIA / ARGUMENTACJA / SYNTEZA / LINKI`.
`ARGUMENTACJA` = całe body po usunięciu frontmatter / lead-blockquote / sekcji „Powiązania" → **lossless**
(rekord ważny nawet gdy ekstrakcja SEDNO/SYNTEZA zawiedzie). Dwa warianty per notatka: **zdekomponowany**
(markery) i **surowy** (`clean_text` body) — ta sama lista notatek → uczciwa ablacja.

## Teza ↔ Antyteza → Synteza
- **T1 (teza):** dekompozycja AST-T2 daje jawny sygnał struktury → GPT-decomp pobije n-gram (generalizacja) i GPT-raw (lepsze reprezentacje).
- **AT1 (antyteza):** przy ~3 MB bardzo repetytywnego korpusu n-gram ≈/> transformer; „zysk" z dekompozycji to przewidywalne markery, nie reprezentacje; ocena na surowym held-out karze model-decomp (off-distribution).
- **S1 (synteza):** rozstrzyga pomiar — dwie falsyfikowalne bramki bits/char na wspólnym held-out. Każdy z 4 wyników (±gen × ±abl) jest deliverable'em.

## Dwie bramki (falsyfikowalne)
- **GENERALIZACJA:** `bits/char(GPT-decomp) < bits/char(n-gram)` → gładka funkcja, nie tablica look-up.
- **ABLACJA:** `bits/char(GPT-decomp) < bits/char(GPT-raw)` → dekompozycja realnie pomaga (test konserwatywny: ocena na surowym held-out).

## Zakres
Rdzeń dialektyczny `{cel, teza, antyteza, synteza, decyzja, fakt, powod}` (tam AST-T2 jest zdefiniowana).
Wyszukiwanie semantyczne / embeddingi świadomie POZA iteracją 1 (tiny char-LM ≠ embedder).

Wynik: [`Wyniki.md`](Wyniki.md). Most do metody Dendrometrii (ta sama dekompozycja AST-T2 co `dendro_decompose.py`).
