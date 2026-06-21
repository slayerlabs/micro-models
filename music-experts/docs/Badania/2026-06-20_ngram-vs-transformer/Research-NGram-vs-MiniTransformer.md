---
type: nauka-research
title: "Research roadmap: n-gram → most → mini-transformer (paper dla Slayer)"
status: w-planowaniu
data: 2026-06-20
created_at: 2026-06-20
author: Arkadiusz Słota
tags: [nauka, llm, research, ngram, transformer, moe, paper]
zlecenie: "Kacper Wikieł — research + paper w LaTeX na bloga Slayer"
---

# Research: n-gram → most → mini-transformer

> Zlecenie Kacpra: pokazać *jak mały model bije n-gram* + zbadać przestrzeń **między** nimi. Deliverable: **paper w LaTeX** (wykres + opis) na bloga Slayer. Baza: [[Capstone-Muzyka-LOG-Realizacja]].

## Teza
**Jeden cel (predykcja następnego tokena), spektrum mechanizmów** — od twardego zliczania do miękkiej uwagi. n-gram i transformer to **dwie różne technologie** (zliczanie vs wagi+gradient), a „most" istnieje właśnie dlatego.

## Mapa (osie, które kolejne techniki „ruszają")
| Technika | Co rusza | Pozycja |
|---|---|---|
| **n-gram** (mamy baseline) | nic — zliczanie, stałe okno | start |
| **Infini-gram** | długość kontekstu (suffix array, zmienne n) | n-gram u ściany |
| **PPM / CTW / variable-order Markov** | adaptacyjna długość | klasyczny przodek |
| **Neural n-gram (Bengio 2003, NPLM)** | wagi + reprezentacja, okno stałe | ⭐ **most** |
| **kNN-LM (2020)** | twarde→miękkie (retrieval po podobieństwie) | rozmyty infini-gram |
| **mini-transformer** (mamy GPT) | uwaga = długi, zmienny kontekst + generalizacja | cel |
| sliding-window attn (Gemma2), Mamba/SSM | okrojony transformer | spotkanie w środku |

## Nasze pomysły (kandydaci na rozdziały 4–5)
- **MoE z małych ekspertów + router** — pojemność bez proporcjonalnego kosztu (Mixtral, DeepSeek). Inteligencja w **selektorze**, nie w ekspertach.
- **Router-na-styku / Branch-Train-Merge (BTM, BTX)** — eksperci trenowani niezależnie na różnych datasetach; mapper uczony na **styku** dwóch domen. Teza Arka: *ile reprezentacji wstrzyknąć = funkcja nakładania się domen*.
- **Routing przez aktywacje** — czytać, który ekspert „przebija się" z aktywacji/koherencji zamiast osobnego routera (interpretowalność jako sygnał).

## Planowane eksperymenty (na naszych danych) — kolejność wg wartość/wysiłek
**Ścieżka pewna (centrum papera-mostu) — skończona-w-zasięgu, rób ją:**
1. **NPLM (most) ⭐** — mały MLP nad embeddingami ostatnich n znaków, bez attention → miękka generalizacja przy stałym oknie. ~50 linii, minuty treningu; prawie na pewno ląduje **monotonicznie** między n-gramem a GPT → wykres-spektrum, który sam opowiada tezę. **To centrum.**
2. **Skalowanie n-gram**: perplexity vs rząd 1–6 vs pamięć/liczba kontekstów → *czemu się nie skaluje* (sparsity, eksplozja pamięci). Tani rozdział.
3. **mini-transformer**: nasz GPT (val ppl 3,80) — punkt docelowy spektrum.
4. **kNN-LM** — **opisać jako punkt na spektrum („rozmyty infini-gram"), NIE budować**: char-level datastore na 2,4M znaków = dużo roboty o niepewnym zysku. NPLM daje więcej za mniej.
5. **Wspólny wykres**: perplexity vs klasa modelu + historia pamięci/parametrów. Ten sam held-out, char-level.

**Ścieżka kompozycji (hazard, osobny profil ryzyka):**
6. **mini-MoE/BTM** — ekspert vs ekspert + router na styku. ⚠️ **Pułapka metrum:** NIE jig(6/8) vs reel(4/4) — router oszuka po nagłówku `M:` (~100% trafności z jednego tokena). Użyj domen w **TYM SAMYM metrum** (walc vs mazur). Najpierw [[Kompozycja-Eksperymenty]] (E_CKA → shared-trunk) i rama: [[Emergencja-i-Wspolna-Reprezentacja]].

## Wyniki spektrum (2026-06-21) — char-level jigi, ten sam split (90/10)
| model | mechanizm | val ppl | rozmiar |
|---|---|---|---|
| n-gram rząd 1 | zliczanie, okno 1 | 11,86 | 52 konteksty |
| n-gram rząd 2 | zliczanie, okno 2 | 7,14 | 1,2K |
| n-gram rząd 3 | zliczanie, okno 3 | 5,19 | 13,9K |
| n-gram rząd 4 | zliczanie, okno 4 | 4,29 | 67,6K |
| n-gram rząd 5 | zliczanie, okno 5 | 3,95 | 215K |
| n-gram rząd 6 | zliczanie, okno 6 | **3,90** | **508K kontekstów** |
| NPLM okno 4 | MLP, stałe okno | 4,52 | 25K param |
| NPLM okno 8 | MLP, stałe okno | 4,41 | 41K param |
| NPLM okno 16 | MLP, stałe okno | 4,38 | 74K param |
| **GPT** | atencja, okno 128 | **3,80** | ~800K param |

Kod: `src/train/ngram_eval.py` (n-gram: interpolacja rzędów 0..M, wagi ∝2^o, floor add-k), `src/train/nplm.py`.

**Wniosek (uczciwie, NIE naiwnie):**
- **n-gram skaluje ppl z rzędem (11,86 → 3,90), ale pamięć EKSPLODUJE** (~2-3× kontekstów na rząd → rząd 6 = 508K) i to **tablica look-up bez generalizacji** poza widziane konteksty.
- **n-gram rząd 6 (3,90) ≈ GPT (3,80)** na tym repetytywnym korpusie — muzyka ma dużo dosłownie powtarzalnych fraz, więc n-gram „pamięta". To **nie** jest „transformer miażdży n-gram"; ppl prawie remis.
- **NPLM kompresuje** n-gram w zwartą, gładką, **generalizującą** funkcję (~40K param, ppl ~4,4) — trochę gorszy ppl, ale uczy reprezentacji, nie tablicy.
- **GPT** wygrywa ppl (3,80) **zmiennym długim kontekstem** (atencja), kosztem ~800K param.
- **Spektrum to ppl vs PAMIĘĆ vs GENERALIZACJA** — nie proste uszeregowanie ppl. To uczciwsza teza papera niż „każdy krok obniża ppl".

## Weryfikacja u źródeł (deep-research, 2026-06-20)
**Teza potwierdzona** (jeden cel, spektrum mechanizmów, most = NPLM/kNN-LM). Cytowania zweryfikowane:
- **kNN-LM** — Khandelwal i in., **ICLR 2020** (arXiv 1911.00172): ppl **15.79** WikiText-103, poprawa **2,86 pkt** (abstrakt zaokrągla do 2,9; „SOTA" wg stanu 2020).
- **NPLM** — Bengio, Ducharme, Vincent, Jauvin, **JMLR 2003** („A Neural Probabilistic Language Model").
- **Infini-gram** — Liu i in., **COLM 2024** (arXiv 2401.17377): suffix array, n dowolnie duże, latencja ms; ~7 B/token (wg treści, nie abstraktu).

**Jeszcze do weryfikacji** (poza zakresem tej rundy): Mixtral/DeepSeek MoE · SOLAR depth up-scaling · model soups/TIES. **Żaden cytat na wiarę.**

> Pełna tabela werdyktów dla technik kompozycji: [[Kompozycja-Malych-Modeli]] (sekcja Referencje).

## Format deliverable
LaTeX, sekcje: Abstrakt → Baseline (n-gram, skalowanie) → Most (NPLM) → Transformer → MoE/BTM → Wyniki+wykres → Dyskusja. Najpierw `eval.py` (liczby), potem skład.
