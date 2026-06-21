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

## Planowane eksperymenty (na naszych danych)
1. **Skalowanie n-gram**: perplexity vs rząd 1–6 vs pamięć/liczba kontekstów → *czemu się nie skaluje* (sparsity, eksplozja pamięci).
2. **NPLM (most)**: mały MLP nad embeddingami ostatnich n znaków, bez attention → miękka generalizacja przy stałym oknie.
3. **mini-transformer**: nasz GPT (val ppl 3,80).
4. **mini-MoE/BTM**: ekspert jigi (6/8) vs ekspert reele (4/4) + mapper na styku; trafność routingu + perplexity vs jeden model.
5. Wspólny wykres: perplexity vs klasa modelu + historia pamięci/parametrów. Ten sam held-out, char-level.

## Weryfikacja u źródeł (deep-research, 2026-06-20)
**Teza potwierdzona** (jeden cel, spektrum mechanizmów, most = NPLM/kNN-LM). Cytowania zweryfikowane:
- **kNN-LM** — Khandelwal i in., **ICLR 2020** (arXiv 1911.00172): ppl **15.79** WikiText-103, poprawa **2,86 pkt** (abstrakt zaokrągla do 2,9; „SOTA" wg stanu 2020).
- **NPLM** — Bengio, Ducharme, Vincent, Jauvin, **JMLR 2003** („A Neural Probabilistic Language Model").
- **Infini-gram** — Liu i in., **COLM 2024** (arXiv 2401.17377): suffix array, n dowolnie duże, latencja ms; ~7 B/token (wg treści, nie abstraktu).

**Jeszcze do weryfikacji** (poza zakresem tej rundy): Mixtral/DeepSeek MoE · SOLAR depth up-scaling · model soups/TIES. **Żaden cytat na wiarę.**

> Pełna tabela werdyktów dla technik kompozycji: [[Kompozycja-Malych-Modeli]] (sekcja Referencje).

## Format deliverable
LaTeX, sekcje: Abstrakt → Baseline (n-gram, skalowanie) → Most (NPLM) → Transformer → MoE/BTM → Wyniki+wykres → Dyskusja. Najpierw `eval.py` (liczby), potem skład.
