---
type: reference
id: REF
status: aktywny
data: 2026-06-22
author: Xavier (DI) — warsztat dla Arkadiusza
---

# Materiał źródłowy (liczby + linki)

## Spektrum (val ppl, char-level, jeden podział)
n-gram: rz.1 11,86 · rz.3 5,19 · rz.6 **3,90** (508 tys. kontekstów) · NPLM ok.8 4,41 / ok.16 4,38 (~41–74K param) · mini-GPT **3,80** (~800K param, okno 128).

## Trójkąt
- n-gram: tani compute, droga pamięć (rośnie ~2–3×/rząd), zero generalizacji (look-up + backoff).
- transformer: drogi compute (~1,8M FLOPs/token), zwarty, generalizuje.
- NPLM: most — tani i zwarty, ograniczony stałym oknem.

## Granica pomiaru (jawnie)
Test ekstrapolacji (val ze świeżych melodii) NIE zrobiony — przewidywanie: n-gram siądzie, sieci utrzymają. To następny eksperyment.

## Most do praktyki (koncepcyjny)
Benchmarki PL labu: LLMzSzŁ, PES, PoQuAD, FLORES, Belebele + dekontaminacja. Ta sama dyscyplina, inna skala/metryka.

## Linki
- pełny rozkład: `../most-ngram-transformer.md`
- kod/wagi: github.com/slayerlabs/micro-models
- pokrewne: `../../docs/Badania/2026-06-20_ngram-vs-transformer/`
