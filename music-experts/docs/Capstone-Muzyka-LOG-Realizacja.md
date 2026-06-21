---
type: nauka-log
title: "Capstone Muzyka — LOG realizacji (n-gram baseline + GPT od zera, etap ABC)"
status: zrobione-etap-1
data: 2026-06-20
created_at: 2026-06-20
author: Arkadiusz Słota
tags: [nauka, llm, muzyka, capstone, gpt, ngram, realizacja]
repo_github: "https://github.com/Maggio333/slay-piano-gpt"
repo_hf: "https://huggingface.co/Maggio33/slay-piano-gpt"
---

# Capstone Muzyka — LOG realizacji (etap 1: ABC)

> Co **realnie** zbudowaliśmy w nocy 19→20.06.2026. Plan: [[Capstone-Muzyka-ABC-MIDI]]. Kod: `C:\ProjektyPublic\MuzykaML`.

## Pipeline (działa end-to-end)
`dane (ABC) → model (następny token) → generacja → MIDI (fortepian)`
Jedno polecenie użycia: `python src/make_midi.py --key G --n 3 --out data/recordings/gpt/x`

## Dane
- Źródło: **thesession.org** (`tunes.csv`, 17 MB, ~54 tys. melodii).
- Filtr + czyszczenie (`src/prepare_data.py`): **jigi 6/8**, usunięte symbole akordów `"..."`, zachowane `~` ozdobniki, `[ace]` akordy, triole.
- Korpus: **12 106 jigów, 2,45 mln znaków, słownik 52** (`data/jigs.abc`, NIE w repo — licencja).

## Model 1 — n-gram (baseline)
- `src/ngram_model.py` — czysty Python, **order-6 z backoffem**, zliczanie (bez wag).
- Metryka nowości (kopiuje vs komponuje): okno 8 znaków → **16% sekwencji nowych** (skleja zapamiętane fragmenty).
- Raport: `reports/raport-baseline.html`.

## Model 2 — GPT od zera (mini-transformer)
- `src/gpt.py` — architektura **L07**: embedding + uwaga Q/K/V z maską + MLP/GELU + residual + LayerNorm, weight tying.
- **816 384 param** (4 warstwy, 4 głowice, dim 128, okno 128, char-level vocab 52).
- Trening (`src/train_gpt.py`, **L03/L08**): cross-entropy, AdamW, warmup+cosine, grad-clip, 2000 iter, CPU (~12 min).
- **Wynik: val loss 1,335 → perplexity 3,80** (train≈val, brak przeuczenia).

## Kluczowe odkrycie (uczciwy pomiar)
**Czyszczenie danych obniżyło perplexity 3,88 → 3,80** i usunęło „akordowy bełkot" w generacji.
Druga lekcja: `[ace]` musi zostać z nawiasami, żeby grało jako **akord** (równoczesne nuty) — udowodnione (2 akordy w → 2 w MIDI). Naprawione w `src/abc_to_midi.py` (`sanitize` nie kasuje już `[]`).

## Rodzina ekspertów (wszystkie modele, stan 2026-06-21)
Wszystkie GPT od zera: **4 warstwy / 4 głowice / dim 128 / okno 128**, char-level, 2000 iter, CPU.
| Ekspert | Dane | Słownik | Val loss / ppl | Uwagi |
|---|---|---|---|---|
| **jig** (`gpt_ckpt.pt`) | 2,45 M znaków | 52 | 1,335 / **3,80** | bazowy; train≈val (brak przeuczenia) |
| **Bach** (`bach_ckpt.pt`) | 79 K (soprany chorałów) | 36 | 0,738 / **2,09** | lekki overfit; dane bardzo powtarzalne → ppl nieporównywalne 1:1 |
| **walc / fortepian** (`waltz_ckpt.pt`) | 789 K | 53 | 1,484 / **~4,4** | liryczny 3/4; render fortepian |
| **reel / skrzypce** (`reel_ckpt.pt`) | 3,90 M | 52 | 1,591 / **~4,9** | żwawy fiddle 4/4; render skrzypce |
| **reel-sv** (`reel_sv_ckpt.pt`) | 3,90 M | **53 (wspólny z walcem)** | 1,582 / ~4,9 | retrening na słowniku walca → do złączenia |

## Złączenie / kompozycja (sandbox dla papera)
- **E0 self-stitch ✅** — mapper na styku bezstratny (Δppl +0,033). Szczegóły: [[Kompozycja-Eksperymenty]].
- **Fuzja (ensemble, `src/fuse.py`)** — mieszanie rozkładów walc×reel (wspólny słownik): `logits = α·walc + (1-α)·reel`. α=0,5 (render fortepian, **odsłuch: ładna**) i α=0,3 (więcej reela, render skrzypce). To baseline „płaskiego ważenia" (KMS5).
- **Duet (`src/duet.py`)** — fortepian (walc) + skrzypce (reel) grają **jednocześnie** (dwie ścieżki, ta sama tonacja). NIE fuzja w jedną linię — dwie niezależne linie złożone naraz.

## Opublikowane (pierwszy raz!)
- 🐙 GitHub: https://github.com/Maggio333/slay-piano-gpt
- 🤗 Hugging Face: https://huggingface.co/Maggio33/slay-piano-gpt (model + card + samples + kod)

## Środowisko
Python **3.10** + `torch 2.11.0+cpu` (CUDA padło — dysk; GPU 3050 na później) + `music21` (ABC→MIDI) + `huggingface_hub`.

## Struktura projektu
`src/` (gpt, train_gpt, sample_gpt, make_midi, gen_samples, ngram_model, prepare_data, **prepare_bach**, abc_to_midi, analyze, upload_hf, **e0_stitch**, **fuse**, **duet**) · `data/` (`models/*_ckpt.pt`, `corpus/*.abc`, `recordings/{ngram,gpt,bach,waltz,reel,fuzja,fuzja_skrzypce,duet}`) · `samples/` · `reports/`.

## Push (plan)
git → **Maggio333/slay-piano-gpt** (prywatne, snapshot „początku") + **slayerlabs/micro-models** (kopia jako podfolder, dom zespołu) · HF → **jedno repo-kolekcja** wszystkich ekspertów. *Po skończonej fuzji, jednym kompletem.*

## Następne kroki
- **E0.5** (stitch niezależny seed) → **E1** (stitch reprezentacji walc×reel — głębsze niż ensemble).
- `eval.py` — perplexity n-gram (rzędy 1–6) vs GPT na wspólnym held-out (twarde liczby do papera).
- **LaTeX paper** dla Kacpra/bloga Slayer → [[Research-NGram-vs-MiniTransformer]] · koncepcja kompozycji → [[Kompozycja-Malych-Modeli]].
- Później: GPU, MIDI (velocity/pedał), fine-tune w stronę Chopina, dane metalowe (DadaGP).
