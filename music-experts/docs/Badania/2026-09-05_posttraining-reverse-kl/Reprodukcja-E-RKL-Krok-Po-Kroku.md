---
type: reprodukcja
title: "E-RKL — reprodukcja krok po kroku"
status: aktywne
data: 2026-09-05
created_at: 2026-09-05
author: Adam Skrodzki
tags: [nauka, llm, muzyka, gpt, realizacja, reprodukcja]
repo_github: "https://github.com/adamskrodzki/micro-models"
---

# E-RKL — reprodukcja krok po kroku

Jak odtworzyć eksperyment „Posttraining: reverse KL z rotującym nauczycielem" od surowych
danych po finalne benchmarki. Kontekst i wyniki: [[Posttraining-ReverseKL-Eksperyment]].
Wszystkie komendy uruchamiamy z katalogu `music-experts/`. Zalecany CUDA (treningi i
benchmarki na CPU działają, ale wolno).

## Wymagania wstępne

- `data/tunes.csv` — dane źródłowe, format jak w repo - oryginalne kolumny `tune_id,type,meter,mode,abc`).
- `data/models/judge_v2.pt` — sędzia (klasyfikator meter/mode/type). Jeśli brak:
  `python src/tools/train_judge.py --csv data/tunes.csv --out data/models/judge_v2.pt`
  (seed splitu 42 — benchmarki zakładają ten sam `--split-seed`).
- `pip install -r requirements.txt`; generator liczb losowych i split są deterministyczne
  (seed 42 wszędzie), więc wyniki powinny być odtwarzalne do szumu generacji.

## Krok 1 — korpusy ABC

Trzy korpusy domenowe + jeden mieszany (wszystko z `tunes.csv`, normalizacja przez
`clean_abc`/`norm_key`):

```bash
python src/data/prepare_data.py jig 6/8 data/jigs.abc
python src/data/prepare_data.py reel 4/4 data/reels.abc
python src/data/prepare_data.py waltz 3/4 data/waltzes.abc
python src/data/prepare_data.py --mixed data/mixed.abc
```

Oczekiwane rzędy wielkości (wersja tunes.csv z 2026-09): jigi ~2.5M znaków, mixed
~10.5M znaków, 54 znaki słownika. `--mixed` nie filtruje typu/metrum — nagłówek `M:`
bierze z wiersza.

## Krok 2 — universalista (wspólny słownik + baseline first-hand)

```bash
python src/train/train_gpt.py data/mixed.abc \
  data/models/universalist_ckpt.pt data/models/universalist_loss_log.csv \
  "" 20260620 --max-iters 8000
```

Parametry domyślne: block 128, batch 32, lr 3e-4, N_LAYER/N_EMBD z ENV (tu: domyślne 4/128).
Kluczowe: to trenowanie BUDUJE słownik 54 znaków — wszyscy nauczyciele w kroku 3
startują z `VOCAB_FROM` na tym ckpcie.

## Krok 3 — nauczyciele na wspólnym słowniku

```bash
python src/train/train_gpt.py data/jigs.abc    data/models/jig_sh_ckpt.pt   data/models/jig_sh_loss.csv   data/models/universalist_ckpt.pt
python src/train/train_gpt.py data/reels.abc   data/models/reel_sh_ckpt.pt  data/models/reel_sh_loss.csv  data/models/universalist_ckpt.pt
python src/train/train_gpt.py data/waltzes.abc data/models/waltz_sh_ckpt.pt data/models/waltz_sh_loss.csv data/models/universalist_ckpt.pt
```

Czwarty argument pozycyjny to `VOCAB_FROM` — ładuje `stoi`/`itos` z universalisty
(wspólny tokenizer; patrz Przebieg krok 1 w [[Posttraining-ReverseKL-Eksperyment]]).
Przy 2000 iteracji każda komenda wypisuje val loss co 200 i zapisze best ckpt.

## Krok 4 — rejestr domen

Dopisz do `data/models/domains.json` (pole `type` decyduje o puli benchmarku i gwiazdce
diagonalnej; dla modeli bez „domeny domowej" wpis jest arbitralny):

```json
"universalist_ckpt.pt": {"type": "jig", "meter": "6/8"},
"jig_sh_ckpt.pt":   {"type": "jig",   "meter": "6/8"},
"reel_sh_ckpt.pt":  {"type": "reel",  "meter": "4/4"},
"waltz_sh_ckpt.pt": {"type": "waltz", "meter": "3/4"}
```

JSON musi być poprawny (`python -c "import json; json.load(open('data/models/domains.json'))"`)
— benchmarki bez zapasu kończą się wyjątkiem parsowania.

## Krok 5 — sanity benchmark nauczycieli i baseline universalisty

```bash
python src/tools/benchmark_ood.py \
  --models data/models/jig_sh_ckpt.pt data/models/reel_sh_ckpt.pt data/models/waltz_sh_ckpt.pt \
  | tee data/benchmarks/ood_teachers_$(date +%Y%m%d_%H%M%S).log

python src/tools/benchmark_ood.py --models data/models/universalist_ckpt.pt \
  | tee data/benchmarks/ood_universalist_$(date +%Y%m%d_%H%M%S).log
```

Oczekiwania: nauczyciele — wysokie diagonale (jig ~0.64, reel ~0.57, waltz ~0.53),
pokrycie 100% w scratch, home bias ~0.9 poza domeną; universalista — przełącza style (reel
scratch ~0.48), ale płytko (waltz ~0.20). Te logi są punktem odniesienia dla kroku 8.

## Krok 6 — trening RKL (pomysł główny)

Najpierw smoke test (mechanika, kształty, zapis — NIE nadpisuj `--last` z właściwego
runu!):

```bash
python src/train/train_rkl.py \
  --teachers jig=data/models/jig_sh_ckpt.pt reel=data/models/reel_sh_ckpt.pt waltz=data/models/waltz_sh_ckpt.pt \
  --max-iters 20 --batch-size 4 --eval-interval 10 --eval-samples 4 \
  --out data/models/_smoke.pt --losslog data/models/_smoke_loss.csv
rm data/models/_smoke.pt data/models/_smoke_loss.csv
```

Pełny run (α=0.1, reverse, obie rotacje: domena×zadanie):

```bash
python src/train/train_rkl.py \
  --teachers jig=data/models/jig_sh_ckpt.pt reel=data/models/reel_sh_ckpt.pt waltz=data/models/waltz_sh_ckpt.pt \
  --max-iters 4000 --batch-size 16 \
  --out data/models/student_rkl_sc_ckpt.pt --last data/models/student_rkl_sc_last_ckpt.pt \
  --losslog data/models/student_rkl_sc_loss.csv
```

Jak czytać przebieg (co 200 iteracji, per domena×zadanie):
- `kl` — spada (cel: < ~0.2 na startcie z universalisty),
- `tlp` — log-prob nauczyciela na rolloutach studenta; ma ROSNĄĆ (nauczyciel coraz lepiej
  rozumie studenta),
- `ent` — entropia generacji; ma być STABILNA (~1.0–1.8 akceptowalne, idealnie < 1.5). Rosnąca entropia przy
  spadającym KL = spłaszczanie do uniform → przerwać, zmniejszyć `--alpha` (np. 0.03).

## Krok 7 — benchmarki finalne

Dopisz checkpointy studenta do `domains.json` (dowolna istniejąca domena — pole służy
tylko jako „home" do hb i gwiazdki):

```json
"student_rkl_sc_ckpt.pt":     {"type": "jig", "meter": "6/8"},
"student_rkl_sc_last_ckpt.pt": {"type": "jig", "meter": "6/8"}
```

```bash
python src/tools/benchmark_judge.py --model data/models/student_rkl_sc_ckpt.pt --samples 100 \
  | tee data/benchmarks/bench_student_rkl_$(date +%Y%m%d_%H%M%S).log

python src/tools/benchmark_ood.py --models data/models/student_rkl_sc_ckpt.pt data/models/student_rkl_sc_last_ckpt.pt \
  | tee data/benchmarks/ood_student_sc_$(date +%Y%m%d_%H%M%S).log
```

Referencje do porównania (seed 42, judge_v2): uniwersalista 0.52/0.62 · 0.48/0.52 ·
0.20/0.29; nauczyciele na przekątnych 0.64 · 0.57 · 0.53 (scratch). Oczekiwany wynik:
student Pareto-dominuje universalistę, waltz scratch na suficie sędziego (~0.54).

## Krok 8 (opcjonalny) — ablacja startu z eksperta

Ten sam trening, ale student startuje z nauczyciela jig (nigdy nie widział reel/waltz
na żadnym etapie) — izoluje wkład pre-RKL ekspozycji universalisty:

```bash
python src/train/train_rkl.py \
  --student data/models/jig_sh_ckpt.pt \
  --teachers jig=data/models/jig_sh_ckpt.pt reel=data/models/reel_sh_ckpt.pt waltz=data/models/waltz_sh_ckpt.pt \
  --max-iters 4000 --batch-size 16 \
  --out data/models/student_jigstart_ckpt.pt --last data/models/student_jigstart_last_ckpt.pt \
  --losslog data/models/student_jigstart_loss.csv
```

(Przy `--student` innym niż universalista asercja słownika musi przejść — dlatego
nauczyciele muszą być z kroku 3, nie stare eksperckie ckpty.) Benchmark jak w kroku 7.

## Artefakty

| ścieżka | co |
|---|---|
| `data/jigs.abc`, `data/reels.abc`, `data/waltzes.abc`, `data/mixed.abc` | korpusy ABC |
| `data/models/universalist_ckpt.pt` | student startowy / wspólny słownik (54 znaki) |
| `data/models/{jig,reel,waltz}_sh_ckpt.pt` | nauczyciele na wspólnym słowniku |
| `data/models/student_rkl_sc_{,_last_}ckpt.pt` | student po RKL (best-by-KL / last) |
| `data/models/student_jigstart_{,_last_}ckpt.pt` | ablation: start z jig_sh |
| `data/models/domains.json` | rejestr domen benchmarku |
| `data/models/*_loss.csv`, `data/benchmarks/*.log` | krzywe treningu i logi benchmarków |
| `src/data/prepare_data.py`, `src/train/train_gpt.py`, `src/train/train_rkl.py` | pipeline |
| `src/tools/{train_judge,benchmark_judge,benchmark_ood}.py` | sędzia i benchmarki |

## Pułapki

1. **Słownik**: `train_rkl.py` wymaga identycznego zestawu znaków studenta i nauczycieli
   (twarda asercja). Stare eksperckie ckpty mają własne, mniejsze słowniki — nie działają
   jako nauczyciele ani (bez przetrenowania) jako student.
2. **`--last`/`--out`**: zawsze podawaj jawne ścieżki; domyślne nadpisują się między
   runami (zdarzyło się: smoke test nadpisał `student_rkl_last.pt`).
3. **Pokrycie w benchmarkach**: pominięte generacje liczą się jako 0; porównuj modele
   tylko przy zbliżonym pokryciu (spadki pokrycia na continuation przy waltz — 83% —
   wynikają ze znaków w surowych wierszach CSV spoza słownika, nie z jakości modelu).
4. **Seed**: benchmarki są deterministyczne przy tym samym seed (identyczne melodie
   w kolumnach); generacja przy ewaluacji i tak wnosi szum ±0.03–0.05 — nie czytaj
   pojedynczych eval-i zbyt dosłownie.
