---
license: mit
tags:
  - music-generation
  - abc-notation
  - symbolic-music
  - gpt
  - char-level
  - from-scratch
  - model-composition
  - stitching
library_name: pytorch
pipeline_tag: text-generation
---

# Slay Micro-Models — tiny from-scratch music experts + composition research

A family of **~0.8M-parameter char-level GPTs**, each trained **from scratch** on monophonic music in
[ABC notation](https://abcnotation.com/), plus experiments in **composing small experts** (stitching,
ensembling, duets). Built as research for the **Slayer** collective (toward a "small models, composed"
paper). Music is the sandbox; the methods generalize.

Every expert shares one architecture: decoder-only Transformer — 4 layers, 4 heads, `d_model=128`,
context 128 chars, character-level. Trained on CPU in minutes.

## The experts
| Expert (`data/models/`) | Style / render | Training data | Val perplexity |
|---|---|---|---|
| `jig_ckpt.pt` | Irish jig, 6/8 | 12.1k tunes (thesession.org) | **3.80** |
| `bach_ckpt.pt` | Baroque chorale soprano | 350 soprano lines (music21) | 2.09\* |
| `waltz_ckpt.pt` | Lyrical waltz, 3/4 → piano | 3.0k tunes | ~4.4 |
| `reel_ckpt.pt` | Driving fiddle, 4/4 → violin | 17.2k tunes | ~4.9 |
| `reel_sv_ckpt.pt` | reel on shared vocab (for composition) | 17.2k tunes | ~4.9 |

\* Bach ppl is **not** directly comparable (smaller vocab + very repetitive data).

## Composition experiments
- **E0 (self-stitch) ✅** — a trained linear mapper at an intermediate seam is lossless (Δppl ≈ 0): the stitching **mechanism** is sound (validates plumbing, not the thesis).
- **Ensemble fusion** (`src/compose/fuse.py`) — blend two experts' next-token distributions (shared vocab) → audible hybrid. This is the **flat-weighting baseline**.
- **Duet** (`src/compose/duet.py`) — two experts layered (piano + violin, simultaneous): multi-track, not model-level fusion.
- **Next — E1:** representation-level stitch (the actual hypothesis, meant to beat these baselines).

## Judge — automatic generation benchmark

Perplexity measures text fit, not whether a generated tune *works as music* (meter? key? dance
type?). The judge is an independent classifier that scores generated melodies like a human
would: seeing only the tune **body** (the `M:`/`K:` headers are stripped — the judge cannot
cheat off the prompt), it predicts **meter**, **mode** and **type** (jig/reel/hornpipe...).
Built from the same building blocks as the experts: `JudgeGPT` = `core/gpt.py` trunk (verbatim)
+ mean-pool + 3 linear heads, pure PyTorch; 55k tunes, leak-free split by `tune_id`.
`judge_v2` val acc: meter **0.92**, mode **0.72**, type **0.82** — with musically honest
confusions (hornpipe↔reel, Bmin↔D).

Two benchmarks built on top of it:

- **In-domain** (`benchmark_judge.py`): each expert is scored only on tunes from its own
  domain (`data/models/domains.json`), failures count as 0 (no survivor bias), with a
  real-melody ceiling for reference. Leaderboard: jigs hold **0.80–0.86** of ceiling,
  waltz 0.86 (soft ceiling), reels 0.69–0.72. The judge's ranking agrees with PPL — two
  independent metrics, one verdict.
- **OOD transfer matrix** (`benchmark_ood.py`): expert × target domain. Diagonal always
  wins (specialization is real), but **headers don't steer off-domain** (home-bias 0.9 —
  a jig expert told `M:4/4` still emits 6/8), while **melody context half-steers** (home-bias
  ~0.5). Reel experts are the most flexible donors; small (e32) models bend easiest,
  big ones are most rigid.

```bash
./run_benchmarks.sh                                   # in-domain sweep, all checkpoints
python src/tools/train_judge.py --iters 5000 --batch 64 --out data/models/judge_v2.pt
python src/tools/benchmark_judge.py --model data/models/jig_ckpt.pt --judge data/models/judge_v2.pt
python src/tools/benchmark_ood.py                     # expert × domain transfer matrix
```

Full WHY / HOW / WHAT, results and limitations (in Polish):
[docs/Badania/2026-09-05_posttraining-reverse-kl/Judge-Sedzia-Generacji.md](docs/Badania/2026-09-05_posttraining-reverse-kl/Judge-Sedzia-Generacji.md)

## Pipeline (`src/`)
`prepare_data.py` / `prepare_bach.py` (build ABC corpus) → `gpt.py` (architecture) → `train_gpt.py`
(train; optional shared vocab) → `make_midi.py` / `gen_samples.py` (generate + render) →
`e0_stitch.py` / `fuse.py` / `duet.py` (composition) · `ngram_model.py` (baseline) · `abc_to_midi.py` (render) ·
`train_judge.py` / `judge_tunes.py` / `benchmark_judge.py` / `benchmark_ood.py` (judge — generation benchmark).

## Usage
```bash
pip install torch music21
python src/generate/gen_samples.py --ckpt data/models/waltz_ckpt.pt --meter 3/4 --keys D,G,Emin --inst piano --out out
```

## Honest scope
**Shown:** a small char-LM learns real musical structure (meter, key signatures, cadences) from
next-token prediction *alone*; data cleaning measurably helps (ppl 3.88→3.80); the stitch mechanism is
lossless; experts can be combined (baseline). **Not yet shown:** that representation-level composition of
small experts beats a single model — the open hypothesis (E1+).

## Data & license
Code & weights: **MIT**. Training data **not redistributed** — folk tunes from
[thesession.org](https://thesession.org/) (rebuild via `prepare_data.py`); Bach chorale sopranos via
`music21`. Please respect source terms.

Built by Arkadiusz Słota for the **Slayer** collective. Educational / research project.
