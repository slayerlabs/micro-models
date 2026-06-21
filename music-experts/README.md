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
| `gpt_ckpt.pt` (jig) | Irish jig, 6/8 | 12.1k tunes (thesession.org) | **3.80** |
| `bach_ckpt.pt` | Baroque chorale soprano | 350 soprano lines (music21) | 2.09\* |
| `waltz_ckpt.pt` | Lyrical waltz, 3/4 → piano | 3.0k tunes | ~4.4 |
| `reel_ckpt.pt` | Driving fiddle, 4/4 → violin | 17.2k tunes | ~4.9 |
| `reel_sv_ckpt.pt` | reel on shared vocab (for composition) | 17.2k tunes | ~4.9 |

\* Bach ppl is **not** directly comparable (smaller vocab + very repetitive data).

## Composition experiments
- **E0 (self-stitch) ✅** — a trained linear mapper at an intermediate seam is lossless (Δppl ≈ 0): the stitching **mechanism** is sound (validates plumbing, not the thesis).
- **Ensemble fusion** (`src/fuse.py`) — blend two experts' next-token distributions (shared vocab) → audible hybrid. This is the **flat-weighting baseline**.
- **Duet** (`src/duet.py`) — two experts layered (piano + violin, simultaneous): multi-track, not model-level fusion.
- **Next — E1:** representation-level stitch (the actual hypothesis, meant to beat these baselines).

## Pipeline (`src/`)
`prepare_data.py` / `prepare_bach.py` (build ABC corpus) → `gpt.py` (architecture) → `train_gpt.py`
(train; optional shared vocab) → `make_midi.py` / `gen_samples.py` (generate + render) →
`e0_stitch.py` / `fuse.py` / `duet.py` (composition) · `ngram_model.py` (baseline) · `abc_to_midi.py` (render).

## Usage
```bash
pip install torch music21
python src/gen_samples.py --ckpt data/models/waltz_ckpt.pt --meter 3/4 --keys D,G,Emin --inst piano --out out
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
