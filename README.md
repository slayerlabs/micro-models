# Slayer Micro-Models

**Small models, honestly measured.**

A research home by the **Slayer** collective for *micro-models* — tiny, from-scratch neural networks —
together with the experiments and the reasoning behind them. We build models small enough to understand
end-to-end, train in minutes, and reason about precisely. The goal isn't leaderboard numbers; it's
**understanding**: what tiny models can learn, where they break, and whether many small experts can be
**composed** into something larger.

## Principles
- **Hard measurement.** Every claim is backed by a number or a primary source — *no thesis without proof.*
- **Honest scope.** We state what we *haven't* shown and we **publish negative results** — they are findings too.
- **Reproducible & transparent.** Code, weights, and the reasoning (dialectic notes) live together; raw data and
  generated outputs stay out of the repo because they regenerate from the scripts.
- **Good taste over benchmaxxing.** A simple baseline must be beaten *honestly*, not gamed.

## Projects
| Project | What it is | Status |
|---|---|---|
| [`music-experts/`](music-experts/) | Tiny char-level music GPTs (Irish jig, Bach, waltz, reel) trained from scratch on [ABC notation](https://abcnotation.com/), plus experiments in **composing small experts** (stitching, ensembling, duets). | active |

*More micro-model projects will be added as sibling folders.*

## Highlight — `music-experts/`
A family of **~0.8M-parameter** decoder-only Transformers — one architecture, each trained on CPU in minutes.

- **Experts:** jig (val ppl 3.80), Bach chorale soprano, waltz (→ rendered on piano), reel (→ rendered on violin).
- **What's shown:** a small char-LM learns real musical structure — meter, key signatures, cadences — from
  next-token prediction *alone* (zero music theory in the code); cleaner data measurably lowers perplexity.
- **Composition so far (honest):** the stitching *mechanism* is sound (a trained linear mapper at an
  intermediate seam is lossless). A flat **ensemble of two experts beats either single model** on a mixed task —
  but a **representation-level stitch does *not* yet beat that ensemble**. The simple baseline is strong; the
  fancier method still has to earn its place. Open directions: enforced shared contracts, richer (non-linear) mappers.
- The full argument — thesis ↔ antithesis → synthesis, with source-verified references — is in
  [`music-experts/docs/`](music-experts/docs/).

## Repository layout
```
micro-models/
└── <project>/             # self-contained
    ├── src/               # pipeline: data → train → generate → compose
    ├── data/models/       # trained checkpoints (weights)
    ├── docs/              # research notes: concept, experiments, honest findings
    └── README.md
```

## Getting started
```bash
cd music-experts
pip install -r requirements.txt
# generate a fresh tune from an expert and render it to MIDI
python src/gen_samples.py --ckpt data/models/waltz_ckpt.pt --meter 3/4 --keys D,G,Emin --inst piano --out out
```

## About
Built by the **Slayer** collective — an open research workshop. Educational / research focus.
**License:** MIT (code & weights). Training data belongs to its original sources (see each project's README).
