---
type: nauka-wynik
id: KM-EKSP
title: "Kompozycja małych modeli — log eksperymentów (E0/E0.5/E1/E2)"
status: w-toku
data: 2026-06-21
created_at: 2026-06-21
author: Arkadiusz Słota
---

# Kompozycja małych modeli — log eksperymentów

> Wyniki E0/E0.5/E1/E2. Kryteria rozstrzygnięcia w syntezach. Kod: `src/e0_stitch.py` (`C:/ProjektyPublic/MuzykaML`).

## E0 — self-stitch (benchmark zerowy) — ✅ PASS — 2026-06-21
**Setup:** model jig (`data/gpt_ckpt.pt`, 4 bloki, n_embd 128). Liniowy mapper **g (128×128)** na **styku po bloku 2**, reszta **zamrożona**. Dane: `jigs.abc` (char-level). Trening **tylko g** (16K param), 500 kroków, AdamW lr 1e-3. Eval na stałych batchach val (ten sam seed).

| krok | val loss | ppl |
|---|---|---|
| baseline (bez g) | 1,3345 | 3,7981 |
| g = identyczność | 1,3345 | **3,7981** (== baseline, co do cyfry) |
| g losowy (przed treningiem) | 4,81 | 122,4 |
| g po treningu (500) | 1,3430 | 3,8306 |

**Δppl = +0,033** (próg <0,05) → **PASS ✅**

**Interpretacja (uczciwie):**
- Sanity (g=identyczność == baseline co do cyfry) potwierdza **poprawność forwardu ze szwem** — zero buga w mechanizmie.
- Losowy mapper niszczy model (ppl **122**), a trening **samych 16K parametrów g** (reszta zamrożona) odbudowuje do ≈baseline → mapper realnie **wyrównuje reprezentację**.
- Δppl ≠ dokładnie 0 (+0,033) = resztka po skończonym treningu; więcej kroków → bliżej zera.
- **Werdykt waliduje MECHANIZM (hydraulikę), NIE tezę kompozycji** (zgodnie z [[KMS1-Kompozycja-Gdy-Kontrakt|KMS1]]). Zielone = „harness działa", nie „kompozycja działa".

## Złączenie walc × reel — ensemble (baseline płaskiego ważenia, KMS5) — 2026-06-21
Prerequisite: **wspólny słownik (53)** — reel przetrenowany na słowniku walca (`reel_sv_ckpt.pt`, val 1,582/ppl ~4,9; retrening nie zaszkodził). `src/fuse.py`: w każdym kroku `logits = α·walc + (1-α)·reel` → sampling (assert wspólnego stoi przechodzi).
- **α=0,5** (render fortepian) — odsłuch: **oceniona jako ładna** (blend stylów zadziałał).
- **α=0,3** (więcej reela, render skrzypce).
- **Duet** (`src/duet.py`): walc (fortepian) + reel (skrzypce) **jednocześnie** — dwie ścieżki, ta sama tonacja; granie razem, nie fuzja w jedną linię.
> To jest **BASELINE (KMS5, płaskie ważenie)** — działa i jest słyszalny. Stitch reprezentacji (E1, nasza nisza) ma to **pobić**; ensemble jest punktem odniesienia.

## E0.5 — niezależny seed (planowane)
Front modelu A × back **niezależnie wytrenowanej kopii** (inny seed, ten sam kontrakt). Test: czy kontrakt **wymusza wspólną geometrię**. Wymaga **jig-v2** (drugi seed). Δppl mały → kontrakt trzyma; duży → naprawić przed E1.

## Powiązania
[[Kompozycja-INDEX]] · [[Kompozycja-Malych-Modeli]] · kryterium: [[KMS1-Kompozycja-Gdy-Kontrakt|KMS1]]
