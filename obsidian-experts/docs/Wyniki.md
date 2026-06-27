# Wyniki — iteracja 1 (NEGATYWNY, robust)

> N-gram (podłoga pamięci) **bije** oba mikro-transformery na wspólnym surowym held-out; dekompozycja jest
> stabilnie minimalnie *gorsza* od surowego tekstu. Negatyw **robust przy ~konwergencji** (8000 iters), nie
> artefakt niedouczenia. Spójne z music-experts („n-gram ≈ GPT na repetytywnym korpusie").

## Co wykonano
- Dekompozycja vaulta: **813 notatek dialektycznych → 731 train / 82 held-out** (seed 20260627, min-chars 200).
  Korpusy: decomp 3,50 MB · raw 3,05 MB · held-out 0,34 MB. ARGUMENTACJA **lossless 731/731**, 0 resztek frontmatter/`[[ ]]`.
- Tokenizery byte-BPE 4096 → vocab **4352** (raw 2,99 zn/tok = 1,19× gęściej niż GPT-2; decomp 3,22; odwracalne 100%).
- 2× GPT od zera, **2,64 mln param**, identyczne hp (`n_layer 4 · n_head 4 · n_embd 192 · block 128`).
- Baseline n-gram char-level (interpolacja rzędów 0..6, floor add-k).

## bits/char na wspólnym surowym held-out
| model | @3000 iters | @8000 iters | val ppl @8000 |
|---|---|---|---|
| n-gram (M=6) | 2,0458 | **2,0458** | — |
| GPT-raw | 2,4613 | **2,1103** | 77,6 |
| GPT-decomp | 2,4961 | **2,1468** | 77,1 |

## Werdykt bramek (robust @8000)
- **GENERALIZACJA ✗ (blisko):** decomp 2,1468 ≥ n-gram 2,0458. Gap zwęził się **0,45 → 0,10** po douczeniu
  (val ppl 160→77). Transformer prawie dorównuje, nie bije → pamięć ≥ generalizacja przy tej skali.
- **ABLACJA ✗ (stabilna):** decomp 2,1468 ≥ raw 2,1103. Kara off-distribution **stabilna** (~0,035 @3000 i @8000).
- **Bramka nauki ✓:** `eval.py` val ppl 78,72 vs baseline 4352 = **55,29×**, overfit train↔val **1,78×**.

## Dlaczego n-gram wygrywa (uczciwie)
1. Korpus bardzo repetytywny (boilerplate: nagłówki, słownictwo IFC/BIM, nazwy id) → 6-gram „pamięta", held-out współdzieli.
2. Retrening 3000→8000 zwęził gap, ale go nie zamknął → negatyw **nie** jest artefaktem niedouczenia (modele ~skonwergowane).
3. Eval decomp konserwatywny (uczony z markerami, oceniany na surowym tekście) → naturalna kara ~0,035.
4. ~3 MB to za mało, by emergowała generalizacja bijąca twardą tablicę n-gramową na tak powtarzalnym tekście.

## Próbka (`sample.py`, temp 0,7)
- **decomp** (seed `### TYP: teza`): odtwarza strukturę — markery, „Action items", ścieżki `../40-Architektura/...`, id, tabele.
- **raw** (seed „Decyzja:"): odtwarza słownictwo dialektyczne — „antyteza", „Synteza:", `../15-Antytezy/...`.
- Wniosek: model nauczył się **stylu i szkieletu**, nie znaczenia — spójne z „pamięć/wzorzec, nie generalizacja".

## Rekomendacje (iteracja 2)
Gap 0,10 jest mały → warto skalować: większy korpus (~1060 notatek + inne vaulty), `n_embd 256`, `iters 15–30k`,
**dedup boilerplate'u** (uczciwszy held-out), char-level GPT (zdejmuje różnicę tokenizera). Most semantyczny:
zasilić front relewancji metody Dendrometrii (E5) embeddingami in-domain.

Kanoniczne notatki dialektyczne: ObsidianVault `NaukaML/Badania/2026-06-27_obsidian-mikromodel-dekompozycja/`,
prywatnie `slayerlabs/PrivateSlayNotes` (`01_ArekSłota/27_06_Obsidian-MicroModel/`).
