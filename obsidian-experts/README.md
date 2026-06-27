# obsidian-experts — mikro-model PL OD ZERA z notatek dialektycznych (Obsidian)

Iteracja badawcza siostrzana do [`../music-experts/`](../music-experts/): bierzemy **vault Obsidiana**
(notatki dialektyczne: cel / teza ↔ antyteza → synteza / decyzja / fakt / powód), rozbijamy je
**deterministycznie** wg schematu **AST-T2** (Teza/SEDNO · Argumentacja · Założenia · Synteza · Linki),
trenujemy **mikro-GPT od zera** i **uczciwie mierzymy**, czy model *uczy się i GENERALIZUJE* (nie tylko
pamięta) — oraz czy **dekompozycja drzewami pomaga** (ablacja decomp-vs-raw). Rama: „od n-gramu do
transformera" (patrz `../REPORT.md`).

Architektura jak w music-experts: decoder-only Transformer — 4 warstwy, 4 głowy, `d_model=192`, kontekst 128,
**byte-level BPE 4096** (vocab 4352). ~2,64 mln parametrów. Trening na GPU (RTX 3050) w minuty.

## Wynik (iteracja 1) — NEGATYWNY, ale ROBUST
bits/char na **wspólnym surowym held-out** (82 notatki, 344 766 znaków). bits/char = Σ nats / liczba znaków / ln2
→ tokenizer-niezależne, więc char-n-gram i dwa BPE-GPT są porównywalne.

| model | bits/char @3000 iters | bits/char @8000 iters | val ppl |
|---|---|---|---|
| **n-gram** (M=6, podłoga pamięci) | 2,0458 | **2,0458** | — |
| GPT-raw (surowe notatki) | 2,4613 | **2,1103** | 77,6 |
| GPT-decomp (zdekomponowane) | 2,4961 | **2,1468** | 77,1 |

- **Generalizacja: ✗ (robust, blisko)** — n-gram wygrywa; gap zwęził się **0,45 → 0,10 bita/znak** po douczeniu
  (model był mocno niedouczony @3000). Przy tej skali: **pamięć ≥ generalizacja**.
- **Ablacja: ✗ (robust, stabilny gap)** — dekompozycja ~0,035 bita/znak **gorsza** od raw (kara off-distribution:
  model-decomp uczony z markerami, oceniany na surowym tekście — test konserwatywny).
- **Bramka nauki: ✓** — val ppl 78,7 vs baseline losowy 4352 = **55× lepiej niż losowo**, overfit train↔val 1,78×.
- **Próbka** (`sample.py`): model-decomp odtwarza **strukturę** (markery `### TYP/TYTUL`, „Action items", ścieżki,
  id), model-raw — **słownictwo dialektyczne**; oba lokalnie-wiarygodne, globalnie niespójne → styl/szkielet, nie znaczenie.

Dlaczego n-gram wygrywa: korpus jest **bardzo repetytywny** (nagłówki sekcji, słownictwo IFC/BIM, nazwy id) →
6-gram z 865 tys. kontekstów „pamięta", a held-out z tych samych projektów to współdzieli. To ten sam efekt co
w music-experts („n-gram rząd 6 ≈ GPT na repetytywnym korpusie"). Pełne rozumowanie: [`docs/`](docs/).

## Pipeline (`./`)
`prepare_obsidian.py` (vault → 3 korpusy: decomp / raw / held-out; dekompozycja AST-T2, `clean_text` vendored) →
`train_tokenizer.py` (byte-BPE 4096) → `train.py` (GPT od zera) → `ngram_obsidian.py` (baseline → bits/char) →
`eval_obsidian.py` (bits/char na wspólnym held-out + dwie falsyfikowalne bramki). `gpt.py` = architektura
(wendor z `academy/01-my-little-llm/lab`).

## Reprodukcja
```bash
pip install -r requirements.txt   # torch (CUDA cu124 dla 3050)
# 1) dekompozycja vaulta -> data/ (surowe dane = własny vault, NIE redystrybuowane)
python prepare_obsidian.py --vault /sciezka/do/ObsidianVault
# 2) tokenizery in-domain (osobny per arm)
python train_tokenizer.py --corpus data/korpus-obsidian.txt     --merges 4096 --out tokenizer-obsidian.json
python train_tokenizer.py --corpus data/korpus-obsidian-raw.txt --merges 4096 --out tokenizer-obsidian-raw.json
# 3) trening 2x GPT od zera (identyczne hp = stały compute)
python train.py --corpus data/korpus-obsidian.txt     --tokenizer tokenizer-obsidian.json     --out model-obsidian.pt     --iters 8000 --n-layer 4 --n-head 4 --n-embd 192 --block 128
python train.py --corpus data/korpus-obsidian-raw.txt --tokenizer tokenizer-obsidian-raw.json --out model-obsidian-raw.pt --iters 8000 --n-layer 4 --n-head 4 --n-embd 192 --block 128
# 4) baseline + główny dowód
python ngram_obsidian.py
python eval_obsidian.py    # -> tabela 3x bits/char + werdykty bramek -> eval_obsidian.json
```

## Uczciwy zakres
**Pokazane:** vault da się zamienić w uczciwy dataset (731/731 rekordów lossless); mikro-GPT *uczy się* (55×
nad losowym); pomiar generalizacji jest falsyfikowalny i reprodukowalny. **NIE pokazane (wynik negatywny,
raportowany):** przy ~3 MB transformer **nie bije** n-gramu (pamięć ≥ generalizacja), a dekompozycja **nie**
buduje lepszych reprezentacji dla surowego tekstu. Wyszukiwanie semantyczne świadomie odłożone (tiny LM ≠ embedder).

**Iteracja 2** (gap 0,10 jest mały → warto): większy korpus (~1060 notatek + inne vaulty), `n_embd 256`,
`iters 15–30k`, **deduplikacja boilerplate'u** (uczciwszy held-out), char-level GPT (zdejmuje różnicę tokenizera).

## Dane i licencja
Kod: **MIT** (jak repo). Korpus **nie jest redystrybuowany** — to prywatny vault Obsidiana autora; odtwarza się
przez `prepare_obsidian.py`. Notatki badawcze (dialektyka): [`docs/`](docs/) + kanonicznie w ObsidianVault
`NaukaML/Badania/2026-06-27_obsidian-mikromodel-dekompozycja/` oraz prywatnie `slayerlabs/PrivateSlayNotes`.

Autor: Arkadiusz Słota, kolektyw **Slayer**. Cel edukacyjno-badawczy.
