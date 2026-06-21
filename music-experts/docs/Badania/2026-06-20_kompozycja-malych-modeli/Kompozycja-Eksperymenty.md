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

> Wyniki E0/E0.5/E1/E2. Kryteria rozstrzygnięcia w syntezach. Kod: `src/compose/e0_stitch.py` (`C:/ProjektyPublic/MuzykaML`).

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
Prerequisite: **wspólny słownik (53)** — reel przetrenowany na słowniku walca (`reel_sv_ckpt.pt`, val 1,582/ppl ~4,9; retrening nie zaszkodził). `src/compose/fuse.py`: w każdym kroku `logits = α·walc + (1-α)·reel` → sampling (assert wspólnego stoi przechodzi).
- **α=0,5** (render fortepian) — odsłuch: **oceniona jako ładna** (blend stylów zadziałał).
- **α=0,3** (więcej reela, render skrzypce).
- **Duet** (`src/compose/duet.py`): walc (fortepian) + reel (skrzypce) **jednocześnie** — dwie ścieżki, ta sama tonacja; granie razem, nie fuzja w jedną linię.
> To jest **BASELINE (KMS5, płaskie ważenie)** — działa i jest słyszalny. Stitch reprezentacji (E1, nasza nisza) ma to **pobić**; ensemble jest punktem odniesienia.

## E1 — stitch reprezentacji (front walc × g × back reel) — 2026-06-21
Wspólny słownik (53). Liniowy mapper **g (128×128)** na styku po bloku 2; reszta zamrożona; trening g 800 kroków. `src/compose/e1_stitch.py`.

**Pomiar 1 (SKRZYWIONY — lekcja):** held-out ze sklejonych korpusów → val wyszedł ~czysty reel → „stitch bije ensemble (5,18<5,60)" było **artefaktem** (reel-alone i tak najlepszy 4,92). Złapane, poprawione.

**Pomiar 2 (zbalansowany, val = 50% walc + 50% reel):**
| model | ppl |
|---|---|
| walc-alone | 6,20 |
| reel-alone | 5,87 |
| **ensemble** | **5,15** |
| stitch | 5,18 |

**Wynik (uczciwie):**
- ✅ **Ensemble bije OBA pojedyncze modele** → łączenie ekspertów realnie pomaga na zadaniu mieszanym.
- ❌ **Stitch ≈ ensemble** (5,18 vs 5,15) → stitch reprezentacji NIE pobił baseline'u. Złożoność nie zarobiła na siebie *w tym wariancie*.

**Interpretacja:** najtrudniejszy przypadek — **post-hoc, liniowy g, BEZ wymuszonego kontraktu** (KMS2), z pominięciem E0.5. Niezależnie trenowane modele mają różne geometrie → jeden liniowy mapper nie wyrównuje ich lepiej niż uśrednianie. **Hipoteza KMS1 niepotwierdzona TĄ drogą** (negatywny, ale wartościowy wynik). Droga do przewagi: wymuszony kontrakt (KMS2) / bogatszy mapper (KMT3) / najpierw E0.5.

## ⚠️ Pułapka metrum — warunek konieczny dla KAŻDEGO eksperymentu z routingiem
Jig (6/8), walc (3/4), reel (4/4) są **rozłączne po nagłówku `M:`**. Router/MoE na takiej parze pokaże ~100% trafności z **trywialnego** powodu: ściąga decyzję z jednego tokena metrum, nie z głębokiej reprezentacji. To zafałszuje E2/MoE/wariancja-routing — „działa" będzie artefaktem pomiaru, nie kompozycji.
- **Dotyczy:** routingu (KMT4, E2). **NIE dotyczy E1** — tam porównujemy perplexity stitcha vs ensemble, bez routera, więc token `M:` nic nie ściąga.
- **Lek:** domeny w **TYM SAMYM metrum** — walc vs mazur (oba 3/4) albo reel vs hornpipe (oba 4/4). Wtedy model nie może oszukać po nagłówku, a test mierzy to, co myślimy.
- Spójne z tezą *„ile reprezentacji wstrzyknąć = funkcja nakładania się domen"* ([[Research-NGram-vs-MiniTransformer]]): test routingu MUSI mieć domeny **nakładające się**.

## Caveat symetrii E1 (dopisany 2026-06-21)
Nasz stitch jest **kierunkowy** (front=walc × g × głowa=reel), ensemble **symetryczny** (miesza obie głowy). Porównaliśmy więc graft-w-jedną-stronę z fuzją-w-obie — lekko jabłka/gruszki. Stitch kierunkowy może być co najwyżej tak dobry jak „back-reela przetwarzający cechy frontu-walca". Uczciwszy test „czy kompozycja reprezentacji bije mieszanie wyjścia" to **symetryczna** fuzja reprezentacji do wspólnej głowy. Nie unieważnia remisu (5,18≈5,15), ale go niuansuje — do dopisania w paperze.

## E_CKA — czy niezależne maluchy mają WSPÓLNĄ geometrię? (NOWY, tani, decydujący — następny w kolejce)
**Pytanie:** czy nasze niezależnie trenowane modele (jig, jig-v2, walc, reel) zbiegają do wspólnej reprezentacji, czy mają rozłączne geometrie? (Robocza nazwa hipotezy: „platońska konwergencja" — cytat w [[Emergencja-i-Wspolna-Reprezentacja]], weryfikacja u źródeł w toku.) Dla dużych modeli badane; **na mikro-skali (~0,8M) — nietknięte**.
**Metoda:** policz wzajemne podobieństwo reprezentacji warstw (CKA / mutual-kNN) między modelami na wspólnym zbiorze próbek. **~godzina roboty, zero treningu.**
**Czemu decydujący:**
- **Zbiegają** (zaskoczenie, mocny sygnał) → *„platońska konwergencja już na 10⁶ param"* + relative representations dają darmową interoperacyjność bez uczonej kotwicy.
- **NIE zbiegają** → spójne z E1-negatywem (brak wspólnej geometrii ⇒ stitch ≈ ensemble).

**⚠️ Uczciwość pomiaru (inaczej wynik nieinterpretowalny):**
- Hipoteza platońska (Huh i in., ICML 2024 — *position paper*) to teza *o zachowaniu wraz ze SKALĄ*: konwergencja ma rosnąć z rozmiarem. **Brak zbieżności na 0,8M NIE falsyfikuje hipotezy** (jest z nią zgodny); *obecność* zbieżności na mikro-skali to mocniejszy sygnał. → mierzyć **trend względem skali** (sweep: 0,2M / 0,8M / 3M…), nie pojedynczy punkt.
- Potrzebny **null/baseline**: losowe (nietrenowane) sieci, permutacje — żeby odróżnić realną zgodność od trywialnej. Sam CKA bez baseline nie rozstrzyga.
> **Wskakuje PRZED shared-trunk:** najpierw zmierz (sweep skali + baseline), czy geometria jest wspólna, zanim budujesz mechanizm jej wykorzystania. Pełny rozkład + cytaty: [[Emergencja-i-Wspolna-Reprezentacja]].

## E0.5 — niezależny seed (uzasadnione wynikiem E1; po E_CKA)
Front modelu A × back **niezależnie wytrenowanej kopii** (inny seed, ten sam kontrakt). Test: czy kontrakt **wymusza wspólną geometrię**. Wymaga **jig-v2** (drugi seed). Δppl mały → kontrakt trzyma; duży → naprawić przed E1. (E_CKA mierzy to **pasywnie** — bez stitchu; E0.5 to ten sam test aktywnie, przez złączenie.)

## Powiązania
[[Kompozycja-INDEX]] · [[Kompozycja-Malych-Modeli]] · kryterium: [[KMS1-Kompozycja-Gdy-Kontrakt|KMS1]]
