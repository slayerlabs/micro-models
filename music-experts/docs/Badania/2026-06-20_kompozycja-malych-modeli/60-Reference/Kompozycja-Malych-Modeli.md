---
type: nauka-koncepcja
title: "Kompozycja małych modeli przez wymuszony kontrakt + mapper + wariancja"
status: koncepcja
data: 2026-06-20
created_at: 2026-06-20
author: Arkadiusz Słota
repo: "https://github.com/slayerlabs/micro-models"
tags: [nauka, llm, kompozycja, moe, stitching, kontrakt, mapper]
weryfikacja: "referencje zweryfikowane u źródeł (deep-research round 1 + dorzut A, 2026-06-20)"
---

# Kompozycja małych modeli (kontrakt · mapper · wariancja)

> **Idea Arka:** zamiast łączyć całe macierze wag „każdy-przez-każdy", komponujemy **wiele małych, poprawnych modeli** przez **stały interfejs (kontrakt)** + lekki mapper. Sygnałem trudnych obszarów (styku) jest **wariancja** między ekspertami. Sandbox: muzyka (darmowe nuty). Cel końcowy: czyszczenie/restrukturyzacja IFC.

> Legenda: ✅ proven (istnieje, działa) · 🔬 spekulacja (do weryfikacji) · ⚠️ ograniczenie.

## Teza
Jeden cel (predykcja następnego tokena), **spektrum mechanizmów**. n-gram i transformer to różne technologie; „most" między nimi istnieje, bo są różne. Nasza hipoteza: **mali, poprawni eksperci + wymuszony wspólny interfejs** dają tanią, składalną alternatywę dla dużego modelu.

## Rozstrzygnięty widelec: gdzie jest styk?
**Kontrakt działa na REPREZENTACJI pośredniej (np. po bloku 2), NIE na logitach.** To decyzja przesądzająca o całej niszy:
- **Styk na logitach** (wspólna `lm_head`) → „kompozycja" = **ensembling**; mapper nic nie robi, płaskie ważenie wygrywa, słownik i wariancja zbędne. Łatwe, ale zbadane.
- **Styk na reprezentacji** → mapper ma realną robotę, słownik i wariancja mają sens. **Tu jest nasza nowość.**
Wspólna `lm_head` zostaje, ale **tylko dla porównywalności wyjścia** — cała robota dzieje się na reprezentacji. „Mapper = `W_O` na poziomie meta" działa wyłącznie na V/reprezentacji.

## Składniki
### 1. Wspólne wyjście / „kontrakt" ✅ (najmocniejsze)
Małe modele dzielą **wspólną przestrzeń reprezentacji na styku** (+ wspólny słownik/`lm_head` dla porównywalności) → są mieszalne.
- ⚠️ Kontrakt trzeba **wymusić w treningu**, nie założyć po fakcie (geometria reprezentacji musi się zgadzać).
- Zweryfikowane u źródeł: **ZipIt!** (Stoica i in., ICLR 2024 — *wymaga tej samej architektury*), **Git Re-Basin** (Ainsworth i in., ICLR 2023 — *reprodukowalność zero-barrier LMC jest przedmiotem debaty: REPAIR/BatchNorm*), **model stitching** (Lenc & Vedaldi, CVPR 2015; Bansal i in., NeurIPS 2021 — *u Bansal cel = porównanie reprezentacji, nie produkcja*), **Stitchable Nets** (Pan i in., CVPR 2023).

### 2. Mapper = `W_O` na poziomie meta (rama pojęciowa)
W atencji: Q/K = gdzie patrzeć, **V = treść („property")**, a **`W_O` scala głowice** w strumień. Nasz „mapper między modelami" = `W_O` na poziomie meta: łączy **reprezentacje** sub-modeli (na styku, nie na logitach).

### 3. Interfejs-słownik z koherencji aktywacji 🔬 (nasza nisza)
„Tokenizer-ale-nie-tokenizer": **wyuczony słownik cech** ze statystyk aktywacji na styku, z którego bierzemy tylko cechy **koherentne między modelami**.
- Komponenty ✅: **VQ-VAE codebook** (van den Oord i in., NeurIPS 2017), **Sparse Autoencoders / dictionary learning** (Bricken i in. 2023, Anthropic „Towards Monosemanticity"; Cunningham i in. 2023, arXiv 2309.08600), **Rosetta Neurons** (Dravid, Gandelsman, Efros, Shocher, **ICCV 2023** — cechy wspólne różnym modelom).
- 🔬 Nowość: użycie tego **jako kontraktu-interfejsu** między *ekstremalnie małymi* modelami.

### 4. Wariancja jako sygnał styku ✅/🔬
**Rozbieżność (wariancja) między ekspertami = niepewność = trudny obszar/granica** → tam koncentrujemy i korygujemy routing (adaptacyjna atencja).
- ⚠️ **Pre-check (obowiązkowy zanim zbudujemy routing):** wariancja musi **korelować z błędem** (wykres wariancja vs CE per token). Przy malutkich ekspertach bywa zaszumiona i może łapać „trudny input" zamiast granicy domen — wtedy routing-przez-wariancję nie ma fundamentu.
- Zweryfikowane: **deep ensembles** (Lakshminarayanan, Pritzel, Blundell, NeurIPS 2017), **Attention on Attention** (Huang i in., ICCV 2019).

### 5. Vindex / statystyki aktywacji (narzędzie analizy)
Do znalezienia styku, profilowania domen, pomiaru rozbieżności. ⚠️ Ograniczenie z **L07 (Hase i in., NeurIPS 2023): gdzie można odczytać/edytować ≠ gdzie wiedza jest przechowywana** (napięcie z premisą locate-then-edit **MEMIT** — Meng i in., ICLR 2023) — traktować jako sygnał, nie dowód.

## Plan eksperymentów
- **E0 — kontrolny (self-stitch):** zszyj **ten sam** model ze sobą. Oczekiwane: **bez zmiany**. Wartość: waliduje **mechanizm mappera**, NIE tezę = benchmark zerowy. Zielone = „harness działa", nie „kompozycja działa".
- **E0.5 — geometria (niezależny seed):** zszyj front jednego modelu z backiem **niezależnie wytrenowanej kopii** (ta sama architektura, inny seed, ten sam kontrakt). Testuje, czy kontrakt **wymusza wspólną geometrię** (Lenc-Vedaldi: afiniczny stitch często wystarcza w obrębie architektury). Zielone → E1 prawie pewne; czerwone → napraw kontrakt przed dalej.
- **E1 — pierwszy prawdziwy:** drugi mały model, **ten sam kontrakt, inny styl** = **BACH** (suity wiolonczelowe BWV 1007–1012, partity/sonaty skrzypcowe BWV 1001–1006 — czysta monofonia pod char-level; chorały SATB **wykluczone**). Stitch → mierz perplexity + jakość vs każdy z osobna. *(Drugi model: Slayer-style, jako kontrast do testu styku.)*
- **E2 — wariancja-routing:** po pre-checku kalibracji (Składnik 4); użyj rozbieżności do routingu; zmierz, czy **przewyższa** naiwne mieszanie.
- **Domena:** muzyka (darmowe, otwarte nuty = stały sandbox) → potem **IFC auto-cleaner**.

## Nasza nisza (uczciwie)
NIE „wymyśliliśmy kompozycję modeli" — **te komponenty istnieją** (ZipIt!, stitching, SAE, Rosetta, AoA, ensembles). Nasza testowalna **luka badawcza**:
> *ekstremalnie małe modele + kontrakt wymuszony od zera NA REPREZENTACJI + interfejs-słownik z koherencji aktywacji + wariancja jako sygnał styku — na konkretnej domenie (muzyka → IFC).*

Opieramy się na istniejących pracach i szukamy konkretnej, niezbadanej luki.

## Referencje (zweryfikowane u źródeł — deep-research round 1 + dorzut A, 2026-06-20)
Wszystkie charakterystyki potwierdzone; cytowania + venue twarde:
- ZipIt! — Stoica i in., **ICLR 2024** (arXiv 2305.03053)
- Git Re-Basin — Ainsworth, Hayase, Srinivasa, **ICLR 2023** (arXiv 2209.04836)
- Model stitching — Lenc & Vedaldi, **CVPR 2015**; Bansal, Nakkiran, Barak, **NeurIPS 2021** (arXiv 2106.07682)
- SN-Net — Pan, Cai, Zhuang, **CVPR 2023** (arXiv 2302.06586)
- VQ-VAE — van den Oord, Vinyals, Kavukcuoglu, **NeurIPS 2017** (arXiv 1711.00937)
- SAE — Bricken i in. 2023 (Anthropic); Cunningham i in. 2023 (arXiv 2309.08600)
- Rosetta Neurons — Dravid, Gandelsman, Efros, Shocher, **ICCV 2023** (arXiv 2306.09346)
- AoANet — Huang, Wang, Chen, Wei, **ICCV 2019** (arXiv 1908.06954)
- Deep ensembles — Lakshminarayanan, Pritzel, Blundell, **NeurIPS 2017** (arXiv 1612.01474)
- MEMIT — Meng i in., **ICLR 2023** (arXiv 2210.07229) · kontra: Hase i in., **NeurIPS 2023** (arXiv 2301.04213)
- NPLM — Bengio, Ducharme, Vincent, Jauvin, **JMLR 2003**
- kNN-LM — Khandelwal i in., **ICLR 2020** (arXiv 1911.00172) — ppl **15.79** na WikiText-103, poprawa **2,86 pkt** (SOTA wg stanu 2020)
- Infini-gram — Liu i in., **COLM 2024** (arXiv 2401.17377) — ~7 B/token (wg treści)

## Powiązania
[[Research-NGram-vs-MiniTransformer]] · [[Capstone-Muzyka-LOG-Realizacja]] · rejestr: [[Badania-INDEX]] · repo: github.com/slayerlabs/micro-models
