---
type: nauka-reference
id: KM-EMERG
title: "Emergencja i wspólna reprezentacja — czy „jednocześnie" da się usunąć?"
status: zywy-dokument
data: 2026-06-21
created_at: 2026-06-21
author: Arkadiusz Słota
---

# Emergencja i wspólna reprezentacja

> Rama teoretyczna dla całego programu kompozycji. Pytanie nośne: czy sieć **niezależnych** mikro-modeli może wejść w koherencję i przeskalować się jak duży LLM — **bez** joint-treningu? Tu: warunki konieczne, czym można zastąpić „jednocześnie", prawo zachowania kotwicy, świeży front i nasz tani eksperyment rozstrzygający. **Wszystkie cytaty zweryfikowane u źródeł pierwotnych (2026-06-21).**

## Dwie tezy, których nie wolno mylić
- **Teza A — neurosymboliczna orkiestra:** mikro-eksperty per funkcja + dużo zweryfikowanego kodu + orkiestrator. Realna, inżynierska. Kierunek **produktu IFC** → [[Cele-Globalne-i-Kotwica]].
- **Teza B — emergencja z niezależnych:** federacja sama wchodzi w koherencję i daje „więcej niż suma". Research-bet, niski historyczny base-rate („society of mind" Minsky'ego); **nasz E1 ją drasnął** (stitch ≈ ensemble — brak „więcej niż suma").

## Warunki konieczne emergencji/transferu (wszystkie muszą zajść)
1. **Wspólna przestrzeń reprezentacji** — jedna geometria, w której cechy się zazębiają.
2. **Sprzężenie gradientów przez wspólne wagi** — ten sam cel ciągnie różne umiejętności w te same parametry. *To JEST słowo „jednocześnie".*
3. **Pojemność ≥ podłoga informacyjna** zadania (~**2 bity/param** — Allen-Zhu & Li 2024).
4. **Szerokość + skala danych** — żadna umiejętność nie dominuje → optymalizator zmuszony znaleźć cechy ogólne, nie zadaniowe.
5. **Kompozycyjna struktura w danych** — transfer wymaga wspólnych pod-struktur (gdyby umiejętności były naprawdę niezależne, transfer jest niemożliwy z definicji).
6. **Optymalizacja znajdująca wspólne rozwiązanie** (indukcyjne biasy + dość treningu).

**Rdzeń:** emergencja = **dzielenie gradientu we wspólnych wagach**. Trenując X i Y o te same parametry, wymuszasz reprezentację służącą obu — stąd reużycie i transfer.

## Czym MOŻNA zastąpić „jednocześnie" (proven → spekulacja)
| Mechanizm | Co robi | Dojrzałość |
|---|---|---|
| Wspólna **zamrożona kotwica** (backbone/embedding/codebook) + adaptery/LoRA | eksperty czytają/piszą do jednej zamrożonej przestrzeni → dziedziczą geometrię bez wspólnego treningu | proven |
| Wspólny **dyskretny codebook** jako interlingua ([[KMT3-Slownik-Z-Aktywacji\|KMT3]]) | wszyscy mówią jednym zamrożonym słownikiem cech; nowy ekspert uczy się go „mówić" → interoperuje | nasza najnowsza wykonalna ścieżka |
| **Naprzemienny co-trening** (Gibbs/EM, Coconet) | „jednocześnie" → „naprzemiennie"; w granicy ≈ joint (block-coordinate descent) | proven |
| **Destylacja** do wspólnej przestrzeni (post-hoc) | = nasz E1; liniowo stratny | słaby liniowo |
| **Orkiestrator** na inferencji (MoE, agenci, zamrożeni eksperci) | koherencja siedzi w routerze, nie w ekspertach | proven (MoE) |

## ⚖️ Prawo zachowania kotwicy
Każda z tych dróg wymaga **wspólnej kotwicy, którą KTOŚ raz nauczył szeroko.** Można uczynić **rozszerzanie** tanim i modularnym — ale wspólna geometria **nie emerguje za darmo** z niezależnych części. **E1 padł dokładnie dlatego, że kotwicy nie było** (post-hoc + liniowy mapper, zero wspólnego podłoża). To nie pech — to ta zasada.

Konsekwencja dla „zmiana w A automatycznie pojawia się w B bez sklejania": wymaga wspólnego punktu styku. Zmień **kotwicę** → zmienia się wszystko, co ją czyta; zmiana lokalnego eksperta **nie propaguje** (brak ścieżki przepływu informacji). Czego naprawdę chcemy („dodaj eksperta, od razu gra z resztą, bez retreningu") = **wspólna zamrożona interlingua + maluchy nią mówiące** = [[KMT3-Slownik-Z-Aktywacji|KMT3]] w roli nośnej.

## Pojemność — uczciwie
Redukujemy **aktywny compute** (MoE = rzadka aktywacja) i koszt treningu/rozszerzania (zamrożona baza + tanie adaptery) — realne wygrane. Ale **nie pobijemy podłogi informacyjnej** (~2 bity/param): „mniejsza pojemność" = „mniej aktywnego compute + tania rozszerzalność", NIE „więcej wiedzy w mniej bitach". Wiedza musi gdzieś leżeć; można ją tylko rzadko aktywować i tanio dokładać.

**Szczepionka na rozczarowanie:** część „emergencji" w LLM to **artefakt metryki** — twarda/nieciągła metryka (np. exact-match) robi pozorny skok tam, gdzie ciągła rosła gładko (Schaeffer i in., NeurIPS 2023). Cel „nagły przeskok zdolności" może być po części złudzeniem pomiarowym; cel „tania, modularna, rozszerzalna sieć przez wspólną kotwicę" jest realny.

## Świeży front (gdzie nowość realnie może żyć)
- **Relative representations** (Moschella i in., **ICLR 2023**, arXiv:2209.15430): reprezentuj próbkę przez podobieństwa (cosine) do zbioru wspólnych **kotwic-próbek** → reprezentacja **niezmiennicza na izometrie latentu** → niezależnie trenowane modele **stitchowalne zero-shot**, bez wspólnego backbone'u i bez joint-treningu. Wspólny język = geometria względem wspólnych przykładów (nie uczony codebook).
- **Platonic Representation Hypothesis** (Huh i in., **ICML 2024 — Position paper**, arXiv:2405.07987): niezależnie trenowane modele (też różnych modalności) zbiegają ku wspólnej geometrii **wraz ze skalą**. To **hipoteza** (argumentowana obserwacjami konwergencji), nie udowodnione twierdzenie. Jeśli prawdziwa — niezależność nie blokuje wspólnej przestrzeni: ona emerguje z dobrego treningu, nie z joint-treningu ani kotwicy.
- **Task arithmetic** (Ilharco i in., **ICLR 2023**, arXiv:2212.04089) + **TIES-merging** (Yadav i in., **NeurIPS 2023**, arXiv:2306.01708) + **DARE** (Yu i in., **ICML 2024**, arXiv:2311.03099): zdolności składają się arytmetycznie w przestrzeni wag („wektor zadania" = wagi_finetune − wagi_baza). Haczyk: nadal **wspólna baza**.
- **CKA — Centered Kernel Alignment** (Kornblith i in., **ICML 2019**, arXiv:1905.00414) + **mutual k-NN** (główna metryka pracy Platonic): metryki podobieństwa reprezentacji między sieciami, niezmiennicze na obrót/izotropowe skalowanie.

## 🔬 Nasz eksperyment rozstrzygający — E_CKA na mikro-skali
**Pytanie:** czy hipoteza platońska zachodzi na ~10⁶ param? Badana tylko na **dużych** modelach — **mikro-skala nietknięta**. Policz CKA / mutual-kNN między niezależnymi maluchami (jig, jig-v2, walc, reel) na wspólnym zbiorze próbek.

**Uczciwość pomiaru (krytyczna — inaczej wynik nieinterpretowalny):**
- PRH przewiduje konwergencję **rosnącą ze skalą** → *brak* zbieżności na 0,8M **jej nie falsyfikuje** (jest z nią zgodny); *obecność* zbieżności na mikro-skali to mocny sygnał. Mierz **trend po skali** (sweep kilku rozmiarów), nie pojedynczy punkt.
- Potrzebny **null/baseline** (losowe sieci, permutacje) — żeby odróżnić realną zgodność od trywialnej.

**Wartość (z poprawnym setupem):** zbiegają → paper „platońska konwergencja już na 10⁶"; nie zbiegają (przez cały sweep) → „konwergencja wymaga progu skali; poniżej kompozycja niezależnych = ensemble" — tłumaczy E1. Detale: [[Kompozycja-Eksperymenty]].

## Powiązania
[[Kompozycja-INDEX]] · [[KMS2-Kontrakt-Vs-Posthoc|KMS2]] (prawo zachowania) · [[KMT3-Slownik-Z-Aktywacji|KMT3]] (interlingua) · [[Wnioski-i-Dowody]] · [[Cele-Globalne-i-Kotwica]]
