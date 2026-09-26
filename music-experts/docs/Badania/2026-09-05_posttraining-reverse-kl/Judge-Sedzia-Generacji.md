---
type: koncepcja
title: "Judge — automatyczny sędzia jakości generacji (benchmark)"
status: aktywne
data: 2026-09-04
author: Adam Skrodzki 
related: "[[README]] · src/core/judge.py · src/tools/train_judge.py · src/tools/judge_tunes.py"
---

# Judge — automatyczny sędzia jakości generacji

## WHY — po co to jest

### Problem: perplexity nie mierzy jakości muzyki

Nasza jedyna dotychczasowa miara jakości eksperta to **perplexity na held-oucie** — czyli
pytanie „jak dobrze model przewiduje następny znak tekstu, na którym się nie trenował". To
uczciwa miara *dopasowania do korpusu*, ale ma dwie ślepe plamy:

1. **PPL nie odpowiada na pytanie „czy to jest muzyka?"** — model może mieć niską PPL i
   generować ciągi znaków ABC, które *wyglądają* jak nuty, ale nie trzymają metrum, tonacji
   ani rytmu typowego dla tańca. Do tej pory to sprawdzaliśmy **uchem** — qualitatywnie,
   niereprodukowalnie, bez liczby do tabeli.
2. **PPL nie porównuje modeli trenowanych na różnych korpusach** (jak Bach vs jig —
   dopisek w README: „ppl nie jest wprost porównywalne").

Potrzebna jest **niezależna, automatyczna, reprodukowalna miara treści muzycznej** — sędzia,
który spojrzy na wygenerowaną melodię i odpowie na pytania, na które odpowiedziałby człowiek
znający irlandzki trad: *jakie to metrum? jaka tonacja? do jakiego tańca to pasuje?*

### Idea: klasyfikator jako sędzia (classifier-as-judge)

Trenujemy klasyfikator na **prawdziwych** melodiach, które mają
etykiety: `meter` (7 klas), `mode` (23 klasy), `type` (12 klas). Potem pokazujemy mu
**wygenerowane** melodie i sprawdzamy zgodność:

- ekspert `jig_ckpt` generuje z promptem `M:6/8 K:D` → sędzia powinien powiedzieć
  `6/8 + D + jig`. **Zgoda = model trzyma styl.**
- sędzia mówi `4/4 + reel` → **dryf stylu** — model „wyszedł poza swoją specjalizację".

To daje liczbę (procent zgodności), którą można kłaść do tabeli obok PPL — i którą można
porównywać *między eksperymentami kompozycji* (E0/E1/ensemble: który spójnie łączy ekspertów,
a który produkuje mush?). Ostatecznie to pierwszy krok do odpowiedzi na pytanie z KMC1:
**czy kompozycja małych modeli daje w ogóle mierzalny zysk?**

### Czemu sędzia widzi tylko ciało melodii

Generowane melodie zawsze mają nagłówek `X:1 M:6/8 K:D` — bo to jest prompt. Gdyby sędzia
widział nagłówek, nauczyłby się czytać odpowiedź z promptu („M:6/8 → meter=6/8", gotowe,
accuracy 100%) i niczego nie mierzył. Dlatego featury budujemy **wyłącznie z ciała melodii**
(nut), a nagłówki `X:/T:/C:/M:/K:/N:/%` są wycinane przed featuryzacją. Sędzia musi wnioskować
metrum i tonację z **rytmu i rozkładu dźwięków** — tak jak człowiek.

## WHAT — co to jest

### Budowa: `JudgeGPT` (`src/core/judge.py`)

Klocki **te same co eksperci** — verbatim reuse `core/gpt.py`

```
ciało melodii (znaki, do 256)  ─┐
                                ├─ trunk GPT (verbatim: tok_emb+pos_emb → 4×Block → ln_f)
maska paddingu ─────────────────┘         │
                                     mean-pool po pozycjach (tylko prawdziwe znaki)
                                          │
                            ┌─────────────┼─────────────┐
                       głowa meter    głowa mode    głowa type     (nn.Linear, ~n_klas wyjść)
```

Kluczowe decyzje projektowe:

- **CausalAttention wystarcza.** Nie pisaliśmy bidirectional attention. Padding jest na
  KOŃCU sekwencji, więc tokeny prawdziwe nigdy nie widzą paddingu (maska dolnotrójkątna to
  gwarantuje), a maska jest potrzebna tylko do poprawnego mean-poolingu. Reprezentacja po
  poolingu zagregowana po całej melodii — kierunkowość uwagi przestaje mieć znaczenie.
- **Trunk = cały GPT**, razem z nieużywaną głową LM (weight tying z embeddingiem sprawia, że
  nie kosztuje dodatkowej pamięci znaczącej). To świadomy wybór: nie „wyciągamy" bloków
  ręcznie, tylko używamy klasy `GPT` taką, jaka jest — mniej kodu, zero rozjazdu z eksper-
  tami, i otwarta droga do `--init-from` (transfer z pretrenowanego eksperta).
- **Okna ciała 256 znaków.** Ciała mają 40–700 znaków. Trening: losowe okno na przykład na
  każdy krok (augmentacja — przez czas treningu model widzi całe ciało, nie tylko początek).
  Eval: wszystkie okna niepokrywające, softmax uśredniony po oknach → jedna predykcja na
  melodię (pełne, deterministyczne pokrycie).

### Dane i split (`data/tunes.csv`)

- Czyszczenie **współdzielone z prepare_data.py** (wyciągnięte do `src/core/abc_corpus.py`):
  usunięcie akordów `"..."`, normalizacja tonacji (`Edorian` → `Edor`), filtr długości ciała
  40–700 znaków. Sędzia widzi dokładnie tę samą dystrybucję, którą widział generator.
- **Split 90/10 po `tune_id`** — to ważne: jedna melodia ma często kilka settingów (rows w
  csv) niemal identycznych. Losowy split po wierszach przeciekałby (copypasta w val),
  split po `tune_id` gwarantuje, że val to melodie, których sędzia nigdy nie widział.

### Zastrzeżenie: split chroni sędziego, ale nie generatory

Split po `tune_id` gwarantuje czystość walidacji **sędziego**. Generatory (eksperci,
universalista, nauczyciele `_sh`) są trenowane na korpusach z `prepare_data.py`, które
biorą CAŁY `tunes.csv` — w tym melodie z val splitu sędziego. Sędzia nie wycieka do
treningu generatorów (żaden generator nie widzi jego wyjść ani wag), ale generator mógł
zapamiętać surowy tekst melodii walidacyjnych. Konsekwencje:

- **Porównania międzymodelami są uczciwe** — wyciek jest wspólny dla wszystkich
  benchmarkowanych generatorów (eksperci, universalista, studenci RKL).
- **Wartości absolutne są optymistyczne** — continuation jest najbardziej narażony
  (prompt = ćwiartka ciała melodii walidacyjnej, którą generator mógł zapamiętać),
  scratch najmniej (same nagłówki). Sufit (`ref`) jest niewrażliwy.
- **Clean-room wariant** (dla paperu): `prepare_data.py` z wykluczeniem val `tune_id`
  splitu sędziego + przetrenowanie wszystkiego.

### Wyniki

| głowa | val acc (judge_v2) | baseline (klasa większościowa) | interpretacja |
|---|---|---|---|
| meter | **0.922** | ≈ 0.5 (4/4) | mocno ponad baseline |
| mode  | **0.717** | ≈ 0.35 (D) | blisko sufitu informacyjnego (patrz niżej) |
| type  | **0.816** | ≈ 0.35 (reel) | mocno ponad baseline |

(pierwsza wersja sędziego, 2000 kroków, dawała 0.891/0.688/0.778 — judge_v2 to 5000 kroków,
batch 64; kanoniczny checkpoint: `data/models/judge_v2.pt`)

**Top pomyłki są muzycznie prawidłowe** — i to jest najważniejszy wynik walidacyjny:

- `hornpipe→reel` (185), `reel→hornpipe` (82): oba to 4/4, różnią się tylko **feelingiem**
  rytmicznego puntowania. Nawet ludzie się spierają.
- `3/4→6/8`, `9/8→12/8`: grupowanie ósemek zamiast samego liczenia — podobieństwo rytmiczne.
- `Bmin→D`, `Edor→Emin`, `Amix→Ador`: **to ten sam zbiór dźwięków** (tonacja względna /
  tryby o tych samych znakach). Z samego rozkładu nut te pary są *nieodróżnialne wprost* —
  rozstrzyga je tylko waga kadencji. ~0.7 na mode jest więc blisko sufitu tego, co można
  wywnioskować z zawartości dźwiękowej.

Sędzia myli się tam, gdzie muzycznie *powinien* się mylić — czyli mierzy strukturę muzyczną,
a nie artefakty formatu. Val acc ≈ train acc: brak overfittingu.

## HOW — jak tego używać

### Trening

```bash
python src/tools/train_judge.py
# warianty:
python src/tools/train_judge.py --init-from data/models/jig_ckpt.pt   # trunk od eksperta (transfer learning — eksperyment do paperu)
python src/tools/train_judge.py --iters 5000 --lr 3e-4                # dłuższy trening
```

Pełna lista flag: `--csv --out --seed --iters --batch --lr --block --eval-every
--train-eval-n`. Co 100 kroków logowane są **train acc** (stała próbka 5000 przykładów) i
**val acc** (cały val); gap między nimi to sygnał overfittingu. Batch = 32 melodie/krok;
próbkowanie losowe bez epoch — ten sam przykład wraca średnio co `len(train)/batch` ≈ 1541
kroków, za każdym razem z nowym losowym oknem. Model zapisuje best-val do
`data/models/judge_v2.pt` (domyślny `--out`).

### Ocena wygenerowanych melodii

```bash
python src/tools/judge_tunes.py out/               # pliki .abc i/lub katalogi (rekurencyjnie)
python src/tools/judge_tunes.py out/ --json        # postać maszynowa (benchmark scripts)
```

Wynik: na każdą melodię predykcja meter/mode/type + pewność (softmax). Melodię z jednego
pliku wielomelodiowego skrypt tnie po liniach `X:` (ta sama konwencja co `first_tune` w
generatorach).

### Interpretacja: zgodność z ekspertem

Dla eksperta `E` generującego z promptu `M:m K:k` sprawdzamy:
- `judge.meter == m` (metryka z promptu),
- `judge.mode == k` (tonacja z promptu),
- `judge.type == typ korpusu E` (jig→jig, reel→reel...).

**Procent zgodności = benchmark jakości generacji E.** Niska zgodność przy dobrym PPL to
sygnał, że model nauczył się powierzchni tekstu, a nie struktury muzycznej. Ten sam miernik
możemy przyłożyć do ensemble'u i stitchy — porównanie zgodności między metodami kompozycji
to główny przewidywany eksperyment (E-JUDGE).

## Benchmark ekspertów (E-JUDGE): domena i OOD

### Benchmark domenowy — `benchmark_judge.py`

Eksperta oceniamy TYLKO na jego własnym terenie. Pula = val split ograniczony do **domeny
checkpointu** (`data/models/domains.json`: substring typu + metrum, jak w `prepare_data.py`).
Dwie próby: **scratch** (tylko nagłówki `M:`/`K:`, model generuje od zera; type pominięty —
model nie ma jak go znać) i **continuation** (ćwiartka prawdziwego ciała w promptcie; sędzia
ocenia całość). Score = średnie P sędziego przy prawdziwych klasach, mianownik = WSZYSTKIE
próbki: porażka modelu w domenie = 0 (bez survivor bias). Referencja = sędzia na prawdziwych
melodiach tej domeny (sufit). Ten sam seed = identyczne melodie dla każdego modelu.

**Wyniki (n=100, judge_v2, score/sufit):**

| rodzina | continuation | stosunek do sufitu | scratch mode P | wniosek |
|---|---|---|---|---|
| jigi (11 wariantów) | 0.61–0.66 / 0.77 | **0.80–0.86** | 0.15–0.27 | najsilniejsza rodzina; `jig_l1` najlepszy |
| reele (2) | 0.53–0.55 / 0.76 | 0.69–0.72 | 0.23–0.27 | najsłabsi u siebie |
| walc | 0.48 / 0.56 | 0.86 (miękki sufit) | 0.17 | type = 1.00 sufitu, ale sufit niski |

Kluczowe obserwacje:
- **Ranking sędziego zgadza się z rankingiem PPL** (jig 3.80 najlepszy) — dwie niezależne
  miary, jeden werdykt. Pierwsza wersja benchmarku (krzyżowa, bez domen) dawała odwrotność —
  artefakt bazowych częstości klas; przejście na pule domenowe to naprawiło.
- **Mode z nagłówka słabo, z kontekstu lepiej**: w scratch sędzia przypisuje prawdziwej
  tonacji tylko ~0.2; w continuation ~0.35 (prefiks niesie materiał dźwiękowy). Modele
  czytają tonację z nut, nie z `K:` — konkretna, sprawdzalna teza o tym, czego nauczył się
  char-LM.
- **Zero śmieci w domenie**: wszystkie pominięcia to niedopasowania słownika (te same 7 wierszy
  dla każdego jiga — deterministyczny seed), nie generacje-garbage.

### Macierz OOD — `benchmark_ood.py`

Wiersze = eksperci, kolumny = domeny celu. Prawda = to, o co *prosimy* (etykiety melodii
celu — model jest proszony, więc nie ma wymówki). Komórka: score/ref, **hb = home-bias**
(P sędzia przypisze KLASIE DOMOWEJ modelu: meter, a w continuation też type) i pokrycie.
hb rozróżnia dwa tryby porażki: wysoki hb = model sztywny, ignoruje prompt i wraca do
domeny; niski hb + niski score = papka. Bach wyłączony (`type: null`) — bez wspólnego
słownika nie da się uczciwie mierzyć (patrz „następny krok").

**Wyniki (n=30, judge_v2; diagonala = domena własna):**

- **Diagonala dominuje wszędzie** — specjalizacja ekspertów jest realna, przeżyła pierwszy
  adwersarialny test.
- **Nagłówki nie sterują poza domeną**: OOD scratch ma hb 0.85–0.97 — jig z promptem `M:4/4`
  generuje 6/8 z pewnością 0.94. Sterowanie promptem jest martwe.
- **Kontekst steruje w połowie**: OOD continuation spada hb do ~0.40–0.55, score rośnie do
  40–63% sufitu. Wniosek projektowy: w kompozycji steruj kontekstem (stitch), nie nagłówkami.
- **Asymetria reele→jigi**: najlepsze komórki OOD w całej macierzy (0.48–0.52, hb type
  zaledwie 0.20–0.24) — eksperci 4/4 są elastyczni, 6/8 to trudniejszy nawyk do zdjęcia.
  Kandydaci na „dawców" w modelach mieszanych.
- **Elastyczność siedzi w małych modelach**: e32 mają najniższy hb wszędzie (0.62–0.79)
  przy najsłabszej diagonali; duże modele hb 0.90–0.97. Nowa, testowalna hipoteza pod E1:
  **komplementarność może wolić małe, „gibkie" modele mimo gorszego PPL.**
- Walc na diagonali osiąga sufit (0.52 vs 0.50) — z zastrzeżeniem, że jego sufit jest miękki
  (mała domena 348 melodii, sędzia najmniej pewny na 3/4).

Ograniczenia: n=30/komórkę (±0.06) i sufit liczony na próbce, nie na całej puli — wersja
pod raport wymaga pełnego przebiegu (n=100) i referencji z całej puli domenowej.

### Pliki

| plik | rola |
|---|---|
| `src/core/abc_corpus.py` | wspólne czyszczenie ABC (reuse w prepare_data i judge) |
| `src/core/judge.py` | `JudgeGPT`: trunk GPT + mean-pool + 3 głowy |
| `src/tools/train_judge.py` | trening sędziego z tunes.csv |
| `src/tools/judge_tunes.py` | ocena plików .abc |
| `src/tools/benchmark_judge.py` | benchmark domenowy ekspertów (score/pokrycie/sufit) |
| `src/tools/benchmark_ood.py` | macierz OOD (ekspert × domena celu) |
| `run_benchmarks.sh` | sweep wszystkich checkpointów sędzią judge_v2 |
| `data/models/domains.json` | domena każdego checkpointu (type/meter) |
| `data/models/judge_v2.pt` | kanoniczny sędzia (5000 kroków) |
| `data/benchmarks/*.log` | logi przebiegów benchmarków |
