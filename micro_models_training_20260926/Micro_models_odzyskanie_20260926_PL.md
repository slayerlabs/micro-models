# Odzyskanie i walidacja projektu Micro-models

Stan: **2026-09-26T02:56:38+02:00** (Europe/Warsaw).

**Odzyskano kompletną pierwszą serię treningów. Nie stwierdzono uszkodzenia
archiwum, danych ani checkpointów.** Robocze kopie zostały usunięte podczas
konserwacji środowiska. Trwały zapis wykonany 26.09.2026 o 00:53 czasu polskiego
zawierał zakończone treningi; nie było przerwanego przebiegu do dokończenia.

## Co odzyskano i sprawdzono

- Archiwum 26 494 665 bajtów, 63 wpisy ZIP. Kontrola CRC bez błędów.
- SHA-256 całego odzyskanego archiwum jest identyczny z wcześniej zapisanym:
  `7f193c38a5e94588a52612c760f8bfc058fa6222a42f9b645e453f7fa0139f49`.
- Wszystkie **62 pliki** wymienione w pierwotnym `SHA256SUMS` mają zgodne sumy.
- Oddzielnie odzyskane raport, notebook i plik kontynuacji są identyczne z ich
  odpowiednikami w archiwum. Wszystkie 33 pliki JSON/JSONL/notebook zostały odczytane.
- Zachowano sześć checkpointów: najlepszy i ostatni dla NPLM oraz dwóch GPT.
  Wszystkie oryginalne pliki danych i przebiegów mają nadal te same bajty.

Oryginalny pakiet był kompletny. Nowa wersja dodaje dowody odzyskania, kontrolę
wznowienia, próbki i aktualny opis stanu. Jej suma całego ZIP będzie inna,
ponieważ zawiera nowe pliki; sumy checkpointów i danych pozostają niezmienione.

## Ponowne sprawdzenie danych

Odtworzono środowisko z `requirements-lock.txt` i pobrano dokładnie przypięty
commit korpusu źródłowego `0fd9e00542445a522c6030c80c687b874aa569d5`.
Ponowny double check zakończył się **19/19 pozytywnych kontroli**. Obejmuje źródła
370 zapisów, konwersję wszystkich 352 zachowanych melodii, rozłączność grup
oraz kompletność ocenianych znaków. Podział pozostaje **281/36/35** melodii
w treningu/walidacji/teście. Nie zmieniono alfabetu, grup ani progu podobieństwa.

To ponowne wykonanie istniejącego niezależnego skryptu kontrolnego, nie opinia
drugiego człowieka. Nie wykryto przecieku według zapisanych kryteriów; nie jest
to gwarancja wykrycia każdej historycznej relacji między melodiami.

## Stan treningów i rzeczywiste odtworzenie wyników

| Model | Wykonane aktualizacje | Najlepszy krok | Walidacyjna PPL części muzycznej |
|---|---:|---:|---:|
| NPLM | 2000/2000 | 1200 | 1,8969843435 |
| GPT A, seed 20260926 | 2000/2000 | 1600 | 1,9828416805 |
| GPT B, seed 20260927 | 2000/2000 | 1800 | 1,9746776664 |

Wagi wczytano w świeżym procesie z `weights_only=True`, sprawdzono ich typ FP32,
skończoność, liczbę parametrów, hashe kodu i stan optymalizatora. Ponownie policzona
pełna walidacja dała **różnicę 0,0** wobec zapisanych wyników wszystkich trzech
modeli. Wybrane checkpointy odpowiadają minimom krzywych walidacyjnych.
N-gramy, ich zliczenia i wyniki również zostały odzyskane bez zmiany bajtów.

Dodatkowo wykonano próbę wznowienia na kopiach w pamięci: porównano dwa kolejne
kroki z przerwaniem po pierwszym, serializacją i odtworzeniem modelu, optymalizatora
oraz obu generatorów losowych. Batch, strata i wszystkie parametry były identyczne
(**maksymalna różnica 0,0** dla każdego modelu). Te kroki były diagnostyką;
nie zapisano ich do właściwych przebiegów. Ich licznik nadal wynosi 2000.

**Zbiór testowy nadal nie był używany do obliczania jakości modeli.** Analiza
struktury testu w audycie nie jest ewaluacją predykcji na teście.

## Notebook

Ponownie wykonano od początku wszystkie **9 komórek kodu**, bez błędów.
Wyniki liczbowe są identyczne, a oba wykresy identyczne nawet na poziomie bajtów
PNG. Sprawdzono format notebooka i obejrzano oba wykresy. Wykonanie odbywa się
przez IPython w świeżym procesie, bez serwera i połączeń sieciowych Jupytera;
nie przeprowadzano osobnego przeglądu pełnego interfejsu Jupyter.

## Uzupełnione próbki generacji

Dodano brakujący element wcześniejszego planu: surowe próbki i pliki MIDI.
Każdy model dostał ten sam nagłówek ABC: G-dur, 4/4, jednostka 1/4. Dwa seedy
losowania ustalono z góry; temperatura 0,8, top-k 20, limit 384 nowych tokenów.
Zachowano **wszystkie sześć** wyników, bez wybierania korzystnych przykładów,
poprawiania ABC ani powtarzania losowań w poszukiwaniu ładniejszego wyniku.
Powtórzenie pierwszej próbki każdego modelu dało identyczne tokeny.

| Model / próbka | Nowe tokeny | Zakończenie | Nieprawidłowe takty wewnętrzne | MIDI odczytane ponownie |
|---|---:|---|---:|---|
| GPT A / 1 | 384 | token_limit | 4/31 | tak |
| GPT A / 2 | 384 | token_limit | 0/36 | tak |
| GPT B / 1 | 377 | EOS | 3/32 | tak |
| GPT B / 2 | 176 | EOS | 0/12 | tak |
| NPLM / 1 | 38 | EOS | 2/2 | tak |
| NPLM / 2 | 139 | EOS | 2/8 | tak |

Wszystkie sześć próbek przyjął parser music21. **Cztery z sześciu** mają
nieprawidłową długość co najmniej jednego wewnętrznego taktu. Dwie próbki osiągnęły
limit tokenów, więc nie przedstawiamy ich jako zakończonych kompozycji. Kontrola
wewnętrznych taktów pomija pierwszy i ostatni, które mogą obejmować przedtakt
lub zakończenie; brak wykrytego błędu nie stanowi pełnej oceny muzykalności.

Bezpośredni eksport oryginalnej struktury taktów do MIDI powiódł się dla dwóch
próbek; dla czterech music21 zgłosił `StreamException`. Dostarczono dla wszystkich
sześciu wspólny, jawnie opisany podgląd MIDI: kolejne odczytane nuty i pauzy,
90 BPM, bez zachowywania pozycji błędnych kresek taktowych. Te sześć plików MIDI
ponownie odczytano. Surowe ABC pozostaje nienaruszone. Pliki `_notation.mid`
są dodatkowymi eksportami oryginalnej struktury tam, gdzie eksport się powiódł.

To diagnoza jakości generowania, a nie objaw uszkodzenia wag. Niska perplexity
znakowa nie zapewnia poprawnego metrum. Sześć próbek nie wystarcza do statystycznej
oceny jakości muzycznej.

## Uporządkowany stan wcześniejszego planu

W pliku kontynuacji starsza lista etapów nadal miała wszędzie `pending`.
Uaktualniono ją, zachowując historyczny zapis:

| Etap | Aktualny stan |
|---|---|
| Środowisko, źródło, manifest i podział | Ukończone dla pilotażu Bach |
| Demonstracja i pliki MIDI | Ukończona diagnostycznie; ograniczenia muzyczne opisane |
| Korpus i n-gramy | Ukończone dla Bach |
| GPT od zera, logi, wagi i próbki | Ukończone; dwa przebiegi po 2000 kroków |
| NPLM i kontrolowane ablacje | NPLM ukończony; szersze ablacje pozostają do zaplanowania |
| Eksperci innych stylów, ensemble, E1 | Wymagają dodatkowych korpusów i eksperymentu |
| CKA i komplementarność | Kolejny etap badawczy |
| Prywatne notatki Obsidian | Brak udostępnionych danych |
| Router / PR 13 / zadanie użytkowe | Zależne od wyników i wybranego zastosowania |

Zamknięto wcześniejsze żądanie audytu, double checku i pierwszych treningów.
Nie oznaczono całej wieloetapowej agendy badawczej jako ukończonej.
Przed końcową oceną testową należy zamrozić listę modeli i reguły porównania;
próbek nie używano do zmiany checkpointów. Dalsza praca nad generowaniem powinna
uwzględniać zachowanie metrum oraz ocenę frazowania i odsłuch.

## Pliki do dalszej pracy

Zacznij od `START_HERE_PL.md` w zaktualizowanej paczce. Zawiera komendy kontroli
sum i ponownej walidacji. `src/verify_package.py` działa bez PyTorch i music21.
Szczegółowe dowody znajdują się w `audit/recovery_validation.json`,
`audit/recovery_double_check.json`, `audit/run_integrity.json`,
`audit/resume_verification.json` i `audit/notebook_recovery.json`.
Próbki i ich opis są w `samples/`; pełny stan dalszej pracy w
`Micro_models_kontynuacja.json`.
