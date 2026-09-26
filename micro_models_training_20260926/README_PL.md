> Aktualny stan po odzyskaniu: **START_HERE_PL.md** oraz **Micro_models_odzyskanie_20260926_PL.md**.

# Micro-models — audyt danych i pierwsza seria treningów

**Seria ukończona.** Wyniki i wnioski są w
`Micro_models_audyt_i_treningi_20260926_PL.md`; wykonany notebook
`Micro_models_audyt_danych_20260926.ipynb` zawiera tabele i wykresy.
`audit/run_integrity.json` potwierdza ponowne wczytanie checkpointów i dokładne
odtworzenie walidacji. `Micro_models_kontynuacja.json` zapisuje stan projektu.

Data rozpoczęcia: 26.09.2026, strefa Europe/Warsaw. To rzeczywiste uczenie na
chorałach Bacha; nie są to dane syntetyczne ani reprodukcja prywatnego korpusu
Obsidiana. Nie wykorzystano danych The Session ani historycznych wytrenowanych
wag. Wszystkie modele tej serii startują od losowej inicjalizacji.

## Źródła i zakres użycia

- Architektura GPT: `slayerlabs/micro-models`, commit
  `da30da3d7486a4b6b3de8d2d4aaa0f183617f751`, plik
  `music-experts/src/core/gpt.py`, skopiowany bez zmian do `src/gpt.py`.
  Licencja MIT jest w `licenses/micro-models-MIT.txt`.
- Dane: Craig Stuart Sapp, *A digital edition of 370 J.S. Bach Chorales*,
  copyright 2009, repozytorium
  <https://github.com/craigsapp/bach-370-chorales>, commit
  `0fd9e00542445a522c6030c80c687b874aa569d5`.
  Licencja **CC BY-NC-SA 4.0**:
  <https://creativecommons.org/licenses/by-nc-sa/4.0/>.
  Oryginalna informacja o licencji: `licenses/Bach-CC-BY-NC-SA-4.0.txt`.
- Przygotowany korpus ABC stanowi adaptację do niekomercyjnego eksperymentu
  edukacyjnego. Zachowuje tę samą licencję CC BY-NC-SA 4.0 i powyższą atrybucję.
  Zmiany: wyodrębnienie sopranu, tekstowy zapis ABC, jedno przejście przez zapisane
  nuty bez rozwijania powtórek, pominięcie tekstu pieśni, artykulacji i ekspresji.
  Wysokości, długości i łuki przedłużające dźwięk zachowano i zweryfikowano.

Nie przypisujemy licencji MIT kodu zewnętrznym danym. Wyniki dotyczą tego
niekomercyjnego eksperymentu, nie stanowią oceny uprawnień do przyszłego produktu.

## Ostateczne dane

| Zbiór | Melodie po deduplikacji | Grupy powiązań | Znaki ABC |
|---|---:|---:|---:|
| Trening | 281 | 135 | 67 016 |
| Walidacja | 36 | 17 | 8 664 |
| Test odłożony | 35 | 17 | 8 299 |
| Razem | 352 | 169 | 83 979 |

Wszystkie 370 oryginalnych zapisów są rozliczone. Osiemnaście identycznych partii
sopranowych połączono z reprezentantem, zachowując pochodzenie i aliasy. Jednostką
podziału jest **cała spójna grupa powiązań**, a nie fragment tekstu. Grupa łączy
wspólne tytuły, identyfikatory utworów BWV, identyczne lub transponowane melodie,
długie wspólne fragmenty oraz bliskie przebiegi interwałów. Powiązanie jest
przechodnie. Największa grupa ma 79 zapisów i obejmuje różne melodie powiązane
tytułami lub utworami źródłowymi; nie oznacza 79 identycznych melodii.

Podział stratyfikowano według metrum, trybu tonalnego, znaków przykluczowych,
długości, liczby rekordów i liczby znaków. Szukanie podziału używało wyłącznie
metadanych, bez wyniku predykcji modeli. Docelowe udziały to 80/10/10; rzeczywiste
udziały melodii wynoszą 79,83% / 10,23% / 9,94%.

W audycie odkryto 48 oznaczeń modalnych nierozpoznawanych jako zwykły obiekt Key
przez music21. Parser odczytuje teraz jawny zapis źródłowy, dzięki czemu nie
odrzuciliśmy tych utworów. Wszystkie 352 zachowane melodie przeszły odczyt ABC
z powrotem do dokładnie tych samych wysokości i długości, z uwzględnieniem łuków.

## Kontrola przecieku

`src/double_check.py` jest osobnym drugim audytem. Nie importuje kodu
przygotowującego podział. Ponownie oblicza identyfikatory i przecięcia, używa
innego algorytmu porównania sekwencji i sprawdza źródłowe pliki oraz konwersję ABC.

W finalnym podziale nie znaleziono między żadną parą zbiorów:

- wspólnych identyfikatorów ani grup powiązań;
- wspólnych tytułów lub rdzeni BWV, również w aliasach po deduplikacji;
- identycznych tekstów, melodii ani dokładnych wersji transponowanych;
- identycznych fragmentów 128 znaków części muzycznej;
- identycznych fragmentów 24 kolejnych interwałów;
- podobieństwa całego przebiegu interwałów co najmniej 80% według sprawdzanych reguł.

Pierwszy próg 88% okazał się zbyt liberalny po ręcznym przeglądzie najbliższych
par: `chor130` i `chor320` miały różne tytuły i podobieństwo około 82%.
Wstępne przebiegi zostały unieważnione. Podział wykonano ponownie przy 80%,
sprawdzono ponownie i uruchomiono modele **od zera**. Starych wag nie przeniesiono.
Historia poprawki jest w `audit/protocol_revision.json`; paczka wynikowa nie
zawiera nieaktualnych wag z tego pierwszego podejścia.

Największe podobieństwo w alternatywnym sprawdzeniu wynosi 65,75% między treningiem
a walidacją i 70,45% między treningiem a testem. Są to wartości miernika, nie
procentowy dowód braku wszelkiego pokrewieństwa muzycznego.

Pełna gwarancja wykrycia każdej historycznej relacji między melodiami nie jest
możliwa za pomocą skończonego zestawu reguł. Krótkie wspólne motywy są naturalną
cechą tego repertuaru. Wynik audytu brzmi: **nie wykryto przecieku według jawnych,
zapisanych kryteriów**, z konserwatywnym grupowaniem przed treningiem.

## Porównywalność i metryki

- Słownik jest stałym alfabetem gramatyki ABC z BOS/EOS/PAD, łącznie 82 tokeny.
  Nie dopasowano tokenizera do walidacji ani testu. W obu tych zbiorach nie ma
  znaków nieobecnych w treningu.
- Każdy utwór ma własny BOS i EOS. Okna mają maksymalnie 128 tokenów i krok 64.
  Kontekst może się nakładać, ale każda pozycja docelowa ma dokładnie jedno
  okno etykietowane. Żadne okno nie łączy dwóch melodii.
- Trening liczy stratę na znakach i EOS. Walidacja obejmuje **każdy rzeczywisty
  znak** ABC, również pierwszy znak po BOS i końcówkę; nie dodaje EOS do
  mianownika bits/char. Pozostałe pozycje mają maskę `-100`.
- Główną metryką wyboru checkpointu jest średnia strata na znak **części muzycznej**,
  z pominięciem nagłówka. Dodatkowo zapisujemy perplexity całego ABC i bits/char.
- Wyniki są ważone liczbą ocenianych znaków. Nie uśredniamy bezpośrednio
  perplexity poszczególnych utworów.
- Wszystkie modele otrzymują dokładnie ten sam zbiór treningowy i walidacyjny.
  N-gram resetuje kontekst między utworami i jest uczony tylko na treningu.
- Test był profilowany strukturalnie, ale **nie obliczono na nim straty żadnego
  trenowanego modelu**. Nie służy do checkpoint selection ani strojenia.

Kontrola wzorem analitycznym dla modelu jednostajnego obejmuje długości 1, 2, 63,
64, 65, 127, 128, 129, 130, 191, 192, 193, 255, 256, 257 i 1000 znaków.
Każdorazowo liczba ocenionych znaków jest poprawna, a perplexity odpowiada
liczbie tokenów słownika. Sprawdzono też, że zmiana przyszłych tokenów nie zmienia
wcześniejszych logitów GPT.

## Ograniczenia reprezentatywności

Zbiór jest mały i obejmuje tylko repertuar chorałowy. Walidacja i test mają po
17 niezależnie przydzielonych grup. Nie uzasadnia to jeszcze szerokich wniosków
o wszystkich rodzajach muzyki. Dwa utwory frygijskie i pojedyncze metrum 3/2
występują tylko w treningu, więc nie mierzymy generalizacji dla tych rzadkich
podkategorii. Długości i główne kategorie są dobrze zbliżone, ale nie identyczne.

Wszystkie wysokości dźwięków walidacji i testu występują w treningu. W każdym
z tych zbiorów jest po pięć zdarzeń o długości niewystępującej w treningu
(po scaleniu łuków, z uwzględnieniem pauz). Szczegóły są w
`audit/musical_support.json`. Nie są to nieznane znaki alfabetu.

Mniejsza strata tekstowa nie jest automatycznie lepszą muzyką. Ocena frazowania,
metrum i słuchowa będzie oddzielnym etapem. Wyników nie należy bezpośrednio
porównywać z liczbami w pierwotnym README: zmieniły się źródło, split, alfabet,
zapis akcydencji oraz sposób oceniania kontekstu.

## Treningi

| Model | Konfiguracja | Budżet |
|---|---|---|
| N-gram | Kontekst 1, 3 i 6 znaków; interpolacja z unigramem add-k 0,01 | Jeden przebieg zliczania treningu |
| NPLM | Kontekst 16, embedding 32, warstwa ukryta 128; 78 866 parametrów | 2000 aktualizacji, seed 20260926 |
| GPT A | 4 bloki, 4 głowy, szerokość 128; 820 224 parametrów | 2000 aktualizacji, seed 20260926 |
| GPT B | Ta sama architektura GPT | 2000 aktualizacji, seed 20260927 |

Precyzja modeli neuronowych: FP32. AdamW, gradient clipping 1, warmup 100 kroków
i kosinusowy learning rate. Batch 32. GPT: lr 0,0003, dropout 0,1.
NPLM: lr 0,001. Równy budżet kroków i okien nie oznacza równych FLOPs.
Walidacja całego zbioru następuje co 200 kroków. Test pozostaje odłożony.

Pole `order` w pliku wyników n-gramów oznacza **długość historii**. Kontekst
6 znaków odpowiada modelowi do 7-gramów, a kontekst 1 modelowi do bigramów.
Nie są to odpowiednio klasyczny 6-gram i unigram.

Pliki przebiegów zawierają identyfikator manifestu danych w nazwie katalogu.
`checkpoint_last.pt` przechowuje wagi, optymalizator, krok, RNG i konfigurację;
`checkpoint_best.pt` zachowuje najlepsze wagi według walidacji.
`status.json` rozstrzyga, czy przebieg się zakończył. `metrics.jsonl` zawiera
rzeczywiste pomiary. Brak pola lub pliku wyniku nie oznacza wyniku zerowego.

## Uruchomienie i wznowienie

Z katalogu tej paczki, w środowisku Python 3.12:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python src/train_ngram.py
.venv/bin/python src/train.py --model nplm --seed 20260926 --steps 2000 --threads 2 --resume
.venv/bin/python src/train.py --model gpt --seed 20260926 --steps 2000 --threads 2 --resume
.venv/bin/python src/train.py --model gpt --seed 20260927 --steps 2000 --threads 2 --resume
```

`--resume` odtwarza istniejący przebieg. Aby rozpocząć nowy niezależny przebieg,
wybierz nowy seed i pomiń `--resume`. Skrypt blokuje przypadkowe nadpisanie
istniejącego checkpointu. Zmiana budżetu lub eksperymentu wymaga nowej, jawnej
konfiguracji; ten tryb resume służy do dokończenia pierwotnych 2000 kroków.

Do odtworzenia danych i kontroli ich źródła:

```bash
git clone https://github.com/craigsapp/bach-370-chorales.git bach-source
git -C bach-source checkout 0fd9e00542445a522c6030c80c687b874aa569d5
.venv/bin/python src/prepare_data.py --source bach-source
.venv/bin/python src/double_check.py --source bach-source
```

Najpierw sprawdź aktualny manifest i hashe. Nie zmieniaj danych pod istniejącym
treningiem. Dla nowych eksperymentów wykorzystuj ten sam podział całych grup.
Nie wolno ponownie wyliczać splitu osobno dla eksperta, mappera, klasyfikatora
czy modelu porównawczego. Powiększenie korpusu wymaga wersji nowego manifestu.

## Następny etap

Zachowujemy obecny test na końcowe, wcześniej zaplanowane porównanie.
Na razie wyniki służą do sprawdzenia uczenia i diagnostyki walidacyjnej.
Po zamrożeniu wyboru modeli możemy wykonać jednorazową ocenę testową,
z niepewnością liczona na poziomie grup, oraz ocenę generowanych melodii.
Obsidian i eksperci jig/walc/reel wymagają własnych dostępnych korpusów oraz
analogicznego grupowania przed treningiem.
