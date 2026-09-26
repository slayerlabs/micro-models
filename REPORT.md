# Slayer Micro-Models — raport stanu prac

> Wejście dla osób z zewnątrz. Prosto, krok po kroku, w kolejności jak idą prace.
> Zasada przez całość: **żadnej tezy bez dowodu; wyniki negatywne też publikujemy.**

## 1. Cel projektu
Budujemy bardzo małe sieci neuronowe (~0,8 mln parametrów, trening w minuty na CPU), na tyle proste,
by w pełni je zrozumieć. Pytanie badawcze: **czy wielu małych, wyspecjalizowanych modeli da się złożyć
w jeden, działający lepiej niż pojedynczy?** Dziedziną testową jest muzyka zapisana tekstowo (notacja ABC):
dane są darmowe, a poprawność łatwo sprawdzić. Cel docelowy (osobny) to model dla budownictwa (pliki `.ifc`).

## 2. Materiał: eksperci
Kilka modeli, każdy na jednym stylu, ta sama architektura (4 warstwy, 4 głowy, `d_model=128`, kontekst 128 znaków, poziom znaków). `jig` ma perplexity (niżej = lepiej) **3,80**; obok `walc`, `reel`, `jig-v2` (inny seed), sopran chorału `Bach`. Każdy uczy się sam z siebie struktury muzycznej (metrum, tonacja, kadencje) z samej predykcji następnego znaku — bez wpisanej teorii muzyki.

## 3. Łuk pierwszy: czy da się ZŁOŻYĆ małe modele?

### E0 — kontrola mechanizmu (self-stitch)
Rozciąć model w środku, wstawić mały liniowy łącznik, resztę zamrozić; trenować tylko łącznik.
**Wynik:** łącznik odtwarza wynik bazowy (3,80 → 3,83). **Mechanizm działa — to nie dowód, że kompozycja pomaga, tylko że hydraulika jest poprawna.**

### Ensemble — baseline do pobicia
Uśrednić przewidywania dwóch modeli. **Wynik** (zadanie mieszane walc+reel): ensemble **5,15** — lepszy niż każdy z osobna (reel 5,87; walc 6,20). **Łączenie ekspertów realnie pomaga; to jest poprzeczka.**

### E1 — łączenie reprezentacji
Front jednego modelu + łącznik + tył drugiego; trenowany tylko łącznik. Po drodze złapano i poprawiono błąd pomiaru (zbiór testowy był przekrzywiony).
**Wynik (po poprawie):** stitch **5,18 ≈ ensemble 5,15** — **nie pobił** baseline'u. Wynik negatywny, zaraportowany uczciwie.

### E_CKA — dlaczego nie pobił? (z self-audytem)
Pytanie: czy niezależnie trenowane małe modele mają **wspólną geometrię reprezentacji**, czy różne. Metoda: zmierzyć podobieństwo reprezentacji (CKA) między modelami, z baselinem losowym; sweep po skali (0,2M–3M, 3 seedy).

**Wynik (po adwersarialnym audycie własnego pomiaru):**

| para | CKA |
|---|---|
| jig vs jig-v2 (te same dane, inny seed) | **0,945** |
| jig vs walc (inny styl) | 0,841 |
| baseline: losowy vs losowy | **0,441** |
| trenowany vs losowy | 0,290 |

Krzywa skali (CKA): 0,915 → 0,936 → 0,947 → 0,945 — wysoka, wczesna, plateau ~0,94; **trend ze skalą mały**.

**Wniosek:** geometrie są **wspólne** (trenowane 0,84–0,94 ≫ losowy baseline 0,44; równomiernie po warstwach → nie artefakt). To **obaliło** wcześniejsze wyjaśnienie remisu E1: nie chodzi o „różne geometrie" (są wspólne), tylko o **redundancję** — modele kodują w dużej części to samo, więc nad uśrednianiem nie ma czego dodać. **Wąskie gardło kompozycji to KOMPLEMENTARNOŚĆ ekspertów, nie wyrównanie.**

**Uczciwość (zakres):** to nie dowód „platońskiej konwergencji" — ta wymaga RÓŻNYCH danych; jig–jig-v2 to **stabilność względem seeda**, cross-styl to słaby surogat. Druga metryka (mutual-kNN) została **wycofana**, bo audyt pokazał, że jej baseline (losowy–losowy 0,68) ≈ sygnał (0,66): mierzyła strukturę wejścia, nie wyuczoną zgodność.

## 4. Łuk drugi: most n-gram → transformer (paper)
Ten sam cel (predykcja następnego znaku), spektrum mechanizmów — od twardego zliczania do uwagi. Pomiar na tym samym korpusie:

| model | mechanizm | val ppl | rozmiar |
|---|---|---|---|
| n-gram rząd 1 | zliczanie, okno 1 | 11,86 | 52 konteksty |
| n-gram rząd 3 | zliczanie, okno 3 | 5,19 | 13,9K |
| n-gram rząd 6 | zliczanie, okno 6 | **3,90** | **508K kontekstów** |
| NPLM okno 8 | MLP, stałe okno | 4,41 | 41K param |
| NPLM okno 16 | MLP, stałe okno | 4,38 | 74K param |
| mini-transformer (GPT) | uwaga, długi kontekst | **3,80** | ~800K param |

**Wnioski (uczciwie):**
- n-gram skaluje ppl z rzędem (11,86 → 3,90), ale **pamięć eksploduje** (rząd 6 = 508K kontekstów) i to tablica look-up **bez generalizacji** poza widziane konteksty.
- **n-gram rząd 6 (3,90) ≈ GPT (3,80)** na tym repetytywnym korpusie — muzyka ma dużo dosłownie powtarzalnych fraz, więc n-gram „pamięta". To **nie** jest „transformer miażdży n-gram"; ppl prawie remis.
- **NPLM jest mostem:** kompresuje n-gram w zwartą, gładką, **generalizującą** funkcję (~40K param), kosztem nieco gorszego ppl — uczy reprezentacji, nie tablicy.
- **GPT** wygrywa ppl zmiennym, długim kontekstem (uwaga), kosztem ~800K param.
Prawdziwe osie spektrum to **pamięć i generalizacja**, nie sam ppl.

## 5. Łuk trzeci: sędzia — mierzenie „czy to w ogóle muzyka" (E-JUDGE)

PPL ma dwie ślepe plamy: nie odpowiada, czy generacja *działa jako muzyka*, i nie porównuje
modeli z różnych korpusów. Zbudowaliśmy więc niezależnego sędziego: klasyfikator `JudgeGPT`
(trunk GPT verbatim + mean-pool + 3 głowy; 55 tys. melodii z thesession.org, split po
`tune_id`), który patrząc TYLKO na ciało melodii (nagłówki `M:`/`K:` wycinane — nie można
oszukiwać promptem) przewiduje **meter / mode / type**. Val acc: meter 0.92, mode 0.72,
type 0.82 — z pomyłkami muzycznie prawidłowymi (hornpipe↔reel to para najtrudniejsza nawet
dla ludzi; Bmin↔D to ten sam zbiór dźwięków — sufit informacyjny).

**Benchmark domenowy.** Każdego eksperta oceniamy wyłącznie na melodiach z jego własnej
domeny (pula walidacyjna ograniczona filtrem type+metrum z `domains.json`). Wynik (score) to
średnie prawdopodobieństwo, jakie sędzia przyznaje prawdziwym etykietom; każda porażka modelu
liczy się jako zero (brak zniekształcenia przez „przeżywających"), a punktem odniesienia
(sufitem) jest ocena przez tego samego sędziego **prawdziwych** melodii tej domeny. Wyniki:
jigi osiągają **80–86% sufitu**, walc 86% (sufit miękki — sędzia sam jest mało pewny na 3/4),
reele 69–72%. Ranking wyznaczony przez sędziego pokrywa się z rankingiem PPL — dwie
niezależne miary, jeden werdykt. Odnotujmy uczciwie: pierwsza wersja benchmarku, oceniająca
wszystkich ekspertów na jednej wspólnej puli melodii, pokazywała dokładnie odwrotnie
(najlepiej wypadały reele). To była wada pomiaru, nie właściwość modeli: niezbalansowane
klasy sprawiały, że ekspert 4/4 „wygrywał" tam, gdzie prawda o metrum była po prostu
najczęstsza. Dopiero pule domenowe dały uczciwy obraz — kolejny pomiar, który obalił nasze
pierwsze odczytanie.

**Macierz OOD** (ekspert × domena celu; prawda = to, o co prosimy):
- Diagonala dominuje wszędzie — **specjalizacja ekspertów jest realna**.
- **Nagłówki nie sterują poza domeną**: jig z promptem `M:4/4` generuje 6/8 z pewnością
  0.94 (home-bias 0.9). Sterowanie promptem jest martwe.
- **Kontekst steruje w połowie**: z prawdziwym prefiksem melodii home-bias spada do ~0.5,
  score rośnie do 40–63% sufitu. Modele czytają styl z nut, nie z nagłówków — w domenie też
  (mode z `K:` słabo, z prefiksu lepiej).
- **Asymetria**: najlepsze komórki OOD to reele→jigi (≈60% sufitu, home-bias type 0.2) —
  eksperci 4/4 są elastyczni, 6/8 to twardy nawyk.
- **Elastyczność siedzi w małych modelach**: e32 mają najniższy home-bias wszędzie
  (najsłabiej „zapatrzone" w swoją domenę) przy najsłabszej diagonali; duże modele są
  najsztywniejsze. Nowa hipoteza pod E1: **komplementarność może wolić małe, gibkie modele
  mimo gorszego PPL**.
- Bach (chorały) jest uczciwie wyłączony: bez wspólnego słownika nie da się go nawet
  zakodować w irlandzkich promptach — wspólny kontrakt słownikowy to warunek wstępny
  dalszych testów.

Ograniczenia: sędzia ma fizyczny sufit na mode (tonacje względne są niedróżnialne z treści
dźwiękowej); macierz OOD policzona na n=30/komórkę — wersja pod paper wymaga pełnego
przebiegu i sufitu z całej puli.

## 6. Co wiemy / czego nie

**Udowodnione (z liczbą):** mały char-LM uczy się struktury muzyki; czyszczenie danych obniża perplexity (3,88 → 3,80); mechanizm szwu bezstratny; ensemble bije pojedyncze modele; **niezależne maluchy mają wspólną geometrię** (CKA, po audycie); zmierzone **spektrum n-gram → NPLM → transformer** (ppl vs pamięć vs generalizacja); **sędzia niezależny od PPL potwierdza ranking ekspertów** i mierzy nową oś (specjalizacja: jigi 80–86% sufitu, reele 69–72%); **nagłówki nie sterują ekspertem poza domeną, kontekst melodyczny tak (w połowie)**; **elastyczność OOD rośnie, gdy model jest mniejszy** (hipoteza pod E1, do potwierdzenia).

**Jeszcze nie:** że łączenie reprezentacji **bije** ensemble — to wymaga ekspertów **komplementarnych** (różne, nakładające się domeny) i/lub wymuszonego wspólnego kontraktu; routing; pełna (n=100) macierz OOD z sufitem z całej puli; wspólny słownik (warunek testów międzykorpusem, m.in. dla Bacha).

## 7. Następne kroki
1. Pre-check: czy rozbieżność (wariancja) między ekspertami koreluje z błędem — bramka przed routingiem.
2. Wymuszony wspólny kontrakt (zamrożony front + głowa) na stylach w **tym samym metrum** (różne metrum = model oszukuje po nagłówku — pułapka pomiaru) + ekspertach **komplementarnych**.
3. **Wspólny słownik przy trenowaniu ekspertów** — usuwa klasę artefaktów OOD (dziś: pominięcia słownikowe, wyłączony Bach) i jest wstępem do stitcha między ekspertami.
4. E-JUDGE ensemble/stitch: te same melodie, ten sam sędzia — czy metoda kompozycji bije pojedynczych ekspertów w oś stylu, nie tylko w PPL.
5. Bogatszy, nieliniowy łącznik — dopiero jeśli (2) pokaże sygnał.
- Osobny kierunek: wiele modeli „grające razem" (polifonia, synchronizacja).

## 8. Po co to
Dwa cele: (a) **budować know-how zespołu Slayer** — tani, jawny, reprodukowalny poligon; (b) **de-ryzykować metody** pod docelowy model budowlany (klasyfikacja / tworzenie / rozumienie `.ifc`), gdzie poprawność jest sprawdzalna kodem (walidator = weryfikowalna nagroda).

## Metoda w akcji (dlaczego to wiarygodne)
W trakcie tych prac **pomiar trzykrotnie obalił wewnętrzne przekonania zespołu** i zostało to przyjęte, nie naciągnięte: (1) pierwszy E1 „bił ensemble" — okazał się artefaktem zbioru testowego; (2) E_CKA przeszedł **adwersarialny audyt własnego pomiaru**, który wycofał jedną z dwóch metryk (skażony baseline); (3) pierwszy benchmark sędziego (wspólna pula dla wszystkich ekspertów) pokazał ranking odwrotny niż PPL — okazał się artefaktem niezbalansowanych klas; naprawa: pule domenowe. To jest sedno tego labu: *najpierw mierz, potem twierdź — i audytuj własny pomiar.*

---
*Pełne rozumowanie (teza ↔ antyteza → synteza, audyt, weryfikacja źródeł) i kod: `music-experts/docs/` oraz `music-experts/src/`.*
