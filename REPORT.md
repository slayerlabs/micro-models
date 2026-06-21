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

## 5. Co wiemy / czego nie

**Udowodnione (z liczbą):** mały char-LM uczy się struktury muzyki; czyszczenie danych obniża perplexity (3,88 → 3,80); mechanizm szwu bezstratny; ensemble bije pojedyncze modele; **niezależne maluchy mają wspólną geometrię** (CKA, po audycie); zmierzone **spektrum n-gram → NPLM → transformer** (ppl vs pamięć vs generalizacja).

**Jeszcze nie:** że łączenie reprezentacji **bije** ensemble — to wymaga ekspertów **komplementarnych** (różne, nakładające się domeny) i/lub wymuszonego wspólnego kontraktu; routing.

## 6. Następne kroki
1. Pre-check: czy rozbieżność (wariancja) między ekspertami koreluje z błędem — bramka przed routingiem.
2. Wymuszony wspólny kontrakt (zamrożony front + głowa) na stylach w **tym samym metrum** (różne metrum = model oszukuje po nagłówku — pułapka pomiaru) + ekspertach **komplementarnych**.
3. Bogatszy, nieliniowy łącznik — dopiero jeśli (2) pokaże sygnał.
- Osobny kierunek: wiele modeli „grające razem" (polifonia, synchronizacja).

## 7. Po co to
Dwa cele: (a) **budować know-how zespołu Slayer** — tani, jawny, reprodukowalny poligon; (b) **de-ryzykować metody** pod docelowy model budowlany (klasyfikacja / tworzenie / rozumienie `.ifc`), gdzie poprawność jest sprawdzalna kodem (walidator = weryfikowalna nagroda).

## Metoda w akcji (dlaczego to wiarygodne)
W trakcie tych prac **pomiar dwukrotnie obalił wewnętrzną hipotezę zespołu** i zostało to przyjęte, nie naciągnięte: (1) pierwszy E1 „bił ensemble" — okazał się artefaktem zbioru testowego; (2) E_CKA przeszedł **adwersarialny audyt własnego pomiaru**, który wycofał jedną z dwóch metryk (skażony baseline). To jest sedno tego labu: *najpierw mierz, potem twierdź — i audytuj własny pomiar.*

---
*Pełne rozumowanie (teza ↔ antyteza → synteza, audyt, weryfikacja źródeł) i kod: `music-experts/docs/` oraz `music-experts/src/`.*
