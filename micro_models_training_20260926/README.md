# Micro-models — treningi i ewaluacja na chorałach Bacha

Kompletna zawartość archiwum `Micro_models_treningi_20260926.zip` (wersja 2): **218 oryginalnych plików**, zachowanych bez zmian. Ten plik README jest dodatkiem na potrzeby GitHuba.

## Od czego zacząć

- [Aktualny stan i odtwarzanie eksperymentu](START_HERE_PL.md)
- [Raport końcowy i ewaluacja (PL)](evaluation_20260926/Micro_models_ewaluacja_PL.md)
- [Wykonany notebook ewaluacji](evaluation_20260926/Micro_models_ewaluacja_PL.ipynb)
- [Samodzielny raport HTML](evaluation_20260926/Micro_models_raport_PL.html) — pobierz plik i otwórz lokalnie.
- [Audyt danych i opis treningów](README_PL.md)

## Wyniki pilotażu

Główna metryka: perplexity części muzycznej ABC, ważona liczbą ocenianych znaków. Zamrożony test obejmuje 35 melodii i 17 rodzin.

| Model | PPL testu (mniej = lepiej) |
|---|---:|
| N-gram, kontekst 6 | 2,2505 |
| NPLM | 1,9501 |
| GPT A | 2,0443 |
| GPT B | 2,0375 |
| NPLM + GPT B, średnia logitów 50/50 | 1,8595 |

To mały pilotaż w jednej domenie, nie bezpośredni ranking wobec historycznych wyników projektu źródłowego. Test został już wykorzystany. Kontrole składni generacji nie zastępują oceny muzycznej ani odsłuchu. Pełne ograniczenia i porównanie metodologii opisano w raporcie.

## Kompletność i odtwarzanie

Paczka zawiera kod, dane i audyty, sześć checkpointów, logi treningowe, raporty, notebooki, generacje i wyniki ewaluacji. Zachowano historyczne raporty i statusy treningów; aktualny stan opisują `START_HERE_PL.md` i `Micro_models_kontynuacja.json`.

Uruchom z tego katalogu:

```bash
python src/verify_package.py
```

Weryfikator sprawdza 217 plików wymienionych w `SHA256SUMS`; sam plik `SHA256SUMS` jest 218. plikiem oryginalnej paczki. README GitHuba nie należy do oryginalnego archiwum.

SHA-256 oryginalnego ZIP-a: `6c23b07d8f87894b6b66e56113359fda5736d118a092b8a8def68d22a29c0bb6`.

## Źródła i licencje

- Architektura GPT: [slayerlabs/micro-models](https://github.com/slayerlabs/micro-models), commit `da30da3d7486a4b6b3de8d2d4aaa0f183617f751`; [zachowana licencja MIT](licenses/micro-models-MIT.txt).
- Korpus: Craig Stuart Sapp, *A digital edition of 370 J.S. Bach Chorales*, 2009, [bach-370-chorales](https://github.com/craigsapp/bach-370-chorales), commit `0fd9e00542445a522c6030c80c687b874aa569d5`; [zachowana informacja o CC BY-NC-SA 4.0](licenses/Bach-CC-BY-NC-SA-4.0.txt). Adaptacja sopranu do ABC zachowuje tę licencję i atrybucję; szczegóły przekształcenia są w `README_PL.md`.
