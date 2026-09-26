# Micro-models — aktualny punkt startowy

Pierwsza seria treningów oraz jej końcowa ewaluacja są ukończone.
Zacznij od `evaluation_20260926/Micro_models_ewaluacja_PL.md` albo wykonanego
`evaluation_20260926/Micro_models_ewaluacja_PL.ipynb`.

Najlepsza pojedyncza PPL testu: NPLM 1,9501. GPT B: 2,0375.
Hybryda NPLM+GPT B (logity 50/50): 1,8595.
Test liczy 35 melodii i 17 rodzin; został już wykorzystany i nie wolno traktować
go jako świeżej puli do strojenia następnych modeli.

## Co jest zachowane

Oryginalne dane, sześć checkpointów, logi i kod treningu są niezmienione.
Stare raporty i `runs/*/status.json` opisują stan w chwili treningu/odzyskania;
ich `test_evaluated: false` jest historyczne. Aktualny stan jest w pliku
`Micro_models_kontynuacja.json` i katalogu `evaluation_20260926/`.

Nowy katalog zawiera protokół zamrożony przed testem, wyniki per melodia i znak,
kontrole, 90 surowych ABC, CKA, pięć wykresów, notebook i odczytane źródła repo.
Nie wykonano nowych treningów, destylacji ani E1 między różnymi domenami.

## Odtworzenie

Zainstaluj zależności z `evaluation_20260926/requirements-evaluation.txt`. Następnie:

```bash
python evaluation_20260926/evaluate_campaign.py
python evaluation_20260926/analyze_representations.py
python evaluation_20260926/generate_evaluation.py
python evaluation_20260926/execute_notebook.py
python evaluation_20260926/double_check_results.py
```

Inferencja jest deterministyczna w zapisanym środowisku. Ponowne wykonanie
nadpisuje wyniki diagnostyczne; zachowaj oryginalną paczkę i protokół zamrożenia.
Notebook można także wykonać w Jupyterze; w tej sesji wykonano 8 komórek kolejno
przez IPython, gdy środowisko zablokowało uruchomienie kernela TCP.

Kontrola paczki: `python src/verify_package.py`.
Szczegółowe nowe kontrole: `evaluation_20260926/double_check_results.json`.
Następny etap i ograniczenia oceny opisano w końcowej części raportu.

Raport WWW (prywatny): https://micro-models-bach-ewaluacja-20260926.rodzina-na-s-0732.chatgpt.site
Wersja HTML offline: `evaluation_20260926/Micro_models_raport_PL.html`.
