"""Create the final Polish report from measured results; update the recovered handoff."""
import argparse
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
def load(p): return json.loads((ROOT/p).read_text())
ap=argparse.ArgumentParser()
ap.add_argument('--continuation',type=Path)
args=ap.parse_args()
summary=load('runs/summary.json'); audit=load('audit/double_check.json')
integrity=load('audit/run_integrity.json'); profile=load('audit/profile_summary.json')
assert audit['passed'] and integrity['passed']
assert not summary['test_evaluated']
assert all(r['state']=='completed' for r in summary['rows'])
now=datetime.datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(timespec='seconds')
lines=['| Model | Parametry | Najlepszy krok | PPL walidacji: muzyka | PPL walidacji: całe ABC |',
       '|---|---:|---:|---:|---:|']
for r in summary['rows']:
    params=f"{r['parameters']:,}".replace(',',' ') if r['parameters'] else '—'
    step=r['best_step'] if r['best_step'] is not None else '—'
    lines.append(f"| {r['model']} | {params} | {step} | {r['val_body_ppl']:.4f} | {r['val_full_ppl']:.4f} |")
winner=min(summary['rows'],key=lambda r:r['val_body_ppl'])
report=f'''# Micro-models — audyt danych, double check i wykonane treningi

Stan: {now}. Wszystkie wyniki dotyczą manifestu
`{summary['manifest_sha256']}`.

**Audyt zakończony, pierwsza seria treningów ukończona.** Zrealizowano kontrolę
danych, osobny double check, trzy porównania n-gramowe, NPLM i dwa GPT od losowych
wag. Każdy model neuronowy wykonał 2000 aktualizacji. Test pozostaje odłożony:
sprawdzono jego strukturę i rozłączność, ale nie obliczano na nim jakości modeli.

## Co odzyskano i jaki eksperyment wykonano

Odzyskano wcześniejszą analizę z 25.09.2026, plik kontynuacji i zapis rozmowy.
Wskazywały m.in. ryzyko przecieku przez obcinanie korpusu przed podziałem oraz
pomijanie końcówek w pomiarze bits/char. Nowy pipeline rozwiązuje te problemy
strukturalnie: najpierw dzieli całe grupy utworów, a później tworzy okna;
każdy rzeczywisty znak walidacji ma dokładnie jedną ocenianą pozycję.
Nie oznacza to poprawienia wszystkich dawnych skryptów upstream.

Wybrano rzeczywisty, publicznie dostępny korpus chorałów Bacha. Prywatnego vaulta
Obsidiana nie było w odzyskanych plikach; nie użyto też danych The Session.
To pierwsza seria na bezpiecznie rozdzielonych danych muzycznych, a nie pełna
reprodukcja wszystkich eksperymentów z pierwotnego repozytorium.

Źródło danych: Craig Stuart Sapp, *A digital edition of 370 J.S. Bach Chorales*
(2009), [bach-370-chorales](https://github.com/craigsapp/bach-370-chorales), commit
`0fd9e00542445a522c6030c80c687b874aa569d5`, licencja **CC BY-NC-SA 4.0**.
Korpus ABC jest adaptacją z zachowaną atrybucją i tą samą licencją do tego
niekomercyjnego eksperymentu. Kod GPT skopiowano bez zmian z
[slayerlabs/micro-models](https://github.com/slayerlabs/micro-models), commit
`da30da3d7486a4b6b3de8d2d4aaa0f183617f751`, MIT. Licencje są w paczce.

## Dane i reprezentatywność

| Zbiór | Melodie | Grupy powiązań | Znaki ABC | Udział melodii |
|---|---:|---:|---:|---:|
| Trening | 281 | 135 | 67 016 | 79,83% |
| Walidacja | 36 | 17 | 8 664 | 10,23% |
| Test odłożony | 35 | 17 | 8 299 | 9,94% |
| Razem | 352 | 169 | 83 979 | 100% |

Rozliczono wszystkie 370 źródeł. Usunięto 18 identycznych realizacji sopranu,
zachowując ich pochodzenie jako aliasy. Parser poprawnie uwzględnił także
48 utworów z oznaczeniami modalnymi. Nie odrzucono ich jako błędnych tonacji.
Każdą z 352 konwersji do ABC odczytano z powrotem i porównano dokładne wysokości
i długości dźwięków. Wyodrębniony sopran nie obejmuje harmonii pozostałych głosów.

Podział uwzględnia tryb, metrum, znaki przykluczowe, długość utworu i wielkość
zbiorów. Optymalizacja podziału wykorzystywała wyłącznie metadane. Mediany długości
wynoszą odpowiednio 219, 216 i 226 znaków. Rozbieżność Jensen–Shannon rozkładu
znaków względem całego korpusu wynosi 0,002086 bitu dla walidacji i 0,002844
dla testu; główne rozkłady są zbliżone.

Ograniczenia są jawne: dwa utwory frygijskie i pojedyncze metrum 3/2 znajdują się
tylko w treningu. Wszystkie wysokości walidacyjne i testowe występują w treningu,
ale po pięć zdarzeń w każdym z tych zbiorów ma długość niewystępującą w treningu.
Nie są to nieznane znaki alfabetu — takich znaków nie ma. To niewielkie, rzadkie
podkategorie, których nie pokrywa idealna stratyfikacja. Nie przenoszono utworów
po zobaczeniu wyników modeli.

## Zabezpieczenia przed przeciekiem i double check

Grupy łączą wspólny tytuł lub rdzeń BWV, identyczne i transponowane melodie,
fragmenty 128 znaków, fragmenty 24 interwałów oraz bliski przebieg interwałów.
Cała spójna grupa trafia do jednego zbioru. Największa ma 79 melodii powiązanych
także pośrednio; nie są to 79 identycznych rekordów.

Ręczny przegląd najbliższych par ujawnił `chor130` / `chor320`: inne tytuły, ale
około 82% podobieństwa. Początkowy próg 88% był zbyt łagodny. Unieważniono pierwsze
przebiegi, zaostrzono próg do 80%, wykonano nowy split i pełen audyt, po czym
uruchomiono treningi od zera. Wagi z pierwszego podejścia nie są użyte ani
dołączone do paczki wynikowej.

Osobny skrypt kontrolny dał **19/19 pozytywnych kontroli**. Ponownie policzył
przecięcia oraz wykorzystał inny algorytm podobieństwa sekwencji niż przygotowanie
danych. Między każdą parą zbiorów uzyskano zero kolizji dla identyfikatorów,
grup, tytułów, rdzeni BWV, dokładnych melodii, transpozycji i długich fragmentów.
Największe podobieństwo według alternatywnego algorytmu to 65,75% dla
trening/walidacja i 70,45% dla trening/test.

Sprawdzono także wszystkie etykietowane pozycje okien, brak łączenia utworów,
brak pominiętej końcówki, przyczynowość maski GPT oraz wzór analityczny metryki
dla 16 długości sekwencji. Słownik 82 tokenów jest ustaloną gramatyką ABC,
bez dopasowania tokenizera do odłożonych danych.

Wniosek brzmi: **nie wykryto przecieku według zapisanych kryteriów**.
Nie jest to gwarancja wykrycia wszystkich historycznych relacji muzycznych.
Krótkie wspólne motywy są naturalną cechą chorałów.

## Wyniki ukończonych treningów

{chr(10).join(lines)}

Niższa perplexity oznacza mniejszy błąd predykcji znaków. Główna kolumna dotyczy
części muzycznej, z pominięciem straty na nagłówkach. Metryka jest ważona liczbą
znaków i obejmuje całą walidację. Pole n-gramowego kontekstu oznacza długość
historii: kontekst 6 wykorzystuje modele do 7-gramów, nie klasyczny 6-gram.

Najlepszy wynik walidacyjny w tej serii: **{winner['model']}**, PPL muzyki
**{winner['val_body_ppl']:.4f}**. Jest to obserwacja dla tego korpusu i budżetu;
nie uzasadnia twierdzenia, że dana architektura zawsze będzie lepsza. Walidacja
służyła również do wyboru checkpointu, więc te liczby nie zastępują końcowej
oceny testowej. Mniejsza strata nie dowodzi lepszego frazowania ani muzykalności.

Treningi neuronowe korzystały z CPU i FP32. GPT: 4 bloki, 4 głowy, szerokość 128,
kontekst 128, dropout 0,1; NPLM: kontekst 16, embedding 32, warstwa ukryta 128.
AdamW, batch 32, warmup 100 kroków i kosinusowy learning rate. Każdy przebieg
wykonał około 3,62 mln prezentacji pozycji docelowych, z powtórzeniami podczas
losowania. Równa liczba aktualizacji nie oznacza równych FLOPs.

Po zakończeniu wczytano wszystkie najlepsze checkpointy w świeżym procesie,
sprawdzono skończone wagi FP32, stany optymalizatora i hashe oraz ponownie
obliczono pełną walidację. Wyniki zgadzają się w tolerancji 0,000001.
Szczegóły: `audit/run_integrity.json`.

## Co zapisano i co dalej

Paczka zawiera dane, manifest, skrypty przygotowania i audytu, kod treningu,
wykonany notebook z wykresami, konfiguracje, pełne logi oraz najlepszy i ostatni
checkpoint każdego modelu neuronowego. Ostatnie checkpointy mają optymalizator
oraz stany generatorów losowych. `README_PL.md` podaje dokładne komendy wznowienia.

Zaktualizowany plik kontynuacji odróżnia historyczne ustalenia od tej ukończonej
serii. Test pozostaje do jednorazowej oceny po zamrożeniu dalszych wyborów.
Przy szacowaniu niepewności należy losować grupy utworów, a nie pojedyncze znaki.
Następnie warto ocenić generowane melodie: poprawność ABC, metrum, frazowanie
i odsłuch. Nie uruchomiono usługi ani publikacji modeli.

Zasoby tej sesji wystarczyły do ukończenia tej serii. Dalsze większe eksperymenty
mogą wymagać dostępu do Twojego środowiska GPU. Dostęp do tego sprzętu nie został
tu potwierdzony. Eksperci jig/walc/reel i Obsidian wymagają oddzielnych dostępnych
korpusów; tych danych nie zastąpiono syntetycznymi przykładami.
'''
report_path=ROOT/'Micro_models_audyt_i_treningi_20260926_PL.md'
report_path.write_text(report)
campaign={'as_of':now,'status':'completed_first_bach_training_campaign',
          'manifest_sha256':summary['manifest_sha256'],'source_commit':load('data/manifest.json')['source_commit'],
          'scope':'Bach soprano; separate real corpus, not all original benchmarks',
          'dataset':profile,'audit_checks_passed':len(audit['checks']),
          'independent_checker':'Separate implementation; RapidFuzz in builder, SequenceMatcher in audit',
          'protocol_revision':'88% initial threshold invalidated; 80% final; all weights restarted',
          'test_evaluated':False,'precision':'FP32','compute':'CPU, two threads per neural run',
          'results':summary['rows'],'checkpoint_integrity':integrity,
          'files':{'bundle':'Micro_models_treningi_20260926.zip','report':report_path.name,
                   'notebook':'Micro_models_audyt_danych_20260926.ipynb'},
          'next_action':'Keep final split fixed. Freeze model choices before a one-time group-aware test assessment; evaluate generated music separately.',
          'upstream_status':'Original E1 and Obsidian scripts not changed; new audited pipeline avoids their identified split/metric issues.',
          'limitations':['Only 17 validation and 17 test groups','Rare modes/meters missing from held-out splits',
                         'Five rare duration events absent from training per held-out split',
                         'No GPU access verified','Do not compare perplexity directly with upstream corpus/encoding']}
if args.continuation:
    previous=json.loads(args.continuation.read_text())
    previous['latest_campaign']=campaign
    previous['updated_at']=now
    previous['next_action']=campaign['next_action']
    previous['scope_completed'].append('2026-09-26: audyt i double check korpusu Bach; ukończone trzy porównania n-gramów, NPLM i dwa GPT po 2000 kroków; checkpointy i metryki zapisane')
    args.continuation.write_text(json.dumps(previous,indent=2,ensure_ascii=False)+'\n')
    (ROOT/'Micro_models_kontynuacja.json').write_text(json.dumps(previous,indent=2,ensure_ascii=False)+'\n')
else:
    (ROOT/'campaign_state.json').write_text(json.dumps(campaign,indent=2,ensure_ascii=False)+'\n')
print(report_path)
