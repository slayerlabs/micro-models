# Slayer Micro-Models

**Małe modele, uczciwie zmierzone.**

Repozytorium kolektywu **Slayer** na *mikro-modele* — małe sieci trenowane od zera — wraz z eksperymentami
i rozumowaniem, które za nimi stoi. Budujemy modele dość małe, by **zrozumieć je od początku do końca**,
wytrenować w minuty i precyzyjnie o nich rozumować. Celem nie są wyniki z rankingów, tylko **zrozumienie**:
czego uczą się malutkie modele, gdzie się łamią i czy wielu małych ekspertów da się **złożyć** w coś większego.

## Zasady
- **Twardy pomiar.** Każde twierdzenie poparte liczbą albo źródłem pierwotnym — *żadnej tezy bez dowodu.*
- **Uczciwy zakres.** Mówimy, czego *nie* pokazaliśmy, i **publikujemy wyniki negatywne** — to też wyniki.
- **Reprodukowalność.** Kod, wagi i rozumowanie (notatki dialektyczne) leżą razem; surowe dane i generacje są
  poza repo, bo odtwarzają się ze skryptów.
- **Dobry smak ponad benchmaxxing.** Prosty baseline trzeba pobić *uczciwie*, nie oszukać.

## Projekty
| Projekt | Co to | Status |
|---|---|---|
| [`music-experts/`](music-experts/) | Małe char-level GPT (jig, Bach, walc, reel) trenowane od zera na [notacji ABC](https://abcnotation.com/) + eksperymenty z **kompozycją małych ekspertów** (stitching, ensemble, duet). | aktywny |
| [`obsidian-experts/`](obsidian-experts/) | Mikro-GPT PL (~2,64 mln param) trenowany od zera na **notatkach dialektycznych** z Obsidiana, rozbitych deterministycznie (AST-T2). Pomiar generalizacji (vs n-gram) + ablacja dekompozycja-vs-surowe. **Wynik: NEGATYWNY robust** (n-gram bije transformer przy ~3 MB; gap 0,10 bita/znak). | iteracja 1 |

*Kolejne projekty mikro-modeli dojdą jako siostrzane foldery.*

## Highlight — `music-experts/`
Rodzina modeli **~0,8 mln parametrów** — jedna architektura, każdy trenowany na CPU w minuty.

- **Eksperci:** jig (val ppl 3,80), sopran chorałów Bacha, walc (→ fortepian), reel (→ skrzypce).
- **Co pokazane:** mały char-LM uczy się realnej struktury muzycznej — metrum, sygnatury tonacji, kadencje —
  z samej predykcji następnego znaku (zero teorii muzyki w kodzie); czystsze dane mierzalnie obniżają perplexity.
- **Kompozycja (uczciwie):** mechanizm stitchu jest **bezstratny** (liniowy mapper na styku odtwarza baseline);
  **ensemble dwóch ekspertów bije każdy pojedynczy model** na zadaniu mieszanym; ale **stitch reprezentacji
  jeszcze NIE bije tego ensemble**. Prosty baseline jest mocny — fancy metoda musi dopiero zasłużyć. Otwarte
  kierunki: wymuszony wspólny kontrakt, bogatsze (nieliniowe) mappery.
- Pełne rozumowanie — teza ↔ antyteza → synteza, z weryfikacją źródeł — w [`music-experts/docs/`](music-experts/docs/).

## Układ repozytorium
```
micro-models/
└── <projekt>/             # samodzielny
    ├── src/               # pipeline: dane -> trening -> generacja -> kompozycja
    ├── data/models/       # wytrenowane checkpointy (wagi)
    ├── docs/              # notatki badawcze: koncepcja, eksperymenty, uczciwe wnioski
    └── README.md
```

## Jak zacząć
```bash
cd music-experts
pip install -r requirements.txt
# wygeneruj melodię od eksperta i wyrenderuj do MIDI
python src/generate/gen_samples.py --ckpt data/models/waltz_ckpt.pt --meter 3/4 --keys D,G,Emin --inst piano --out out
```

## O nas
Tworzone przez kolektyw **Slayer** — otwarty warsztat badawczy. Cel edukacyjno-badawczy.
**Licencja:** MIT (kod i wagi). Dane treningowe należą do swoich źródeł (patrz README projektów).
