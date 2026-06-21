---
type: nauka-koncepcja
id: GRP
title: "Granie razem — sprzężone oscylatory i polifonia (oś pionowa)"
status: koncepcja
data: 2026-06-21
created_at: 2026-06-21
author: Arkadiusz Słota
---

# Granie razem — sprzężone oscylatory i polifonia

> Nowy kierunek, **prostopadły do stitcha**. Stitch był poziomy (dwa modele → jedna linia w czasie → zapadał się do ensemble). To jest **pionowe**: N linii, które się **nawzajem słuchają** i układają w harmonię — definicja zespołu (kontrapunkt), nie zlana linia. Cytaty zweryfikowane u źródeł (2026-06-21).

## Dlaczego to inna oś niż stitch
- **Stitch (poziomy):** fuzja dwóch modeli w jedną linię → dlatego ≈ ensemble ([[Kompozycja-Eksperymenty]] E1).
- **Granie razem (pionowy):** osobne głosy trzymane osobno + sprzężenie, by brzmiały razem. Muzyka potrzebuje obu osi; ta jest bardziej muzyczna (harmonia/kontrapunkt) — stitch nigdy jej nie dosięgał.

## Matematyka, nie poezja: sprzężone oscylatory
Analogia „dostają bodziec, ktoś się rozjeżdża, potem znów się składają" = **model Kuramoto** (Kuramoto 1975, *Self-entrainment of a population of coupled non-linear oscillators*): sprzężone oscylatory fazowe spontanicznie synchronizują się po przekroczeniu krytycznej siły sprzężenia (metronomy na wspólnym stole, świetliki). Realne, mierzalne zjawisko.

## Gdzie to już istnieje (i co jest nasze)
- **Coconet / Bach Doodle** (Huang i in., **ISMIR 2017**, arXiv:1903.07227): harmonizacja 4-głosu (chorały SATB) przez **iteracyjny blocked-Gibbs** (maskuj-i-przepróbkuj wszystkie głosy łącznie aż do zbieżności) na **jednym wspólnym** modelu. Czyli „rozjazd i ponowne złożenie" jest już zrobione — ale na jednym modelu.
- **Nasz twist:** osobne, **zamrożone** eksperty sprzężone w locie, nie jeden wspólny model. Rekombinacja istniejących klocków pod nowym kątem.

## Uczciwa zmiana ramy
To **nie** „darmowa kompozycja zamrożonych ekspertów", tylko „naucz lekkiej **warstwy-dyrygenta** nad ekspertami, na **polifonii**". Koszt niezerowy; ocena **uchem** (konsonans, prowadzenie głosów), nie perplexity.

## Trzy składniki konieczne
1. **Wspólny zegar** — każdy głos wie, GDZIE jest w takcie (wspólny token pozycji/beatu podawany wszystkim). Bez tego głosy dryfują bez ograniczeń. To „wspólny bodziec".
2. **Sprzężenie** — każdy czyta, co inni właśnie zagrali (lekka cross-atencja/adapter). Haczyk: nasze eksperty trenowane **monofonicznie** — nie znają harmonii pionowej → sprzężenie musi być **uczone na danych POLIFONICZNYCH** → **chorały Bacha (4-głos SATB) wracają do gry** (wykluczone przy kontrakcie monofonicznym — TU są idealne).
3. **Sygnał harmonii** — miara konsonansu karząca dysonans między równoczesnymi nutami; inaczej nic nie spina głosów pionowo.

## 🔬 Najsmaczniejszy eksperyment: perturbacja → re-synchronizacja
Rozbij jeden głos celowo (wymuś dryf) i zmierz: **czy zespół wraca do wspólnej fazy/harmonii i jak szybko** (czas + amplituda powrotu po zaburzeniu). Czysta metryka, **nowatorska dla sprzężonych generatorów językowych** — dokładnie „rozjeżdżą się i znów muszą się złożyć". Tego nikt porządnie nie zmierzył dla osobnych modeli językowych.

## Minimalny start (kolejność ma znaczenie!)
**Zegar najpierw, potem słuchanie, potem harmonia.** Bez wspólnego zegara dwa zamrożone monofoniczne modele NIE zsynchronizują się same — kakofonia, nie jam.
1. 2 głosy + wspólny token zegara.
2. Lekki adapter cross-atencji uczony na 2-głosowym chorale Bacha.
3. Mierz: (a) konsonans, (b) powrót po perturbacji.

## Spięcie z produktem (IFC)
Odpowiednik w IFC: wiele modeli-dyscyplin spójnych bez kolizji. „Harmonia" = model bez clashy; „rozjazd i złożenie" = clash detection + koordynacja (ból #1 BIM). → [[Cele-Globalne-i-Kotwica]].

## Powiązania
[[Badania-INDEX]] · [[Kompozycja-INDEX]] · [[Emergencja-i-Wspolna-Reprezentacja]] · [[Cele-Globalne-i-Kotwica]]
