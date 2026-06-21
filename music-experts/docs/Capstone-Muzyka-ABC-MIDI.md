---
type: nauka-capstone
title: "Capstone — model generujący muzykę fortepianową (ABC → MIDI), styl Chopina"
status: zaplanowany
data: 2026-06-19
created_at: 2026-06-19
author: Arkadiusz Słota
tags: [nauka, llm, muzyka, capstone, chopin]
---

# Capstone — Muzyka (ABC → MIDI)

> Projekt końcowy reużywający cały pipeline kursu (tokenizer → model → trening → ewaluacja). Cel kulturowy: generowanie utworów fortepianowych w stylu Chopina; podtrzymanie dziedzictwa. **Po dokończeniu kursu.**

## Idea
Muzyka symboliczna = sekwencja zdarzeń → to samo zadanie co tekst (next-token). Ten sam GPT-2, inne tokeny.
Dowód, że działa: MuseNet (OpenAI, GPT-2, „w stylu Chopina"), Music Transformer (Magenta, MAESTRO), folk-rnn (ABC jako tekst).

## Plan dwuetapowy (decyzja Arka)
1. **ABC najpierw** — muzyka jako tekst; reużywa wszystko z kursu; szybki „hello world", słychać efekt.
2. **MIDI potem** (przez `miditok`: REMI / MIDI-Like) — bogatsze (velocity, pedał, timing); **porównanie ABC vs MIDI = widać różnicę**.

## Kluczowa decyzja projektowa (z lekcji L01–L03)
Pure-Chopin-from-scratch = **za mało danych** (kilkaset utworów → niedouczony, L03/Chinchilla).
**Ścieżka:** pretrenuj na DUŻO pianistyki (MAESTRO / klasyka) → **fine-tune na Chopinie** (transfer stylu, L01/L02).

## Dane
MAESTRO (Magenta, MIDI), zbiory MIDI Chopina (Kunstderfuge/Classical Archives), ewentualnie ABC. Held-out test (bez wycieku — L03).

## Ewaluacja
Ucho Arka (5 lat fortepianu = ekspert w pętli) + metryki (rozkłady wysokości, długość spójnej frazy) + held-out.

## Realizm (uczciwie)
Osiągalne: spójne, często ładne utwory z chopinowskim zabarwieniem. Nieosiągalne przy tej skali: nieodróżnialne arcydzieło. Sprzęt (RTX 3050 → RX 7900 XTX) z zapasem — modele muzyczne bywają mniejsze niż tekstowy 124M.

## Prior art
Music Transformer (Huang i in. 2018) · MuseNet (OpenAI 2019) · folk-rnn (ABC) · Anticipatory Music Transformer (2023). Narzędzia: `miditok`.

## 🔗 Powiązania
[[INDEX]] (mapa kursu) · reużywa: tokenizer (Odc.3), architektura (L-Architektura), trening, ewaluacja.
