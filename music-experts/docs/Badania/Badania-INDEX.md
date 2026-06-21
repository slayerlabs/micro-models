---
type: index-badania
title: "Slayer Micro-Models — rejestr badań"
status: aktywny
data: 2026-06-20
created_at: 2026-06-20
author: Arkadiusz Słota
repo: "https://github.com/slayerlabs/micro-models"
---

# Slayer Micro-Models — rejestr badań

> Wszystkie badania nad małymi modelami w jednym miejscu. Repo programu: **github.com/slayerlabs/micro-models**.
> **Konwencja:** `Badania/<RRRR-MM-DD>_<temat>/` (data z przodu = sortowanie chronologiczne) + **unikalne nazwy plików** (Obsidian linkuje po nazwie, więc bez powtórek typu `00-Koncepcja`).

## Rejestr badań
| Data | Temat | Status | Dokument |
|---|---|---|---|
| 2026-06-20 | Kompozycja małych modeli (kontrakt · mapper · wariancja) | warsztat dialektyczny | [[Kompozycja-INDEX]] |
| 2026-06-20 | N-gram → mini-transformer (most) | koncepcja / roadmap | [[Research-NGram-vs-MiniTransformer]] |

## Baza zrealizowana (fundament)
- [[Capstone-Muzyka-LOG-Realizacja]] — pierwszy model: n-gram baseline + GPT od zera (val ppl 3,80), opublikowany na HF/GitHub. To na nim stawiamy eksperymenty kompozycji.

## Referencje wspólne
- **Repo programu:** https://github.com/slayerlabs/micro-models

## Jak dodać nowe badanie
1. Folder `Badania/<data>_<krótki-temat>/`.
2. Dokument koncepcji o **unikalnej** nazwie (np. `Temat-Koncepcja.md`).
3. Wiersz w rejestrze powyżej.
4. (Później) `...-Eksperymenty.md` i `...-Wyniki.md`, gdy będą dane.
