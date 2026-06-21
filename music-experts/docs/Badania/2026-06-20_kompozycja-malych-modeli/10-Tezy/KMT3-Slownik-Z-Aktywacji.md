---
type: teza
id: KMT3
title: "Interfejs = wyuczony słownik cech z koherencji aktywacji na styku (VQ/SAE), nie zwykły tokenizer"
status: w-dyskusji
data: 2026-06-20
created_at: 2026-06-20
parents: ["KMC1"]
author: Arkadiusz Słota
---

# KMT3 — Słownik z koherencji aktywacji

## Teza
„Tokenizer-ale-nie-tokenizer": **wyuczony słownik cech** ze statystyk aktywacji na styku modeli, z którego bierzemy tylko cechy **koherentne między modelami**. Bogata, wspólna reprezentacja wymiany (interfejs). Komponenty istnieją: **VQ-VAE codebook**, **Sparse Autoencoders/dictionary learning**, **Rosetta Neurons** (cechy wspólne różnym modelom).

## Antyteza
[[../15-Antytezy/KMAT3-Liniowy-Mapper|KMAT3]] — prosty **liniowy mapper** (Procrustes) wystarczy, jest tańszy i stabilniejszy; słownik = przerost.

## Powiązania
[[../00-Cele/KMC1-Cel-Kompozycja-Malych-Modeli|KMC1]] · synteza: [[../25-Syntezy/KMS3-Liniowy-Najpierw|KMS3]]
