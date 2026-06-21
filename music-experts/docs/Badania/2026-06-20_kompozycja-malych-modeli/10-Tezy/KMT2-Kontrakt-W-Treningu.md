---
type: teza
id: KMT2
title: "Wymuś wspólny kontrakt wyjścia PODCZAS treningu (zamrożona głowica / wspólna przestrzeń)"
status: w-dyskusji
data: 2026-06-20
created_at: 2026-06-20
parents: ["KMC1"]
author: Arkadiusz Słota
---

# KMT2 — Kontrakt wymuszony w treningu

## Teza
Eksperci uczą się **przeciw wspólnej, zamrożonej przestrzeni reprezentacji** (+ wspólna `lm_head` dla porównywalności) → ich reprezentacje są **mieszalne z konstrukcji** (nie trzeba ich potem mozolnie godzić). Kontrakt jak interfejs w programowaniu: ustalony z góry, wszyscy go respektują.

**Gdzie styk (rozstrzygnięty widelec):** kontrakt działa na **reprezentacji pośredniej** (np. po bloku 2), NIE na logitach. Wspólna `lm_head` służy tylko porównywalności; robota mappera/słownika/wariancji dzieje się na reprezentacji. Styk na logitach zredukowałby program do ensemblingu.

## Antyteza
[[../15-Antytezy/KMAT2-Wyrownanie-Posthoc|KMAT2]] — trenuj niezależnie, wyrównaj **po fakcie** (ZipIt!/Git Re-Basin); elastyczniejsze, nie wymaga koordynacji treningu.

## Powiązania
[[../00-Cele/KMC1-Cel-Kompozycja-Malych-Modeli|KMC1]] · synteza: [[../25-Syntezy/KMS2-Kontrakt-Vs-Posthoc|KMS2]]
