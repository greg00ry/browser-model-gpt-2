---
date: 2026-06-20
tags: [sampler, temperature, topk, generowanie]
---

# Strategia samplowania — temperature=0 to nie softmax

`temperature=0` to specjalny przypadek: nie ma sensu dzielić przez 0 i liczyć softmax — zwracamy po prostu `argmax`. Implementacja sprawdza ten warunek osobno.

**Kolejność operacji ma znaczenie:**
```
logits → temperature scaling → top-k masking → softmax → multinomial
```
Zamiana kolejności (np. softmax przed temperature) daje inne rozkłady.

**`lastTokenLogits`** wyciąga ostatni wiersz z `[seq, vocabSize]` — tylko ostatni token ma sens przy autoregresywnej generacji. Wywołanie `forward()` zwraca logity dla KAŻDEJ pozycji, ale interesuje nas wyłącznie ostatnia.

**Top-k przez próg:** zamiast pełnego sortowania i kopiowania, liczymy próg z `sorted[k-1]` i zerujemy (`-Infinity`) poniżej. Działa poprawnie przy równych wartościach (threshold = min z top-k, nie strict `>`).

## Powiązane

- [[index]]
- [[architecture/weight-tying]]
