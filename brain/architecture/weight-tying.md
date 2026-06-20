---
date: 2026-06-20
tags: [wagi, lm-head, wte, gpt2, memory]
---

# Weight tying — lm_head współdzieli wagi z wte

GPT-2 nie ma osobnej macierzy `lm_head`. Zamiast tego używa transponowanego `wte` (`[vocabSize, nEmbd]` → `[nEmbd, vocabSize]`).

```
logits = x @ wte.T
```

**Dlaczego to ważne:**  
Pliki `.safetensors` GPT-2 nie zawierają klucza `lm_head.weight` — próba załadowania go jako osobnej macierzy skończy się błędem. Trzeba explicite użyć `wte` z transponowaniem.

**Koszt pamięci:** zero — nie alokujemy nowej macierzy, tylko iterujemy po wte w odwróconej kolejności (w `transposeWeight` tworzymy kopię, do optymalizacji w Etapie 1b przez widok).

**Gdzie:** `model.ts` → `transposeWeight()` + `forward()`.

## Powiązane

- [[index]]
- [[architecture/float32array-tensor-backend]]
- [[architecture/tensor-shape-convention]]
