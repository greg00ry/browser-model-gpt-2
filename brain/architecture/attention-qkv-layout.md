---
date: 2026-06-20
tags: [attention, weights, gpt2]
---

# Layout wag attention w GPT-2

GPT-2 łączy Q, K, V w jedną macierz projekcji `c_attn.weight: [768, 2304]`.

**Kolejność w macierzy:**
```
[Q: 0..768 | K: 768..1536 | V: 1536..2304]
```

Jeden `matmul` zamiast trzech — mniej wywołań, lepsze wykorzystanie cache.  
Po projekcji `sliceCols` tnie wynik na trzy tensory `[seq, 768]`.

**Ryzyko:**  
`sliceCols` kopiuje dane — w krytycznej ścieżce inferencji można zastąpić widokiem (subarray). Odkładamy do optymalizacji po pierwszym poprawnym benchmarku.

## Powiązane

- [[index]]
- [[architecture/tensor-shape-convention]]
