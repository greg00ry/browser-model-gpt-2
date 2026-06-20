---
date: 2026-06-20
tags: [benchmark, baseline, node, tok/s, cpu]
---

# Baseline Node.js — pierwszy pomiar

**Środowisko:** Node.js v25.2.1, MacOS, CPU only, bez Wasm SIMD  
**Prompt:** `"The brain is"`  
**Tokeny:** 50  

| Metryka | Wynik |
|---|---|
| Czas ładowania wag | 0.48s |
| Tok/s | **0.3 tok/s** |
| Czas generacji (50 tokenów) | 184.5s |

**Cel benchmarku (z propozycji badawczej):** ≥ 5 tok/s

**Wniosek:** aktualny wynik to ~17× poniżej celu. Wąskie gardło: naiwna pętla matmul w czystym JS (O(M×K×N) bez SIMD, bez wielowątkowości).

**Obserwacja:** output repetytywny (`the brain is the brain is...`) — typowe dla GPT-2 bez repetition penalty. Nie jest to błąd implementacji.

**Kolejne kroki optymalizacji:**
1. Wasm SIMD (`f32x4`) dla matmul — potencjalnie 4× przyspieszenie
2. KV-cache — eliminuje ponowne liczenie attention dla poprzednich tokenów
3. WebGPU — Etap 2

## Powiązane

- [[index]]
- [[architecture/generation-loop]]
- [[architecture/tensor-shape-convention]]
