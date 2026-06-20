---
date: 2026-06-20
tags: [benchmark, tiling, cache-blocking, apple-silicon, wasm, negatywny-wynik]
---

# Cache-blocking (tiling) — negatywny wynik na Apple Silicon

## Eksperyment

Zaimplementowano cache-blocking matmul2d z różnymi rozmiarami bloków:
- TILE_K=64, TILE_N=128 → 1.91 tok/s
- TILE_K=32, TILE_N=64  → 1.86 tok/s
- SIMD bez tilingu       → **2.01 tok/s** (najlepszy)

## Dlaczego tiling nie pomógł

Apple Silicon (M-series) ma znacznie większe cache niż typowy x86:
- L1 data: **128KB** (vs 32KB na x86)
- L2: **12MB shared** (vs 256KB na x86)

Macierze wag GPT-2 Small: 768×2304 = 7MB — nie mieszczą się w L2, ale hardware prefetcher M-series jest agresywny i obsługuje sekwencyjny wzorzec dostępu `b[k*bCols+n]` bez potrzeby jawnego tilingu.

Tiling dodaje overhead (dodatkowe pętle, bardziej złożone indeksowanie) który na Apple Silicon nie jest kompensowany przez lepsze cache-hit.

## Wniosek

Tiling jest optymalizacją dla x86 z małym L1 (32KB). Na Apple Silicon wyniki mogą być inne niż na serwerach/przeglądarce użytkownika końcowego. Warto sprawdzić na typowym x86 laptop/desktop (Intel/AMD).

## Następny krok

Benchmark w przeglądarce Chrome (issue #3) pokaże rzeczywistą wydajność na x86, gdzie tiling może mieć sens.

## Powiązane

- [[benchmarks/wasm-vs-js-baseline]]
- [[benchmarks/three-way-comparison]]
