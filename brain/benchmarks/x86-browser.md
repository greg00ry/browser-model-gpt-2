---
date: 2026-06-23
tags: [benchmark, x86, ryzen, intel, browser, wasm, cross-platform]
---

# GPT-2 Wasm — wyniki na x86 (browser)

## Wyniki

| Platforma | Backend | tok/s |
|---|---|---|
| Apple Silicon M-series | Wasm SIMD (Chrome) | 2.00 |
| Ryzen 5 (Windows) | Wasm SIMD (Chrome) | 0.53 |
| Intel Core i5 2013 (macOS) | Safari | nie działa* |

\* Safari na starym macOS nie wspiera WASM SIMD. Sprzęt z 2013 niereprezentacyjny — pominięty.

## Wniosek

Luka Apple Silicon vs x86: **3.8×** (2.00 vs 0.53 tok/s).

Różnica wynika prawdopodobnie z:
- Wyższa memory bandwidth na Apple Silicon (unified memory)
- Lepsza implementacja WASM SIMD w V8 na ARM vs x86 w tej generacji CPU
- Ryzen 5 laptopowy — throttling termiczny

Realny x86 target = Windows + Chrome/Firefox (WASM SIMD wspierane od Chrome 91 / Firefox 89, 2021).

## Nota o tiling

Wynik 0.53 tok/s na Ryzenie (L1=32KB) sugeruje że cache-blocking (tiling) **mógłby** pomóc na x86, gdzie L1 jest 4× mniejsze niż Apple Silicon. Nie zbadane — potencjalny temat.

## Powiązane

- [[benchmarks/mobile-browser]]
- [[benchmarks/tiling-experiment]]
- [[benchmarks/three-way-comparison]]
