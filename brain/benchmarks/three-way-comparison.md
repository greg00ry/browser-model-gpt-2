---
date: 2026-06-20
tags: [benchmark, python, pytorch, blas, wasm, js, trzy-benchmarki]
---

# Trójstronne porównanie: Python vs JS vs Wasm

**Środowisko:** MacOS, CPU only, float32, GPT-2 Small 124M  
**Prompt:** `"The brain is"`, 20 tokenów, temperature=1.0, top-k=40

| Backend | Tok/s | vs Python | Stack |
|---|---|---|---|
| Python (HuggingFace / PyTorch 2.12) | **17.40** | 1.00× | OpenBLAS + SIMD |
| TypeScript / V8 JS | **0.80** | 0.046× | JIT + auto-vectorize |
| AssemblyScript / Wasm | **0.30** | 0.017× | naive scalar matmul |

## Dlaczego Python jest 21× szybszy od JS?

PyTorch deleguje matmul do **OpenBLAS** (lub MKL), który:
- używa AVX2 (8× f32 naraz) lub AVX-512 (16× f32)
- jest cache-blockowany (tiling, lepsze wykorzystanie L2/L3)
- może używać wielu wątków (OpenMP)

Efektywnie osiąga ~100 GFLOP/s na nowoczesnym CPU.

Nasze implementacje:
- JS V8: ~5 GFLOP/s (JIT auto-vectorizes simple loops, 1 wątek)
- Wasm: ~2 GFLOP/s (scalar, brak SIMD, brak threadów)

## Otwarta kwestia badawcza

Pytanie: **czy da się zbliżyć do Python wydajnością w Wasm na CPU bez GPU?**

Ścieżki:
1. **Wasm SIMD** (`f32x4`, 4× throughput) → ~1.2 tok/s → 14× off
2. **Cache-blocking matmul** (tiling) → kolejne 2-4× → ~5 tok/s → 3.5× off
3. **WebNN API** (deleguje do przeglądarki/systemu BLAS) → może =Python
4. **XNNPACK w Wasm** (Google's optimized NN ops) → ~=Python na CPU

Wasm SIMD + tiling to prawdopodobnie maksimum co można osiągnąć czystym Wasm.
WebNN/XNNPACK to delegacja — nie "od zera" ale realna opcja w przeglądarce.

## Co to oznacza dla projektu badawczego

Wyniki potwierdzają tezę: naiwna implementacja w Wasm nie jest konkurencyjna z zoptymalizowanym ML stack. Żeby Wasm był użyteczny w przeglądarce dla LLM, potrzebne są te same optymalizacje co w BLAS (SIMD + tiling + threading).

Projekt "The Brain" odpowiada na pytanie: **ile kosztuje reimplementacja od zera** i które optymalizacje są kluczowe.

## Powiązane

- [[benchmarks/wasm-vs-js-baseline]]
- [[architecture/wasm-interop]]
- [[research/optimization-roadmap]]
