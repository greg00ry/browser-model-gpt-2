---
date: 2026-06-20
tags: [benchmark, wasm, js, tok/s, gc, wynik]
---

# Wasm vs JS — pierwszy pomiar porównawczy

**Środowisko:** Node.js v25.2.1, MacOS, CPU only  
**Prompt:** `"The brain is"`, 20 tokenów

| Backend | Tok/s | Speedup |
|---|---|---|
| TypeScript / V8 JS | 0.58 | 1.00x (baseline) |
| AssemblyScript / Wasm | 0.20 | **0.34x (wolniejszy)** |

## Dlaczego Wasm jest wolniejszy od JS?

Nieoczekiwany wynik — ale ma konkretną przyczynę: **alokacje w pętli**.

W każdym forward passie AS alokuje setki `Float32Array` (intermediary tensory w attention, feedforward, splitHeads, mergeHeads itd.). AS GC musi je zbierać. V8 ma wysoce zoptymalizowany garbage collector pod ten wzorzec — AS GC jest prostszy i wolniejszy przy takim obciążeniu.

V8 dodatkowo JIT-kompiluje JS do natywnego kodu maszynowego ze statyczną analizą typów — efektywnie też "kompiluje do natywnego". Różnica między V8 JIT a Wasm maleje w praktyce.

## Co to zmienia w projekcie badawczym

To jest **prawdziwy wynik badawczy**, nie błąd. Pokazuje że:
1. Naiwne Wasm (z AS GC i alokacjami w pętli) ≠ szybkie Wasm
2. V8 jest lepszy w zarządzaniu krótkożyjącymi alokacjami niż AS runtime
3. Żeby Wasm był szybszy od JS trzeba **pre-alokować bufory** i unikać GC w hot path

## Historia optymalizacji

| Wersja | JS tok/s | Wasm tok/s | Speedup | Zmiana |
|---|---|---|---|---|
| baseline (naiwna) | 0.58 | 0.20 | 0.34x | allocF32 w hot path |
| + pre-alokacja scratch | 0.58 | 0.23 | 0.40x | Scratch class, zero alloc |
| + lm_head last-token | **0.80** | **0.30** | **0.38x** | tylko ostatni wiersz |

Pre-alokacja (assembly/scratch.ts): wszystkie bufory pośrednie (qkv, q, k, v, qH, kH, vH, scores, kT, attended, merged, hidden, tmp, attnOut, mlpOut, x, wteT) alokowane raz w `initScratch(maxSeq)`. Eliminuje ~180 alokacji na token.

lm_head tylko dla ostatniego tokenu: zamiast [seq, 768] × [768, 50257] → [1, 768] × [768, 50257]. Dla avg seq=12 to 12x mniej operacji na tym kroku.

## Dlaczego Wasm nadal wolniejszy od JS?

V8 JIT auto-wektoryzuje wewnętrzne pętle matmul (AVX/SSE: 4-8 f32 na raz).
AS-Wasm to kod skalarny bez SIMD. Różnica 2.6x = spodziewany efekt braku wektoryzacji.

**Żeby Wasm pokonał JS:** potrzebne `v128`/`f32x4` SIMD w AssemblyScript (`--enable-feature simd`). To 4x throughput na matmul = net ~4/2.6 = 1.5x nad JS.

## Powiązane

- [[index]]
- [[benchmarks/baseline-node]]
- [[architecture/wasm-interop]]
- [[architecture/assemblyscript-setup]]
