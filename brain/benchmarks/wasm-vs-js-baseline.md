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

## Następny krok optymalizacji

Pre-alokacja: wszystkie intermediary tensory (qkv, q, k, v, qH, kH, vH, scores, itp.) alokowane raz przed pętlą generacji i reużywane. Eliminuje GC pressure.

## Powiązane

- [[index]]
- [[benchmarks/baseline-node]]
- [[architecture/wasm-interop]]
- [[architecture/assemblyscript-setup]]
