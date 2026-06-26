# browser-model-gpt-2

> **Projekt The Brain** — GPT-2 Small (124M) zaimplementowany od zera w TypeScript / AssemblyScript i uruchomiony w przeglądarce jako WebAssembly. Bez API. Bez chmury. Bez bibliotek ML.

## Pytanie badawcze

**Jaki jest realny koszt uruchomienia modelu językowego bezpośrednio w przeglądarce?**

Nie chodzi tylko o tok/s. Chodzi o to, co musisz wiedzieć i co musisz zoptymalizować, żeby naiwna implementacja stała się czymś używalnym — i jak daleko od Pythona możesz dojść czystym Wasm na CPU.

## Historia projektu

### Etap 1 — implementacja i pierwsze pomiary

Cały model GPT-2 Small napisany ręcznie: embedding, attention z multi-head, feedforward, layer norm, weight tying. Zero zewnętrznych zależności ML. Kompilacja TypeScript 7 → AssemblyScript → Wasm przez `tsc-go`.

Pierwsze pomiary (Node.js, CPU, 20 tokenów):

| Backend | tok/s |
|---|---|
| Python / PyTorch (OpenBLAS) | 12.24 |
| TypeScript / V8 JS | 0.58 |
| AssemblyScript / Wasm (naiwny) | 0.20 |

**Zaskakujący wynik:** Wasm był *wolniejszy* od JS — 0.34× zamiast oczekiwanego przyspieszenia.

Przyczyna: naiwna implementacja alokuje setki `Float32Array` na token (q, k, v, scores, bufory pośrednie attention i feedforward). V8 ma wysoce zoptymalizowany GC pod ten wzorzec. AS runtime jest prostszy i wolniejszy przy takiej liczbie krótkotrwałych alokacji.

### Optymalizacje pre-alokacji

Zamiast alokować w hot path — jeden `Scratch` obiekt z wszystkimi buforami alokowanymi raz przy inicjalizacji (`initScratch(maxSeq)`). Eliminacja ~180 alokacji na token.

Dodatkowo: `lm_head` tylko dla ostatniego tokenu — zamiast mnożenia `[seq, 768] × [768, 50257]` tylko `[1, 768] × [768, 50257]`. Dla sekwencji 12 tokenów to 12× mniej operacji na tym kroku.

Po optymalizacjach:

| Backend | tok/s |
|---|---|
| JS V8 | 0.80 |
| Wasm (pre-alokacja + lm_head) | 0.30 |

Wasm nadal wolniejszy od JS — V8 JIT auto-wektoryzuje proste pętle matmul (AVX/SSE). Wasm skalarny nie ma wektoryzacji. Różnica 2.6× = spodziewany efekt braku SIMD.

### Etap 2 — SIMD i cross-platform

Wdrożenie `f32x4` SIMD w AssemblyScript (`--enable-feature simd`). Zamiast jednej operacji na float — cztery naraz, 4× throughput matmul w teorii.

Wynik na Apple Silicon (MacOS, Chrome):

| Backend | tok/s |
|---|---|
| Wasm (naiwny, bez SIMD) | 0.30 |
| Wasm SIMD f32x4 | **2.00** |

**6.7× przyspieszenie** relative do naiwnego Wasm. JS V8 z auto-wektoryzacją: 0.80 tok/s — Wasm SIMD przebija JS 2.5×.

### Negatywny wynik: tiling nie pomógł

Cache-blocking (tiling) matmul z różnymi rozmiarami bloków na Apple Silicon:

| Konfiguracja | tok/s |
|---|---|
| TILE_K=64, TILE_N=128 | 1.91 |
| TILE_K=32, TILE_N=64 | 1.86 |
| SIMD bez tilingu | **2.01** |

Tiling pogorszył wyniki. Przyczyna: Apple Silicon ma L1 data cache 128 KB (vs 32 KB na x86). Hardware prefetcher M-series radzi sobie z sekwencyjnym wzorcem dostępu bez jawnego tilingu. Overhead dodatkowych pętli nie jest kompensowany przez lepszy cache-hit.

Tiling jest optymalizacją dla x86 — na własnym sprzęcie daje odwrotny efekt.

### Cross-platform

| Platforma | Backend | tok/s |
|---|---|---|
| MacOS / Apple Silicon | Wasm SIMD (Node.js) | 2.01 |
| MacOS / Apple Silicon | Wasm SIMD (Chrome) | 2.00 |
| iOS/Android | Wasm SIMD (mobile browser) | 1.30 |
| Windows / Ryzen 5 | Wasm SIMD (Chrome) | 0.53 |
| macOS 2013 / Intel | Safari | nie działa* |

\* Safari na starym macOS nie obsługuje Wasm SIMD. Wasm SIMD jest wspierane od Chrome 91 / Firefox 89 (2021) — sprzęt pre-2019 jest niereprezentacyjny.

Luka Apple Silicon vs x86: **3.8×** (2.00 vs 0.53 tok/s). Prawdopodobna przyczyna: unified memory i lepsza implementacja Wasm SIMD w V8 na ARM.

Model działa na telefonie — 1.30 tok/s to ~1 token/s, 20 tokenów w ~15 sekund. Użyteczne jako demo.

## Pełne zestawienie wyników

| Backend | tok/s | vs Python/PyTorch | Środowisko |
|---|---|---|---|
| Python / PyTorch (OpenBLAS) | 12.24 | 1.00× (baseline) | MacOS, CPU |
| Python / numpy (BLAS) | 4.80 | 0.39× | MacOS, CPU |
| Wasm SIMD (Node.js) | 2.01 | 0.16× | MacOS, Apple Silicon |
| Wasm SIMD (Chrome desktop) | 2.00 | 0.16× | MacOS, Apple Silicon |
| Wasm SIMD (mobile browser) | 1.30 | 0.11× | iOS/Android |
| TypeScript / V8 JS | 0.80 | 0.065× | MacOS, Node.js |
| Wasm SIMD (Chrome, x86) | 0.53 | 0.043× | Windows, Ryzen 5 |
| Wasm (naiwny, bez SIMD) | 0.30 | 0.025× | MacOS, Node.js |
| Wasm (bez optymalizacji) | 0.20 | 0.016× | MacOS, Node.js |

## Wnioski

Naiwna implementacja Wasm osiąga 0.016× wydajności Pythona. Po optymalizacjach (pre-alokacja + SIMD) — 0.16×. Luka: **6×** do Pythona na tym samym sprzęcie.

Żeby Wasm dorównał Pythonowi na CPU potrzebne są te same techniki co w BLAS: SIMD + cache-blocking + wielowątkowość (SharedArrayBuffer + Atomics). Każda z osobna daje 2-6×, razem mogłyby zamknąć lukę. Nie sprawdzono.

Projekt odpowiada na pytanie empirycznie: **uruchomienie GPT-2 w przeglądarce jest możliwe, użyteczne jako demo, ale wymaga wiedzy o optymalizacji niskiego poziomu** — tej samej co przy pisaniu zoptymalizowanego BLAS.

## Stos technologiczny

| Warstwa | Narzędzie |
|---|---|
| Implementacja modelu | TypeScript 7 + `tsc-go` (strict, zero zależności ML) |
| Wasm backend | AssemblyScript z `f32x4` SIMD |
| Wagi modelu | GPT-2 Small 124M `.safetensors` (OpenAI / HuggingFace) |
| Tokenizer | `tiktoken` (JS port, BPE) |
| Środowisko testowe | Chrome 125+, Firefox 126+, Node.js v25, CPU only |
| Referencja Python | HuggingFace Transformers 4.x, PyTorch 2.x, numpy |

## Roadmapa

| Etap | Opis | Status |
|---|---|---|
| **1** | GPT-2 w TS/Wasm — implementacja + benchmarki | zakończony |
| **2** | SIMD + cross-platform | zakończony |
| 1b (opcjonalny) | Kwantyzacja wag (INT8/INT4) | rozważany |
| 3 | Fine-tuning → generowanie żądań HTTP z języka naturalnego | planowany |
| 4 | Pełny agent webowy: wysyła żądania, przetwarza odpowiedzi | planowany |

## Gałęzie

- `main` — kod projektu
- `paper` — raporty badawcze (Etap 1, Etap 2)
