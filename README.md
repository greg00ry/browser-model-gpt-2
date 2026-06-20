# browser-model-gpt-2

> **Projekt The Brain — Etap 1**
> Implementacja GPT-2 (124M) w TypeScript 7 / WebAssembly z benchmarkiem inferencji w przeglądarce.

## O projekcie

Celem Etapu 1 jest implementacja modelu GPT-2 Small (124M parametrów) od zera w TypeScript 7 (kompilator `tsc-go`), kompilacja do WebAssembly oraz porównanie wydajności i jakości inferencji z referencyjną implementacją w Pythonie (HuggingFace Transformers).

Projekt odpowiada na pytanie: **jaki jest realny koszt uruchomienia modelu językowego bezpośrednio w przeglądarce, bez zależności od API w chmurze?**

Docelowa wizja projektu wykracza poza benchmarki: chodzi o zbudowanie środowiska uruchamiania modeli językowych w przeglądarce, w którym agent traktowany jest jako **kompilator kodu maszynowego** — przyjmuje język naturalny i produkuje wykonywalny kod (żądania HTTP, bajtkod Wasm, sekwencje akcji).

## Benchmarki

| Metryka | Python / HuggingFace | TypeScript / Wasm |
|---|---|---|
| Tokeny / sekundę | baseline (empiryczny) | cel: ≥ 5 tok/s |
| Pamięć RAM | ~500 MB (RSS) | ~500–700 MB (heap JS) |
| Perplexity (ΔPP) | baseline | akceptowalne: < 0.5 |

## Stos technologiczny

| Warstwa | Narzędzie |
|---|---|
| Implementacja modelu | TypeScript 7 + `tsc-go` (strict mode, zero zależności ML) |
| Kompilacja do Wasm | `tsc-go` native compiler |
| Wagi modelu | GPT-2 Small 124M `.safetensors` (OpenAI / HuggingFace) |
| Tokenizer | `tiktoken` (JS port, BPE) |
| Środowisko testowe | Chrome 125+ / Firefox 126+, CPU only |
| Referencja Python | HuggingFace Transformers 4.x, PyTorch 2.x, CPU |

## Roadmapa

| Etap | Opis |
|---|---|
| **1 (obecny)** | GPT-2 w TS/Wasm — implementacja + benchmarki |
| 1b (opcjonalny) | Kwantyzacja wag (INT8/INT4) jeśli RAM > 1 GB lub tok/s < 5 |
| 2 | Fine-tuning → generowanie żądań HTTP z języka naturalnego |
| 3 | Pełny agent webowy: wysyła żądania, przetwarza odpowiedzi |

## Gałęzie

- `main` — kod projektu
- `paper` — propozycja badawcza (Etap 1)

## Kontekst

Etap 1 jest weryfikowalnym proof of concept z jasno zdefiniowanymi metrykami. Wyniki — niezależnie od tego czy pozytywne czy negatywne — dostarczą empirycznej podstawy dla kolejnych etapów i decyzji o kierunku rozwoju projektu.
