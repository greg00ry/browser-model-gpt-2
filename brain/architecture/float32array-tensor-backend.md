---
date: 2026-06-20
tags: [tensor, memory, fp32, wasm]
---

# Float32Array jako backend tensora

Dane tensora trzymamy w `Float32Array` (natywny typ JS).

**Dlaczego:**  
FP32 = identyczne z wagami GPT-2 z HuggingFace (`.safetensors`). Zero konwersji przy ładowaniu wag. IEEE 754 FP32 w Wasm jest zgodne z PyTorch CPU — ΔPerplexity powinno być bliskie zeru jeśli implementacja jest poprawna.

**Konsekwencje:**  
GPT-2 Small (124M) zajmie ~500 MB RAM. Jeśli benchmark pokaże przekroczenie 1 GB → Etap 1b: kwantyzacja INT8.

## Powiązane

- [[index]]
- [[architecture/tensor-shape-convention]]
