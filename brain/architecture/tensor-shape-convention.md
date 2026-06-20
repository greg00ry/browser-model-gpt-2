---
date: 2026-06-20
tags: [tensor, konwencja, attention]
---

# Konwencja kształtów tensorów

Flat `Float32Array` + `shape: readonly number[]`. Brak strides — layout zawsze row-major (C-order).

**Kluczowe kształty w GPT-2 Small:**

| Tensor | Shape |
|---|---|
| Wejście bloku | `[seq, 768]` |
| Q, K, V (po split) | `[seq, 768]` |
| Po splitHeads | `[12, seq, 64]` |
| Scores | `[12, seq, seq]` |
| Po mergeHeads | `[seq, 768]` |
| FeedForward (inner) | `[seq, 3072]` |

**Dlaczego flat array zamiast zagnieżdżonych tablic:**  
Wasm operuje na liniowej pamięci — flat layout bez pośrednich wskaźników to zero narzutu przy ewentualnej migracji do Wasm SIMD.

## Powiązane

- [[index]]
- [[architecture/float32array-tensor-backend]]
- [[architecture/attention-qkv-layout]]
