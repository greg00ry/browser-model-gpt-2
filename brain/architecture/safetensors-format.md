---
date: 2026-06-20
tags: [safetensors, loader, wagi, format, alignment]
---

# Format safetensors — pułapki przy ładowaniu

**Struktura pliku:**
```
[8 bajtów: uint64 LE = długość nagłówka JSON]
[N bajtów: nagłówek JSON]
[dane tensorów: ciągły blok FP32]
```

**Offsety tensorów są relatywne do sekcji danych**, nie do początku pliku. Błąd: użycie surowego offsetu z nagłówka bez dodania `8 + headerLen` = błędne wagi, brak błędu w runtime.

**Zero-copy:** `new Float32Array(buffer, dataStart + offset, length / 4)` tworzy widok bez kopiowania ~500 MB. SafeTensors gwarantuje wyrównanie danych do 8 bajtów, więc widok FP32 (4 bajty) jest bezpieczny.

**Klucze HuggingFace GPT-2:**
- `transformer.wte.weight` — token embeddings
- `transformer.wpe.weight` — position embeddings
- `transformer.h.{i}.attn.c_attn.weight` — Q+K+V razem
- `transformer.ln_f.weight` / `transformer.ln_f.bias` — końcowy LayerNorm
- `lm_head.weight` — **nie istnieje** (patrz [[architecture/weight-tying]])

**`@types/node` wymagany** dla `fs/promises` w `loadFromFile()` (tylko Node.js helper, nie używany w przeglądarce).

## Powiązane

- [[index]]
- [[architecture/weight-tying]]
- [[architecture/float32array-tensor-backend]]
