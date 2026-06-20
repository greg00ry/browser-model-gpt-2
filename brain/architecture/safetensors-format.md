---
date: 2026-06-20
tags: [safetensors, loader, wagi, format, alignment, klucze]
---

# Format safetensors — pułapki przy ładowaniu

**Struktura pliku:**
```
[8 bajtów: uint64 LE = długość nagłówka JSON]
[N bajtów: nagłówek JSON]
[dane tensorów: ciągły blok FP32]
```

**Offsety tensorów są relatywne do sekcji danych**, nie do początku pliku. Błąd: użycie surowego offsetu z nagłówka bez dodania `8 + headerLen` = błędne wagi, brak błędu w runtime.

**Alignment gotcha:** zero-copy `new Float32Array(buffer, dataStart + offset, ...)` rzuca `RangeError` gdy `dataStart + offset` nie jest wielokrotnością 4. SafeTensors specyfikacja gwarantuje wyrównanie wewnątrz sekcji danych, ale `dataStart` (8 + headerLen) może nie być wyrównany. Bezpieczne rozwiązanie: kopiuj przez `Uint8Array` → `Float32Array`.

**Klucze w pliku `openai-community/gpt2` (bez prefiksu `transformer.`):**
- `wte.weight` — token embeddings
- `wpe.weight` — position embeddings
- `h.{i}.ln_1.weight` / `h.{i}.ln_1.bias`
- `h.{i}.attn.c_attn.weight` — Q+K+V razem
- `h.{i}.attn.bias` — **zapamiętana maska kauzalna, ignorujemy** (nie jest parametrem treningowym)
- `ln_f.weight` / `ln_f.bias` — końcowy LayerNorm
- `lm_head.weight` — **nie istnieje** (patrz [[architecture/weight-tying]])

**Uwaga:** niektóre checkpointy GPT-2 na HuggingFace używają prefiksu `transformer.` (np. fine-tuned modele). Oficjalny `openai-community/gpt2` go nie ma.

**`@types/node` wymagany** dla `fs/promises` w `loadFromFile()` (tylko Node.js helper, nie używany w przeglądarce).

## Powiązane

- [[index]]
- [[architecture/weight-tying]]
- [[architecture/float32array-tensor-backend]]
