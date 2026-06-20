---
date: 2026-06-20
tags: [assemblyscript, wasm, tensor, f32, kompilacja]
---

# AssemblyScript tensor.ts — różnice vs TypeScript

**Wynik kompilacji:** `build/tensor.wasm` = 9.7 KB prawdziwego WebAssembly.

**Kluczowe różnice w stosunku do `src/tensor.ts`:**

| TypeScript | AssemblyScript |
|---|---|
| `number` | `f32` / `i32` |
| `Math.tanh()` | `Mathf.tanh()` (zwraca f32) |
| `Math.exp()` | `Mathf.exp()` |
| `a[i]` | `unchecked(a[i])` (brak bounds check = szybciej) |
| `new Float32Array(n)` | `new Float32Array(n)` (działa tak samo) |
| klasa z `readonly` | klasa bez `readonly` |

**API oparte na flat arrays zamiast obiektów Tensor:**  
Funkcje przyjmują `Float32Array` + wymiary jako osobne parametry (np. `aRows: i32, aCols: i32`). Łatwiejszy interop z hostem JS — nie trzeba zarządzać wskaźnikami do klas AS.

**`unchecked()`** — usuwa bounds checking przy dostępie do tablic. Krytyczne dla matmul (miliony dostępów w pętli). Bezpieczne bo wymiary są walidowane po stronie hosta.

**`Mathf` vs `Math`** — AS ma dwie wersje funkcji matematycznych: `Math` (f64) i `Mathf` (f32). Używamy `Mathf` bo wagi GPT-2 są w FP32 — mieszanie f32/f64 spowalnia i może zmieniać wyniki.

## Powiązane

- [[index]]
- [[architecture/assemblyscript-setup]]
- [[architecture/gelu-approximation]]
- [[architecture/float32array-tensor-backend]]
