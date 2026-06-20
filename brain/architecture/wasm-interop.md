---
date: 2026-06-20
tags: [wasm, interop, memory, pinning, as-runtime]
---

# Wasm interop — pamięć i pinowanie

**Architektura:**
```
Host JS (TypeScript)          Wasm (AssemblyScript)
  WasmRunner.loadWeights() →  initWeights() — kopiuje wagi raz
  WasmRunner.forward()     →  runForward()  — każdy krok generacji
```

**AS Float32Array w pamięci Wasm:** obiekt ma 20-bajtowy nagłówek przed danymi:
`[class_id: i32][gc_flags: i32][buffer_ptr: i32][element_count: i32][byteLength: i32]`
Stąd offset `ptr + 20` przy tworzeniu widoku `Float32Array(memory.buffer, ptr + 20, length)`.

**`__pin` / `__unpin`:** AS ma garbage collector — żeby GC nie zebrał tablic podczas gdy JS trzyma wskaźnik, używamy `__pin(ptr)`. Bez tego Wasm GC może zwolnić pamięć wag między wywołaniami.

**Wagi kopiowane raz:** `loadWeights()` alokuje i kopiuje ~500MB do Wasm memory przy starcie. `runForward()` tylko odczytuje — zero kopiowania podczas generacji.

**Dlaczego tokenizer/sampler zostają w JS:** mają dostęp do API przeglądarki (Math.random, TextDecoder, fetch) których Wasm nie ma. Czas wykonania < 1ms vs ~5s forward pass — nie wpływają na benchmark.

## Powiązane

- [[index]]
- [[architecture/assemblyscript-setup]]
- [[architecture/safetensors-format]]
