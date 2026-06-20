---
date: 2026-06-20
tags: [assemblyscript, wasm, esm, kompilator]
---

# AssemblyScript — setup i ESM

Wersja: `0.28.19`. Kompilator: `asc`.

**Brak konfliktu ESM:** `assemblyscript` ma już `"type": "module"` w swoim `package.json` — przewidywany problem z propozycji badawczej nie wystąpił w tej wersji.

**Dwie niezależne ścieżki kompilacji:**
```
assembly/*.ts  →  asc  →  build/model.wasm  (model)
src/*.ts       →  tsgo  →  dist/*.js         (host: ładuje wasm, UI, benchmark)
```

`tsc-go` nigdy nie widzi kodu AssemblyScript. Są to całkowicie oddzielne kompilatory.

**Ograniczenia AS vs TypeScript:**
- Brak `Math.random()` — trzeba seedować własny PRNG lub importować z hosta
- Brak klas generycznych
- Brak `any`, `unknown`
- Typy numeryczne są ścisłe: `f32`, `f64`, `i32`, `i64` (nie `number`)
- Brak GC — zarządzanie pamięcią ręczne lub przez `--runtime` (managed/stub)

## Powiązane

- [[index]]
- [[architecture/tensor-shape-convention]]
- [[architecture/float32array-tensor-backend]]
