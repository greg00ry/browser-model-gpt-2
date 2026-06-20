---
date: 2026-06-20
tags: [gelu, aktivacja, numeryczne, gpt2]
---

# GELU — aproksymacja tanh, nie erf

GPT-2 używa aproksymacji GELU przez `tanh`, nie dokładnej wersji przez `erf`:

```
0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Dlaczego to ważne:**  
Jeśli użyjesz `erf` (dokładna wersja z ES2026 lub math lib) zamiast tej aproksymacji, ΔPerplexity wzrośnie powyżej progu 0.5 nawet przy poprawnej implementacji reszty modelu. HuggingFace GPT-2 używa dokładnie tej aproksymacji.

**Gdzie zaimplementowane:** `tensor.ts` → `gelu()`.

## Powiązane

- [[index]]
- [[architecture/float32array-tensor-backend]]
