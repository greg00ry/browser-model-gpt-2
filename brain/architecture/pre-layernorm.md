---
date: 2026-06-20
tags: [layernorm, architektura, gpt2, stabilizacja]
---

# Pre-LayerNorm — GPT-2 normalizuje PRZED blokiem, nie po

GPT-2 używa Pre-LN (normalizacja przed attention i feedforward), nie Post-LN jak w oryginalnym paperze "Attention is All You Need".

```
x = x + attn(ln1(x))   ← ln przed attention
x = x + mlp(ln2(x))    ← ln przed feedforward
```

**Dlaczego to ważne:**  
Post-LN (oryginał) wymaga warmup schedulera — bez niego trening jest niestabilny. Pre-LN jest bardziej stabilne i stało się standardem od GPT-2 wzwyż. Implementacja błędna (Post-LN) da inne wagi residualne i złą perplexity.

**Gdzie:** `block.ts` → `transformerBlock()`.

## Powiązane

- [[index]]
- [[architecture/tensor-shape-convention]]
- [[architecture/gelu-approximation]]
