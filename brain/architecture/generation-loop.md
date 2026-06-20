---
date: 2026-06-20
tags: [generowanie, autoregresja, eos, streaming]
---

# Pętla generowania — autoregresja i streaming

**Autoregresja:** każdy krok forward pass dostaje PEŁNY kontekst `[prompt + generated]`, nie tylko ostatni token. GPT-2 nie ma KV-cache — to pierwsze miejsce do optymalizacji w Etapie 1b.

**EOS token:** GPT-2 nie ma oficjalnego EOS w sensie treningowym — token `50256` (`<|endoftext|>`) jest używany przez konwencję przy fine-tuningu. Przy generacji bazowej model rzadko go emituje — `maxNewTokens` jest głównym warunkiem stopu.

**Streaming przez callback:** `onToken?: (token: string) => void` pozwala pisać tokeny do `process.stdout` w Node.js lub do DOM w przeglądarce bez czekania na koniec generacji. Kluczowe dla UX w przeglądarce.

**Benchmark:** `performance.now()` działa w Node.js i przeglądarce — ta sama metryka w obu środowiskach. Tok/s liczymy od pierwszego tokenu, nie od załadowania wag.

## Powiązane

- [[index]]
- [[architecture/sampling-strategy]]
- [[architecture/weight-tying]]
