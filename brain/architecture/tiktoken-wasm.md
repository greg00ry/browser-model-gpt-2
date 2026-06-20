---
date: 2026-06-20
tags: [tokenizer, tiktoken, wasm, bpe, api]
---

# Tiktoken — API i pułapki

Poprawna nazwa eksportu to `get_encoding` (snake_case), nie `getEncoding`. Pomyłka daje błąd w runtime, nie w kompilacji jeśli typy nie są ścisłe.

**Lazy init:** WASM ładuje się przy pierwszym wywołaniu `enc()`. W przeglądarce może to zająć chwilę — warto wywołać `encode("")` przy starcie aplikacji żeby WASM był gotowy zanim użytkownik wpisze prompt.

**`free()`:** tiktoken alokuje pamięć w WASM — należy wywołać `free()` przy zamknięciu aplikacji żeby uniknąć wycieku. W przeglądarce przy zamknięciu zakładki przeglądarka to posprzata, ale w Node.js testery mogą zgłaszać wycieki.

**`decode` zwraca `Uint8Array`**, nie string — trzeba przepuścić przez `TextDecoder`. GPT-2 tokenizuje bajty UTF-8, więc pojedynczy token może być fragmentem znaku wielobajtowego — `TextDecoder` radzi sobie z tym poprawnie.

## Powiązane

- [[index]]
- [[architecture/generation-loop]]
