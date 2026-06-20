# CLAUDE.md — browser-model-gpt-2

## Kiedy używać skilli

### /build
Uruchom po każdej zmianie w `src/`, gdy:
- użytkownik prosi o kompilację lub sprawdzenie czy kod działa
- kończysz implementację nowego komponentu modelu
- naprawiasz błąd typów i chcesz potwierdzić że build przechodzi

### /typecheck
Uruchom zamiast `/build` gdy:
- użytkownik pyta o poprawność typów bez potrzeby emitowania plików
- zrobiłeś refactor interfejsów lub typów i chcesz to zweryfikować przed commitem

### /brain
Uruchom **proaktywnie** — bez czekania na prośbę użytkownika — gdy:
- zapada decyzja architektoniczna (wybór biblioteki, format wag, target Wasm, struktura modułów)
- pojawia się nieoczywista obserwacja podczas implementacji (zachowanie numeryczne, problem z ESM, edge case w attention)
- użytkownik opisuje wyniki benchmarku lub eksperymentu
- dyskusja ujawnia coś wartego zapamiętania na potrzeby kolejnych etapów projektu

## Kontekst projektu

- Język: TypeScript 7, kompilator `tsgo` (`npx tsgo`)
- Cel: inferencja GPT-2 Small (124M) w przeglądarce via WebAssembly
- Wizja długoterminowa: agent jako kompilator kodu maszynowego
- Dokumentacja: vault Obsidian w `brain/`, indeks w `brain/index.md`
- Wagi modelu: GPT-2 Small `.safetensors` (OpenAI / HuggingFace)
- Środowisko docelowe: Chrome 125+ / Firefox 126+, CPU only, bez GPU
