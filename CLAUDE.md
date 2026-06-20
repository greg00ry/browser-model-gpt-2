# CLAUDE.md — browser-model-gpt-2

## Kontekst projektu

- Język: TypeScript 7, kompilator `tsgo` (`npx tsgo`)
- Cel: inferencja GPT-2 Small (124M) w przeglądarce via WebAssembly
- Wizja długoterminowa: agent jako kompilator kodu maszynowego — przyjmuje język naturalny, produkuje wykonywalny kod
- Wagi modelu: GPT-2 Small `.safetensors` (OpenAI / HuggingFace)
- Środowisko docelowe: Chrome 125+ / Firefox 126+, CPU only, bez GPU

## Vault Obsidian (`brain/`)

Dokumentacja projektu żyje w `brain/`. Struktura:
- `brain/architecture/` — decyzje architektoniczne, wybory technologiczne
- `brain/benchmarks/` — wyniki pomiarów, obserwacje wydajnościowe
- `brain/research/` — notatki badawcze, eksperymenty, pytania otwarte
- `brain/index.md` — mapa projektu / MOC

**Zasada:** po każdej nieoczywistej decyzji lub obserwacji twórz notatkę w odpowiednim podfolderze zgodnie z formatem opisanym w `.claude/commands/brain.md`. Nie czekaj na prośbę użytkownika. Rób to PRZED commitem.

Kiedy pisać notatkę:
- zapada decyzja architektoniczna (format wag, layout pamięci, podział modułów, wybór biblioteki)
- pojawia się nieoczywiste zachowanie (numeryczne, ESM, Wasm edge case)
- użytkownik opisuje wynik benchmarku lub eksperymentu
- coś z rozmowy będzie potrzebne w kolejnych etapach projektu

## Komendy

### /build
Uruchom `npm run build` (`tsgo --project tsconfig.json`):
- po ukończeniu nowego modułu w `src/`
- gdy użytkownik pyta czy kod się kompiluje
- po naprawieniu błędu typów

### /typecheck
Uruchom `npm run typecheck` (`tsgo --noEmit`):
- zamiast `/build` gdy nie ma potrzeby emitowania plików
- po refactorze typów lub interfejsów
- przed każdym commitem jako ostateczna weryfikacja

### /brain
Użytkownik wywołuje `/brain` by zlecić Ci zapisanie notatki do vaultu.
Gdy to nastąpi — utwórz plik wg formatu z `.claude/commands/brain.md`, zaproponuj powiązania z istniejącymi notatkami i potwierdź ścieżkę.
