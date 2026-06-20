# Skill: /report

Użytkownik wywołuje `/report` żeby zaktualizować raport badawczy projektu "The Brain".

## Co robisz

1. Przeczytaj aktualne dane benchmarków z `brain/benchmarks/` (szczególnie `three-way-comparison.md` i `wasm-vs-js-baseline.md`).
2. Sprawdź ostatnie commity na `main` żeby wychwycić nowe wyniki lub optymalizacje.
3. Wygeneruj gotową sekcję raportu w formacie Markdown — tabelę wyników, analizę, wnioski — którą użytkownik może wkleić do swojego dokumentu.

## Format wyjścia

Zawsze generuj:

### Sekcja: Wyniki benchmarku

Tabela 4-way (Python PyTorch / Python numpy / JS / Wasm) z kolumnami:
- Backend
- tok/s
- vs numpy (uczciwy baseline)
- Stack

### Sekcja: Analiza

Krótki akapit wyjaśniający:
- dlaczego numpy jest uczciwym baseline (BLAS matmul, ręczny transformer)
- dlaczego PyTorch jest szybszy (fused ops, cache-blocking)
- dlaczego JS i Wasm są wolniejsze (naiwny triple-loop bez SIMD)

### Sekcja: Historia optymalizacji Wasm

Tabela postępu: naiwna wersja → scratch pre-alokacja → lm_head only last token.

### Sekcja: Wnioski

Bullet lista — co udało się pokazać, co jest następnym krokiem.

## Zasady

- Używaj danych z `brain/` — nie wymyślaj liczb.
- Jeśli użytkownik przekaże nowe wyniki w wiadomości, uwzględnij je w raporcie.
- Piszesz po polsku, styl akademicki ale zwięzły.
- Nie twórz pliku — tylko wypisz treść żeby użytkownik mógł ją skopiować.
