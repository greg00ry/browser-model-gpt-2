Zbadaj dogłębnie temat w internecie — odpowiednik `/brain-deep` ale zamiast vault przeszukuje sieć.

Argumenty: $ARGUMENTS — pytanie lub temat do zbadania (np. "czy tiling matmul pomaga na x86 w Wasm?", "WebNN API vs Wasm SIMD wydajność")

## Jak badać

1. Sformułuj 2-3 konkretne zapytania wyszukiwania na podstawie $ARGUMENTS. Szukaj precyzyjnie — techniczne terminy, numery wersji, nazwy API jeśli znane.

2. Wykonaj WebSearch dla każdego zapytania.

3. Z wyników wyszukiwania wybierz 3-5 najbardziej obiecujących źródeł (dokumentacja, artykuły techniczne, benchmarki, dyskusje). Użyj WebFetch żeby przeczytać ich treść.

4. Jeśli znalezione informacje odsyłają do czegoś konkretnego (spec, paper, issue tracker) — idź tam i przeczytaj.

## Format odpowiedzi

Napisz wyczerpującą odpowiedź na zadane pytanie na podstawie znalezionych źródeł. Struktura:

1. **Bezpośrednia odpowiedź** — co mówi internet na ten temat, konkrety i liczby
2. **Kontekst i mechanizmy** — dlaczego tak jest, jak to działa
3. **Co jest aktualne** — daty, wersje, stan wsparcia w przeglądarkach/środowiskach
4. **Co to oznacza dla projektu** — jak znalezione informacje odnoszą się do tego co robimy w browser-model-gpt-2

## Zasady

- Podawaj źródła przy kluczowych twierdzeniach (nazwa strony + URL)
- Jeśli źródła są sprzeczne — wskaż to i oceń które jest bardziej wiarygodne
- Jeśli czegoś nie ma w internecie lub wyniki są słabe — powiedz wprost
- Skup się na tym co użyteczne dla projektu, nie na encyklopedycznym przeglądzie
