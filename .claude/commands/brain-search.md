Wyszukaj notatki w vaulcie Obsidian w `brain/`.

Argumenty: $ARGUMENTS — słowo kluczowe, fraza lub tag do wyszukania.

## Jak szukać

1. Znajdź pasujące pliki:

```bash
grep -ril "$ARGUMENTS" brain/ --include="*.md" | grep -v ".obsidian"
```

2. Przeczytaj **pełną treść** każdego trafionego pliku (użyj Read).

3. Jeśli brak wyników — spróbuj rozbić zapytanie na słowa i szukaj każdego osobno.

## Format odpowiedzi

Napisz **jedną długą, spójną odpowiedź** na temat zapytania — tak jakbyś tłumaczył komuś pełny stan projektu w tym obszarze. Nie listuj plików jeden po drugim. Syntetyzuj wszystkie notatki w ciągłą narrację.

Struktura odpowiedzi:
1. **Co wiemy** — aktualny stan wiedzy, kluczowe liczby i wyniki
2. **Jak do tego doszliśmy** — historia decyzji i odkryć, łącznie z negatywnymi wynikami
3. **Co jest otwarte** — niezbadane pytania, potencjalne kierunki

Odpowiedź powinna być na tyle pełna, żeby osoba czytająca ją po raz pierwszy rozumiała cały kontekst bez zaglądania do notatek.

## Zasady

- Szukaj case-insensitive (`-i`)
- Jeśli $ARGUMENTS zawiera `#tag` — szukaj po tagach w frontmatter
- Czytaj pełne pliki, nie urywki — kontekst jest ważny
- Używaj konkretnych liczb z notatek
- Nie wymieniaj nazw plików w odpowiedzi — pisz o wiedzy, nie o dokumentach
