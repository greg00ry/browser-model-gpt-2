Zbadaj dogłębnie konkretne pytanie lub element wynikający z wcześniejszego `/brain-search`.

Argumenty: $ARGUMENTS — pytanie lub temat do zbadania (np. "czy tiling pomógłby na x86?", "jak działa pre-alokacja scratch buforów")

## Jak badać

1. Znajdź notatki bezpośrednio powiązane z tematem:
```bash
grep -ril "$ARGUMENTS" brain/ --include="*.md" | grep -v ".obsidian"
```

2. Przeczytaj pełną treść trafionych notatek (użyj Read).

3. Ze znalezionych notatek wyciągnij wszystkie wikilinki (`[[...]]`) i przeczytaj też te pliki — nawet jeśli nie trafiły w grep. Wikilinki to ścieżki kontekstowe do powiązanej wiedzy.

4. Przeszukaj kod źródłowy pod kątem implementacji związanej z tematem:
```bash
grep -ril "$ARGUMENTS" src/ --include="*.ts" 2>/dev/null
grep -ril "$ARGUMENTS" assembly/ --include="*.ts" 2>/dev/null
```
Jeśli coś znajdziesz — przeczytaj te pliki i skonfrontuj implementację z tym co mówią notatki.

5. Jeśli pytanie dotyczy liczb lub wyników — sprawdź czy są spójne między notatkami.

## Format odpowiedzi

Napisz wyczerpującą odpowiedź na zadane pytanie. Nie streszczaj notatek — odpowiedz na pytanie korzystając z nich jako źródła. Struktura:

1. **Bezpośrednia odpowiedź** — co wiemy na ten temat, konkrety
2. **Głębszy kontekst** — dlaczego tak jest, mechanizmy, przyczyny
3. **Co kod mówi** — jeśli znalazłeś implementację, skonfrontuj z wiedzą z vault
4. **Luki i niezbadane** — czego nie wiemy, co byłoby warte sprawdzenia

## Zasady

- Odpowiadaj na pytanie, nie opisuj dokumentów
- Używaj konkretnych liczb i nazw z notatek i kodu
- Jeśli notatki są sprzeczne — wskaż to wprost
- Jeśli pytanie wykracza poza to co jest w vault i kodzie — powiedz to wprost zamiast spekulować
