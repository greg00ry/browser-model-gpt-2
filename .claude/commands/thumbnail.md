Wygeneruj miniaturkę YouTube dla filmu na podstawie tytułu i opisu.

## Krok 1 — zbierz dane

Zanim cokolwiek wygenerujesz, zapytaj użytkownika o:

1. **Tytuł filmu** — główny tekst na miniaturce (krótki, max 5-6 słów)
2. **Kluczowe liczby / wyniki** — max 4 wiersze do tabeli benchmarków (np. "Python: 12.24 tok/s", "Wasm SIMD: 2.00 tok/s")
3. **Podtytuł** — 1-2 linijki opisu pod tytułem (np. "TypeScript + WebAssembly", "pełne benchmarki vs Python")

Jeśli użytkownik poda za mało — dopytaj. Nie generuj z domysłów.

## Krok 2 — wygeneruj hook

Na podstawie tytułu, podtytułu i wyników **sam wygeneruj hook** — krótkie, przykuwające uwagę zdanie które pojawi się jako badge na miniaturce. Hook powinien:
- Być zaskakujący lub kontraintuicyjny (np. "Wasm wolniejszy od JS?")
- Max 5-6 słów + strzałka →
- Odwoływać się do największego twista lub zaskakującego wyniku z danych

Zaproponuj użytkownikowi 2-3 opcje hooka i zapytaj który wybrać lub czy coś zmienić.

## Krok 3 — wygeneruj skrypt

Na podstawie zebranych danych napisz skrypt Python używający Pillow i zapisz go jako `youtube/generate-thumbnail.py` (nadpisz istniejący).

Styl wizualny (zachowaj spójny z projektem):
- Tło: ciemne `(10, 12, 18)` z gradientem
- Lewy pasek akcentowy: `(0, 200, 120)`
- Tytuł: biały, duży (font ~90-110pt), lewa strona górnej połowy
- Podtytuł: szary `(140, 150, 165)`
- Hook badge: czerwone tło `(40, 20, 20)`, obramowanie `(255, 80, 80)`, czerwony tekst
- Prawa strona: ciemny panel z tabelą wyników — zielony dla najlepszego wyniku, czerwony dla najgorszego, szary dla reszty
- Dolny pasek: `(14, 16, 24)` z "The Brain Project" po lewej
- Rozmiar: 1280×720px, zapis do `youtube/thumbnail.png`

## Krok 4 — uruchom i pokaż

Uruchom skrypt:
```bash
python3 youtube/generate-thumbnail.py
```

Wyświetl wygenerowany plik `youtube/thumbnail.png` używając Read.

Zapytaj czy coś zmienić i iteruj jeśli trzeba.
