# Skill: /report

Użytkownik wywołuje `/report` żeby dodać nowy plik raportu do brancha `paper`.

## Kroki — wykonaj je po kolei

### 1. Przełącz na branch `paper`
```bash
git checkout paper
```
Potwierdź użytkownikowi że jesteś na `paper`.

### 2. Poproś o plik
Napisz do użytkownika:
> Jestem na branchu `paper`. Wklej treść pliku raportu — powiedz też jak ma się nazywać.

Czekaj. Nie rób nic więcej dopóki użytkownik nie wklei treści.

### 3. Zapisz plik
Gdy użytkownik wklei treść i poda nazwę pliku — zapisz plik w katalogu głównym projektu (obok istniejących `.docx`).

### 4. Dodaj notatkę do brain
Utwórz plik `brain/research/report-<slug>.md` wg formatu:
```
---
date: <dzisiaj>
tags: [raport, paper, etap-1]
---

# <tytuł raportu>

Dodano plik: `<nazwa pliku>`
Branch: `paper`

<krótkie 2-3 zdania streszczenie tego co raport zawiera — na podstawie wklejonej treści>

## Powiązane
- [[benchmarks/three-way-comparison]]
- [[benchmarks/wasm-vs-js-baseline]]
```

### 5. Commit i push na `paper`
```bash
git add <plik raportu> brain/research/report-<slug>.md
git commit -m "Add report: <tytuł>"
git push origin paper
```

### 6. Wróć na `main`
```bash
git checkout main
```
Potwierdź użytkownikowi że wróciłeś na `main`.
