---
date: 2026-06-20
tags: [benchmark, mobile, browser, wasm, ios]
---

# GPT-2 Wasm — wynik na telefonie (mobile browser)

## Konfiguracja

- Urządzenie: telefon (iOS/Android, dokładny model nieznany)
- Sieć: WiFi LAN, serwer `python3 http.server --bind 0.0.0.0` na komputerze
- URL: `http://192.168.0.143:8080/browser/`
- Wagi: pobrane przez telefon (~548 MB)

## Wynik

| Backend | tok/s |
|---|---|
| Wasm (desktop Node.js) | 2.01 |
| Wasm (desktop Chrome) | 2.00 |
| **Wasm (mobile browser)** | **1.30** |

## Obserwacje

- 1.30 tok/s vs 2.00 tok/s desktop Chrome — **35% wolniej**
- WASM SIMD (`f32x4`) działa na telefonie (iOS 16.4+ / Chrome Android 91+)
- Różnica prawdopodobnie wynika z: wolniejszego CPU, mniejszej pamięci bandwidth, bardziej agresywnego throttlingu przeglądarki mobilnej

## Wniosek

Model jest używalny na telefonie — 1.30 tok/s to ok. 1 token na sekundę, co dla krótkiej generacji (20 tokenów ≈ 15s) jest akceptowalne jako demo.

## Powiązane

- [[benchmarks/three-way-comparison]]
- [[benchmarks/tiling-experiment]]
