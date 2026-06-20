---
date: 2026-06-20
tags: [raport, etap2, simd, cross-platform]
---

# Raport Etap 2 — SIMD + Cross-Platform

## Plik

`The_Brain_Raport_Etap2_SIMD_CrossPlatform.docx` — branch `paper`.

## Zakres

Raport obejmuje wyniki po wdrożeniu SIMD f32x4 i uruchomieniu modelu cross-platform:
- Node.js (JS i Wasm)
- Desktop Chrome (native ESM, bez bundlera)
- Mobile browser (telefon, WiFi LAN)

## Kluczowe wyniki

| Backend | tok/s |
|---|---|
| PyTorch (OpenBLAS) | 12.24 |
| numpy (BLAS) | 4.80 |
| Wasm SIMD (Node.js) | 2.01 |
| Wasm SIMD (Chrome) | 2.00 |
| Wasm SIMD (mobile) | 1.30 |
| JS V8 | 0.80 |

## Powiązane

- [[benchmarks/mobile-browser]]
- [[benchmarks/tiling-experiment]]
- [[benchmarks/three-way-comparison]]
