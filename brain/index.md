# The Brain — GPT-2 / Browser

## Mapa projektu

- [[architecture/overview]] — architektura modelu i stosu technicznego
- [[research/gpt2-layers]] — warstwy do implementacji
- [[benchmarks/plan]] — plan pomiarów
- [[benchmarks/mobile-browser]] — wynik na telefonie (1.30 tok/s)

## Etapy

- **Etap 1** — GPT-2 (124M) w TypeScript 7 / WebAssembly, benchmark inferencji
- **Etap 1b** *(opcjonalny)* — kwantyzacja wag INT8/INT4
- **Etap 2** — fine-tuning → generowanie żądań HTTP
- **Etap 3** — agent webowy jako kompilator kodu maszynowego
