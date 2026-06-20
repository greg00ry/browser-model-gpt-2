#!/usr/bin/env bash
# Uruchom wszystkie 4 benchmarki i pokaż tabelę wyników.
# Użycie: ./scripts/benchmark_all.sh ["prompt"] [max_tokens]
set -euo pipefail

PROMPT="${1:-The brain is}"
MAX_TOKENS="${2:-20}"
WEIGHTS="weights/gpt2.safetensors"
WASM="build/model.wasm"

sep() { printf '═%.0s' {1..62}; echo; }

sep
echo "  GPT-2 Small (124M) — benchmark porównawczy"
printf '  Prompt: "%s" | %s tokenów | CPU float32\n' "$PROMPT" "$MAX_TOKENS"
sep
echo ""

# ── 1/4 Python / PyTorch ─────────────────────────────────────
echo "[ 1/4 ] Python — PyTorch + HuggingFace (OpenBLAS, fused ops)"
echo ""
PY_RAW=$(python3 scripts/benchmark_python.py "$PROMPT" "$MAX_TOKENS" 2>/dev/null)
echo "$PY_RAW"
PY_TPS=$(echo "$PY_RAW" | grep -oE '[0-9]+\.[0-9]+ tok/s' | grep -oE '[0-9]+\.[0-9]+')
echo ""

# ── 2/4 Python / numpy ────────────────────────────────────────
echo "[ 2/4 ] Python — numpy od zera (BLAS matmul, ręczny transformer)"
echo ""
NP_RAW=$(python3 scripts/benchmark_numpy.py "$WEIGHTS" "$PROMPT" "$MAX_TOKENS" 2>/dev/null)
echo "$NP_RAW"
NP_TPS=$(echo "$NP_RAW" | grep -oE '[0-9]+\.[0-9]+ tok/s' | grep -oE '[0-9]+\.[0-9]+')
echo ""

# ── 3/4 TypeScript / JS ───────────────────────────────────────
echo "[ 3/4 ] TypeScript — V8 JavaScript (Node.js, ręczny transformer)"
echo ""
JS_RAW=$(node dist/main.js "$WEIGHTS" "$WASM" "$PROMPT" js 2>/dev/null)
echo "$JS_RAW" | grep -v '^RESULT_TPS'
JS_TPS=$(echo "$JS_RAW" | grep '^RESULT_TPS' | grep -oE '[0-9]+\.[0-9]+')
echo ""

# ── 4/4 WebAssembly ───────────────────────────────────────────
echo "[ 4/4 ] AssemblyScript — WebAssembly (Node.js, ręczny transformer)"
echo ""
WASM_RAW=$(node dist/main.js "$WEIGHTS" "$WASM" "$PROMPT" wasm 2>/dev/null)
echo "$WASM_RAW" | grep -v '^RESULT_TPS'
WASM_TPS=$(echo "$WASM_RAW" | grep '^RESULT_TPS' | grep -oE '[0-9]+\.[0-9]+')
echo ""

# ── Tabela wyników ────────────────────────────────────────────
sep
echo "  Wyniki końcowe (tok/s — im więcej tym lepiej)"
sep
printf "  %-42s %6s   %s\n" "Backend" "tok/s" "vs numpy"
printf "  %-42s %6s   %s\n" "------" "-----" "--------"

printf "  %-42s %6s   %s\n" "Python / PyTorch (OpenBLAS + fused ops)" "$PY_TPS" \
  "$(LC_ALL=C awk "BEGIN { printf \"%.2f×\", ${PY_TPS}/${NP_TPS} }")"

printf "  %-42s %6s   %s\n" "Python / numpy (od zera)" "$NP_TPS" "1.00×"

printf "  %-42s %6s   %s\n" "TypeScript / V8 JS (od zera)" "$JS_TPS" \
  "$(LC_ALL=C awk "BEGIN { printf \"%.2f×\", ${JS_TPS}/${NP_TPS} }")"

printf "  %-42s %6s   %s\n" "AssemblyScript / WebAssembly (od zera)" "$WASM_TPS" \
  "$(LC_ALL=C awk "BEGIN { printf \"%.2f×\", ${WASM_TPS}/${NP_TPS} }")"

sep
