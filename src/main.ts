import { forward, GPT2Weights, GPT2Config, GPT2_SMALL } from "./model.js";
import { loadSafetensors, loadFromFile } from "./loader.js";
import { sample, greedy, lastTokenLogits } from "./sampler.js";
import { WasmRunner } from "./wasm-runner.js";

export interface Tokenizer {
  encode(text: string): number[];
  decode(tokens: number[]): string;
}

export interface GenerateOptions {
  maxNewTokens?: number;   // default 100
  temperature?: number;    // default 1.0
  topK?: number;           // default 40
  greedy?: boolean;        // default false — użyj greedy zamiast sampling
  onToken?: (token: string) => void; // streaming callback
}

export async function generate(
  prompt: string,
  tokenizer: Tokenizer,
  weights: GPT2Weights,
  config: GPT2Config = GPT2_SMALL,
  opts: GenerateOptions = {},
): Promise<string> {
  const {
    maxNewTokens = 100,
    temperature = 1.0,
    topK = 40,
    greedy: useGreedy = false,
    onToken,
  } = opts;

  const tokens = tokenizer.encode(prompt);

  if (tokens.length > config.nCtx) {
    throw new Error(`Prompt too long: ${tokens.length} tokens, max ${config.nCtx}`);
  }

  const generated: number[] = [];

  for (let i = 0; i < maxNewTokens; i++) {
    const input = [...tokens, ...generated];
    const logits = forward(input, weights, config);
    const nextLogits = lastTokenLogits(logits);

    const nextToken = useGreedy
      ? greedy(nextLogits)
      : sample(nextLogits, { temperature, topK });

    generated.push(nextToken);
    onToken?.(tokenizer.decode([nextToken]));

    // GPT-2 nie ma oficjalnego EOS — 50256 (<|endoftext|>) używany jako konwencja
    if (nextToken === 50256) break;
  }

  return tokenizer.decode(generated);
}

// --- Node.js entry point (testy i benchmark) ---

async function main() {
  const args        = process.argv.slice(2);
  const weightsPath = args[0] ?? "weights/gpt2.safetensors";
  const wasmPath    = args[1] ?? "build/model.wasm";
  const prompt      = args[2] ?? "The brain is";
  const mode        = args[3] ?? "all"; // all | js | wasm

  console.log(`Loading weights from ${weightsPath}...`);
  const t0 = performance.now();
  const weights = await loadFromFile(weightsPath, GPT2_SMALL);
  console.log(`Loaded in ${((performance.now() - t0) / 1000).toFixed(2)}s\n`);

  const tokenizer = await loadTokenizer();

  let jsResult:   { tokens: number; elapsed: number; tps: number } | null = null;
  let wasmResult: { tokens: number; elapsed: number; tps: number } | null = null;

  // ── benchmark JS ──────────────────────────────────────────────────
  if (mode === "all" || mode === "js") {
    console.log(`[ JS ] Prompt: "${prompt}"`);
    jsResult = await runBenchmark(prompt, tokenizer, weights, (input, w) =>
      forward(input, w, GPT2_SMALL),
    );
    console.log(`[ JS ] ${jsResult.tokens} tokens in ${jsResult.elapsed.toFixed(2)}s = ${jsResult.tps.toFixed(2)} tok/s\n`);
  }

  // ── benchmark Wasm ────────────────────────────────────────────────
  if (mode === "all" || mode === "wasm") {
    console.log(`[Wasm] Loading model.wasm...`);
    const tw = performance.now();
    const runner = await WasmRunner.load(wasmPath, weights, GPT2_SMALL);
    console.log(`[Wasm] Wasm init in ${((performance.now() - tw) / 1000).toFixed(2)}s`);
    console.log(`[Wasm] Prompt: "${prompt}"`);
    wasmResult = await runBenchmark(prompt, tokenizer, weights, (input) =>
      runner.forward(input),
    );
    console.log(`[Wasm] ${wasmResult.tokens} tokens in ${wasmResult.elapsed.toFixed(2)}s = ${wasmResult.tps.toFixed(2)} tok/s\n`);
  }

  if (jsResult && wasmResult) {
    console.log(`Speedup Wasm vs JS: ${(wasmResult.tps / jsResult.tps).toFixed(2)}x`);
  }

  // wypisz tok/s jako ostatnią linię dla parsowania przez skrypt
  const tps = (jsResult ?? wasmResult)!.tps;
  process.stdout.write(`RESULT_TPS=${tps.toFixed(2)}\n`);
}

async function runBenchmark(
  prompt: string,
  tokenizer: Tokenizer,
  weights: GPT2Weights,
  forwardFn: (tokens: number[], w: GPT2Weights) => ReturnType<typeof forward>,
) {
  const MAX_TOKENS = 20;
  const tokens = tokenizer.encode(prompt);
  const generated: number[] = [];
  let tokenCount = 0;
  process.stdout.write("  output: ");

  const t0 = performance.now();
  for (let i = 0; i < MAX_TOKENS; i++) {
    const input = [...tokens, ...generated];
    const logits = forwardFn(input, weights);
    const nextToken = sample(lastTokenLogits(logits), { temperature: 1.0, topK: 40 });
    generated.push(nextToken);
    process.stdout.write(tokenizer.decode([nextToken]));
    tokenCount++;
    if (nextToken === 50256) break;
  }
  const elapsed = (performance.now() - t0) / 1000;
  process.stdout.write("\n");

  return { tokens: tokenCount, elapsed, tps: tokenCount / elapsed };
}

async function loadTokenizer(): Promise<Tokenizer> {
  const { tokenizer } = await import("./tokenizer.js");
  return tokenizer;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
