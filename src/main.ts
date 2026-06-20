import { forward, GPT2Weights, GPT2Config, GPT2_SMALL } from "./model.js";
import { loadSafetensors, loadFromFile } from "./loader.js";
import { sample, greedy, lastTokenLogits } from "./sampler.js";

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
  const weightsPath = process.argv[2] ?? "weights/gpt2.safetensors";
  const prompt = process.argv[3] ?? "The brain is";

  console.log(`Loading weights from ${weightsPath}...`);
  const t0 = performance.now();
  const weights = await loadFromFile(weightsPath, GPT2_SMALL);
  console.log(`Loaded in ${((performance.now() - t0) / 1000).toFixed(2)}s`);

  // placeholder — zastąp prawdziwym tiktoken w tokenizer.ts
  const tokenizer = await loadTokenizer();

  console.log(`\nPrompt: "${prompt}"\n`);

  let tokenCount = 0;
  const t1 = performance.now();

  const output = await generate(prompt, tokenizer, weights, GPT2_SMALL, {
    maxNewTokens: 50,
    temperature: 1.0,
    topK: 40,
    onToken: (t) => {
      process.stdout.write(t);
      tokenCount++;
    },
  });

  const elapsed = (performance.now() - t1) / 1000;
  console.log(`\n\n--- ${tokenCount} tokens in ${elapsed.toFixed(2)}s = ${(tokenCount / elapsed).toFixed(1)} tok/s`);
}

async function loadTokenizer(): Promise<Tokenizer> {
  // TODO: zastąp tiktoken gdy tokenizer.ts będzie gotowy
  // import { getEncoding } from "tiktoken";
  // const enc = getEncoding("gpt2");
  // return { encode: (t) => Array.from(enc.encode(t)), decode: (ids) => enc.decode(new Uint32Array(ids)) };
  throw new Error("Tokenizer not implemented yet — see tokenizer.ts");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
