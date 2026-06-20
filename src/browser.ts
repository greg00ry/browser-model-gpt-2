import { loadFromUrl } from "./loader.js";
import { GPT2_SMALL } from "./model.js";
import { WasmRunner } from "./wasm-runner.js";
import { sample, lastTokenLogits } from "./sampler.js";
import { tokenizer } from "./tokenizer.js";

const $ = (id: string) => document.getElementById(id)!;

let runner: WasmRunner | null = null;

async function init() {
  const status = $("status");
  const btn    = $("btn") as HTMLButtonElement;
  btn.disabled = true;

  status.textContent = "Ładowanie wag (548 MB)…";
  const weights = await loadFromUrl("weights/gpt2.safetensors", GPT2_SMALL, (pct) => {
    status.textContent = `Ładowanie wag… ${(pct * 100).toFixed(0)}%`;
  });

  status.textContent = "Ładowanie modelu Wasm…";
  runner = await WasmRunner.loadUrl("build/model.wasm", weights, GPT2_SMALL);

  status.textContent = "Gotowy.";
  btn.disabled = false;
}

async function generate() {
  if (!runner) return;
  const prompt  = ($("prompt") as HTMLInputElement).value.trim() || "The brain is";
  const output  = $("output");
  const tpsEl   = $("tps");
  const btn     = $("btn") as HTMLButtonElement;

  output.textContent = "";
  tpsEl.textContent  = "";
  btn.disabled = true;

  const tokens    = tokenizer.encode(prompt);
  const generated: number[] = [];
  const t0 = performance.now();

  for (let i = 0; i < 20; i++) {
    const input  = [...tokens, ...generated];
    const logits = runner.forward(input);
    const next   = sample(lastTokenLogits(logits), { temperature: 1.0, topK: 40 });
    generated.push(next);
    output.textContent += tokenizer.decode([next]);
    if (next === 50256) break;
    await new Promise(r => setTimeout(r, 0)); // yield do UI
  }

  const elapsed = (performance.now() - t0) / 1000;
  tpsEl.textContent = `${(generated.length / elapsed).toFixed(2)} tok/s`;
  btn.disabled = false;
}

$("btn").addEventListener("click", generate);
init();
