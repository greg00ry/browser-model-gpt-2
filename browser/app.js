import { loadFromUrl }    from "../dist/loader.js";
import { GPT2_SMALL }     from "../dist/model.js";
import { WasmRunner }     from "../dist/wasm-runner.js";
import { sample, lastTokenLogits } from "../dist/sampler.js";
import { encode, decode } from "./tok.js";

const $ = (id) => document.getElementById(id);

let runner = null;

async function init() {
  const status = $("status");
  const btn    = $("btn");
  btn.disabled = true;

  status.textContent = "Ładowanie wag (548 MB)…";
  const weights = await loadFromUrl("../weights/gpt2.safetensors", GPT2_SMALL, (pct) => {
    status.textContent = `Ładowanie wag… ${(pct * 100).toFixed(0)}%`;
  });

  status.textContent = "Ładowanie modelu Wasm…";
  runner = await WasmRunner.loadUrl("../build/model.wasm", weights, GPT2_SMALL);

  status.textContent = "Gotowy.";
  btn.disabled = false;
}

async function generate() {
  if (!runner) return;
  const prompt  = $("prompt").value.trim() || "The brain is";
  const output  = $("output");
  const tpsEl   = $("tps");
  const btn     = $("btn");

  output.textContent = "";
  tpsEl.textContent  = "";
  btn.disabled = true;

  const tokens    = encode(prompt);
  const generated = [];
  const t0 = performance.now();

  for (let i = 0; i < 20; i++) {
    const input  = [...tokens, ...generated];
    const logits = runner.forward(input);
    const next   = sample(lastTokenLogits(logits), { temperature: 1.0, topK: 40 });
    generated.push(next);
    output.textContent += decode([next]);
    if (next === 50256) break;
    await new Promise(r => setTimeout(r, 0));
  }

  const elapsed = (performance.now() - t0) / 1000;
  tpsEl.textContent = `${(generated.length / elapsed).toFixed(2)} tok/s`;
  btn.disabled = false;
}

$("btn").addEventListener("click", generate);
init();
