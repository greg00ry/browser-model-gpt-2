import { createWriteStream, existsSync, mkdirSync } from "fs";
import { pipeline } from "stream/promises";

const URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors";
const OUT  = "weights/gpt2.safetensors";
const EXPECTED_MB = 548;

if (existsSync(OUT)) {
  console.log(`Already exists: ${OUT}`);
  process.exit(0);
}

mkdirSync("weights", { recursive: true });

console.log(`Downloading GPT-2 Small weights (~${EXPECTED_MB} MB)...`);
console.log(`Source: ${URL}\n`);

const res = await fetch(URL);
if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

const total = Number(res.headers.get("content-length") ?? 0);
let downloaded = 0;
let lastPrint = 0;

const reader = res.body.getReader();
const writer = createWriteStream(OUT);

const t0 = Date.now();

await pipeline(
  (async function* () {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      downloaded += value.byteLength;

      const now = Date.now();
      if (now - lastPrint > 500 || downloaded === total) {
        const pct = total ? ((downloaded / total) * 100).toFixed(1) : "?";
        const mb  = (downloaded / 1024 / 1024).toFixed(1);
        const speed = (downloaded / 1024 / 1024 / ((now - t0) / 1000)).toFixed(1);
        process.stdout.write(`\r${pct}%  ${mb} MB  ${speed} MB/s  `);
        lastPrint = now;
      }

      yield Buffer.from(value);
    }
  })(),
  writer,
);

const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
console.log(`\n\nSaved to ${OUT} in ${elapsed}s`);
