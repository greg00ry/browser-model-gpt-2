import { Tensor } from "./tensor.js";

export interface SamplerOptions {
  temperature?: number; // default 1.0 — <1 bardziej deterministyczny, >1 bardziej losowy
  topK?: number;        // default 0 (wyłączone) — ogranicz do K najbardziej prawdopodobnych
}

// logits [vocabSize] -> następny token
export function sample(logits: Tensor, opts: SamplerOptions = {}): number {
  const { temperature = 1.0, topK = 0 } = opts;

  if (temperature === 0) {
    return argmax(logits.data);
  }

  let scores = applyTemperature(logits.data, temperature);

  if (topK > 0) {
    scores = applyTopK(scores, topK);
  }

  return multinomial(softmaxRaw(scores));
}

// greedy — zawsze najwyższy logit, bez losowości
export function greedy(logits: Tensor): number {
  return argmax(logits.data);
}

// wyciągnij logity ostatniego tokenu z [seq, vocabSize] -> [vocabSize]
export function lastTokenLogits(logits: Tensor): Tensor {
  const vocabSize = logits.shape[1];
  const seq = logits.shape[0];
  const data = logits.data.slice((seq - 1) * vocabSize, seq * vocabSize);
  return new Tensor(data, [vocabSize]);
}

function argmax(data: Float32Array): number {
  let best = 0;
  for (let i = 1; i < data.length; i++) {
    if (data[i] > data[best]) best = i;
  }
  return best;
}

function applyTemperature(data: Float32Array, temperature: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = data[i] / temperature;
  return out;
}

function applyTopK(data: Float32Array, k: number): Float32Array {
  // znajdź próg k-tego największego logitu i wyzeruj resztę (-Infinity)
  const sorted = Array.from(data).sort((a, b) => b - a);
  const threshold = sorted[Math.min(k, sorted.length) - 1];
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] = data[i] >= threshold ? data[i] : -Infinity;
  }
  return out;
}

function softmaxRaw(data: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
  const out = new Float32Array(data.length);
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    out[i] = Math.exp(data[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < data.length; i++) out[i] /= sum;
  return out;
}

function multinomial(probs: Float32Array): number {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1;
}
