import { layerNorm, matmul2d } from "./tensor";
import { transformerBlock } from "./block";
import { getScratch } from "./scratch";

const N_LAYERS:   i32 = 12;
const N_HEADS:    i32 = 12;
const N_EMBD:     i32 = 768;
const VOCAB_SIZE: i32 = 50257;

export function forward(
  tokens: Int32Array,
  wte: Float32Array,
  wpe: Float32Array,
  ln1Weights: Float32Array[], ln1Biases: Float32Array[],
  ln2Weights: Float32Array[], ln2Biases: Float32Array[],
  cAttnWeights: Float32Array[], cAttnBiases: Float32Array[],
  cProjWeights: Float32Array[], cProjBiases: Float32Array[],
  cFcWeights: Float32Array[],   cFcBiases: Float32Array[],
  cFcProjWeights: Float32Array[], cFcProjBiases: Float32Array[],
  lnFWeight: Float32Array,
  lnFBias: Float32Array,
  out: Float32Array,
): void {
  const s   = getScratch();
  const seq: i32 = tokens.length;

  // token + position embeddings -> scratch.x [seq, nEmbd]
  for (let i: i32 = 0; i < seq; i++) {
    const tokenOffset: i32 = unchecked(tokens[i]) * N_EMBD;
    const posOffset:   i32 = i * N_EMBD;
    for (let d: i32 = 0; d < N_EMBD; d++) {
      unchecked(s.x[i * N_EMBD + d] = wte[tokenOffset + d] + wpe[posOffset + d]);
    }
  }

  for (let i: i32 = 0; i < N_LAYERS; i++) {
    transformerBlock(
      s.x, seq, N_EMBD,
      unchecked(ln1Weights[i]),    unchecked(ln1Biases[i]),
      unchecked(ln2Weights[i]),    unchecked(ln2Biases[i]),
      unchecked(cAttnWeights[i]),  unchecked(cAttnBiases[i]),
      unchecked(cProjWeights[i]),  unchecked(cProjBiases[i]),
      unchecked(cFcWeights[i]),    unchecked(cFcBiases[i]),
      unchecked(cFcProjWeights[i]), unchecked(cFcProjBiases[i]),
      N_HEADS,
    );
  }

  // final layer norm (in-place na scratch.x)
  layerNorm(s.x, seq, N_EMBD, lnFWeight, lnFBias, s.x);

  // lm_head: tylko ostatni token [1, nEmbd] x wteT -> out [vocabSize]
  // (sampling używa tylko ostatniego tokenu)
  const lastOff: i32 = (seq - 1) * N_EMBD;
  memory.fill(out.dataStart, 0, VOCAB_SIZE << 2);
  for (let d: i32 = 0; d < N_EMBD; d++) {
    const xVal = unchecked(s.x[lastOff + d]);
    for (let v: i32 = 0; v < VOCAB_SIZE; v++) {
      unchecked(out[v] += xVal * s.wteT[d * VOCAB_SIZE + v]);
    }
  }
}

// wywołaj raz po załadowaniu wag — wypełnia scratch.wteT transponowanym wte
export function precomputeWteT(wte: Float32Array): void {
  const s = getScratch();
  for (let v: i32 = 0; v < VOCAB_SIZE; v++) {
    for (let d: i32 = 0; d < N_EMBD; d++) {
      unchecked(s.wteT[d * VOCAB_SIZE + v] = wte[v * N_EMBD + d]);
    }
  }
}
