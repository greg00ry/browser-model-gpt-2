import { layerNorm, matmul2d, allocF32 } from "./tensor";
import { transformerBlock } from "./block";

// config GPT-2 Small — stałe żeby nie przekazywać ich przez parametry
const N_LAYERS:    i32 = 12;
const N_HEADS:     i32 = 12;
const N_EMBD:      i32 = 768;
const VOCAB_SIZE:  i32 = 50257;

// tokens [seq] -> logits [seq, vocabSize]
// wagi bloków przekazywane jako płaskie tablice w ustalonej kolejności (patrz niżej)
export function forward(
  tokens: Int32Array,
  wte: Float32Array,        // [vocabSize, nEmbd]
  wpe: Float32Array,        // [nCtx, nEmbd]
  // bloki: po 14 tablic każdy (ln1w, ln1b, ln2w, ln2b, cAttnW, cAttnB, cProjW, cProjB, cFcW, cFcB, cFcProjW, cFcProjB)
  ln1Weights: Float32Array[], ln1Biases: Float32Array[],
  ln2Weights: Float32Array[], ln2Biases: Float32Array[],
  cAttnWeights: Float32Array[], cAttnBiases: Float32Array[],
  cProjWeights: Float32Array[], cProjBiases: Float32Array[],
  cFcWeights: Float32Array[],   cFcBiases: Float32Array[],
  cFcProjWeights: Float32Array[], cFcProjBiases: Float32Array[],
  lnFWeight: Float32Array,  // [nEmbd]
  lnFBias: Float32Array,    // [nEmbd]
  out: Float32Array,        // [seq, vocabSize]
): void {
  const seq: i32 = tokens.length;

  // token + position embeddings -> x [seq, nEmbd]
  const x = tokenAndPositionEmbeddings(tokens, wte, wpe, seq);

  // 12x transformer blocks (in-place na x)
  for (let i: i32 = 0; i < N_LAYERS; i++) {
    transformerBlock(
      x, seq, N_EMBD,
      unchecked(ln1Weights[i]),    unchecked(ln1Biases[i]),
      unchecked(ln2Weights[i]),    unchecked(ln2Biases[i]),
      unchecked(cAttnWeights[i]),  unchecked(cAttnBiases[i]),
      unchecked(cProjWeights[i]),  unchecked(cProjBiases[i]),
      unchecked(cFcWeights[i]),    unchecked(cFcBiases[i]),
      unchecked(cFcProjWeights[i]), unchecked(cFcProjBiases[i]),
      N_HEADS,
    );
  }

  // final layer norm (in-place)
  layerNorm(x, seq, N_EMBD, lnFWeight, lnFBias, x);

  // lm_head: x [seq, nEmbd] x wte^T [nEmbd, vocabSize] -> out [seq, vocabSize]
  const wteT = transposeWeight(wte, VOCAB_SIZE, N_EMBD);
  matmul2d(x, seq, N_EMBD, wteT, VOCAB_SIZE, out);
}

// token embedding + positional embedding -> [seq, nEmbd]
function tokenAndPositionEmbeddings(
  tokens: Int32Array,
  wte: Float32Array,
  wpe: Float32Array,
  seq: i32,
): Float32Array {
  const x = allocF32(seq * N_EMBD);
  for (let i: i32 = 0; i < seq; i++) {
    const tokenOffset: i32 = unchecked(tokens[i]) * N_EMBD;
    const posOffset:   i32 = i * N_EMBD;
    for (let d: i32 = 0; d < N_EMBD; d++) {
      unchecked(x[i * N_EMBD + d] = wte[tokenOffset + d] + wpe[posOffset + d]);
    }
  }
  return x;
}

// wte [vocabSize, nEmbd] -> wteT [nEmbd, vocabSize]
function transposeWeight(wte: Float32Array, vocabSize: i32, nEmbd: i32): Float32Array {
  const out = allocF32(nEmbd * vocabSize);
  for (let v: i32 = 0; v < vocabSize; v++) {
    for (let d: i32 = 0; d < nEmbd; d++) {
      unchecked(out[d * vocabSize + v] = wte[v * nEmbd + d]);
    }
  }
  return out;
}
