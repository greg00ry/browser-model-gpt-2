import { Tensor, add, layerNorm, matmul, reshape } from "./tensor.js";
import { transformerBlock, BlockWeights } from "./block.js";

export interface GPT2Config {
  nLayers: number;   // 12
  nHeads: number;    // 12
  nEmbd: number;     // 768
  nCtx: number;      // 1024
  vocabSize: number; // 50257
}

export interface GPT2Weights {
  wte: Tensor;          // token embeddings [vocabSize, nEmbd]
  wpe: Tensor;          // position embeddings [nCtx, nEmbd]
  blocks: BlockWeights[];
  lnFWeight: Tensor;    // final layer norm weight [nEmbd]
  lnFBias: Tensor;      // final layer norm bias [nEmbd]
}

export const GPT2_SMALL: GPT2Config = {
  nLayers: 12,
  nHeads: 12,
  nEmbd: 768,
  nCtx: 1024,
  vocabSize: 50257,
};

// tokens [seq] -> logits [seq, vocabSize]
export function forward(tokens: number[], weights: GPT2Weights, config: GPT2Config): Tensor {
  const seq = tokens.length;
  const { nEmbd, nHeads, vocabSize } = config;

  // token + position embeddings -> [seq, nEmbd]
  let x = tokenAndPositionEmbeddings(tokens, weights.wte, weights.wpe, nEmbd);

  // 12x transformer blocks
  for (const block of weights.blocks) {
    x = transformerBlock(x, block, nHeads);
  }

  // final layer norm
  x = layerNorm(x, weights.lnFWeight, weights.lnFBias);

  // lm_head: [seq, nEmbd] x [nEmbd, vocabSize] -> [seq, vocabSize]
  // GPT-2 shares weights: lm_head = wte^T
  const wteT = transposeWeight(weights.wte, vocabSize, nEmbd);
  return matmul(x, wteT);
}

function tokenAndPositionEmbeddings(
  tokens: number[],
  wte: Tensor,
  wpe: Tensor,
  nEmbd: number,
): Tensor {
  const seq = tokens.length;
  const data = new Float32Array(seq * nEmbd);
  for (let i = 0; i < seq; i++) {
    const tokenOffset = tokens[i] * nEmbd;
    const posOffset = i * nEmbd;
    for (let d = 0; d < nEmbd; d++) {
      data[i * nEmbd + d] = wte.data[tokenOffset + d] + wpe.data[posOffset + d];
    }
  }
  return new Tensor(data, [seq, nEmbd]);
}

// [vocabSize, nEmbd] -> [nEmbd, vocabSize]  (weight tying: reuse wte transposed)
function transposeWeight(wte: Tensor, vocabSize: number, nEmbd: number): Tensor {
  const out = new Float32Array(nEmbd * vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    for (let d = 0; d < nEmbd; d++) {
      out[d * vocabSize + v] = wte.data[v * nEmbd + d];
    }
  }
  return new Tensor(out, [nEmbd, vocabSize]);
}
