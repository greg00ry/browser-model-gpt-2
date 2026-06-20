import { Tensor, add, gelu, matmul } from "./tensor.js";

export interface FeedForwardWeights {
  cFcWeight: Tensor;    // [768, 3072]
  cFcBias: Tensor;      // [3072]
  cProjWeight: Tensor;  // [3072, 768]
  cProjBias: Tensor;    // [768]
}

// [seq, 768] -> [seq, 3072] -> [seq, 768]
export function feedForward(x: Tensor, w: FeedForwardWeights): Tensor {
  const hidden = gelu(add(matmul(x, w.cFcWeight), w.cFcBias));
  return add(matmul(hidden, w.cProjWeight), w.cProjBias);
}
