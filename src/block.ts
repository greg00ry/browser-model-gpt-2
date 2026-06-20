import { Tensor, add, layerNorm } from "./tensor.js";
import { multiHeadAttention, AttentionWeights } from "./attention.js";
import { feedForward, FeedForwardWeights } from "./feedforward.js";

export interface BlockWeights {
  ln1Weight: Tensor;
  ln1Bias: Tensor;
  ln2Weight: Tensor;
  ln2Bias: Tensor;
  attn: AttentionWeights;
  mlp: FeedForwardWeights;
}

// x = x + attn(ln1(x))
// x = x + mlp(ln2(x))
export function transformerBlock(x: Tensor, w: BlockWeights, nHeads: number): Tensor {
  x = add(x, multiHeadAttention(layerNorm(x, w.ln1Weight, w.ln1Bias), w.attn, nHeads));
  x = add(x, feedForward(layerNorm(x, w.ln2Weight, w.ln2Bias), w.mlp));
  return x;
}
