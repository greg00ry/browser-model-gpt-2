import { add, layerNorm, allocF32 } from "./tensor";
import { multiHeadAttention } from "./attention";
import { feedForward } from "./feedforward";

// x = x + attn(ln1(x))
// x = x + mlp(ln2(x))
export function transformerBlock(
  x: Float32Array, seq: i32, dModel: i32,
  ln1Weight: Float32Array, ln1Bias: Float32Array,
  ln2Weight: Float32Array, ln2Bias: Float32Array,
  cAttnWeight: Float32Array, cAttnBias: Float32Array,
  cProjWeight: Float32Array, cProjBias: Float32Array,
  cFcWeight: Float32Array,   cFcBias: Float32Array,
  cFcProjWeight: Float32Array, cFcProjBias: Float32Array,
  nHeads: i32,
): void {
  const tmp = allocF32(seq * dModel);

  // attn branch: ln1 -> attention -> residual
  layerNorm(x, seq, dModel, ln1Weight, ln1Bias, tmp);
  const attnOut = allocF32(seq * dModel);
  multiHeadAttention(tmp, seq, dModel, cAttnWeight, cAttnBias, cProjWeight, cProjBias, nHeads, attnOut);
  add(x, attnOut, x);

  // mlp branch: ln2 -> feedforward -> residual
  layerNorm(x, seq, dModel, ln2Weight, ln2Bias, tmp);
  const mlpOut = allocF32(seq * dModel);
  feedForward(tmp, seq, dModel, cFcWeight, cFcBias, cFcProjWeight, cFcProjBias, mlpOut);
  add(x, mlpOut, x);
}
