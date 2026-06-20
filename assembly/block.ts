import { add, layerNorm } from "./tensor";
import { multiHeadAttention } from "./attention";
import { feedForward } from "./feedforward";
import { getScratch } from "./scratch";

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
  const s = getScratch();
  const seqDM: i32 = seq * dModel;

  // attn branch: ln1 -> attention -> residual
  layerNorm(x, seq, dModel, ln1Weight, ln1Bias, s.tmp);
  multiHeadAttention(s.tmp, seq, dModel, cAttnWeight, cAttnBias, cProjWeight, cProjBias, nHeads, s.attnOut);
  add(x, s.attnOut, x, seqDM);

  // mlp branch: ln2 -> feedforward -> residual
  layerNorm(x, seq, dModel, ln2Weight, ln2Bias, s.tmp);
  feedForward(s.tmp, seq, dModel, cFcWeight, cFcBias, cFcProjWeight, cFcProjBias, s.mlpOut);
  add(x, s.mlpOut, x, seqDM);
}
