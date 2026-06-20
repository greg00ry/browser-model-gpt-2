import { matmul2d, add, gelu } from "./tensor";
import { getScratch } from "./scratch";

export function feedForward(
  x: Float32Array, seq: i32, dModel: i32,
  cFcWeight: Float32Array,
  cFcBias: Float32Array,
  cProjWeight: Float32Array,
  cProjBias: Float32Array,
  out: Float32Array,
): void {
  const s = getScratch();
  const dInner: i32 = dModel * 4;
  const nHidden: i32 = seq * dInner;

  // [seq, dModel] x [dModel, 4*dModel] + bias -> scratch.hidden
  matmul2d(x, seq, dModel, cFcWeight, dInner, s.hidden);
  add(s.hidden, cFcBias, s.hidden, nHidden);
  gelu(s.hidden, s.hidden, nHidden);

  // [seq, 4*dModel] x [4*dModel, dModel] + bias -> out
  matmul2d(s.hidden, seq, dInner, cProjWeight, dModel, out);
  add(out, cProjBias, out, seq * dModel);
}
