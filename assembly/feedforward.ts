import { matmul2d, add, gelu, allocF32 } from "./tensor";

// [seq, 768] -> [seq, 3072] -> [seq, 768]
export function feedForward(
  x: Float32Array, seq: i32, dModel: i32,
  cFcWeight: Float32Array,    // [dModel, 4*dModel]
  cFcBias: Float32Array,      // [4*dModel]
  cProjWeight: Float32Array,  // [4*dModel, dModel]
  cProjBias: Float32Array,    // [dModel]
  out: Float32Array,          // [seq, dModel]
): void {
  const dInner: i32 = dModel * 4;

  // [seq, dModel] x [dModel, 4*dModel] + bias -> [seq, 4*dModel]
  const hidden = allocF32(seq * dInner);
  matmul2d(x, seq, dModel, cFcWeight, dInner, hidden);
  add(hidden, cFcBias, hidden);
  gelu(hidden, hidden);

  // [seq, 4*dModel] x [4*dModel, dModel] + bias -> [seq, dModel]
  matmul2d(hidden, seq, dInner, cProjWeight, dModel, out);
  add(out, cProjBias, out);
}
