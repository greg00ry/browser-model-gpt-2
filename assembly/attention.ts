import { matmul2d, matmul3d, transpose, add, scale, softmax, causalMask, allocF32 } from "./tensor";

// [seq, dModel] x [dModel, 2304] + [2304] -> [seq, 2304]  -> split Q,K,V -> splitHeads -> scaledDotProduct -> mergeHeads -> proj
export function multiHeadAttention(
  x: Float32Array, seq: i32, dModel: i32,
  cAttnWeight: Float32Array,              // [dModel, 3*dModel]
  cAttnBias: Float32Array,                // [3*dModel]
  cProjWeight: Float32Array,              // [dModel, dModel]
  cProjBias: Float32Array,                // [dModel]
  nHeads: i32,
  out: Float32Array,                      // [seq, dModel]
): void {
  const dHead: i32 = dModel / nHeads;
  const dModel3: i32 = dModel * 3;

  // QKV projection: [seq, dModel] x [dModel, 3*dModel] -> [seq, 3*dModel]
  const qkv = allocF32(seq * dModel3);
  matmul2d(x, seq, dModel, cAttnWeight, dModel3, qkv);
  add(qkv, cAttnBias, qkv);

  // split Q, K, V: each [seq, dModel]
  const q = sliceCols(qkv, seq, dModel3, 0,       dModel);
  const k = sliceCols(qkv, seq, dModel3, dModel,   dModel * 2);
  const v = sliceCols(qkv, seq, dModel3, dModel * 2, dModel3);

  // split heads: [seq, dModel] -> [nHeads, seq, dHead]
  const qH = splitHeads(q, seq, nHeads, dHead);
  const kH = splitHeads(k, seq, nHeads, dHead);
  const vH = splitHeads(v, seq, nHeads, dHead);

  // scores: [nHeads, seq, seq]
  const mask = allocF32(seq * seq);
  causalMask(seq, mask);
  const scores = scaledDotProduct(qH, kH, nHeads, seq, dHead, mask);

  // weighted sum: [nHeads, seq, dHead]
  const attended = allocF32(nHeads * seq * dHead);
  matmul3d(scores, nHeads, seq, seq, vH, dHead, attended);

  // merge heads: [nHeads, seq, dHead] -> [seq, dModel]
  const merged = mergeHeads(attended, nHeads, seq, dHead);

  // output projection: [seq, dModel] x [dModel, dModel] -> [seq, dModel]
  matmul2d(merged, seq, dModel, cProjWeight, dModel, out);
  add(out, cProjBias, out);
}

// softmax(Q @ K^T / sqrt(dHead) + mask)
function scaledDotProduct(
  q: Float32Array, k: Float32Array,
  nHeads: i32, seq: i32, dHead: i32,
  mask: Float32Array,
): Float32Array {
  // [nHeads, seq, dHead] x [nHeads, dHead, seq] -> [nHeads, seq, seq]
  const kT = allocF32(nHeads * seq * dHead);
  transpose(k, nHeads, seq, dHead, kT);

  const raw = allocF32(nHeads * seq * seq);
  matmul3d(q, nHeads, seq, dHead, kT, seq, raw);

  const factor: f32 = 1.0 / Mathf.sqrt(f32(dHead));
  scale(raw, factor, raw);

  // broadcast mask [seq, seq] across heads
  const seqSq: i32 = seq * seq;
  for (let h: i32 = 0; h < nHeads; h++) {
    for (let i: i32 = 0; i < seqSq; i++) {
      unchecked(raw[h * seqSq + i] += mask[i]);
    }
  }

  softmax(raw, nHeads * seq, seq, raw);
  return raw;
}

// [seq, dModel] -> [nHeads, seq, dHead]
function splitHeads(src: Float32Array, seq: i32, nHeads: i32, dHead: i32): Float32Array {
  const dst = allocF32(nHeads * seq * dHead);
  for (let s: i32 = 0; s < seq; s++) {
    for (let h: i32 = 0; h < nHeads; h++) {
      for (let d: i32 = 0; d < dHead; d++) {
        unchecked(dst[h * seq * dHead + s * dHead + d] = src[s * nHeads * dHead + h * dHead + d]);
      }
    }
  }
  return dst;
}

// [nHeads, seq, dHead] -> [seq, dModel]
function mergeHeads(src: Float32Array, nHeads: i32, seq: i32, dHead: i32): Float32Array {
  const dModel: i32 = nHeads * dHead;
  const dst = allocF32(seq * dModel);
  for (let h: i32 = 0; h < nHeads; h++) {
    for (let s: i32 = 0; s < seq; s++) {
      for (let d: i32 = 0; d < dHead; d++) {
        unchecked(dst[s * dModel + h * dHead + d] = src[h * seq * dHead + s * dHead + d]);
      }
    }
  }
  return dst;
}

// slice columns [colStart, colEnd) from [rows, srcCols]
function sliceCols(
  src: Float32Array, rows: i32, srcCols: i32,
  colStart: i32, colEnd: i32,
): Float32Array {
  const outCols: i32 = colEnd - colStart;
  const dst = allocF32(rows * outCols);
  for (let r: i32 = 0; r < rows; r++) {
    for (let c: i32 = 0; c < outCols; c++) {
      unchecked(dst[r * outCols + c] = src[r * srcCols + colStart + c]);
    }
  }
  return dst;
}
