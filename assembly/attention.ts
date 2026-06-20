import { matmul2d, matmul3d, transpose, add, scale, softmax, causalMask } from "./tensor";
import { getScratch } from "./scratch";

export function multiHeadAttention(
  x: Float32Array, seq: i32, dModel: i32,
  cAttnWeight: Float32Array,
  cAttnBias: Float32Array,
  cProjWeight: Float32Array,
  cProjBias: Float32Array,
  nHeads: i32,
  out: Float32Array,
): void {
  const s = getScratch();
  const dHead: i32   = dModel / nHeads;
  const dModel3: i32 = dModel * 3;
  const seqDM3: i32  = seq * dModel3;
  const seqDM: i32   = seq * dModel;

  // QKV projection: [seq, dModel] x [dModel, 3*dModel] -> scratch.qkv
  matmul2d(x, seq, dModel, cAttnWeight, dModel3, s.qkv);
  add(s.qkv, cAttnBias, s.qkv, seqDM3);

  // split Q, K, V — każdy [seq, dModel] zapisany do osobnego bufora scratch
  sliceColsTo(s.qkv, seq, dModel3, 0,           dModel,     s.q);
  sliceColsTo(s.qkv, seq, dModel3, dModel,       dModel * 2, s.k);
  sliceColsTo(s.qkv, seq, dModel3, dModel * 2,  dModel3,    s.v);

  // split heads: [seq, dModel] -> [nHeads, seq, dHead]
  splitHeadsTo(s.q, seq, nHeads, dHead, s.qH);
  splitHeadsTo(s.k, seq, nHeads, dHead, s.kH);
  splitHeadsTo(s.v, seq, nHeads, dHead, s.vH);

  // refresh maska kauzalna gdy seq się zmienia
  if (s.maskSeq !== seq) {
    causalMask(seq, s.mask);
    s.maskSeq = seq;
  }

  // scores: softmax(Q @ K^T / sqrt(dHead) + mask) -> scratch.scores
  scaledDotProductTo(s.qH, s.kH, nHeads, seq, dHead, s.mask, s.kT, s.scores);

  // weighted sum: [nHeads, seq, seq] x [nHeads, seq, dHead] -> scratch.attended
  matmul3d(s.scores, nHeads, seq, seq, s.vH, dHead, s.attended);

  // merge heads: [nHeads, seq, dHead] -> scratch.merged
  mergeHeadsTo(s.attended, nHeads, seq, dHead, s.merged);

  // output projection: [seq, dModel] x [dModel, dModel] -> out
  matmul2d(s.merged, seq, dModel, cProjWeight, dModel, out);
  add(out, cProjBias, out, seqDM);
}

// softmax(Q @ K^T / sqrt(dHead) + mask) — zapisuje do scores, używa kT jako scratch
function scaledDotProductTo(
  q: Float32Array, k: Float32Array,
  nHeads: i32, seq: i32, dHead: i32,
  mask: Float32Array,
  kT: Float32Array,
  scores: Float32Array,
): void {
  const nElems: i32 = nHeads * seq * seq;
  const seqSq: i32  = seq * seq;

  // transpose K: [nHeads, seq, dHead] -> [nHeads, dHead, seq]
  transpose(k, nHeads, seq, dHead, kT);

  // Q @ K^T: [nHeads, seq, dHead] x [nHeads, dHead, seq] -> scores [nHeads, seq, seq]
  matmul3d(q, nHeads, seq, dHead, kT, seq, scores);

  // scale + mask
  const factor: f32 = 1.0 / Mathf.sqrt(f32(dHead));
  scale(scores, factor, scores, nElems);

  for (let h: i32 = 0; h < nHeads; h++) {
    for (let i: i32 = 0; i < seqSq; i++) {
      unchecked(scores[h * seqSq + i] += mask[i]);
    }
  }

  softmax(scores, nHeads * seq, seq, scores);
}

// [seq, dModel] -> [nHeads, seq, dHead] (direct write, bez allocacji)
function splitHeadsTo(
  src: Float32Array, seq: i32, nHeads: i32, dHead: i32,
  dst: Float32Array,
): void {
  for (let s: i32 = 0; s < seq; s++) {
    for (let h: i32 = 0; h < nHeads; h++) {
      for (let d: i32 = 0; d < dHead; d++) {
        unchecked(dst[h * seq * dHead + s * dHead + d] = src[s * nHeads * dHead + h * dHead + d]);
      }
    }
  }
}

// [nHeads, seq, dHead] -> [seq, dModel] (direct write)
function mergeHeadsTo(
  src: Float32Array, nHeads: i32, seq: i32, dHead: i32,
  dst: Float32Array,
): void {
  const dModel: i32 = nHeads * dHead;
  for (let h: i32 = 0; h < nHeads; h++) {
    for (let s: i32 = 0; s < seq; s++) {
      for (let d: i32 = 0; d < dHead; d++) {
        unchecked(dst[s * dModel + h * dHead + d] = src[h * seq * dHead + s * dHead + d]);
      }
    }
  }
}

// slice columns [colStart, colEnd) z [rows, srcCols] (direct write)
function sliceColsTo(
  src: Float32Array, rows: i32, srcCols: i32,
  colStart: i32, colEnd: i32,
  dst: Float32Array,
): void {
  const outCols: i32 = colEnd - colStart;
  for (let r: i32 = 0; r < rows; r++) {
    for (let c: i32 = 0; c < outCols; c++) {
      unchecked(dst[r * outCols + c] = src[r * srcCols + colStart + c]);
    }
  }
}
