import { Tensor, zeros, matmul, transpose, add, scale, softmax, reshape, causalMask } from "./tensor.js";

export interface AttentionWeights {
  cAttnWeight: Tensor;  // [768, 2304] — Q, K, V projected together
  cAttnBias: Tensor;    // [2304]
  cProjWeight: Tensor;  // [768, 768]
  cProjBias: Tensor;    // [768]
}

export function multiHeadAttention(
  x: Tensor,            // [seq, 768]
  w: AttentionWeights,
  nHeads: number,
): Tensor {
  const [seq, dModel] = x.shape as [number, number];
  const dHead = dModel / nHeads;

  // [seq, 768] x [768, 2304] + [2304] -> [seq, 2304]
  const qkv = add(matmul(x, w.cAttnWeight), w.cAttnBias);

  // split into Q, K, V — each [seq, 768]
  const q = sliceCols(qkv, 0, dModel);
  const k = sliceCols(qkv, dModel, 2 * dModel);
  const v = sliceCols(qkv, 2 * dModel, 3 * dModel);

  // split heads: [seq, 768] -> [nHeads, seq, dHead]
  const qH = splitHeads(q, nHeads, dHead);
  const kH = splitHeads(k, nHeads, dHead);
  const vH = splitHeads(v, nHeads, dHead);

  // scores: [nHeads, seq, seq]
  const mask = causalMask(seq);
  const scores = scaledDotProduct(qH, kH, dHead, mask);

  // weighted sum: [nHeads, seq, dHead]
  const attended = matmul(scores, vH);

  // merge heads: [seq, 768]
  const merged = mergeHeads(attended, nHeads, dHead, seq);

  // output projection: [seq, 768]
  return add(matmul(merged, w.cProjWeight), w.cProjBias);
}

// scores = softmax(Q @ K^T / sqrt(dHead) + mask)
function scaledDotProduct(q: Tensor, k: Tensor, dHead: number, mask: Tensor): Tensor {
  // [nHeads, seq, dHead] x [nHeads, dHead, seq] -> [nHeads, seq, seq]
  const raw = matmul(q, transpose(k));
  const scaled = scale(raw, 1 / Math.sqrt(dHead));

  // broadcast mask [seq, seq] across heads
  const nHeads = q.shape[0];
  const seq = q.shape[1];
  const masked = zeros([nHeads, seq, seq]);
  for (let h = 0; h < nHeads; h++) {
    for (let i = 0; i < seq * seq; i++) {
      masked.data[h * seq * seq + i] = scaled.data[h * seq * seq + i] + mask.data[i];
    }
  }

  // softmax along last axis (per row per head)
  return softmax(masked);
}

// [seq, dModel] -> [nHeads, seq, dHead]
function splitHeads(t: Tensor, nHeads: number, dHead: number): Tensor {
  const seq = t.shape[0];
  const out = zeros([nHeads, seq, dHead]);
  for (let s = 0; s < seq; s++) {
    for (let h = 0; h < nHeads; h++) {
      for (let d = 0; d < dHead; d++) {
        out.data[h * seq * dHead + s * dHead + d] = t.data[s * nHeads * dHead + h * dHead + d];
      }
    }
  }
  return out;
}

// [nHeads, seq, dHead] -> [seq, dModel]
function mergeHeads(t: Tensor, nHeads: number, dHead: number, seq: number): Tensor {
  const dModel = nHeads * dHead;
  const out = zeros([seq, dModel]);
  for (let h = 0; h < nHeads; h++) {
    for (let s = 0; s < seq; s++) {
      for (let d = 0; d < dHead; d++) {
        out.data[s * dModel + h * dHead + d] = t.data[h * seq * dHead + s * dHead + d];
      }
    }
  }
  return out;
}

// slice columns [start, end) from a 2D tensor [rows, cols]
function sliceCols(t: Tensor, start: number, end: number): Tensor {
  const rows = t.shape[0];
  const srcCols = t.shape[1];
  const outCols = end - start;
  const out = zeros([rows, outCols]);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < outCols; c++) {
      out.data[r * outCols + c] = t.data[r * srcCols + start + c];
    }
  }
  return out;
}
