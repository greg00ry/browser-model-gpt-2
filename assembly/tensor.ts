// AssemblyScript — kompiluje do prawdziwego .wasm
// Typy: f32/i32 zamiast number, Mathf zamiast Math, unchecked() dla wydajności

// ---- stałe f32 ----
const GELU_COEF: f32     = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
const NEG_INF: f32        = -Infinity;

// ---- matmul ----
// [M, K] x [K, N] -> [out]   (out musi być zaalokowany przez wywołującego)
export function matmul2d(
  a: Float32Array, aRows: i32, aCols: i32,
  b: Float32Array, bCols: i32,
  out: Float32Array,
): void {
  for (let m: i32 = 0; m < aRows; m++) {
    for (let k: i32 = 0; k < aCols; k++) {
      const aVal = unchecked(a[m * aCols + k]);
      for (let n: i32 = 0; n < bCols; n++) {
        unchecked(out[m * bCols + n] += aVal * unchecked(b[k * bCols + n]));
      }
    }
  }
}

// batched: [B, M, K] x [B, K, N] -> [B, M, N]
export function matmul3d(
  a: Float32Array, B: i32, M: i32, K: i32,
  b: Float32Array, N: i32,
  out: Float32Array,
): void {
  const sliceA: i32 = M * K;
  const sliceB: i32 = K * N;
  const sliceO: i32 = M * N;
  for (let i: i32 = 0; i < B; i++) {
    const aSlice = a.subarray(i * sliceA, (i + 1) * sliceA);
    const bSlice = b.subarray(i * sliceB, (i + 1) * sliceB);
    const oSlice = out.subarray(i * sliceO, (i + 1) * sliceO);
    matmul2d(aSlice, M, K, bSlice, N, oSlice);
  }
}

// ---- transpose — swap ostatnich dwóch osi ----
export function transpose(
  src: Float32Array, batches: i32, rows: i32, cols: i32,
  dst: Float32Array,
): void {
  for (let b: i32 = 0; b < batches; b++) {
    const offset: i32 = b * rows * cols;
    for (let r: i32 = 0; r < rows; r++) {
      for (let c: i32 = 0; c < cols; c++) {
        unchecked(dst[offset + c * rows + r] = src[offset + r * cols + c]);
      }
    }
  }
}

// ---- add element-wise (b broadcastuje po ostatniej osi) ----
export function add(
  a: Float32Array, b: Float32Array, out: Float32Array,
): void {
  const bLen: i32 = b.length;
  for (let i: i32 = 0; i < out.length; i++) {
    unchecked(out[i] = a[i] + b[i % bLen]);
  }
}

// ---- scale przez skalar ----
export function scale(src: Float32Array, factor: f32, dst: Float32Array): void {
  for (let i: i32 = 0; i < src.length; i++) {
    unchecked(dst[i] = src[i] * factor);
  }
}

// ---- softmax po ostatniej osi ----
export function softmax(src: Float32Array, rows: i32, cols: i32, dst: Float32Array): void {
  for (let r: i32 = 0; r < rows; r++) {
    const offset: i32 = r * cols;
    let max: f32 = NEG_INF;
    for (let c: i32 = 0; c < cols; c++) {
      const v = unchecked(src[offset + c]);
      if (v > max) max = v;
    }
    let sum: f32 = 0.0;
    for (let c: i32 = 0; c < cols; c++) {
      const e = Mathf.exp(unchecked(src[offset + c]) - max);
      unchecked(dst[offset + c] = e);
      sum += e;
    }
    for (let c: i32 = 0; c < cols; c++) {
      unchecked(dst[offset + c] /= sum);
    }
  }
}

// ---- GELU (aproksymacja tanh — identyczna z GPT-2 / HuggingFace) ----
export function gelu(src: Float32Array, dst: Float32Array): void {
  for (let i: i32 = 0; i < src.length; i++) {
    const x = unchecked(src[i]);
    const inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
    unchecked(dst[i] = 0.5 * x * (1.0 + Mathf.tanh(inner)));
  }
}

// ---- LayerNorm po ostatniej osi ----
export function layerNorm(
  src: Float32Array, rows: i32, cols: i32,
  weight: Float32Array, bias: Float32Array,
  dst: Float32Array,
  eps: f32 = 1e-5,
): void {
  for (let r: i32 = 0; r < rows; r++) {
    const offset: i32 = r * cols;
    let mean: f32 = 0.0;
    for (let c: i32 = 0; c < cols; c++) mean += unchecked(src[offset + c]);
    mean /= f32(cols);
    let variance: f32 = 0.0;
    for (let c: i32 = 0; c < cols; c++) {
      const diff = unchecked(src[offset + c]) - mean;
      variance += diff * diff;
    }
    variance /= f32(cols);
    const std = Mathf.sqrt(variance + eps);
    for (let c: i32 = 0; c < cols; c++) {
      unchecked(dst[offset + c] = ((src[offset + c] - mean) / std) * weight[c] + bias[c]);
    }
  }
}

// ---- maska kauzalna — górny trójkąt = -Infinity ----
export function causalMask(size: i32, dst: Float32Array): void {
  for (let r: i32 = 0; r < size; r++) {
    for (let c: i32 = 0; c < size; c++) {
      unchecked(dst[r * size + c] = c > r ? NEG_INF : 0.0);
    }
  }
}

// ---- alokacja (host wywołuje przez __new) ----
export function allocF32(length: i32): Float32Array {
  return new Float32Array(length);
}
