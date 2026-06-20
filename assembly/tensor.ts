// AssemblyScript — kompiluje do prawdziwego .wasm
// Typy: f32/i32 zamiast number, Mathf zamiast Math, unchecked() dla wydajności

const GELU_COEF: f32     = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.7978845608;
const NEG_INF: f32        = -Infinity;

// [M, K] x [K, N] -> out — SIMD f32x4 (4 elementy naraz), scalar tail dla reszty
export function matmul2d(
  a: Float32Array, aRows: i32, aCols: i32,
  b: Float32Array, bCols: i32,
  out: Float32Array,
): void {
  memory.fill(out.dataStart, 0, (aRows * bCols) << 2);
  const bCols4: i32 = bCols & ~3;

  for (let m: i32 = 0; m < aRows; m++) {
    for (let k: i32 = 0; k < aCols; k++) {
      const aVec = f32x4.splat(unchecked(a[m * aCols + k]));
      const bRow = b.dataStart   + <usize>((k * bCols) << 2);
      const oRow = out.dataStart + <usize>((m * bCols) << 2);

      let n: i32 = 0;
      for (; n < bCols4; n += 4) {
        const nb = <usize>(n << 2);
        v128.store(oRow + nb,
          f32x4.add(v128.load(oRow + nb), f32x4.mul(aVec, v128.load(bRow + nb)))
        );
      }
      const aVal = f32x4.extract_lane(aVec, 0);
      for (; n < bCols; n++) {
        unchecked(out[m * bCols + n] += aVal * b[k * bCols + n]);
      }
    }
  }
}

// batched SIMD — [B, M, K] x [B, K, N] -> [B, M, N]
export function matmul3d(
  a: Float32Array, B: i32, M: i32, K: i32,
  b: Float32Array, N: i32,
  out: Float32Array,
): void {
  memory.fill(out.dataStart, 0, (B * M * N) << 2);
  const N4: i32 = N & ~3;

  for (let i: i32 = 0; i < B; i++) {
    const aBatch: i32 = i * M * K;
    const bBatch: i32 = i * K * N;
    const oBatch: i32 = i * M * N;
    for (let m: i32 = 0; m < M; m++) {
      for (let k: i32 = 0; k < K; k++) {
        const aVec = f32x4.splat(unchecked(a[aBatch + m * K + k]));
        const bRow = b.dataStart   + <usize>((bBatch + k * N) << 2);
        const oRow = out.dataStart + <usize>((oBatch + m * N) << 2);

        let n: i32 = 0;
        for (; n < N4; n += 4) {
          const nb = <usize>(n << 2);
          v128.store(oRow + nb,
            f32x4.add(v128.load(oRow + nb), f32x4.mul(aVec, v128.load(bRow + nb)))
          );
        }
        const aVal = f32x4.extract_lane(aVec, 0);
        for (; n < N; n++) {
          unchecked(out[oBatch + m * N + n] += aVal * b[bBatch + k * N + n]);
        }
      }
    }
  }
}

// swap last two axes
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

// add element-wise (b broadcastuje po ostatniej osi) — nElems zamiast out.length
export function add(
  a: Float32Array, b: Float32Array, out: Float32Array, nElems: i32,
): void {
  const bLen: i32 = b.length;
  for (let i: i32 = 0; i < nElems; i++) {
    unchecked(out[i] = a[i] + b[i % bLen]);
  }
}

// scale — nElems zamiast src.length
export function scale(src: Float32Array, factor: f32, dst: Float32Array, nElems: i32): void {
  for (let i: i32 = 0; i < nElems; i++) {
    unchecked(dst[i] = src[i] * factor);
  }
}

// softmax po ostatniej osi — już ma explicit rows/cols
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

// GELU — nElems zamiast src.length
export function gelu(src: Float32Array, dst: Float32Array, nElems: i32): void {
  for (let i: i32 = 0; i < nElems; i++) {
    const x = unchecked(src[i]);
    const inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
    unchecked(dst[i] = 0.5 * x * (1.0 + Mathf.tanh(inner)));
  }
}

// LayerNorm — explicit rows/cols
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

// kauzalna maska — górny trójkąt = -Infinity
export function causalMask(size: i32, dst: Float32Array): void {
  for (let r: i32 = 0; r < size; r++) {
    for (let c: i32 = 0; c < size; c++) {
      unchecked(dst[r * size + c] = c > r ? NEG_INF : 0.0);
    }
  }
}

export function allocF32(length: i32): Float32Array {
  return new Float32Array(length);
}
