export type Shape = readonly number[];

export class Tensor {
  readonly data: Float32Array;
  readonly shape: Shape;

  constructor(data: Float32Array, shape: Shape) {
    this.data = data;
    this.shape = shape;
  }

  get ndim(): number { return this.shape.length; }
  get size(): number { return this.data.length; }
}

export function zeros(shape: Shape): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  return new Tensor(new Float32Array(size), shape);
}

export function reshape(t: Tensor, shape: Shape): Tensor {
  return new Tensor(t.data, shape);
}

// [M, K] x [K, N] -> [M, N]  |  [B, M, K] x [B, K, N] -> [B, M, N]
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.ndim === 2 && b.ndim === 2) {
    return matmul2d(a, b);
  }
  if (a.ndim === 3 && b.ndim === 3) {
    const B = a.shape[0];
    const out = zeros([B, a.shape[1], b.shape[2]]);
    const sliceA = a.shape[1] * a.shape[2];
    const sliceB = b.shape[1] * b.shape[2];
    const sliceOut = a.shape[1] * b.shape[2];
    for (let i = 0; i < B; i++) {
      const ta = new Tensor(a.data.subarray(i * sliceA, (i + 1) * sliceA), [a.shape[1], a.shape[2]]);
      const tb = new Tensor(b.data.subarray(i * sliceB, (i + 1) * sliceB), [b.shape[1], b.shape[2]]);
      const tc = matmul2d(ta, tb);
      out.data.set(tc.data, i * sliceOut);
    }
    return out;
  }
  throw new Error(`matmul: unsupported shapes ${a.shape} x ${b.shape}`);
}

function matmul2d(a: Tensor, b: Tensor): Tensor {
  const M = a.shape[0], K = a.shape[1], N = b.shape[1];
  const out = zeros([M, N]);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < K; k++) {
      const aVal = a.data[m * K + k];
      for (let n = 0; n < N; n++) {
        out.data[m * N + n] += aVal * b.data[k * N + n];
      }
    }
  }
  return out;
}

// swap last two dimensions
export function transpose(t: Tensor): Tensor {
  const ndim = t.ndim;
  const rows = t.shape[ndim - 2];
  const cols = t.shape[ndim - 1];
  const batches = t.size / (rows * cols);
  const newShape = [...t.shape.slice(0, -2), cols, rows] as number[];
  const out = zeros(newShape);
  for (let b = 0; b < batches; b++) {
    const offset = b * rows * cols;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        out.data[offset + c * rows + r] = t.data[offset + r * cols + c];
      }
    }
  }
  return out;
}

export function add(a: Tensor, b: Tensor): Tensor {
  const out = new Tensor(new Float32Array(a.data), a.shape);
  for (let i = 0; i < out.data.length; i++) {
    out.data[i] += b.data[i % b.data.length];
  }
  return out;
}

export function scale(t: Tensor, factor: number): Tensor {
  const out = new Tensor(new Float32Array(t.data), t.shape);
  for (let i = 0; i < out.data.length; i++) {
    out.data[i] *= factor;
  }
  return out;
}

// softmax along last axis
export function softmax(t: Tensor): Tensor {
  const cols = t.shape[t.ndim - 1];
  const rows = t.size / cols;
  const out = zeros(t.shape);
  for (let r = 0; r < rows; r++) {
    const offset = r * cols;
    let max = -Infinity;
    for (let c = 0; c < cols; c++) max = Math.max(max, t.data[offset + c]);
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      out.data[offset + c] = Math.exp(t.data[offset + c] - max);
      sum += out.data[offset + c];
    }
    for (let c = 0; c < cols; c++) out.data[offset + c] /= sum;
  }
  return out;
}

// GELU approximation used by GPT-2
export function gelu(t: Tensor): Tensor {
  const out = new Tensor(new Float32Array(t.size), t.shape);
  for (let i = 0; i < t.size; i++) {
    const x = t.data[i];
    out.data[i] = 0.5 * x * (1 + Math.tanh(Math.SQRT2 / Math.PI * (x + 0.044715 * x * x * x)));
  }
  return out;
}

// layer norm along last axis
export function layerNorm(t: Tensor, weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
  const cols = t.shape[t.ndim - 1];
  const rows = t.size / cols;
  const out = zeros(t.shape);
  for (let r = 0; r < rows; r++) {
    const offset = r * cols;
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += t.data[offset + c];
    mean /= cols;
    let variance = 0;
    for (let c = 0; c < cols; c++) {
      const diff = t.data[offset + c] - mean;
      variance += diff * diff;
    }
    variance /= cols;
    const std = Math.sqrt(variance + eps);
    for (let c = 0; c < cols; c++) {
      out.data[offset + c] = ((t.data[offset + c] - mean) / std) * weight.data[c] + bias.data[c];
    }
  }
  return out;
}

// upper-triangular causal mask: positions beyond current token = -inf
export function causalMask(size: number): Tensor {
  const out = zeros([size, size]);
  for (let r = 0; r < size; r++) {
    for (let c = r + 1; c < size; c++) {
      out.data[r * size + c] = -Infinity;
    }
  }
  return out;
}
