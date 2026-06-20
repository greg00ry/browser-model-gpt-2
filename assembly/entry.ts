import { forward } from "./model";

// zwraca wskaźnik do surowych bajtów Float32Array — JS pisze bezpośrednio do Wasm memory
export function dataPtr(arr: Float32Array): usize {
  return arr.dataStart;
}

export function newF32(length: i32): Float32Array {
  return new Float32Array(length);
}

export function newI32(length: i32): Int32Array {
  return new Int32Array(length);
}

export function i32DataPtr(arr: Int32Array): usize {
  return arr.dataStart;
}

// globalne referencje do wag — kopiowane raz przy init, trzymane w pamięci Wasm
let g_wte: Float32Array = new Float32Array(0);
let g_wpe: Float32Array = new Float32Array(0);
let g_ln1Weights:    Float32Array[] = [];
let g_ln1Biases:     Float32Array[] = [];
let g_ln2Weights:    Float32Array[] = [];
let g_ln2Biases:     Float32Array[] = [];
let g_cAttnWeights:  Float32Array[] = [];
let g_cAttnBiases:   Float32Array[] = [];
let g_cProjWeights:  Float32Array[] = [];
let g_cProjBiases:   Float32Array[] = [];
let g_cFcWeights:    Float32Array[] = [];
let g_cFcBiases:     Float32Array[] = [];
let g_cFcProjWeights: Float32Array[] = [];
let g_cFcProjBiases:  Float32Array[] = [];
let g_lnFWeight: Float32Array = new Float32Array(0);
let g_lnFBias:   Float32Array = new Float32Array(0);

// wywołaj raz z JS po załadowaniu wag
export function initWeights(
  wte: Float32Array, wpe: Float32Array,
  ln1w: Float32Array, ln1b: Float32Array,
  ln2w: Float32Array, ln2b: Float32Array,
  cattnW: Float32Array, cattnB: Float32Array,
  cprojW: Float32Array, cprojB: Float32Array,
  cfcW: Float32Array,   cfcB: Float32Array,
  cfcprojW: Float32Array, cfcprojB: Float32Array,
  lnfw: Float32Array, lnfb: Float32Array,
  layerIndex: i32,
): void {
  if (layerIndex === 0) {
    g_wte = wte;
    g_wpe = wpe;
    g_lnFWeight = lnfw;
    g_lnFBias   = lnfb;
    g_ln1Weights    = [];
    g_ln1Biases     = [];
    g_ln2Weights    = [];
    g_ln2Biases     = [];
    g_cAttnWeights  = [];
    g_cAttnBiases   = [];
    g_cProjWeights  = [];
    g_cProjBiases   = [];
    g_cFcWeights    = [];
    g_cFcBiases     = [];
    g_cFcProjWeights = [];
    g_cFcProjBiases  = [];
  }
  g_ln1Weights.push(ln1w);
  g_ln1Biases.push(ln1b);
  g_ln2Weights.push(ln2w);
  g_ln2Biases.push(ln2b);
  g_cAttnWeights.push(cattnW);
  g_cAttnBiases.push(cattnB);
  g_cProjWeights.push(cprojW);
  g_cProjBiases.push(cprojB);
  g_cFcWeights.push(cfcW);
  g_cFcBiases.push(cfcB);
  g_cFcProjWeights.push(cfcprojW);
  g_cFcProjBiases.push(cfcprojB);
}

// wywołaj dla każdego kroku generacji
export function runForward(tokens: Int32Array, out: Float32Array): void {
  forward(
    tokens,
    g_wte, g_wpe,
    g_ln1Weights,    g_ln1Biases,
    g_ln2Weights,    g_ln2Biases,
    g_cAttnWeights,  g_cAttnBiases,
    g_cProjWeights,  g_cProjBiases,
    g_cFcWeights,    g_cFcBiases,
    g_cFcProjWeights, g_cFcProjBiases,
    g_lnFWeight, g_lnFBias,
    out,
  );
}
