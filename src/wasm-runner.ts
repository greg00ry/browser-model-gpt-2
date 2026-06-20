import { readFile } from "fs/promises";
import { GPT2Weights, GPT2Config, GPT2_SMALL } from "./model.js";
import { Tensor } from "./tensor.js";

interface AsmExports {
  memory: WebAssembly.Memory;
  initWeights(
    wte: number, wpe: number,
    ln1w: number, ln1b: number,
    ln2w: number, ln2b: number,
    cattnW: number, cattnB: number,
    cprojW: number, cprojB: number,
    cfcW: number,   cfcB: number,
    cfcprojW: number, cfcprojB: number,
    lnfw: number, lnfb: number,
    layerIndex: number,
  ): void;
  runForward(tokens: number, out: number): void;
  allocF32(length: number): number;
  __newArray(id: number, values: ArrayLike<number>): number;
  __getArray(ptr: number): number[];
  __new(size: number, id: number): number;
  __pin(ptr: number): number;
  __unpin(ptr: number): void;
  readonly Int32Array_ID: number;
  readonly Float32Array_ID: number;
}

export class WasmRunner {
  private exp!: AsmExports;
  private outPtr!: number;
  private seqLen: number = 0;
  private readonly vocabSize: number;

  private constructor(private config: GPT2Config) {
    this.vocabSize = config.vocabSize;
  }

  static async load(
    wasmPath: string,
    weights: GPT2Weights,
    config: GPT2Config = GPT2_SMALL,
  ): Promise<WasmRunner> {
    const runner = new WasmRunner(config);
    const bytes = await readFile(wasmPath);
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { abort: () => { throw new Error("Wasm abort"); } },
    });
    runner.exp = instance.exports as unknown as AsmExports;
    runner.loadWeights(weights);
    return runner;
  }

  // tokens [seq] -> logits [seq, vocabSize] (taki sam interfejs jak TS forward())
  forward(tokens: number[], config: GPT2Config = this.config): Tensor {
    const { exp, vocabSize } = this;
    const seq = tokens.length;

    // (re)alokuj bufor wyjściowy jeśli seq się zmienił
    if (seq !== this.seqLen) {
      if (this.outPtr) exp.__unpin(this.outPtr);
      this.outPtr = exp.__pin(exp.allocF32(seq * vocabSize));
      this.seqLen = seq;
    }

    // utwórz Int32Array z tokenami w pamięci Wasm
    const tokPtr = exp.__pin(exp.__newArray(exp.Int32Array_ID, tokens));

    exp.runForward(tokPtr, this.outPtr);
    exp.__unpin(tokPtr);

    // odczytaj wynik z pamięci Wasm
    const view = new Float32Array(exp.memory.buffer, this.outPtr + 20, seq * vocabSize);
    return new Tensor(new Float32Array(view), [seq, vocabSize]);
  }

  private loadWeights(w: GPT2Weights): void {
    const { exp } = this;
    const pin = (arr: Float32Array) => {
      const ptr = exp.__pin(exp.allocF32(arr.length));
      new Float32Array(exp.memory.buffer, ptr + 20, arr.length).set(arr);
      return ptr;
    };

    const wtePtr = pin(w.wte.data);
    const wpePtr = pin(w.wpe.data);
    const lnfwPtr = pin(w.lnFWeight.data);
    const lnfbPtr = pin(w.lnFBias.data);

    for (let i = 0; i < w.blocks.length; i++) {
      const b = w.blocks[i];
      exp.initWeights(
        wtePtr, wpePtr,
        pin(b.ln1Weight.data), pin(b.ln1Bias.data),
        pin(b.ln2Weight.data), pin(b.ln2Bias.data),
        pin(b.attn.cAttnWeight.data), pin(b.attn.cAttnBias.data),
        pin(b.attn.cProjWeight.data), pin(b.attn.cProjBias.data),
        pin(b.mlp.cFcWeight.data),    pin(b.mlp.cFcBias.data),
        pin(b.mlp.cProjWeight.data),  pin(b.mlp.cProjBias.data),
        lnfwPtr, lnfbPtr,
        i,
      );
    }
  }
}
