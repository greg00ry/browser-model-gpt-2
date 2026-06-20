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
  newF32(length: number): number;
  dataPtr(arr: number): number;
  newI32(length: number): number;
  i32DataPtr(arr: number): number;
  __pin(ptr: number): number;
  __unpin(ptr: number): void;
}

export class WasmRunner {
  private exp!: AsmExports;
  private outObj: number = 0;   // AS Float32Array object pointer (dla runForward)
  private outRaw: number = 0;   // raw data pointer (dla odczytu wyników)
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
  forward(tokens: number[]): Tensor {
    const { exp, vocabSize } = this;
    const seq = tokens.length;

    // (re)alokuj bufor wyjściowy jeśli seq się zmienił
    if (seq !== this.seqLen) {
      if (this.outObj) exp.__unpin(this.outObj);
      this.outObj = exp.__pin(exp.newF32(seq * vocabSize));   // managed object
      this.outRaw = exp.dataPtr(this.outObj);                  // raw data ptr
      this.seqLen = seq;
    }

    // utwórz Int32Array z tokenami w pamięci Wasm
    const tokArr = exp.__pin(exp.newI32(seq));
    new Int32Array(exp.memory.buffer, exp.i32DataPtr(tokArr), seq).set(tokens);

    exp.runForward(tokArr, this.outObj);  // przekazuj managed object, nie raw ptr
    exp.__unpin(tokArr);

    // odczytaj wynik z raw data pointer
    const view = new Float32Array(exp.memory.buffer, this.outRaw, seq * vocabSize);
    return new Tensor(new Float32Array(view), [seq, vocabSize]);
  }

  private loadWeights(w: GPT2Weights): void {
    const { exp } = this;

    // alokuje Float32Array w Wasm, kopiuje dane JS → Wasm, zwraca obiekt pointer (pinned)
    const copyToWasm = (data: Float32Array): number => {
      const arr = exp.__pin(exp.newF32(data.length));
      const raw = exp.dataPtr(arr);
      new Float32Array(exp.memory.buffer, raw, data.length).set(data);
      return arr;
    };

    const wtePtr  = copyToWasm(w.wte.data);
    const wpePtr  = copyToWasm(w.wpe.data);
    const lnfwPtr = copyToWasm(w.lnFWeight.data);
    const lnfbPtr = copyToWasm(w.lnFBias.data);

    for (let i = 0; i < w.blocks.length; i++) {
      const b = w.blocks[i];
      exp.initWeights(
        wtePtr, wpePtr,
        copyToWasm(b.ln1Weight.data),      copyToWasm(b.ln1Bias.data),
        copyToWasm(b.ln2Weight.data),      copyToWasm(b.ln2Bias.data),
        copyToWasm(b.attn.cAttnWeight.data), copyToWasm(b.attn.cAttnBias.data),
        copyToWasm(b.attn.cProjWeight.data), copyToWasm(b.attn.cProjBias.data),
        copyToWasm(b.mlp.cFcWeight.data),  copyToWasm(b.mlp.cFcBias.data),
        copyToWasm(b.mlp.cProjWeight.data), copyToWasm(b.mlp.cProjBias.data),
        lnfwPtr, lnfbPtr,
        i,
      );
    }
  }
}
