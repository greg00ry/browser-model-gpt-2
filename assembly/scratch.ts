// Wszystkie bufory pośrednie alokowane raz — zero alokacji w hot path
export class Scratch {
  // attention
  qkv:      Float32Array; // [seq, 3*dModel]
  q:        Float32Array; // [seq, dModel]
  k:        Float32Array; // [seq, dModel]
  v:        Float32Array; // [seq, dModel]
  qH:       Float32Array; // [nHeads, seq, dHead]
  kH:       Float32Array; // [nHeads, seq, dHead]
  vH:       Float32Array; // [nHeads, seq, dHead]
  kT:       Float32Array; // [nHeads, dHead, seq]
  scores:   Float32Array; // [nHeads, seq, seq]
  attended: Float32Array; // [nHeads, seq, dHead]
  merged:   Float32Array; // [seq, dModel]
  mask:     Float32Array; // [seq, seq] — refill gdy seq się zmienia
  // feedforward
  hidden:   Float32Array; // [seq, 4*dModel]
  // block
  tmp:      Float32Array; // [seq, dModel]
  attnOut:  Float32Array; // [seq, dModel]
  mlpOut:   Float32Array; // [seq, dModel]
  // model
  x:        Float32Array; // [seq, dModel]
  wteT:     Float32Array; // [dModel, vocabSize] — pre-computed raz
  maskSeq:  i32 = 0;

  constructor(maxSeq: i32, nHeads: i32, dModel: i32, vocabSize: i32) {
    const dHead: i32  = dModel / nHeads;
    const dInner: i32 = dModel * 4;
    this.qkv      = new Float32Array(maxSeq * dModel * 3);
    this.q        = new Float32Array(maxSeq * dModel);
    this.k        = new Float32Array(maxSeq * dModel);
    this.v        = new Float32Array(maxSeq * dModel);
    this.qH       = new Float32Array(nHeads * maxSeq * dHead);
    this.kH       = new Float32Array(nHeads * maxSeq * dHead);
    this.vH       = new Float32Array(nHeads * maxSeq * dHead);
    this.kT       = new Float32Array(nHeads * dHead * maxSeq);
    this.scores   = new Float32Array(nHeads * maxSeq * maxSeq);
    this.attended = new Float32Array(nHeads * maxSeq * dHead);
    this.merged   = new Float32Array(maxSeq * dModel);
    this.mask     = new Float32Array(maxSeq * maxSeq);
    this.hidden   = new Float32Array(maxSeq * dInner);
    this.tmp      = new Float32Array(maxSeq * dModel);
    this.attnOut  = new Float32Array(maxSeq * dModel);
    this.mlpOut   = new Float32Array(maxSeq * dModel);
    this.x        = new Float32Array(maxSeq * dModel);
    this.wteT     = new Float32Array(dModel * vocabSize);
  }
}

let g: Scratch = new Scratch(1, 12, 768, 1);

export function initScratch(maxSeq: i32): void {
  g = new Scratch(maxSeq, 12, 768, 50257);
}

export function getScratch(): Scratch { return g; }
