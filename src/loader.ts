import { Tensor } from "./tensor.js";
import { GPT2Weights, GPT2Config } from "./model.js";
import { BlockWeights } from "./block.js";

interface TensorMeta {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

type SafetensorsHeader = Record<string, TensorMeta | Record<string, string>>;

// accepts an ArrayBuffer — works in browser (fetch) and Node.js (fs.readFile)
export function loadSafetensors(buffer: ArrayBuffer, config: GPT2Config): GPT2Weights {
  const view = new DataView(buffer);

  // 8-byte little-endian uint64 = JSON header length
  const headerLen = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen));
  const header = JSON.parse(headerJson) as SafetensorsHeader;

  // data section starts after 8 (length prefix) + headerLen bytes
  // safetensors pads the header to guarantee 8-byte alignment here
  const dataStart = 8 + headerLen;

  function tensor(name: string): Tensor {
    const meta = header[name] as TensorMeta | undefined;
    if (!meta) throw new Error(`Missing tensor: ${name}`);
    if (meta.dtype !== "F32") throw new Error(`Expected F32, got ${meta.dtype} for ${name}`);
    const [start, end] = meta.data_offsets;
    const src = new Uint8Array(buffer, dataStart + start, end - start);
    const data = new Float32Array(src.length / 4);
    new Uint8Array(data.buffer).set(src);
    return new Tensor(data, meta.shape);
  }

  const blocks: BlockWeights[] = [];
  for (let i = 0; i < config.nLayers; i++) {
    const h = `h.${i}`;
    blocks.push({
      ln1Weight:   tensor(`${h}.ln_1.weight`),
      ln1Bias:     tensor(`${h}.ln_1.bias`),
      ln2Weight:   tensor(`${h}.ln_2.weight`),
      ln2Bias:     tensor(`${h}.ln_2.bias`),
      attn: {
        cAttnWeight: tensor(`${h}.attn.c_attn.weight`),
        cAttnBias:   tensor(`${h}.attn.c_attn.bias`),
        cProjWeight: tensor(`${h}.attn.c_proj.weight`),
        cProjBias:   tensor(`${h}.attn.c_proj.bias`),
      },
      mlp: {
        cFcWeight:   tensor(`${h}.mlp.c_fc.weight`),
        cFcBias:     tensor(`${h}.mlp.c_fc.bias`),
        cProjWeight: tensor(`${h}.mlp.c_proj.weight`),
        cProjBias:   tensor(`${h}.mlp.c_proj.bias`),
      },
    });
  }

  return {
    wte:       tensor("wte.weight"),
    wpe:       tensor("wpe.weight"),
    blocks,
    lnFWeight: tensor("ln_f.weight"),
    lnFBias:   tensor("ln_f.bias"),
  };
}

// Node.js helper — not used in browser
export async function loadFromFile(path: string, config: GPT2Config): Promise<GPT2Weights> {
  const { readFile } = await import("fs/promises");
  const bytes = await readFile(path);
  return loadSafetensors(bytes.buffer, config);
}
