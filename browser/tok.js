// tiktoken init shim dla native browser ESM
// ładuje tiktoken_bg.wasm ręcznie przez fetch — omija problem "import * from .wasm"
const bg = await import("../node_modules/tiktoken/tiktoken_bg.js");

const { instance } = await WebAssembly.instantiateStreaming(
  fetch("../node_modules/tiktoken/tiktoken_bg.wasm"),
  { "./tiktoken_bg.js": bg },
);
bg.__wbg_set_wasm(instance.exports);

const enc_data = await fetch("../node_modules/tiktoken/encoders/gpt2.json").then(r => r.json());
const enc = new bg.Tiktoken(enc_data.bpe_ranks, enc_data.special_tokens, enc_data.pat_str);

export const encode = (text) => Array.from(enc.encode(text));
export const decode = (tokens) => new TextDecoder().decode(enc.decode(new Uint32Array(tokens)));
