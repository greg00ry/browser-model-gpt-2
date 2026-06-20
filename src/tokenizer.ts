import { get_encoding, Tiktoken } from "tiktoken";
import type { Tokenizer } from "./main.js";

let _enc: Tiktoken | null = null;

// lazy init — WASM loads once on first call
function enc(): Tiktoken {
  if (!_enc) _enc = get_encoding("gpt2");
  return _enc!;
}

export function encode(text: string): number[] {
  return Array.from(enc().encode(text));
}

export function decode(tokens: number[]): string {
  const bytes = enc().decode(new Uint32Array(tokens));
  return new TextDecoder().decode(bytes);
}

export function free(): void {
  _enc?.free();
  _enc = null;
}

export const tokenizer: Tokenizer = { encode, decode };
