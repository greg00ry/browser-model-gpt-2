#!/usr/bin/env python3
"""
Benchmark Python GPT-2 Small — punkt odniesienia dla JS/Wasm.
Używa HuggingFace transformers z PyTorch (CPU, float32 — te same warunki co JS/Wasm).
"""

import time
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROMPT      = sys.argv[1] if len(sys.argv) > 1 else "The brain is"
MAX_TOKENS  = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TEMPERATURE = 1.0
TOP_K       = 40

print(f"[Python] Loading GPT-2 Small (HuggingFace)...")
t0 = time.time()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
model.eval()
print(f"[Python] Loaded in {time.time()-t0:.2f}s")
print(f"[Python] Backend: PyTorch {torch.__version__}, CPU, float32")
print(f"[Python] Prompt: \"{PROMPT}\"")

input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
generated = input_ids.clone()

print(f"  output: ", end="", flush=True)
t0 = time.time()

with torch.no_grad():
    for _ in range(MAX_TOKENS):
        outputs = model(generated)
        logits  = outputs.logits[:, -1, :] / TEMPERATURE   # tylko ostatni token

        # top-k sampling
        top_k_vals, top_k_idx = torch.topk(logits, TOP_K, dim=-1)
        probs = torch.softmax(top_k_vals, dim=-1)
        chosen = torch.multinomial(probs, num_samples=1)
        next_token = top_k_idx.gather(-1, chosen)

        generated = torch.cat([generated, next_token], dim=-1)
        word = tokenizer.decode(next_token[0])
        print(word, end="", flush=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

elapsed = time.time() - t0
tps = MAX_TOKENS / elapsed
print()
print(f"[Python] {MAX_TOKENS} tokens in {elapsed:.2f}s = {tps:.2f} tok/s")
