#!/usr/bin/env python3
"""
GPT-2 Small od zera w numpy — uczciwe porównanie z JS/Wasm.
Ten sam algorytm co assembly/ i src/, bez PyTorch ani HuggingFace.
"""

import sys
import time
import struct
import json
import numpy as np
import tiktoken

WEIGHTS_PATH = sys.argv[1] if len(sys.argv) > 1 else "weights/gpt2.safetensors"
PROMPT       = sys.argv[2] if len(sys.argv) > 2 else "The brain is"
MAX_TOKENS   = int(sys.argv[3]) if len(sys.argv) > 3 else 20
TEMPERATURE  = 1.0
TOP_K        = 40

# ── loader safetensors ────────────────────────────────────────────────────────

def load_safetensors(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header     = json.loads(f.read(header_len))
        data_start = 8 + header_len
        raw        = f.read()

    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype  = meta["dtype"]
        shape  = meta["shape"]
        b0, b1 = meta["data_offsets"]
        buf    = raw[b0:b1]
        if dtype == "F32":
            arr = np.frombuffer(buf, dtype=np.float32).reshape(shape)
        else:
            raise ValueError(f"unsupported dtype: {dtype}")
        tensors[name] = arr
    return tensors

# ── primitives ────────────────────────────────────────────────────────────────

def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return w * (x - mean) / np.sqrt(var + eps) + b

# ── attention ─────────────────────────────────────────────────────────────────

def attention(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, n_heads):
    seq, d_model = x.shape
    d_head = d_model // n_heads

    qkv  = x @ c_attn_w + c_attn_b          # [seq, 3*d_model]
    q, k, v = np.split(qkv, 3, axis=-1)     # każdy [seq, d_model]

    # split heads: [seq, d_model] -> [n_heads, seq, d_head]
    def split_heads(t):
        return t.reshape(seq, n_heads, d_head).transpose(1, 0, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    # scores: [n_heads, seq, seq]
    scale  = 1.0 / np.sqrt(d_head)
    scores = q @ k.transpose(0, 2, 1) * scale

    # causal mask
    mask = np.triu(np.full((seq, seq), -np.inf), k=1)
    scores = scores + mask

    scores = softmax(scores)

    # attended: [n_heads, seq, d_head] -> [seq, d_model]
    out = (scores @ v).transpose(1, 0, 2).reshape(seq, d_model)

    return out @ c_proj_w + c_proj_b

# ── feedforward ───────────────────────────────────────────────────────────────

def feedforward(x, c_fc_w, c_fc_b, c_proj_w, c_proj_b):
    return gelu(x @ c_fc_w + c_fc_b) @ c_proj_w + c_proj_b

# ── transformer block ─────────────────────────────────────────────────────────

def transformer_block(x, block, n_heads):
    x = x + attention(
        layer_norm(x, block["ln_1.weight"], block["ln_1.bias"]),
        block["attn.c_attn.weight"], block["attn.c_attn.bias"],
        block["attn.c_proj.weight"], block["attn.c_proj.bias"],
        n_heads,
    )
    x = x + feedforward(
        layer_norm(x, block["ln_2.weight"], block["ln_2.bias"]),
        block["mlp.c_fc.weight"],   block["mlp.c_fc.bias"],
        block["mlp.c_proj.weight"], block["mlp.c_proj.bias"],
    )
    return x

# ── forward ───────────────────────────────────────────────────────────────────

def forward(tokens, weights, n_layers=12, n_heads=12):
    wte    = weights["wte.weight"]    # [vocab, d_model]
    wpe    = weights["wpe.weight"]    # [ctx, d_model]
    ln_f_w = weights["ln_f.weight"]
    ln_f_b = weights["ln_f.bias"]

    x = wte[tokens] + wpe[np.arange(len(tokens))]

    for i in range(n_layers):
        block = {k.split(f"h.{i}.")[1]: v
                 for k, v in weights.items()
                 if k.startswith(f"h.{i}.")}
        x = transformer_block(x, block, n_heads)

    x = layer_norm(x, ln_f_w, ln_f_b)

    # lm_head — tylko ostatni token (weight tying: lm_head = wte^T)
    return x[-1] @ wte.T    # [vocab]

# ── main ──────────────────────────────────────────────────────────────────────

print("[numpy] Loading weights...")
t0 = time.time()
weights = load_safetensors(WEIGHTS_PATH)
print(f"[numpy] Loaded in {time.time()-t0:.2f}s")
print(f"[numpy] Backend: numpy {np.__version__}, CPU, float32")
print(f"[numpy] Prompt: \"{PROMPT}\"")

enc    = tiktoken.get_encoding("gpt2")
tokens = enc.encode(PROMPT)

print("  output: ", end="", flush=True)
t0 = time.time()

for _ in range(MAX_TOKENS):
    logits = forward(tokens, weights)

    # top-k sampling
    top_k_idx  = np.argpartition(logits, -TOP_K)[-TOP_K:]
    top_k_vals = logits[top_k_idx]
    top_k_vals = top_k_vals - top_k_vals.max()
    probs      = np.exp(top_k_vals / TEMPERATURE)
    probs      = probs / probs.sum()
    next_token = np.random.choice(top_k_idx, p=probs)

    tokens.append(int(next_token))
    print(enc.decode([int(next_token)]), end="", flush=True)

    if next_token == 50256:
        break

elapsed = time.time() - t0
tps = MAX_TOKENS / elapsed
print()
print(f"[numpy] {MAX_TOKENS} tokens in {elapsed:.2f}s = {tps:.2f} tok/s")
