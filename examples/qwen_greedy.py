"""Minimal GPUOpt Runtime example."""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpuopt import CUDAGraphPool, SafetyPolicy


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map={"": "cuda"}, attn_implementation="sdpa"
).eval()
inputs = tokenizer("Explain why GPU launch overhead matters.", return_tensors="pt").to("cuda")

pool = CUDAGraphPool(
    max_entries=4,
    safety=SafetyPolicy(max_vram_percent=85, min_free_mib=1024),
)
for request in range(2):
    result = pool.generate_greedy(model, inputs, new_tokens=64)
    print(f"request={request + 1}: {tokenizer.decode(result.token_ids[0])}")
    print(json.dumps(result.metrics(), indent=2))
print(pool.stats())

