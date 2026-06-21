# GPUOpt Runtime

I built this because a lot of smaller LLM workloads waste time launching the
same CUDA operations over and over. GPUOpt captures that decode work once,
checks the output, and reuses it when the next request has a compatible shape.

It is not a driver replacement and it does not pretend every model gets faster.
If graph capture or validation fails, it drops back to eager SDPA instead of
silently returning questionable output.

## What it does

- pools CUDA Graphs by model, GPU, dtype, batch, and capacity
- validates every token for a new shape before trusting it
- resets and refills the KV cache after capture
- reuses compatible graphs with an LRU cache
- tracks capture time, cache hits, speed, and VRAM
- enforces configurable VRAM limits

## Install

For development:

```bash
python -m pip install -e ".[examples,test]"
```

Or install the built wheel:

```bash
python -m pip install gpuopt_runtime-0.1.0-py3-none-any.whl
```

## Use it

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpuopt import CUDAGraphPool

name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name,
    dtype=torch.float16,
    device_map={"": "cuda"},
    attn_implementation="sdpa",
).eval()
inputs = tokenizer("Explain GPU launch overhead.", return_tensors="pt").to("cuda")

pool = CUDAGraphPool(max_entries=4)
result = pool.generate_greedy(model, inputs, new_tokens=64)
print(tokenizer.decode(result.token_ids[0]))
print(result.metrics())
```

The first compatible request captures and validates the graph. Later requests
reuse it without paying that cost again.

## Results so far

| GPU | Eager SDPA | CUDA Graph | Steady state speedup | Exact |
|---|---:|---:|---:|---:|
| RTX 5070 | 60.06 tok/s | 281.18 tok/s | 4.68x | 64/64 |
| A100 1g.10gb MIG | 70.55 tok/s | 130.14 tok/s | 1.84x | 64/64 |

These are Qwen2.5-0.5B greedy decode measurements, not a promise for every GPU
or model. Cold requests also pay capture and validation costs.

## Current limits

Version 0.1 is for NVIDIA CUDA, Hugging Face causal models, greedy decoding, and
unpadded prompts. Sampling, padded batches, continuous batching, and serving
across multiple GPUs still need proper tests before I call them supported.

Licensed under Apache-2.0.
