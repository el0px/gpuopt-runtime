"""Reusable, correctness-gated CUDA Graph decoding for Hugging Face models."""

from __future__ import annotations

import gc
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Mapping

import torch
from transformers import StaticCache


def bucket_capacity(value: int, minimum: int = 32) -> int:
    """Round a positive capacity to a reusable power-of-two bucket."""
    if value < 1:
        raise ValueError("capacity must be positive")
    return max(minimum, 1 << (value - 1).bit_length())


@dataclass(frozen=True)
class SafetyPolicy:
    max_vram_percent: float = 85.0
    min_free_mib: float = 1024.0

    def __post_init__(self) -> None:
        if not 50.0 <= self.max_vram_percent <= 95.0:
            raise ValueError("max_vram_percent must be between 50 and 95")
        if self.min_free_mib < 256.0:
            raise ValueError("min_free_mib must be at least 256")


@dataclass(frozen=True)
class GraphKey:
    model_name: str
    device_index: int
    batch_size: int
    cache_capacity: int
    generation_capacity: int
    dtype: str
    model_identity: int = field(repr=False)


@dataclass
class GraphGenerationResult:
    token_ids: torch.Tensor
    backend: str
    key: GraphKey
    cache_hit: bool
    capture_ms: float
    prefill_ms: float
    decode_ms: float
    validation_ms: float
    validated: bool
    peak_vram_mib: float
    free_vram_mib: float
    used_vram_percent: float
    fallback_reason: str | None = None

    @property
    def decode_tokens_per_second(self) -> float:
        steps = max(0, int(self.token_ids.shape[1]) - 1)
        return steps * 1000.0 / self.decode_ms if steps and self.decode_ms > 0 else 0.0

    def metrics(self) -> dict:
        key = asdict(self.key)
        key.pop("model_identity", None)
        return {
            "backend": self.backend,
            "key": key,
            "cache_hit": self.cache_hit,
            "capture_ms": self.capture_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "validation_ms": self.validation_ms,
            "validated": self.validated,
            "peak_vram_mib": self.peak_vram_mib,
            "free_vram_mib": self.free_vram_mib,
            "used_vram_percent": self.used_vram_percent,
            "fallback_reason": self.fallback_reason,
            "decode_tokens_per_second": self.decode_tokens_per_second,
        }


class GraphValidationError(RuntimeError):
    def __init__(self, matching: int, total: int, first_mismatch: int | None):
        super().__init__(
            f"CUDA Graph output failed exact validation: {matching}/{total} tokens match; "
            f"first mismatch={first_mismatch}"
        )
        self.matching = matching
        self.total = total
        self.first_mismatch = first_mismatch


def _memory_snapshot() -> tuple[float, float, float]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    free_mib = free / 1048576.0
    used_percent = 100.0 * (total - free) / total
    peak_mib = torch.cuda.max_memory_allocated() / 1048576.0
    return free_mib, used_percent, peak_mib


class _GraphRunner:
    def __init__(self, model, key: GraphKey, policy: SafetyPolicy):
        self.model = model
        self.key = key
        self.policy = policy
        self.cache = StaticCache(config=model.config, max_cache_len=key.cache_capacity)
        device = torch.device("cuda", key.device_index)
        self.static_token = torch.empty((key.batch_size, 1), device=device, dtype=torch.long)
        self.static_position = torch.empty((1,), device=device, dtype=torch.long)
        self.static_step = torch.empty((1,), device=device, dtype=torch.long)
        self.generated = torch.empty(
            (key.batch_size, key.generation_capacity), device=device, dtype=torch.long
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_ms = 0.0
        self.validated_shapes: set[tuple[int, int]] = set()

    @staticmethod
    def _normalize_inputs(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized = dict(inputs)
        if "input_ids" not in normalized:
            raise ValueError("inputs must contain input_ids")
        if normalized["input_ids"].ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        mask = normalized.get("attention_mask")
        if mask is not None and not bool(torch.all(mask == 1).item()):
            raise ValueError("v0.1 supports only unpadded prompts")
        return normalized

    def _prefill(self, inputs: dict[str, torch.Tensor], reset: bool) -> tuple[torch.Tensor, float]:
        if reset:
            self.cache.reset()
        prompt_tokens = int(inputs["input_ids"].shape[1])
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = self.model(
            **inputs,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=torch.arange(prompt_tokens, device=inputs["input_ids"].device),
            logits_to_keep=1,
        )
        torch.cuda.synchronize()
        return outputs.logits[:, -1:].argmax(dim=-1), (time.perf_counter() - start) * 1000.0

    def _set_controls(self, first_token: torch.Tensor, prompt_tokens: int) -> None:
        self.static_token.copy_(first_token)
        self.static_position.fill_(prompt_tokens)
        self.static_step.fill_(1)
        self.generated.zero_()
        self.generated[:, :1].copy_(first_token)

    def _capture(self, first_token: torch.Tensor, prompt_tokens: int) -> None:
        self._set_controls(first_token, prompt_tokens)
        self.model(
            input_ids=self.static_token,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=self.static_position,
            logits_to_keep=1,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            logits = self.model(
                input_ids=self.static_token,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=self.static_position,
                logits_to_keep=1,
            ).logits
            next_token = logits[:, -1:].argmax(dim=-1)
            indices = self.static_step.view(1, 1).expand(self.key.batch_size, 1)
            self.generated.scatter_(1, indices, next_token)
            self.static_token.copy_(next_token)
            self.static_position.add_(1)
            self.static_step.add_(1)
        torch.cuda.synchronize()
        self.capture_ms = (time.perf_counter() - start) * 1000.0

    def _eager_reference(
        self, inputs: dict[str, torch.Tensor], new_tokens: int
    ) -> torch.Tensor:
        prompt_tokens = int(inputs["input_ids"].shape[1])
        cache = StaticCache(config=self.model.config, max_cache_len=self.key.cache_capacity)
        position = torch.empty((1,), device=inputs["input_ids"].device, dtype=torch.long)
        outputs = self.model(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(prompt_tokens, device=inputs["input_ids"].device),
            logits_to_keep=1,
        )
        token = outputs.logits[:, -1:].argmax(dim=-1)
        generated = [token]
        for step in range(new_tokens - 1):
            position.fill_(prompt_tokens + step)
            outputs = self.model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                cache_position=position,
                logits_to_keep=1,
            )
            token = outputs.logits[:, -1:].argmax(dim=-1)
            generated.append(token)
        torch.cuda.synchronize()
        return torch.cat(generated, dim=1)

    def generate(
        self,
        inputs: Mapping[str, torch.Tensor],
        new_tokens: int,
        cache_hit: bool,
        validate: bool,
    ) -> GraphGenerationResult:
        normalized = self._normalize_inputs(inputs)
        prompt_tokens = int(normalized["input_ids"].shape[1])
        batch = int(normalized["input_ids"].shape[0])
        if batch != self.key.batch_size:
            raise ValueError("batch size does not match graph key")
        if new_tokens < 2 or new_tokens > self.key.generation_capacity:
            raise ValueError("new_tokens is outside this graph's capacity")
        if prompt_tokens + new_tokens > self.key.cache_capacity:
            raise ValueError("prompt plus generation exceeds cache capacity")

        self.model.config._attn_implementation = "sdpa"
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            first_token, initial_prefill_ms = self._prefill(
                normalized, reset=self.graph is not None
            )
            captured_now = self.graph is None
            if captured_now:
                self._capture(first_token, prompt_tokens)
                # Warm-up and capture mutate the bound cache. Always rebuild the
                # prompt state before using a newly captured graph.
                first_token, prefill_ms = self._prefill(normalized, reset=True)
            else:
                prefill_ms = initial_prefill_ms

            self._set_controls(first_token, prompt_tokens)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _step in range(new_tokens - 1):
                self.graph.replay()
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - start) * 1000.0
            token_ids = self.generated[:, :new_tokens].clone()

            shape = (prompt_tokens, new_tokens)
            validation_ms = 0.0
            validated = shape in self.validated_shapes
            if validate and not validated:
                start = time.perf_counter()
                reference = self._eager_reference(normalized, new_tokens)
                validation_ms = (time.perf_counter() - start) * 1000.0
                equal = token_ids == reference
                matching = int(equal.sum().item())
                total = int(equal.numel())
                if matching != total:
                    mismatch = (~equal).nonzero(as_tuple=False)
                    first = int(mismatch[0, 1].item()) if int(mismatch.numel()) else None
                    raise GraphValidationError(matching, total, first)
                self.validated_shapes.add(shape)
                validated = True

        free_mib, used_percent, peak_mib = _memory_snapshot()
        if used_percent > self.policy.max_vram_percent or free_mib < self.policy.min_free_mib:
            raise RuntimeError(
                f"VRAM policy exceeded: used={used_percent:.2f}%, free={free_mib:.1f} MiB"
            )
        return GraphGenerationResult(
            token_ids=token_ids,
            backend="cuda_graph_sdpa",
            key=self.key,
            cache_hit=cache_hit,
            capture_ms=self.capture_ms if captured_now else 0.0,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            validation_ms=validation_ms,
            validated=validated,
            peak_vram_mib=peak_mib,
            free_vram_mib=free_mib,
            used_vram_percent=used_percent,
        )


def _eager_fallback(
    model,
    inputs: Mapping[str, torch.Tensor],
    new_tokens: int,
    key: GraphKey,
    policy: SafetyPolicy,
    reason: Exception,
    cache_hit: bool,
) -> GraphGenerationResult:
    """Return a safe eager StaticCache result when graph execution fails."""
    normalized = _GraphRunner._normalize_inputs(inputs)
    prompt_tokens = int(normalized["input_ids"].shape[1])
    cache = StaticCache(config=model.config, max_cache_len=key.cache_capacity)
    position = torch.empty((1,), device=normalized["input_ids"].device, dtype=torch.long)
    model.config._attn_implementation = "sdpa"
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(
            **normalized,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(
                prompt_tokens, device=normalized["input_ids"].device
            ),
            logits_to_keep=1,
        )
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - start) * 1000.0
        token = outputs.logits[:, -1:].argmax(dim=-1)
        generated = [token]
        start = time.perf_counter()
        for step in range(new_tokens - 1):
            position.fill_(prompt_tokens + step)
            outputs = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                cache_position=position,
                logits_to_keep=1,
            )
            token = outputs.logits[:, -1:].argmax(dim=-1)
            generated.append(token)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - start) * 1000.0
        token_ids = torch.cat(generated, dim=1)

    free_mib, used_percent, peak_mib = _memory_snapshot()
    if used_percent > policy.max_vram_percent or free_mib < policy.min_free_mib:
        raise RuntimeError(
            f"VRAM policy exceeded during fallback: used={used_percent:.2f}%, "
            f"free={free_mib:.1f} MiB"
        ) from reason
    return GraphGenerationResult(
        token_ids=token_ids,
        backend="static_sdpa_fallback",
        key=key,
        cache_hit=cache_hit,
        capture_ms=0.0,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        validation_ms=0.0,
        validated=True,
        peak_vram_mib=peak_mib,
        free_vram_mib=free_mib,
        used_vram_percent=used_percent,
        fallback_reason=f"{type(reason).__name__}: {reason}",
    )


class CUDAGraphPool:
    """Thread-safe LRU of captured greedy-decode graphs.

    v0.1 serializes access to each pool and supports unpadded prompts. Use one
    pool per worker process; do not share a model across independent processes.
    """

    def __init__(
        self,
        max_entries: int = 4,
        safety: SafetyPolicy | None = None,
        validate_first_shape: bool = True,
        fallback_on_error: bool = True,
    ):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.safety = safety or SafetyPolicy()
        self.validate_first_shape = validate_first_shape
        self.fallback_on_error = fallback_on_error
        self._entries: OrderedDict[GraphKey, _GraphRunner] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.fallbacks = 0

    @staticmethod
    def _key(model, inputs: Mapping[str, torch.Tensor], new_tokens: int) -> GraphKey:
        input_ids = inputs["input_ids"]
        if not input_ids.is_cuda:
            raise ValueError("input_ids must be on CUDA")
        parameter = next(model.parameters())
        if not parameter.is_cuda:
            raise ValueError("model must be on CUDA")
        device_index = parameter.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        prompt_tokens = int(input_ids.shape[1])
        model_name = str(getattr(model.config, "_name_or_path", model.__class__.__name__))
        return GraphKey(
            model_name=model_name,
            device_index=device_index,
            batch_size=int(input_ids.shape[0]),
            cache_capacity=bucket_capacity(prompt_tokens + new_tokens),
            generation_capacity=bucket_capacity(new_tokens),
            dtype=str(parameter.dtype),
            model_identity=id(model),
        )

    def generate_greedy(
        self,
        model,
        inputs: Mapping[str, torch.Tensor],
        new_tokens: int,
        validate: bool | None = None,
    ) -> GraphGenerationResult:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        key = self._key(model, inputs, new_tokens)
        should_validate = self.validate_first_shape if validate is None else validate
        with self._lock:
            runner = self._entries.get(key)
            cache_hit = runner is not None
            if cache_hit:
                self.hits += 1
                self._entries.move_to_end(key)
            else:
                self.misses += 1
                runner = _GraphRunner(model, key, self.safety)
                self._entries[key] = runner
                if len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
                    self.evictions += 1
                    gc.collect()
                    torch.cuda.empty_cache()
            try:
                return runner.generate(inputs, new_tokens, cache_hit, should_validate)
            except (GraphValidationError, RuntimeError) as error:
                # Never reuse a capture that failed validation or execution.
                self._entries.pop(key, None)
                gc.collect()
                torch.cuda.empty_cache()
                if not self.fallback_on_error:
                    raise
                self.fallbacks += 1
                return _eager_fallback(
                    model, inputs, new_tokens, key, self.safety, error, cache_hit
                )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "fallbacks": self.fallbacks,
            }
