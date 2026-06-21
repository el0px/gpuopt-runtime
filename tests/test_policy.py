import pytest
import torch

from gpuopt import GraphGenerationResult, GraphKey, SafetyPolicy, bucket_capacity


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 32), (32, 32), (33, 64), (80, 128), (129, 256)],
)
def test_bucket_capacity(value, expected):
    assert bucket_capacity(value) == expected


def test_bucket_rejects_nonpositive_values():
    with pytest.raises(ValueError):
        bucket_capacity(0)


def test_safety_policy_validation():
    assert SafetyPolicy().max_vram_percent == 85.0
    with pytest.raises(ValueError):
        SafetyPolicy(max_vram_percent=99)
    with pytest.raises(ValueError):
        SafetyPolicy(min_free_mib=128)


def test_metrics_are_small_and_do_not_expose_process_identity():
    key = GraphKey("model", 0, 1, 64, 32, "torch.float16", 12345)
    result = GraphGenerationResult(
        token_ids=torch.zeros((1, 4), dtype=torch.long),
        backend="static_sdpa_fallback",
        key=key,
        cache_hit=True,
        capture_ms=0.0,
        prefill_ms=1.0,
        decode_ms=3.0,
        validation_ms=0.0,
        validated=True,
        peak_vram_mib=100.0,
        free_vram_mib=1000.0,
        used_vram_percent=10.0,
        fallback_reason="RuntimeError: test",
    )
    metrics = result.metrics()
    assert "token_ids" not in metrics
    assert "model_identity" not in metrics["key"]
    assert metrics["fallback_reason"] == "RuntimeError: test"
