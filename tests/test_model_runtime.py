from __future__ import annotations

from typing import Mapping

import numpy as np
import pytest
import torch

from util.model_runtime import (
    bool_from_flag,
    fallback_average_precision,
    fallback_roc_auc,
    is_compile_runtime_error,
    normalize_checkpoint_state_dict,
    resolve_mps_max_batch_size,
    resolve_num_workers,
    sigmoid_np,
)


def test_bool_from_flag_supports_int_and_bool() -> None:
    assert bool_from_flag(True) is True
    assert bool_from_flag(False) is False
    assert bool_from_flag(1) is True
    assert bool_from_flag(0) is False


def test_resolve_num_workers_auto_for_cpu_is_zero() -> None:
    resolved = resolve_num_workers("auto", device="cpu")
    assert resolved == 0


def test_resolve_num_workers_rejects_invalid_text() -> None:
    with pytest.raises(ValueError, match="integer or 'auto'"):
        _ = resolve_num_workers("oops", device="cuda")


def test_resolve_mps_max_batch_size_invalid_env_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_MPS_MAX_BATCH_SIZE", "bad")
    assert resolve_mps_max_batch_size("cnn", default_batch_size=2048) == 2048


def test_is_compile_runtime_error_matches_known_keyword() -> None:
    exc = RuntimeError("backend_hash failed in torch._inductor")
    assert is_compile_runtime_error(exc) is True


def test_normalize_checkpoint_state_dict_strips_orig_mod_prefix() -> None:
    raw_state: Mapping[str, torch.Tensor] = {
        "_orig_mod.layer.weight": torch.tensor([1.0]),
        "layer.bias": torch.tensor([2.0]),
    }
    normalized = normalize_checkpoint_state_dict(raw_state)
    assert "layer.weight" in normalized
    assert "layer.bias" in normalized


def test_sigmoid_np_large_values_are_finite() -> None:
    logits = np.array([-1000.0, 0.0, 1000.0], dtype=np.float64)
    probs = sigmoid_np(logits)
    assert np.isfinite(probs).all()
    assert probs[0] < 1e-6
    assert probs[2] > 1.0 - 1e-6


def test_fallback_metrics_raise_for_invalid_labels() -> None:
    labels = np.array([0, 0, 0], dtype=np.int32)
    probs = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="positive"):
        _ = fallback_average_precision(labels, probs)
    with pytest.raises(ValueError, match="Both positive and negative"):
        _ = fallback_roc_auc(labels, probs)
