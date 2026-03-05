from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest
import torch

from util.model_runtime import (
    bool_from_flag,
    configure_triton_tool_paths,
    fallback_average_precision,
    fallback_max_f1,
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


def test_configure_triton_tool_paths_sets_cuda_env_for_conda_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_root = tmp_path / "env"
    ptxas_path = env_root / "bin" / "ptxas"
    ptxas_path.parent.mkdir(parents=True, exist_ok=True)
    ptxas_path.write_text("", encoding="utf-8")
    cuda_header = env_root / "targets" / "x86_64-linux" / "include" / "cuda.h"
    cuda_header.parent.mkdir(parents=True, exist_ok=True)
    cuda_header.write_text("", encoding="utf-8")

    def _fake_which(name: str) -> str | None:
        if name == "ptxas":
            return str(ptxas_path)
        return None

    monkeypatch.setattr("util.model_runtime.shutil.which", _fake_which)
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    monkeypatch.delenv("TRITON_PTXAS_BLACKWELL_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(env_root))

    configure_triton_tool_paths()

    assert os.environ["TRITON_PTXAS_PATH"] == str(ptxas_path)
    assert os.environ["TRITON_PTXAS_BLACKWELL_PATH"] == str(ptxas_path)
    assert os.environ["CUDA_HOME"] == str(env_root.resolve())
    assert os.environ["CUDA_PATH"] == str(env_root.resolve())
    assert os.environ["CPATH"] == str(cuda_header.parent.resolve())


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
    with pytest.raises(ValueError, match="positive"):
        _ = fallback_max_f1(labels, probs)
    with pytest.raises(ValueError, match="Both positive and negative"):
        _ = fallback_roc_auc(labels, probs)


def test_fallback_max_f1_returns_high_score_for_separable_data() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    probs = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    max_f1 = fallback_max_f1(labels, probs)
    assert max_f1 == pytest.approx(1.0)
