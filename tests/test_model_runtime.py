from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pytest
import torch

import util.model_runtime as model_runtime
from util.model_runtime import (
    bool_from_flag,
    compile_model_with_fallback,
    configure_torch_compile_runtime,
    configure_triton_tool_paths,
    fallback_average_precision,
    fallback_max_f1,
    fallback_roc_auc,
    is_compile_runtime_error,
    log10_sigmoid_np,
    normalize_checkpoint_state_dict,
    probabilities_to_log10_scores_np,
    resolve_compile_enabled,
    resolve_auto_num_workers,
    resolve_mps_max_batch_size,
    resolve_num_workers,
    sigmoid_np,
)


@pytest.fixture(autouse=True)
def _reset_compile_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    prior_skip_dynamic_graphs: bool | None = None
    inductor_module = getattr(torch, "_inductor", None)
    if inductor_module is not None:
        config_obj = getattr(inductor_module, "config", None)
        triton_config_obj = getattr(config_obj, "triton", None)
        if triton_config_obj is not None:
            value_obj = getattr(
                triton_config_obj,
                "cudagraph_skip_dynamic_graphs",
                None,
            )
            if isinstance(value_obj, bool):
                prior_skip_dynamic_graphs = value_obj

    monkeypatch.delenv("INTRONMODEL_TORCH_COMPILE_STRATEGY", raising=False)
    monkeypatch.delenv("INTRONMODEL_TORCH_COMPILE_STICKY_MODE", raising=False)
    monkeypatch.delenv("INTRONMODEL_TORCH_COMPILE_DISABLED_MODES", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", raising=False)
    model_runtime._COMPILE_RUNTIME_CACHE_LOADED = False
    model_runtime._COMPILE_STICKY_MODE_CACHE = None
    model_runtime._COMPILE_DISABLED_MODES_CACHE.clear()
    yield
    if prior_skip_dynamic_graphs is not None:
        inductor_module = getattr(torch, "_inductor", None)
        if inductor_module is not None:
            config_obj = getattr(inductor_module, "config", None)
            triton_config_obj = getattr(config_obj, "triton", None)
            if triton_config_obj is not None:
                setattr(
                    triton_config_obj,
                    "cudagraph_skip_dynamic_graphs",
                    prior_skip_dynamic_graphs,
                )


def test_bool_from_flag_supports_int_and_bool() -> None:
    assert bool_from_flag(True) is True
    assert bool_from_flag(False) is False
    assert bool_from_flag(1) is True
    assert bool_from_flag(0) is False


def test_probabilities_to_log10_scores_np_converts_zero_and_one() -> None:
    scores = probabilities_to_log10_scores_np(
        np.asarray([0.0, 0.1, 1.0], dtype=np.float32)
    )

    assert np.isneginf(scores[0])
    assert scores[1] == pytest.approx(-1.0)
    assert scores[2] == pytest.approx(0.0)


def test_log10_sigmoid_np_matches_expected_probabilities() -> None:
    logits = np.asarray([0.0, math.log(9.0)], dtype=np.float32)

    scores = log10_sigmoid_np(logits)

    assert scores[0] == pytest.approx(np.log10(0.5), abs=1e-6)
    assert scores[1] == pytest.approx(np.log10(0.9), abs=1e-6)


def test_resolve_num_workers_auto_for_cpu_is_zero() -> None:
    resolved = resolve_num_workers("auto", device="cpu")
    assert resolved == 0


def test_resolve_num_workers_auto_for_cuda_caps_at_eight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_runtime.os, "cpu_count", lambda: 64)
    resolved = resolve_num_workers("auto", device="cuda")
    assert resolved == 8


def test_resolve_auto_num_workers_targets_four_to_eight_per_parallel_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_runtime.os, "cpu_count", lambda: 32)
    assert resolve_auto_num_workers(max_parallel_trials=1) == 8
    assert resolve_auto_num_workers(max_parallel_trials=4) == 4


def test_resolve_auto_num_workers_caps_by_per_trial_cpu_budget() -> None:
    assert resolve_auto_num_workers(cpu_count=12, max_parallel_trials=4) == 3
    assert resolve_auto_num_workers(cpu_count=64, max_parallel_trials=4) == 8


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


def test_is_compile_runtime_error_matches_cuda_oom() -> None:
    exc = RuntimeError("CUDA error: out of memory")
    assert is_compile_runtime_error(exc) is True


def test_resolve_compile_enabled_auto_no_epoch_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/tmp/ptxas")
    enabled = resolve_compile_enabled(
        compile_mode="auto",
        compile_flag=False,
        quick_phase=False,
        device="cuda",
        epochs=1,
    )
    assert enabled is True


def test_compile_model_with_fallback_uses_default_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode_calls: list[tuple[str | None, bool | None]] = []
    model = torch.nn.Linear(4, 2)

    def _fake_compile(
        module: torch.nn.Module,
        mode: str | None = None,
        dynamic: bool | None = None,
    ) -> torch.nn.Module:
        mode_calls.append((mode, dynamic))
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)
    compiled, enabled, selected_mode, setup_error = compile_model_with_fallback(model)
    assert compiled is model
    assert enabled is True
    assert selected_mode == "reduce-overhead"
    assert setup_error is None
    assert mode_calls == [("default", False)]


def test_configure_torch_compile_runtime_enables_dynamic_cudagraph_skip() -> None:
    inductor_module = getattr(torch, "_inductor", None)
    if inductor_module is None:
        pytest.skip("torch._inductor is unavailable in this torch build.")
    config_obj = getattr(inductor_module, "config", None)
    triton_config_obj = getattr(config_obj, "triton", None)
    if triton_config_obj is None:
        pytest.skip("torch._inductor.config.triton is unavailable.")

    value_obj = getattr(triton_config_obj, "cudagraph_skip_dynamic_graphs", None)
    if not isinstance(value_obj, bool):
        pytest.skip("cudagraph_skip_dynamic_graphs is unavailable.")

    setattr(triton_config_obj, "cudagraph_skip_dynamic_graphs", False)
    max_autotune_obj = getattr(config_obj, "max_autotune", None)
    max_autotune_gemm_obj = getattr(config_obj, "max_autotune_gemm", None)
    if isinstance(max_autotune_obj, bool):
        setattr(config_obj, "max_autotune", True)
    if isinstance(max_autotune_gemm_obj, bool):
        setattr(config_obj, "max_autotune_gemm", True)
    configure_torch_compile_runtime()
    assert triton_config_obj.cudagraph_skip_dynamic_graphs is True
    if isinstance(max_autotune_obj, bool):
        assert config_obj.max_autotune is False
    if isinstance(max_autotune_gemm_obj, bool):
        assert config_obj.max_autotune_gemm is False


def test_compile_model_with_fallback_applies_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    model = torch.nn.Linear(4, 2)

    def _fake_configure_triton() -> None:
        calls.append("triton")

    def _fake_configure() -> None:
        calls.append("compile")

    def _fake_compile(
        module: torch.nn.Module,
        mode: str | None = None,
        dynamic: bool | None = None,
    ) -> torch.nn.Module:
        del mode, dynamic
        return module

    monkeypatch.setattr(
        model_runtime,
        "configure_triton_tool_paths",
        _fake_configure_triton,
    )
    monkeypatch.setattr(
        model_runtime,
        "configure_torch_compile_runtime",
        _fake_configure,
    )
    monkeypatch.setattr(torch, "compile", _fake_compile)
    _ = compile_model_with_fallback(model)
    assert calls == ["triton", "compile"]


def test_compile_model_with_fallback_max_then_default_skips_small_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Props:
        multi_processor_count = 20

    mode_calls: list[tuple[str | None, bool | None]] = []
    model = torch.nn.Linear(3, 1)

    def _fake_compile(
        module: torch.nn.Module,
        mode: str | None = None,
        dynamic: bool | None = None,
    ) -> torch.nn.Module:
        mode_calls.append((mode, dynamic))
        return module

    monkeypatch.setenv(
        "INTRONMODEL_TORCH_COMPILE_STRATEGY",
        "max-then-default-then-off",
    )
    monkeypatch.setattr(torch, "compile", _fake_compile)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: _Props())

    _, enabled, selected_mode, setup_error = compile_model_with_fallback(model)
    assert enabled is True
    assert selected_mode == "reduce-overhead"
    assert setup_error is None
    assert mode_calls == [("default", False)]


def test_compile_model_with_fallback_on_ignores_max_autotune_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode_calls: list[tuple[str | None, bool | None]] = []
    model = torch.nn.Linear(2, 1)

    def _fake_compile(
        module: torch.nn.Module,
        mode: str | None = None,
        dynamic: bool | None = None,
    ) -> torch.nn.Module:
        mode_calls.append((mode, dynamic))
        if dynamic is not False:
            raise AssertionError("compile should force dynamic=False")
        return module

    monkeypatch.setenv(
        "INTRONMODEL_TORCH_COMPILE_STRATEGY",
        "max-then-default-then-off",
    )
    monkeypatch.setattr(torch, "compile", _fake_compile)

    first = compile_model_with_fallback(model, compile_mode="on")
    second = compile_model_with_fallback(model, compile_mode="on")

    assert first[1] is True
    assert first[2] == "reduce-overhead"
    assert first[3] is None
    assert second[1] is True
    assert second[2] == "reduce-overhead"
    assert second[3] is None
    assert mode_calls == [
        ("default", False),
        ("default", False),
    ]


def test_compile_model_with_fallback_uses_default_backend_mode_for_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode_calls: list[tuple[str | None, bool | None]] = []
    model = torch.nn.Linear(4, 2)
    model.eval()

    def _fake_compile(
        module: torch.nn.Module,
        mode: str | None = None,
        dynamic: bool | None = None,
    ) -> torch.nn.Module:
        mode_calls.append((mode, dynamic))
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)
    _ = compile_model_with_fallback(model)
    assert mode_calls == [("default", False)]


def test_compile_model_with_fallback_disables_max_autotune_for_default_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(4, 2)
    observed_env: list[str | None] = []
    observed_config: list[tuple[bool | None, bool | None]] = []

    inductor_module = getattr(torch, "_inductor", None)
    config_obj = getattr(inductor_module, "config", None) if inductor_module else None
    if config_obj is not None:
        if isinstance(getattr(config_obj, "max_autotune", None), bool):
            setattr(config_obj, "max_autotune", True)
        if isinstance(getattr(config_obj, "max_autotune_gemm", None), bool):
            setattr(config_obj, "max_autotune_gemm", True)

    def _fake_compile(
        module: torch.nn.Module,
        mode: str | None = None,
        dynamic: bool | None = None,
    ) -> torch.nn.Module:
        del mode, dynamic
        observed_env.append(os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"))
        if config_obj is not None:
            max_autotune = getattr(config_obj, "max_autotune", None)
            max_autotune_gemm = getattr(config_obj, "max_autotune_gemm", None)
            observed_config.append((max_autotune, max_autotune_gemm))
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)
    _ = compile_model_with_fallback(model)
    assert observed_env == ["0"]
    if observed_config:
        assert observed_config == [(False, False)]


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


def test_configure_triton_tool_paths_finds_ptxas_without_path_lookup(
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

    monkeypatch.setattr("util.model_runtime.shutil.which", lambda _name: None)
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    monkeypatch.delenv("TRITON_PTXAS_BLACKWELL_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(env_root))

    configure_triton_tool_paths()

    assert os.environ["TRITON_PTXAS_PATH"] == str(ptxas_path.resolve())
    assert os.environ["TRITON_PTXAS_BLACKWELL_PATH"] == str(
        ptxas_path.resolve()
    )
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
