"""Shared runtime helpers for model training and inference modules."""

from __future__ import annotations

from contextlib import contextmanager
import math
import os
import random
import shutil
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
import torch.nn as nn

_COMPILE_MODE_OFF: str = "off"
_COMPILE_MODE_REDUCE_OVERHEAD: str = "reduce-overhead"
_COMPILE_MODE_MAX_AUTOTUNE: str = "max-autotune"

_COMPILE_MODE_CHOICES: tuple[str, ...] = (
    _COMPILE_MODE_REDUCE_OVERHEAD,
    _COMPILE_MODE_MAX_AUTOTUNE,
)
_COMPILE_STRATEGY_CHOICES: tuple[str, ...] = (
    "off",
    "default-only",
    "default-then-off",
    "max-only",
    "max-then-default-then-off",
)

_COMPILE_STRATEGY_ENV: str = "INTRONMODEL_TORCH_COMPILE_STRATEGY"
_COMPILE_STICKY_MODE_ENV: str = "INTRONMODEL_TORCH_COMPILE_STICKY_MODE"
_COMPILE_DISABLED_MODES_ENV: str = "INTRONMODEL_TORCH_COMPILE_DISABLED_MODES"
_TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_ENV: str = "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"

_MAX_AUTOTUNE_MIN_SM_COUNT_CUDA: int = 68

_COMPILE_RUNTIME_CACHE_LOADED: bool = False
_COMPILE_STICKY_MODE_CACHE: str | None = None
_COMPILE_DISABLED_MODES_CACHE: set[str] = set()


def binary_clf_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative false/true positives at each score threshold.

    Complexity
    ----------
    O(n log n) time and O(n) additional memory.
    """
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and scores must be 1-D arrays.")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length.")
    if labels.size == 0:
        raise ValueError("labels and scores must be non-empty.")

    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(np.int64, copy=False)
    scores_sorted = scores[order]

    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_indices = np.r_[distinct_indices, labels_sorted.size - 1]

    true_positives = np.cumsum(labels_sorted)[threshold_indices]
    false_positives = (threshold_indices + 1) - true_positives
    return (
        false_positives.astype(np.float64, copy=False),
        true_positives.astype(np.float64, copy=False),
    )


def fallback_average_precision(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute binary average precision without scikit-learn.

    Complexity
    ----------
    O(n log n) time and O(n) memory.
    """
    positives = float(np.sum(labels == 1))
    if positives <= 0.0:
        raise ValueError("At least one positive label is required.")

    false_positives, true_positives = binary_clf_curve(labels, probs)
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / positives

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def fallback_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute binary ROC-AUC without scikit-learn.

    Complexity
    ----------
    O(n log n) time and O(n) memory.
    """
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    if positives <= 0.0 or negatives <= 0.0:
        raise ValueError("Both positive and negative labels are required.")

    false_positives, true_positives = binary_clf_curve(labels, probs)
    fpr = np.r_[0.0, false_positives / negatives, 1.0]
    tpr = np.r_[0.0, true_positives / positives, 1.0]
    return float(np.trapezoid(tpr, fpr))


def fallback_max_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute binary max-F1 over score thresholds without scikit-learn.

    Complexity
    ----------
    O(n log n) time and O(n) memory.
    """
    positives = float(np.sum(labels == 1))
    if positives <= 0.0:
        raise ValueError("At least one positive label is required.")

    false_positives, true_positives = binary_clf_curve(labels, probs)
    false_negatives = positives - true_positives

    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / np.maximum(true_positives + false_negatives, 1.0)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
    if f1.size == 0:
        raise ValueError("At least one prediction is required.")
    return float(np.max(f1))


def bool_from_flag(flag: bool | int) -> bool:
    """Convert integer/boolean flags from CLI to a strict bool."""
    if isinstance(flag, bool):
        return flag
    return int(flag) != 0


def resolve_auto_num_workers(
    *,
    cpu_count: int | None = None,
    max_parallel_trials: int = 1,
) -> int:
    """Resolve a conservative default ``num_workers`` value for ``auto``.

    Parameters
    ----------
    cpu_count : int | None, default=None
        Logical CPU count used for budgeting. When ``None``, the function uses
        ``os.cpu_count()`` and falls back to ``4`` if the platform cannot
        report a count.
    max_parallel_trials : int, default=1
        Number of concurrent GPU trials sharing the CPU budget.

    Returns
    -------
    int
        Conservative worker count per trial. The result is capped at ``8`` and
        never exceeds the per-trial CPU budget.

    Raises
    ------
    ValueError
        If ``cpu_count`` or ``max_parallel_trials`` is not positive.

    Complexity
    ----------
    O(1) time and O(1) memory.
    """

    if cpu_count is None:
        resolved_cpu_count = os.cpu_count() or 4
    else:
        resolved_cpu_count = int(cpu_count)
    if resolved_cpu_count <= 0:
        raise ValueError("cpu_count must be > 0.")

    parallel = int(max_parallel_trials)
    if parallel < 1:
        raise ValueError("max_parallel_trials must be >= 1.")

    per_trial_cpu_budget = max(1, resolved_cpu_count // parallel)
    workers = max(1, per_trial_cpu_budget // 4)
    if resolved_cpu_count >= 64 and parallel >= 4:
        workers = max(workers, 4)
    current_default = min(8, workers)
    return min(current_default, per_trial_cpu_budget)


def resolve_num_workers(raw: str | int, device: str) -> int:
    """Resolve DataLoader worker count from int or ``auto``.

    Parameters
    ----------
    raw : str | int
        User-supplied ``--num_workers`` value.
    device : str
        Runtime device name.

    Returns
    -------
    int
        Concrete worker count. ``auto`` resolves to zero on non-CUDA devices
        and to a conservative shared default on CUDA.

    Raises
    ------
    ValueError
        If ``raw`` is invalid or negative.

    Complexity
    ----------
    O(1) time and O(1) memory.
    """
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError("--num_workers must be >= 0.")
        return raw

    text = str(raw).strip().lower()
    if text == "auto":
        if device != "cuda":
            return 0
        return resolve_auto_num_workers()

    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError("--num_workers must be an integer or 'auto'.") from exc
    if parsed < 0:
        raise ValueError("--num_workers must be >= 0.")
    return parsed


def resolve_amp_dtype(name: str, device: str) -> torch.dtype | None:
    """Resolve AMP dtype from a user-facing string."""
    if device != "cuda":
        return None
    lowered = name.strip().lower()
    if lowered == "bf16":
        return torch.bfloat16
    if lowered == "fp16":
        return torch.float16
    if lowered != "auto":
        raise ValueError("--amp_dtype must be one of: auto, bf16, fp16.")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_compile_enabled(
    compile_mode: str,
    compile_flag: bool,
    quick_phase: bool,
    device: str,
    epochs: int,
) -> bool:
    """Resolve final ``torch.compile`` usage policy."""
    del epochs
    if device != "cuda":
        return False
    if compile_flag:
        return True
    mode = compile_mode.strip().lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode != "auto":
        raise ValueError("--compile_mode must be one of: off, on, auto.")
    if quick_phase:
        return False
    ptxas_env = os.environ.get("TRITON_PTXAS_PATH")
    ptxas_blackwell_env = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH")
    if ptxas_env or ptxas_blackwell_env:
        return True
    return shutil.which("ptxas") is not None


def _normalize_compile_mode_token(raw: str) -> str | None:
    token = raw.strip().lower().replace("_", "-")
    aliases = {
        "default": _COMPILE_MODE_REDUCE_OVERHEAD,
        "normal": _COMPILE_MODE_REDUCE_OVERHEAD,
        "reduce-overhead": _COMPILE_MODE_REDUCE_OVERHEAD,
        "reduce": _COMPILE_MODE_REDUCE_OVERHEAD,
        "max": _COMPILE_MODE_MAX_AUTOTUNE,
        "max-autotune": _COMPILE_MODE_MAX_AUTOTUNE,
        "max-autotune-gemm": _COMPILE_MODE_MAX_AUTOTUNE,
        "off": _COMPILE_MODE_OFF,
        "none": _COMPILE_MODE_OFF,
        "false": _COMPILE_MODE_OFF,
    }
    resolved = aliases.get(token)
    return resolved


def _normalize_compile_strategy(raw: str) -> str:
    token = raw.strip().lower().replace("_", "-")
    aliases = {
        "off": "off",
        "none": "off",
        "false": "off",
        "default": "default-then-off",
        "auto": "default-then-off",
        "default-only": "default-only",
        "default-then-off": "default-then-off",
        "max": "max-only",
        "max-only": "max-only",
        "max-then-default": "max-then-default-then-off",
        "max-then-default-then-off": "max-then-default-then-off",
    }
    resolved = aliases.get(token)
    if resolved is None:
        choices = ", ".join(_COMPILE_STRATEGY_CHOICES)
        raise ValueError(f"{_COMPILE_STRATEGY_ENV} must be one of: {choices}.")
    return resolved


def _compile_modes_for_strategy(strategy: str) -> tuple[str, ...]:
    if strategy == "off":
        return ()
    if strategy in {"default-only", "default-then-off"}:
        return (_COMPILE_MODE_REDUCE_OVERHEAD,)
    if strategy == "max-only":
        return (_COMPILE_MODE_MAX_AUTOTUNE,)
    if strategy == "max-then-default-then-off":
        return (_COMPILE_MODE_MAX_AUTOTUNE, _COMPILE_MODE_REDUCE_OVERHEAD)
    raise ValueError(
        f"Unrecognized compile strategy '{strategy}'. "
        f"Expected one of: {', '.join(_COMPILE_STRATEGY_CHOICES)}."
    )


def _load_compile_runtime_cache_from_env() -> None:
    global _COMPILE_RUNTIME_CACHE_LOADED
    global _COMPILE_STICKY_MODE_CACHE
    if _COMPILE_RUNTIME_CACHE_LOADED:
        return
    sticky_raw = os.environ.get(_COMPILE_STICKY_MODE_ENV)
    if sticky_raw is not None and sticky_raw.strip():
        sticky_mode = _normalize_compile_mode_token(sticky_raw)
        if sticky_mode is not None:
            _COMPILE_STICKY_MODE_CACHE = sticky_mode
    disabled_raw = os.environ.get(_COMPILE_DISABLED_MODES_ENV)
    if disabled_raw is not None and disabled_raw.strip():
        for item in disabled_raw.split(","):
            disabled_mode = _normalize_compile_mode_token(item)
            if disabled_mode in _COMPILE_MODE_CHOICES:
                _COMPILE_DISABLED_MODES_CACHE.add(disabled_mode)
    _COMPILE_RUNTIME_CACHE_LOADED = True


def _persist_compile_runtime_cache_to_env() -> None:
    sticky_mode = _COMPILE_STICKY_MODE_CACHE
    if sticky_mode is None:
        os.environ.pop(_COMPILE_STICKY_MODE_ENV, None)
    else:
        os.environ[_COMPILE_STICKY_MODE_ENV] = sticky_mode
    if _COMPILE_DISABLED_MODES_CACHE:
        os.environ[_COMPILE_DISABLED_MODES_ENV] = ",".join(
            sorted(_COMPILE_DISABLED_MODES_CACHE)
        )
    else:
        os.environ.pop(_COMPILE_DISABLED_MODES_ENV, None)


def _set_compile_sticky_mode(mode: str | None) -> None:
    global _COMPILE_STICKY_MODE_CACHE
    _COMPILE_STICKY_MODE_CACHE = mode
    _persist_compile_runtime_cache_to_env()


def _disable_compile_mode(mode: str) -> None:
    _COMPILE_DISABLED_MODES_CACHE.add(mode)
    if _COMPILE_STICKY_MODE_CACHE == mode:
        _set_compile_sticky_mode(None)
    _persist_compile_runtime_cache_to_env()


def _can_use_max_autotune_mode() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        current_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_index)
    except Exception:
        return False
    avail_sms = int(getattr(props, "multi_processor_count", 0))
    return avail_sms >= _MAX_AUTOTUNE_MIN_SM_COUNT_CUDA


@contextmanager
def _temporary_max_autotune_setting(enabled: bool) -> Iterator[None]:
    """Temporarily align env/config max-autotune flags with one compile mode."""
    key = _TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_ENV
    previous = os.environ.get(key)
    inductor_module = getattr(torch, "_inductor", None)
    config_obj = getattr(inductor_module, "config", None) if inductor_module else None
    max_autotune_prev = None
    max_autotune_gemm_prev = None
    if config_obj is not None:
        value_obj = getattr(config_obj, "max_autotune", None)
        if isinstance(value_obj, bool):
            max_autotune_prev = value_obj
            setattr(config_obj, "max_autotune", enabled)
        value_obj = getattr(config_obj, "max_autotune_gemm", None)
        if isinstance(value_obj, bool):
            max_autotune_gemm_prev = value_obj
            setattr(config_obj, "max_autotune_gemm", enabled)
    os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if config_obj is not None:
            if max_autotune_prev is not None:
                setattr(config_obj, "max_autotune", max_autotune_prev)
            if max_autotune_gemm_prev is not None:
                setattr(config_obj, "max_autotune_gemm", max_autotune_gemm_prev)
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _compile_model_once_with_mode(model: nn.Module, mode: str) -> nn.Module:
    """Compile one model with a fixed-shape graph preference.

    Parameters
    ----------
    model : nn.Module
        Module to compile.
    mode : str
        Torch compile mode token.

    Returns
    -------
    nn.Module
        Compiled module.

    Raises
    ------
    RuntimeError
        If ``torch.compile`` is unavailable.
    ValueError
        If ``mode`` is unsupported.
    """
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        raise RuntimeError("torch.compile is unavailable in this torch build.")
    if mode not in _COMPILE_MODE_CHOICES:
        raise ValueError(
            f"Unsupported compile mode '{mode}'. "
            f"Expected one of: {', '.join(_COMPILE_MODE_CHOICES)}."
        )
    torch_mode = mode
    if mode == _COMPILE_MODE_REDUCE_OVERHEAD:
        # ``reduce-overhead`` is implemented as cudagraph-enabled Inductor. On
        # these sequence models that tends to fragment into several partitions,
        # producing noisy perf hints without a matching runtime benefit. Keep
        # the same public compile policy, but compile with the default
        # no-cudagraph Inductor mode under the hood.
        torch_mode = "default"
    elif model.training and mode == _COMPILE_MODE_MAX_AUTOTUNE:
        # Keep autotuning for training, but skip cudagraph capture to avoid the
        # same graph partition issue as ``reduce-overhead``.
        torch_mode = "max-autotune-no-cudagraphs"
    with _temporary_max_autotune_setting(mode == _COMPILE_MODE_MAX_AUTOTUNE):
        # The training/inference entry points that enable compile already keep
        # loaders and padded inference batches shape-stable. Asking Inductor to
        # avoid dynamic shape graphs lets cudagraph capture remain intact.
        return compile_fn(model, mode=torch_mode, dynamic=False)


def compile_model_with_fallback(
    model: nn.Module,
    *,
    compile_mode: str = "auto",
) -> tuple[nn.Module, bool, str | None, Exception | None]:
    """Compile one model with strategy-based fallback and process-local caching.

    Parameters
    ----------
    model : nn.Module
        Model to compile with ``torch.compile``.
    compile_mode : str, default="auto"
        High-level compile policy passed from the training or inference
        function.  When ``"auto"``, the strategy is always capped to
        ``"default-then-off"`` (i.e. ``reduce-overhead`` only), ignoring
        ``INTRONMODEL_TORCH_COMPILE_STRATEGY``.  For any other value (e.g.
        ``"on"``) the env-var strategy is respected as usual.

    Returns
    -------
    tuple[nn.Module, bool, str | None, Exception | None]
        ``(effective_model, compile_enabled, selected_mode, last_error)``.
        ``selected_mode`` is ``None`` when compile is disabled.
    """
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return model, False, None, None

    # Apply shared runtime guards once per compile attempt so models that call
    # this helper directly also inherit the same stable Triton/Inductor
    # settings even when callers do not preconfigure tool paths.
    configure_triton_tool_paths()
    configure_torch_compile_runtime()

    _load_compile_runtime_cache_from_env()
    if compile_mode.strip().lower() == "auto":
        strategy = "default-then-off"
    else:
        strategy_raw = os.environ.get(_COMPILE_STRATEGY_ENV, "default-then-off")
        strategy = _normalize_compile_strategy(strategy_raw)
    if strategy == "off":
        _set_compile_sticky_mode(_COMPILE_MODE_OFF)
        return model, False, None, None

    if _COMPILE_STICKY_MODE_CACHE == _COMPILE_MODE_OFF:
        return model, False, None, None

    if _COMPILE_STICKY_MODE_CACHE in _COMPILE_MODE_CHOICES:
        candidate_modes = [_COMPILE_STICKY_MODE_CACHE]
    else:
        candidate_modes = list(_compile_modes_for_strategy(strategy))

    last_error: Exception | None = None
    for mode in candidate_modes:
        if mode in _COMPILE_DISABLED_MODES_CACHE:
            continue
        if mode == _COMPILE_MODE_MAX_AUTOTUNE and not _can_use_max_autotune_mode():
            _disable_compile_mode(mode)
            continue
        try:
            compiled_model = _compile_model_once_with_mode(model, mode)
        except Exception as exc:
            last_error = exc
            _disable_compile_mode(mode)
            continue
        _set_compile_sticky_mode(mode)
        return compiled_model, True, mode, None

    _set_compile_sticky_mode(_COMPILE_MODE_OFF)
    return model, False, None, last_error


def record_compile_runtime_failure(selected_mode: str | None) -> None:
    """Record one runtime compile failure and adjust future compile attempts."""
    _load_compile_runtime_cache_from_env()
    if selected_mode is None:
        _set_compile_sticky_mode(_COMPILE_MODE_OFF)
        return
    normalized_mode = _normalize_compile_mode_token(selected_mode)
    if normalized_mode not in _COMPILE_MODE_CHOICES:
        _set_compile_sticky_mode(_COMPILE_MODE_OFF)
        return
    _disable_compile_mode(normalized_mode)
    if normalized_mode == _COMPILE_MODE_MAX_AUTOTUNE:
        _set_compile_sticky_mode(None)
        return
    _set_compile_sticky_mode(_COMPILE_MODE_OFF)


def _iter_cuda_root_candidates() -> list[Path]:
    """Return plausible CUDA toolkit root directories."""
    roots: list[Path] = []
    seen: set[Path] = set()

    def _append(candidate: Path) -> None:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    env_roots = (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        os.environ.get("CONDA_PREFIX"),
    )
    for root in env_roots:
        if root is None or not root.strip():
            continue
        _append(Path(root.strip()))

    tool_names = ("ptxas", "nvcc")
    for tool_name in tool_names:
        tool_path_text = shutil.which(tool_name)
        if tool_path_text is None:
            continue
        tool_path = Path(tool_path_text).resolve()
        if tool_path.parent.name == "bin":
            _append(tool_path.parent.parent)

    explicit_tool_env = (
        os.environ.get("TRITON_PTXAS_PATH"),
        os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
    )
    for path_text in explicit_tool_env:
        if path_text is None or not path_text.strip():
            continue
        tool_path = Path(path_text.strip()).expanduser().resolve()
        if tool_path.parent.name == "bin":
            _append(tool_path.parent.parent)

    common_roots = (
        Path("/usr/local/cuda"),
        Path("/opt/cuda"),
    )
    for root in common_roots:
        _append(root)
    return roots


def _iter_cuda_header_candidates() -> list[Path]:
    """Return plausible ``cuda.h`` candidates for multiple toolkit layouts."""
    headers: list[Path] = []
    seen: set[Path] = set()

    def _append(candidate: Path) -> None:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        headers.append(resolved)

    for root in _iter_cuda_root_candidates():
        _append(root / "include" / "cuda.h")
        targets_root = root / "targets"
        _append(targets_root / "x86_64-linux" / "include" / "cuda.h")
        if targets_root.exists():
            for include_dir in targets_root.glob("*/include"):
                _append(include_dir / "cuda.h")

    _append(Path("/usr/include/cuda.h"))
    return headers


def _find_cuda_header() -> Path | None:
    """Return first existing ``cuda.h`` path from known candidate locations."""
    for candidate in _iter_cuda_header_candidates():
        if candidate.exists():
            return candidate
    return None


def _resolve_cuda_root_from_header(cuda_header: Path) -> Path:
    """Infer CUDA toolkit root from ``cuda.h`` location."""
    include_dir = cuda_header.parent
    arch_dir = include_dir.parent
    targets_dir = arch_dir.parent
    if include_dir.name == "include" and targets_dir.name == "targets":
        return targets_dir.parent
    return include_dir.parent


def _prepend_env_path(key: str, new_path: Path) -> None:
    """Prepend a path-like environment variable entry without duplicates."""
    resolved_new = str(new_path.expanduser().resolve())
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        os.environ[key] = resolved_new
        return
    parts: list[str] = []
    seen: set[str] = set()
    for item in raw.split(os.pathsep):
        text = item.strip()
        if not text:
            continue
        resolved = str(Path(text).expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        parts.append(text)
    if resolved_new in seen:
        return
    os.environ[key] = os.pathsep.join([resolved_new, *parts])


def _find_ptxas_path() -> Path | None:
    """Return first existing ``ptxas`` binary from known CUDA locations."""
    explicit_env = (
        os.environ.get("TRITON_PTXAS_PATH"),
        os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
    )
    for path_text in explicit_env:
        if path_text is None or not path_text.strip():
            continue
        candidate = Path(path_text.strip()).expanduser().resolve()
        if candidate.exists():
            return candidate

    which_path = shutil.which("ptxas")
    if which_path is not None:
        return Path(which_path).expanduser().resolve()

    for root in _iter_cuda_root_candidates():
        for candidate in (
            root / "bin" / "ptxas",
            root / "targets" / "x86_64-linux" / "bin" / "ptxas",
        ):
            resolved = candidate.expanduser().resolve()
            if resolved.exists():
                return resolved
    return None


def configure_triton_tool_paths() -> None:
    """Configure Triton/CUDA tool paths for ``torch.compile`` stability."""
    ptxas_path = _find_ptxas_path()
    if ptxas_path is None:
        pass
    elif "TRITON_PTXAS_PATH" not in os.environ:
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas_path)
    if ptxas_path is not None and "TRITON_PTXAS_BLACKWELL_PATH" not in os.environ:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(ptxas_path)

    cuda_header = _find_cuda_header()
    if cuda_header is None:
        return
    cuda_include_dir = cuda_header.parent
    cuda_root = _resolve_cuda_root_from_header(cuda_header)
    if "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = str(cuda_root)
    if "CUDA_PATH" not in os.environ:
        os.environ["CUDA_PATH"] = str(cuda_root)
    _prepend_env_path("CPATH", cuda_include_dir)


def configure_torch_compile_runtime() -> None:
    """Apply conservative ``torch.compile`` runtime settings for stability."""
    dynamo_module = getattr(torch, "_dynamo", None)
    if dynamo_module is None:
        return
    config_obj = getattr(dynamo_module, "config", None)
    if config_obj is None:
        return
    capture_scalar_outputs = getattr(config_obj, "capture_scalar_outputs", None)
    if isinstance(capture_scalar_outputs, bool) and not capture_scalar_outputs:
        setattr(config_obj, "capture_scalar_outputs", True)
    if "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS" not in os.environ:
        os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"

    inductor_module = getattr(torch, "_inductor", None)
    if inductor_module is None:
        return
    inductor_config_obj = getattr(inductor_module, "config", None)
    if inductor_config_obj is None:
        return
    max_autotune = getattr(inductor_config_obj, "max_autotune", None)
    if isinstance(max_autotune, bool) and max_autotune:
        setattr(inductor_config_obj, "max_autotune", False)
    max_autotune_gemm = getattr(inductor_config_obj, "max_autotune_gemm", None)
    if isinstance(max_autotune_gemm, bool) and max_autotune_gemm:
        setattr(inductor_config_obj, "max_autotune_gemm", False)
    triton_config_obj = getattr(inductor_config_obj, "triton", None)
    if triton_config_obj is None:
        return
    skip_dynamic_graphs = getattr(
        triton_config_obj,
        "cudagraph_skip_dynamic_graphs",
        None,
    )
    if isinstance(skip_dynamic_graphs, bool) and not skip_dynamic_graphs:
        setattr(triton_config_obj, "cudagraph_skip_dynamic_graphs", True)


def is_cuda_oom_error(exc: RuntimeError) -> bool:
    """Return whether a runtime error is likely a CUDA OOM error."""
    message = str(exc).lower()
    keywords = (
        "out of memory",
        "cuda error: out of memory",
        "cudnn_status_alloc_failed",
    )
    return any(keyword in message for keyword in keywords)


def is_mps_oom_error(exc: RuntimeError) -> bool:
    """Return whether a runtime error is likely an MPS OOM error."""
    message = str(exc).lower()
    keywords = (
        "mps backend out of memory",
        "mps out of memory",
        "not enough memory",
    )
    return any(keyword in message for keyword in keywords)


def empty_device_cache(device: str) -> None:
    """Best-effort cache cleanup for the given device."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if device != "mps":
        return
    mps_module = getattr(torch, "mps", None)
    if mps_module is None:
        return
    empty_cache_fn = getattr(mps_module, "empty_cache", None)
    if callable(empty_cache_fn):
        empty_cache_fn()


def resolve_mps_max_batch_size(model_tag: str, default_batch_size: int) -> int:
    """Resolve MPS batch-size cap from env with a safe default."""
    raw = os.environ.get("INTRONMODEL_MPS_MAX_BATCH_SIZE")
    if raw is None or raw.strip() == "":
        return default_batch_size
    try:
        parsed = int(raw)
    except ValueError:
        print(
            f"[{model_tag}] invalid INTRONMODEL_MPS_MAX_BATCH_SIZE. "
            f"Use default {default_batch_size}.",
        )
        return default_batch_size
    if parsed <= 0:
        print(
            f"[{model_tag}] non-positive INTRONMODEL_MPS_MAX_BATCH_SIZE. "
            f"Use default {default_batch_size}.",
        )
        return default_batch_size
    return parsed


def is_compile_runtime_error(exc: Exception) -> bool:
    """Return whether an exception is likely from ``torch.compile`` runtime."""
    diagnostic = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    keywords = (
        "inductorerror",
        "torch._inductor",
        "torch._dynamo",
        "torchdynamo",
        "triton",
        "cudagraph",
        "cuda graph",
        "cudagraph_trees",
        "ptxas",
        "backend_hash",
        "cuda error: out of memory",
        "out of memory",
        "cannot copy out of meta tensor",
        "meta tensor",
    )
    return any(keyword in diagnostic for keyword in keywords)


def export_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Export model parameters in a stable, non-compiled key format."""
    original_model = getattr(model, "_orig_mod", None)
    if isinstance(original_model, nn.Module):
        return dict(original_model.state_dict())
    return dict(model.state_dict())


def normalize_checkpoint_state_dict(
    raw_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize legacy/compiled checkpoint keys for plain model loading."""
    compiled_prefix = "_orig_mod."
    normalized: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        if not key.startswith(compiled_prefix):
            normalized[key] = value
    for key, value in raw_state_dict.items():
        if key.startswith(compiled_prefix):
            stripped_key = key[len(compiled_prefix) :]
            normalized.setdefault(stripped_key, value)
    return normalized


def set_seed(
    seed: int = 1337,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
    allow_tf32: bool = False,
) -> None:
    """Set random seeds and backend runtime flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark and (not deterministic)
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def seed_worker(worker_id: int) -> None:
    """Seed dataloader worker-local RNG states."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pick_device(preference: str = "auto") -> str:
    """Resolve runtime device preference."""
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid values with numerically stable branches."""
    logits = np.asarray(x, dtype=np.float64)
    clipped = np.clip(logits, -500.0, 500.0)
    out = np.empty_like(clipped, dtype=np.float64)
    positive_mask = clipped >= 0.0
    out[positive_mask] = 1.0 / (1.0 + np.exp(-clipped[positive_mask]))
    negative_logits = clipped[~positive_mask]
    exp_values = np.exp(negative_logits)
    out[~positive_mask] = exp_values / (1.0 + exp_values)
    return out.astype(np.float32, copy=False)


def probabilities_to_log10_scores_np(values: np.ndarray) -> np.ndarray:
    """Convert probability values in ``[0, 1]`` to log10 scores."""
    probabilities = np.asarray(values, dtype=np.float64)
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probability values must lie in [0, 1].")
    out = np.full_like(probabilities, fill_value=-np.inf, dtype=np.float64)
    positive_mask = probabilities > 0.0
    out[positive_mask] = np.log10(probabilities[positive_mask])
    return out.astype(np.float32, copy=False)


def log10_sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Compute ``log10(sigmoid(x))`` with numerically stable branches."""
    logits = np.asarray(x, dtype=np.float64)
    out = np.empty_like(logits, dtype=np.float64)
    positive_mask = logits >= 0.0
    out[positive_mask] = -np.log1p(np.exp(-logits[positive_mask]))
    negative_logits = logits[~positive_mask]
    out[~positive_mask] = negative_logits - np.log1p(np.exp(negative_logits))
    out /= math.log(10.0)
    return out.astype(np.float32, copy=False)
