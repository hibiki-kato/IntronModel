"""Shared runtime helpers for model training and inference modules."""

from __future__ import annotations

import os
import random
import shutil
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn


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


def bool_from_flag(flag: bool | int) -> bool:
    """Convert integer/boolean flags from CLI to a strict bool."""
    if isinstance(flag, bool):
        return flag
    return int(flag) != 0


def resolve_num_workers(raw: str | int, device: str) -> int:
    """Resolve DataLoader worker count from int or ``auto``."""
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError("--num_workers must be >= 0.")
        return raw

    text = str(raw).strip().lower()
    if text == "auto":
        if device != "cuda":
            return 0
        cpu_count = os.cpu_count() or 4
        return max(0, cpu_count // 2)

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
    if epochs < 10:
        return False
    ptxas_env = os.environ.get("TRITON_PTXAS_PATH")
    ptxas_blackwell_env = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH")
    if ptxas_env or ptxas_blackwell_env:
        return True
    return shutil.which("ptxas") is not None


def configure_triton_tool_paths() -> None:
    """Configure Triton tool paths for ``torch.compile`` stability."""
    ptxas_path = shutil.which("ptxas")
    if ptxas_path is None:
        return
    if "TRITON_PTXAS_PATH" not in os.environ:
        os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    if "TRITON_PTXAS_BLACKWELL_PATH" not in os.environ:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_path


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
        "ptxas",
        "backend_hash",
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
