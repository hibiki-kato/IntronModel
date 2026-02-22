"""CNN model implementation for site-level splice scoring.

This module contains CNN-specific components:
- model architecture
- training and validation loop
- checkpoint loading
- site-level inference for donor/acceptor sequences
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import os
import random
import shutil
import time
from typing import (
    ContextManager,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from util.data_proc import (
    build_run_name,
    infer_default_train_paths,
    read_examples_single_task,
    read_test_site_rows,
    resolve_effective_window_lengths,
    resolve_test_tsv,
    resolve_train_paths,
    species_data_dirs,
    validate_window_args,
)
from util.losses import LOSS_NAME_CHOICES, build_binary_classification_loss
from util.training_control import (
    resolve_early_stopping_params,
    resolve_training_epoch_budget,
)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

DEFAULT_MPS_MAX_BATCH_SIZE: int = 2048


def _binary_clf_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative false/true positives at each score threshold.

    This follows the same core idea as scikit-learn:
    sort by score descending, then take cumulative sums at distinct score
    boundaries.

    Complexity:
        O(n log n) time due to sorting and O(n) additional memory.
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


def _fallback_average_precision(
    labels: np.ndarray,
    probs: np.ndarray,
) -> float:
    """Compute binary average precision without scikit-learn.

    This integrates a step-wise precision-recall curve:
    ``AP = sum((R_n - R_{n-1}) * P_n)``.

    Complexity:
        O(n log n) time and O(n) memory.
    """
    positives = float(np.sum(labels == 1))
    if positives <= 0.0:
        raise ValueError("At least one positive label is required.")

    false_positives, true_positives = _binary_clf_curve(labels, probs)
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / positives

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _fallback_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute binary ROC-AUC without scikit-learn.

    This computes ROC points from cumulative true/false positives and applies
    the trapezoidal rule.

    Complexity:
        O(n log n) time and O(n) memory.
    """
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    if positives <= 0.0 or negatives <= 0.0:
        raise ValueError("Both positive and negative labels are required.")

    false_positives, true_positives = _binary_clf_curve(labels, probs)
    fpr = np.r_[0.0, false_positives / negatives, 1.0]
    tpr = np.r_[0.0, true_positives / positives, 1.0]
    return float(np.trapezoid(tpr, fpr))


def _bool_from_flag(flag: Union[bool, int]) -> bool:
    """Convert integer/boolean flags from CLI to a strict bool."""
    if isinstance(flag, bool):
        return flag
    return int(flag) != 0


def _resolve_num_workers(raw: Union[str, int], device: str) -> int:
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


def _resolve_amp_dtype(name: str, device: str) -> Optional[torch.dtype]:
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


def _resolve_compile_enabled(
    compile_mode: str,
    compile_flag: bool,
    quick_phase: bool,
    device: str,
    epochs: int,
) -> bool:
    """Resolve final torch.compile usage policy."""
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


def _configure_triton_tool_paths() -> None:
    """Configure Triton NVIDIA tool paths for torch.compile stability.

    Some Triton versions query a ``ptxas-blackwell`` binary and can fail with a
    ``TypeError`` if the path resolution falls back to ``None``. This function
    provides conservative defaults from ``ptxas`` when available.
    """
    ptxas_path = shutil.which("ptxas")
    if ptxas_path is None:
        return
    if "TRITON_PTXAS_PATH" not in os.environ:
        os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    if "TRITON_PTXAS_BLACKWELL_PATH" not in os.environ:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_path


def _configure_torch_compile_runtime() -> None:
    """Apply conservative torch.compile runtime settings for stability."""
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


def _is_cuda_oom_error(exc: RuntimeError) -> bool:
    """Return whether a runtime error is likely a CUDA OOM error."""
    message = str(exc).lower()
    keywords = (
        "out of memory",
        "cuda error: out of memory",
        "cudnn_status_alloc_failed",
    )
    return any(keyword in message for keyword in keywords)


def _is_mps_oom_error(exc: RuntimeError) -> bool:
    """Return whether a runtime error is likely an MPS OOM error."""
    message = str(exc).lower()
    keywords = (
        "mps backend out of memory",
        "out of memory (mps)",
        "out of memory on mps",
        "metal out of memory",
    )
    return any(keyword in message for keyword in keywords)


def _empty_device_cache(device: str) -> None:
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


def _resolve_mps_max_batch_size() -> int:
    """Resolve MPS batch-size cap from env with a safe default."""
    raw = os.environ.get("INTRONMODEL_MPS_MAX_BATCH_SIZE")
    if raw is None or raw.strip() == "":
        return DEFAULT_MPS_MAX_BATCH_SIZE
    try:
        parsed = int(raw)
    except ValueError:
        print(
            "[cnn] invalid INTRONMODEL_MPS_MAX_BATCH_SIZE. "
            f"Use default {DEFAULT_MPS_MAX_BATCH_SIZE}.",
        )
        return DEFAULT_MPS_MAX_BATCH_SIZE
    if parsed <= 0:
        print(
            "[cnn] non-positive INTRONMODEL_MPS_MAX_BATCH_SIZE. "
            f"Use default {DEFAULT_MPS_MAX_BATCH_SIZE}.",
        )
        return DEFAULT_MPS_MAX_BATCH_SIZE
    return parsed


def _is_compile_runtime_error(exc: RuntimeError) -> bool:
    """Return whether a runtime error is likely from torch.compile runtime."""
    diagnostic = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    keywords = (
        "inductorerror",
        "torch._inductor",
        "torch._dynamo",
        "triton",
        "ptxas",
        "backend_hash",
    )
    return any(keyword in diagnostic for keyword in keywords)


def _export_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Export model parameters in a stable, non-compiled key format.

    Parameters
    ----------
    model : nn.Module
        Training model instance. This may be a plain module or a
        ``torch.compile`` wrapper exposing ``_orig_mod``.

    Returns
    -------
    dict[str, torch.Tensor]
        State dictionary keyed by canonical module names.
    """
    original_model = getattr(model, "_orig_mod", None)
    if isinstance(original_model, nn.Module):
        return dict(original_model.state_dict())
    return dict(model.state_dict())


def _normalize_checkpoint_state_dict(
    raw_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize legacy/compiled checkpoint keys for plain model loading.

    Parameters
    ----------
    raw_state_dict : Mapping[str, torch.Tensor]
        Raw checkpoint state dictionary.

    Returns
    -------
    dict[str, torch.Tensor]
        Normalized keys where ``_orig_mod.`` prefixes are stripped.
    """
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


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader worker-local RNG states."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pick_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid values with numerically stable branches.

    Parameters
    ----------
    x : np.ndarray
        Input logits array. Shape: ``(N,)`` or any broadcastable shape.

    Returns
    -------
    np.ndarray
        Sigmoid probabilities in ``[0, 1]`` with dtype ``float32`` and the
        same shape as ``x``.
    """
    logits = np.asarray(x, dtype=np.float64)
    clipped = np.clip(logits, -500.0, 500.0)
    out = np.empty_like(clipped, dtype=np.float64)
    positive_mask = clipped >= 0.0
    out[positive_mask] = 1.0 / (1.0 + np.exp(-clipped[positive_mask]))
    negative_logits = clipped[~positive_mask]
    exp_values = np.exp(negative_logits)
    out[~positive_mask] = exp_values / (1.0 + exp_values)
    return out.astype(np.float32, copy=False)


def one_hot_encode_dna(seq: str, window_len: int = 50) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((4, window_len), dtype=np.float32)

    for i, base in enumerate(seq[:window_len]):
        if base in mapping:
            encoded[mapping[base], i] = 1.0

    return encoded


def parse_conv_channels(
    raw: Optional[str],
    arg_name: str = "--conv_channels",
) -> Optional[List[int]]:
    """Parse comma-separated convolution channel sizes.

    Parameters
    ----------
    raw : str | None
        Comma-separated channel sizes like ``"64,128,256"``.

    Returns
    -------
    list[int] | None
        Parsed positive channel sizes, or ``None`` when not specified.

    Raises
    ------
    ValueError
        If the string has invalid format or non-positive sizes.
    """
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{arg_name} must include at least one integer.")

    channels: List[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {arg_name} item '{part}'. Use integers like 64,128,256."
            ) from exc
        if value <= 0:
            raise ValueError(f"{arg_name} values must be positive.")
        channels.append(value)
    return channels


@dataclass(frozen=True)
class TaskTrainParams:
    """Resolved train-time hyperparameters for one task."""

    batch_size: int
    lr: float
    loss_name: str
    conv_channels: Optional[Sequence[int]]
    kernel_size: int
    tcn_block_repeats: int
    tcn_causal: int
    dropout: float
    fc_hidden: int
    weight_decay: float
    eta_min_ratio: float
    val_frac: float
    grad_clip: float
    pos_weight_cap: float
    focal_gamma: float
    focal_alpha_pos: Optional[float]
    asym_gamma_pos: float
    asym_gamma_neg: float
    asym_alpha_pos: Optional[float]


def _resolve_task_train_params(
    *,
    task: str,
    model_args: argparse.Namespace,
    shared_conv_channels: Optional[Sequence[int]],
    donor_conv_channels: Optional[Sequence[int]],
    acceptor_conv_channels: Optional[Sequence[int]],
) -> TaskTrainParams:
    """Resolve task-specific train parameters with fallback to shared values.

    Parameters
    ----------
    task : str
        One of ``donor`` or ``acceptor``.
    model_args : argparse.Namespace
        Parsed model arguments.
    shared_conv_channels : Sequence[int] | None
        Shared ``--conv_channels`` values.
    donor_conv_channels : Sequence[int] | None
        Parsed ``--donor_conv_channels`` values.
    acceptor_conv_channels : Sequence[int] | None
        Parsed ``--acceptor_conv_channels`` values.

    Returns
    -------
    TaskTrainParams
        Fully resolved per-task parameter set.

    Raises
    ------
    ValueError
        If task name is unsupported.
    """
    if task not in {"donor", "acceptor"}:
        raise ValueError(f"Unsupported task: {task}")

    prefix = f"{task}_"

    def _override_or_default(name: str, default: object) -> object:
        override = getattr(model_args, f"{prefix}{name}", None)
        return default if override is None else override

    task_specific_conv = (
        donor_conv_channels if task == "donor" else acceptor_conv_channels
    )
    resolved_conv_channels = (
        task_specific_conv if task_specific_conv is not None else shared_conv_channels
    )

    return TaskTrainParams(
        batch_size=int(_override_or_default("batch_size", model_args.batch_size)),
        lr=float(_override_or_default("lr", model_args.lr)),
        loss_name=str(_override_or_default("loss", model_args.loss)),
        conv_channels=resolved_conv_channels,
        kernel_size=int(_override_or_default("kernel_size", model_args.kernel_size)),
        tcn_block_repeats=int(
            _override_or_default(
                "tcn_block_repeats", model_args.tcn_block_repeats
            )
        ),
        tcn_causal=int(_override_or_default("tcn_causal", model_args.tcn_causal)),
        dropout=float(_override_or_default("dropout", model_args.dropout)),
        fc_hidden=int(_override_or_default("fc_hidden", model_args.fc_hidden)),
        weight_decay=float(
            _override_or_default("weight_decay", model_args.weight_decay)
        ),
        eta_min_ratio=float(
            _override_or_default("eta_min_ratio", model_args.eta_min_ratio)
        ),
        val_frac=float(_override_or_default("val_frac", model_args.val_frac)),
        grad_clip=float(_override_or_default("grad_clip", model_args.grad_clip)),
        pos_weight_cap=float(
            _override_or_default("pos_weight_cap", model_args.pos_weight_cap)
        ),
        focal_gamma=float(_override_or_default("focal_gamma", model_args.focal_gamma)),
        focal_alpha_pos=_override_or_default(
            "focal_alpha_pos", model_args.focal_alpha_pos
        ),
        asym_gamma_pos=float(
            _override_or_default("asym_gamma_pos", model_args.asym_gamma_pos)
        ),
        asym_gamma_neg=float(
            _override_or_default("asym_gamma_neg", model_args.asym_gamma_neg)
        ),
        asym_alpha_pos=_override_or_default(
            "asym_alpha_pos", model_args.asym_alpha_pos
        ),
    )


class DNADataset(Dataset):
    """DNA sequence dataset with optional pre-encoded feature cache.

    Parameters
    ----------
    examples : Sequence[tuple[str, int]]
        Sequence/label pairs.
    window_len : int, default=50
        Effective window length for one-hot encoding.
    preencode : bool, default=False
        If ``True``, precompute all one-hot tensors once at dataset creation.
        This reduces CPU overhead during training, especially on MPS.
    """

    def __init__(
        self,
        examples: Sequence[Tuple[str, int]],
        window_len: int = 50,
        preencode: bool = False,
    ) -> None:
        self.examples: list[Tuple[str, int]] = list(examples)
        self.window_len: int = window_len
        self.preencode: bool = preencode
        self._cached_x: Optional[torch.Tensor]
        self._cached_y: Optional[torch.Tensor]
        if preencode:
            encoded = np.stack(
                [
                    one_hot_encode_dna(seq, self.window_len)
                    for seq, _ in self.examples
                ]
            ).astype(np.float32, copy=False)
            labels = np.asarray(
                [label for _, label in self.examples],
                dtype=np.float32,
            )
            self._cached_x = torch.from_numpy(encoded)
            self._cached_y = torch.from_numpy(labels)
        else:
            self._cached_x = None
            self._cached_y = None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_x is not None and self._cached_y is not None:
            return self._cached_x[idx], self._cached_y[idx]
        seq, label = self.examples[idx]
        x = one_hot_encode_dna(seq, self.window_len)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


class _TCNBlock(nn.Module):
    """Temporal convolutional 1D convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if causal:
            padding = (kernel_size - 1) * dilation
            self.chomp_size = padding
        else:
            padding = (kernel_size // 2) * dilation
            self.chomp_size = 0
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.proj: Optional[nn.Module]
        if in_channels == out_channels:
            self.proj = None
        else:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        """Trim right-side padded frames when causal mode is enabled."""
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self._chomp(out)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self._chomp(out)
        out = self.bn2(out)
        out = self.dropout(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        return F.relu(out, inplace=True)


class TCNSpliceCNN(nn.Module):
    """CNN for splice-site scoring with temporal convolutional blocks."""

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: Optional[Sequence[int]] = None,
        kernel_size: int = 7,
        dropout: float = 0.3,
        fc_hidden: int = 128,
        block_repeats: int = 2,
        causal: bool = False,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = [64, 128, 256]
        channels = list(conv_channels)
        if not channels:
            raise ValueError("conv_channels must not be empty.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding.")
        if block_repeats <= 0:
            raise ValueError("block_repeats must be positive.")

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                channels[0],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        prev_channels = channels[0]
        dilation = 1
        for _ in range(block_repeats):
            for out_channels in channels:
                blocks.append(
                    _TCNBlock(
                        in_channels=prev_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                        causal=causal,
                    )
                )
                prev_channels = out_channels
                dilation = min(dilation * 2, 16384)
        self.blocks = nn.ModuleList(blocks)

        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(prev_channels, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        logits = self.fc(x).squeeze(-1)
        return logits


def stratified_split(
    examples: Sequence[Tuple[str, int]], val_frac: float = 0.1, seed: int = 1337
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    rng = random.Random(seed)
    pos = [(s, y) for s, y in examples if y == 1]
    neg = [(s, y) for s, y in examples if y == 0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_val_pos = max(1, int(len(pos) * val_frac))
    n_val_neg = max(1, int(len(neg) * val_frac))

    train = pos[n_val_pos:] + neg[n_val_neg:]
    val = pos[:n_val_pos] + neg[:n_val_neg]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    """Evaluate a model on a validation loader."""
    model.eval()
    all_logits = []
    all_labels = []
    use_non_blocking = device == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=use_non_blocking)
        y = y.to(device, non_blocking=use_non_blocking)
        if use_amp and device == "cuda" and amp_dtype is not None:
            amp_context: ContextManager[object] = torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            )
        else:
            amp_context = nullcontext()
        with amp_context:
            logits = model(x)
        # numpy does not support torch.bfloat16 directly.
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(y.float().cpu().numpy())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    probs = sigmoid_np(logits) if logits.size else np.array([])

    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    labels = labels.astype(np.int32)

    metrics: Dict[str, float] = {}
    if labels.size:
        metrics["acc@0.5"] = float(np.mean((probs >= 0.5) == (labels >= 0.5)))

        if len(np.unique(labels)) > 1:
            roc_auc_value: Optional[float] = None
            if roc_auc_score is not None:
                try:
                    roc_auc_value = float(roc_auc_score(labels, probs))
                except Exception:
                    roc_auc_value = None
            if roc_auc_value is None:
                try:
                    roc_auc_value = _fallback_roc_auc(labels, probs)
                except ValueError:
                    roc_auc_value = None
            if roc_auc_value is not None:
                metrics["roc_auc"] = roc_auc_value

            pr_auc_value: Optional[float] = None
            if average_precision_score is not None:
                try:
                    pr_auc_value = float(average_precision_score(labels, probs))
                except Exception:
                    pr_auc_value = None
            if pr_auc_value is None:
                try:
                    pr_auc_value = _fallback_average_precision(labels, probs)
                except ValueError:
                    pr_auc_value = None
            if pr_auc_value is not None:
                metrics["pr_auc"] = pr_auc_value

    return metrics


def train_task_model(
    task: str,
    pos_path: str,
    neg_path: str,
    checkpoint_path: str,
    window_len: int,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    epochs: int = 20,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    batch_size: int = 512,
    lr: float = 5e-4,
    seed: int = 1337,
    lightweight: bool = False,
    conv_channels: Optional[Sequence[int]] = None,
    kernel_size: int = 7,
    tcn_block_repeats: int = 2,
    tcn_causal: Union[bool, int] = 0,
    dropout: float = 0.3,
    fc_hidden: int = 128,
    weight_decay: float = 0.01,
    eta_min_ratio: float = 0.01,
    val_frac: float = 0.1,
    grad_clip: float = 5.0,
    compile_model: bool = False,
    compile_mode: str = "auto",
    device: str = "auto",
    loss_name: str = "weighted_bce",
    pos_weight_cap: float = 20.0,
    focal_gamma: float = 2.0,
    focal_alpha_pos: Optional[float] = None,
    asym_gamma_pos: float = 0.0,
    asym_gamma_neg: float = 4.0,
    asym_alpha_pos: Optional[float] = None,
    use_amp: Union[bool, int] = 1,
    amp_dtype: str = "auto",
    allow_tf32: Union[bool, int] = 1,
    cudnn_benchmark: Union[bool, int] = 1,
    deterministic: Union[bool, int] = 0,
    num_workers: Union[str, int] = "auto",
    prefetch_factor: int = 4,
    persistent_workers: Union[bool, int] = 1,
    pin_memory: Union[bool, int] = 1,
    min_batch_size: int = 64,
    max_oom_retries: int = 8,
    quick_phase: bool = False,
    gpu_id: Optional[int] = None,
) -> Dict[str, object]:
    """Train one task model with GPU-oriented runtime settings.

    Parameters
    ----------
    task : str
        Task name (``donor`` or ``acceptor``).
    pos_path : str
        Positive training examples path.
    neg_path : str
        Negative training examples path.
    checkpoint_path : str
        Output checkpoint path.
    window_len : int
        Sequence window length.
    donor_len : int | None
        Donor window length.
    acceptor_len : int | None
        Acceptor window length.
    epochs : int, default=20
        Number of epochs.
    batch_size : int, default=512
        Initial batch size.
    lr : float, default=5e-4
        Learning rate.
    seed : int, default=1337
        Random seed.
    lightweight : bool, default=False
        Use lightweight architecture preset.
    conv_channels : Sequence[int] | None, default=None
        Convolution channels.
    kernel_size : int, default=7
        Convolution kernel size.
    tcn_block_repeats : int, default=2
        Number of times the channel schedule is repeated across dilated blocks.
    tcn_causal : bool | int, default=0
        Whether to use causal convolutions in temporal blocks.
    dropout : float, default=0.3
        Dropout rate.
    fc_hidden : int, default=128
        Hidden units in fully connected block.
    weight_decay : float, default=0.01
        AdamW weight decay.
    eta_min_ratio : float, default=0.01
        Scheduler eta_min ratio.
    val_frac : float, default=0.1
        Validation fraction.
    grad_clip : float, default=5.0
        Gradient clip max norm.
    compile_model : bool, default=False
        Legacy compile flag.
    compile_mode : str, default="auto"
        Compile mode: ``off|on|auto``.
    device : str, default="auto"
        Device preference.
    loss_name : str, default="weighted_bce"
        Loss function name.
    pos_weight_cap : float, default=20.0
        Max positive class weight.
    focal_gamma : float, default=2.0
        Focal gamma.
    focal_alpha_pos : float | None, default=None
        Focal positive alpha.
    asym_gamma_pos : float, default=0.0
        Asymmetric focal positive gamma.
    asym_gamma_neg : float, default=4.0
        Asymmetric focal negative gamma.
    asym_alpha_pos : float | None, default=None
        Asymmetric focal positive alpha.
    use_amp : bool | int, default=1
        Whether to use AMP on CUDA.
    amp_dtype : str, default="auto"
        AMP dtype strategy: ``auto|bf16|fp16``.
    allow_tf32 : bool | int, default=1
        Whether to enable TF32 on CUDA.
    cudnn_benchmark : bool | int, default=1
        Whether to enable cuDNN benchmark.
    deterministic : bool | int, default=0
        Whether to force deterministic algorithms.
    num_workers : str | int, default="auto"
        DataLoader workers.
    prefetch_factor : int, default=4
        DataLoader prefetch factor when workers are enabled.
    persistent_workers : bool | int, default=1
        Keep DataLoader workers alive between epochs.
    pin_memory : bool | int, default=1
        Enable pin memory for DataLoader.
    min_batch_size : int, default=64
        Minimum batch size when retrying after CUDA OOM.
    max_oom_retries : int, default=8
        Maximum OOM retries.
    quick_phase : bool, default=False
        Whether this run is a quick-phase trial.
    gpu_id : int | None, default=None
        Assigned GPU id for sweep logs.

    Returns
    -------
    dict[str, object]
        Task training summary with validation and runtime metadata.

    Raises
    ------
    ValueError
        If public arguments are invalid.
    RuntimeError
        If training fails for non-recoverable runtime errors.
    """
    if kernel_size <= 0:
        raise ValueError("--kernel_size must be positive.")
    if tcn_block_repeats <= 0:
        raise ValueError("--tcn_block_repeats must be positive.")
    if fc_hidden <= 0:
        raise ValueError("--fc_hidden must be positive.")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("--dropout must satisfy 0 <= dropout < 1.")
    if weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative.")
    if eta_min_ratio < 0.0:
        raise ValueError("--eta_min_ratio must be non-negative.")
    if val_frac <= 0.0 or val_frac >= 1.0:
        raise ValueError("--val_frac must satisfy 0 < val_frac < 1.")
    if grad_clip < 0.0:
        raise ValueError("--grad_clip must be non-negative.")
    if prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive.")
    if min_batch_size <= 0:
        raise ValueError("--min_batch_size must be positive.")
    if max_oom_retries < 0:
        raise ValueError("--max_oom_retries must be >= 0.")
    if batch_size < min_batch_size:
        raise ValueError("--batch_size must be >= --min_batch_size.")
    tcn_causal_bool = _bool_from_flag(tcn_causal)

    device = pick_device(device)
    resolved_num_workers = _resolve_num_workers(num_workers, device=device)
    use_pin_memory = _bool_from_flag(pin_memory) and device == "cuda"
    use_persistent_workers = (
        _bool_from_flag(persistent_workers) and resolved_num_workers > 0
    )
    use_amp_bool = _bool_from_flag(use_amp) and device == "cuda"
    allow_tf32_bool = _bool_from_flag(allow_tf32)
    deterministic_bool = _bool_from_flag(deterministic)
    cudnn_benchmark_bool = _bool_from_flag(cudnn_benchmark)
    amp_dtype_resolved = _resolve_amp_dtype(amp_dtype, device)
    compile_enabled = _resolve_compile_enabled(
        compile_mode=compile_mode,
        compile_flag=compile_model,
        quick_phase=quick_phase,
        device=device,
        epochs=epochs,
    )

    set_seed(
        seed=seed,
        deterministic=deterministic_bool,
        cudnn_benchmark=cudnn_benchmark_bool,
        allow_tf32=allow_tf32_bool,
    )
    total_started_at = time.perf_counter()
    timing_sec: dict[str, float] = {
        "read_examples": 0.0,
        "split_examples": 0.0,
        "dataset_build": 0.0,
        "loader_build": 0.0,
        "model_setup": 0.0,
        "train_data_wait": 0.0,
        "train_step": 0.0,
        "validation": 0.0,
        "total": 0.0,
    }
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    read_started_at = time.perf_counter()
    examples = read_examples_single_task(
        pos_path,
        neg_path,
        task,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )
    timing_sec["read_examples"] = time.perf_counter() - read_started_at

    n_pos = sum(y for _, y in examples)
    n_neg = len(examples) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Insufficient training examples for {task}: pos={n_pos}, neg={n_neg}."
        )

    split_started_at = time.perf_counter()
    train_ex, val_ex = stratified_split(examples, val_frac=val_frac, seed=seed)
    timing_sec["split_examples"] = time.perf_counter() - split_started_at
    print(
        f"[{task}] device={device} total={len(examples)} "
        f"(pos={n_pos}, neg={n_neg}) train={len(train_ex)} val={len(val_ex)}"
    )
    preencode_dataset = device == "mps"
    if preencode_dataset:
        print(f"[{task}] dataset pre-encoding enabled for mps.")
    dataset_started_at = time.perf_counter()
    train_ds = DNADataset(
        train_ex,
        window_len=window_len,
        preencode=preencode_dataset,
    )
    val_ds = DNADataset(
        val_ex,
        window_len=window_len,
        preencode=preencode_dataset,
    )
    timing_sec["dataset_build"] = time.perf_counter() - dataset_started_at

    if conv_channels is None:
        conv_channels = [64, 128] if lightweight else [64, 128, 256]
    else:
        conv_channels = list(conv_channels)
    train_pos = sum(y for _, y in train_ex)
    train_neg = len(train_ex) - train_pos
    criterion, loss_meta = build_binary_classification_loss(
        loss_name=loss_name,
        train_pos=train_pos,
        train_neg=train_neg,
        device=device,
        pos_weight_cap=pos_weight_cap,
        focal_gamma=focal_gamma,
        focal_alpha_pos=focal_alpha_pos,
        asym_gamma_pos=asym_gamma_pos,
        asym_gamma_neg=asym_gamma_neg,
        asym_alpha_pos=asym_alpha_pos,
    )

    effective_batch_size = batch_size
    if device == "mps":
        mps_max_batch_size = _resolve_mps_max_batch_size()
        if effective_batch_size > mps_max_batch_size:
            print(
                f"[{task}] mps batch clamp: {effective_batch_size} -> "
                f"{mps_max_batch_size} "
                "(set INTRONMODEL_MPS_MAX_BATCH_SIZE to change)."
            )
            effective_batch_size = mps_max_batch_size
    oom_retries = 0
    use_non_blocking = device == "cuda"
    while True:
        saw_training_batch = False
        compile_enabled_attempt = compile_enabled and hasattr(torch, "compile")
        loader_started_at = time.perf_counter()
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
        train_loader_kwargs: dict[str, object] = {
            "dataset": train_ds,
            "batch_size": effective_batch_size,
            "shuffle": True,
            "num_workers": resolved_num_workers,
            "pin_memory": use_pin_memory,
            "worker_init_fn": _seed_worker if resolved_num_workers > 0 else None,
            "generator": loader_generator,
        }
        if resolved_num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = prefetch_factor
            train_loader_kwargs["persistent_workers"] = use_persistent_workers
        train_loader = DataLoader(**train_loader_kwargs)

        val_loader_kwargs: dict[str, object] = {
            "dataset": val_ds,
            "batch_size": effective_batch_size,
            "shuffle": False,
            "num_workers": resolved_num_workers,
            "pin_memory": use_pin_memory,
        }
        if resolved_num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = prefetch_factor
            val_loader_kwargs["persistent_workers"] = use_persistent_workers
        val_loader = DataLoader(**val_loader_kwargs)
        timing_sec["loader_build"] += time.perf_counter() - loader_started_at
        print(
            f"[{task}] loader train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} batch_size={effective_batch_size} "
            f"workers={resolved_num_workers}"
        )

        try:
            model_setup_started_at = time.perf_counter()
            model = TCNSpliceCNN(
                in_channels=4,
                conv_channels=conv_channels,
                kernel_size=kernel_size,
                block_repeats=tcn_block_repeats,
                causal=tcn_causal_bool,
                dropout=dropout,
                fc_hidden=fc_hidden,
            ).to(device)

            if compile_enabled_attempt:
                _configure_triton_tool_paths()
                _configure_torch_compile_runtime()
                ptxas_path = os.environ.get("TRITON_PTXAS_PATH")
                ptxas_blackwell_path = os.environ.get(
                    "TRITON_PTXAS_BLACKWELL_PATH"
                )
                print(
                    f"[{task}] torch.compile requested "
                    f"(ptxas={ptxas_path}, ptxas_blackwell={ptxas_blackwell_path})."
                )
                try:
                    model = torch.compile(model)
                except Exception as exc:
                    compile_enabled_attempt = False
                    compile_enabled = False
                    print(
                        f"[{task}] torch.compile setup failed "
                        f"({exc.__class__.__name__}). Continue without compile."
                    )

            optimizer_impl = "adamw"
            adamw_kwargs: dict[str, object] = {
                "params": model.parameters(),
                "lr": lr,
                "weight_decay": weight_decay,
            }
            if device == "cuda":
                try:
                    optimizer = torch.optim.AdamW(
                        **adamw_kwargs,
                        fused=True,
                    )
                    optimizer_impl = "adamw_fused"
                except (TypeError, RuntimeError):
                    optimizer = torch.optim.AdamW(**adamw_kwargs)
            else:
                optimizer = torch.optim.AdamW(**adamw_kwargs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=lr * eta_min_ratio,
            )
            timing_sec["model_setup"] += (
                time.perf_counter() - model_setup_started_at
            )
            scaler_enabled = (
                use_amp_bool
                and device == "cuda"
                and amp_dtype_resolved == torch.float16
            )
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                scaler = torch.amp.GradScaler(
                    "cuda",
                    enabled=scaler_enabled,
                )
            else:
                scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

            best_score = -1e9
            best_metric_name = "acc@0.5"
            best_epoch = 0
            best_pr_auc: Optional[float] = None
            best_roc_auc: Optional[float] = None
            best_acc_at_0_5: Optional[float] = None
            epoch_history: list[dict[str, object]] = []
            log_every = max(1, epochs // 5)
            epochs_completed = 0
            epochs_since_improvement = 0
            stopped_early = False

            for epoch in range(1, epochs + 1):
                epochs_completed = epoch
                if device == "mps":
                    print(f"[{task}] epoch {epoch}/{epochs} start")
                model.train()
                running_loss = torch.zeros((), dtype=torch.float64)
                train_iterator = iter(train_loader)
                for batch_idx in range(1, len(train_loader) + 1):
                    wait_started_at = time.perf_counter()
                    x, y = next(train_iterator)
                    timing_sec["train_data_wait"] += (
                        time.perf_counter() - wait_started_at
                    )
                    step_started_at = time.perf_counter()
                    saw_training_batch = True
                    x = x.to(device, non_blocking=use_non_blocking)
                    y = y.to(device, non_blocking=use_non_blocking)

                    optimizer.zero_grad(set_to_none=True)
                    if (
                        use_amp_bool
                        and device == "cuda"
                        and amp_dtype_resolved is not None
                    ):
                        amp_context: ContextManager[object] = torch.autocast(
                            device_type="cuda",
                            dtype=amp_dtype_resolved,
                            enabled=True,
                        )
                    else:
                        amp_context = nullcontext()

                    with amp_context:
                        logits = model(x)
                        loss = criterion(logits, y)

                    if scaler_enabled:
                        scaler.scale(loss).backward()
                        if grad_clip > 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                grad_clip,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                grad_clip,
                            )
                        optimizer.step()
                    if device == "mps" and batch_idx == 1:
                        print(f"[{task}] epoch {epoch}/{epochs} first batch done")
                    running_loss = running_loss + loss.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                    timing_sec["train_step"] += (
                        time.perf_counter() - step_started_at
                    )

                scheduler.step()
                train_loss = float(running_loss / max(1, len(train_loader)))

                val_started_at = time.perf_counter()
                val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    use_amp=use_amp_bool,
                    amp_dtype=amp_dtype_resolved,
                )
                timing_sec["validation"] += time.perf_counter() - val_started_at
                pr_auc = val_metrics.get("pr_auc")
                roc_auc = val_metrics.get("roc_auc")
                acc_at_0_5 = val_metrics.get("acc@0.5")
                if pr_auc is not None:
                    best_pr_auc = (
                        pr_auc if best_pr_auc is None else max(best_pr_auc, pr_auc)
                    )
                if roc_auc is not None:
                    best_roc_auc = (
                        roc_auc
                        if best_roc_auc is None
                        else max(best_roc_auc, roc_auc)
                    )
                if acc_at_0_5 is not None:
                    best_acc_at_0_5 = (
                        acc_at_0_5
                        if best_acc_at_0_5 is None
                        else max(best_acc_at_0_5, acc_at_0_5)
                    )

                if pr_auc is not None:
                    score = pr_auc
                    score_name = "pr_auc"
                elif roc_auc is not None:
                    score = roc_auc
                    score_name = "roc_auc"
                else:
                    score = float(val_metrics.get("acc@0.5", 0.0))
                    score_name = "acc@0.5"

                improved = score > (best_score + early_stop_min_delta)
                if improved:
                    epochs_since_improvement = 0
                    best_score = score
                    best_metric_name = score_name
                    best_epoch = epoch
                    state_dict_to_save = _export_model_state_dict(model)
                    torch.save(
                        {
                            "task": task,
                            "window_len": window_len,
                            "model_config": {
                                "conv_channels": list(conv_channels),
                                "kernel_size": kernel_size,
                                "tcn_block_repeats": tcn_block_repeats,
                                "tcn_causal": tcn_causal_bool,
                                "dropout": dropout,
                                "fc_hidden": fc_hidden,
                            },
                            "model_state": state_dict_to_save,
                        },
                        checkpoint_path,
                    )
                else:
                    epochs_since_improvement += 1

                epoch_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "pr_auc": pr_auc,
                        "roc_auc": roc_auc,
                        "acc@0.5": acc_at_0_5,
                        "objective_metric": score_name,
                        "objective_score": score,
                        "improved": improved,
                        "best_metric": best_metric_name,
                        "best_score": float(best_score),
                        "best_epoch": best_epoch,
                    }
                )

                should_log = (
                    epoch == 1
                    or epoch == epochs
                    or epoch % log_every == 0
                    or improved
                )
                if should_log:
                    mark = "*" if improved else "-"
                    print(
                        f"[{task}] {mark} epoch {epoch}/{epochs} "
                        f"loss={train_loss:.4f} {score_name}={score:.4f} "
                        f"best={best_score:.4f} (ep {best_epoch})"
                    )

                if early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience:
                    stopped_early = True
                    print(
                        f"[{task}] early stop at epoch {epoch} "
                        f"(patience={early_stop_patience}, min_delta={early_stop_min_delta:g})"
                    )
                    break

            print(
                f"[{task}] done best_{best_metric_name}={best_score:.4f} "
                f"at epoch {best_epoch}"
            )
            timing_sec["total"] = time.perf_counter() - total_started_at
            total_profile_sec = (
                timing_sec["read_examples"]
                + timing_sec["split_examples"]
                + timing_sec["dataset_build"]
                + timing_sec["loader_build"]
                + timing_sec["model_setup"]
                + timing_sec["train_data_wait"]
                + timing_sec["train_step"]
                + timing_sec["validation"]
            )
            timing_ratio: dict[str, float] = {}
            if total_profile_sec > 0.0:
                for key, value in timing_sec.items():
                    if key == "total":
                        continue
                    timing_ratio[key] = value / total_profile_sec
            print(
                f"[{task}] timing total={timing_sec['total']:.3f}s "
                f"data_wait={timing_sec['train_data_wait']:.3f}s "
                f"train_step={timing_sec['train_step']:.3f}s "
                f"val={timing_sec['validation']:.3f}s"
            )
            return {
                "task": task,
                "num_examples": len(examples),
                "num_pos": n_pos,
                "num_neg": n_neg,
                "best_metric": best_metric_name,
                "best_epoch": best_epoch,
                "best_score": float(best_score),
                "best_pr_auc": best_pr_auc,
                "best_roc_auc": best_roc_auc,
                "best_acc_at_0_5": best_acc_at_0_5,
                "epoch_history": epoch_history,
                "epochs_completed": epochs_completed,
                "stopped_early": stopped_early,
                "early_stop_patience": early_stop_patience,
                "early_stop_min_delta": early_stop_min_delta,
                "checkpoint": checkpoint_path,
                "loss": loss_name,
                "pos_weight": loss_meta["pos_weight"],
                "focal_gamma": loss_meta["focal_gamma"],
                "focal_alpha_pos": loss_meta["focal_alpha_pos"],
                "asym_gamma_pos": loss_meta["asym_gamma_pos"],
                "asym_gamma_neg": loss_meta["asym_gamma_neg"],
                "asym_alpha_pos": loss_meta["asym_alpha_pos"],
                "conv_channels": list(conv_channels),
                "kernel_size": kernel_size,
                "tcn_block_repeats": tcn_block_repeats,
                "tcn_causal": tcn_causal_bool,
                "dropout": dropout,
                "fc_hidden": fc_hidden,
                "weight_decay": weight_decay,
                "eta_min_ratio": eta_min_ratio,
                "val_frac": val_frac,
                "grad_clip": grad_clip,
                "compile_enabled": compile_enabled_attempt,
                "use_amp": use_amp_bool,
                "amp_dtype": (
                    str(amp_dtype_resolved).replace("torch.", "")
                    if amp_dtype_resolved is not None
                    else None
                ),
                "allow_tf32": allow_tf32_bool,
                "cudnn_benchmark": cudnn_benchmark_bool,
                "deterministic": deterministic_bool,
                "num_workers": resolved_num_workers,
                "prefetch_factor": (
                    prefetch_factor if resolved_num_workers > 0 else None
                ),
                "persistent_workers": use_persistent_workers,
                "pin_memory": use_pin_memory,
                "effective_batch_size": effective_batch_size,
                "oom_retries": oom_retries,
                "gpu_id": gpu_id,
                "quick_phase": quick_phase,
                "optimizer_impl": optimizer_impl,
                "timing_sec": timing_sec,
                "timing_ratio": timing_ratio,
            }
        except RuntimeError as exc:
            is_compile_failure = (
                compile_enabled_attempt and _is_compile_runtime_error(exc)
            )
            if is_compile_failure:
                compile_enabled = False
                print(
                    f"[{task}] torch.compile runtime failed "
                    f"({exc.__class__.__name__}). Retry without compile."
                )
                _empty_device_cache(device)
                continue

            is_device_oom = False
            if device == "cuda":
                is_device_oom = _is_cuda_oom_error(exc)
            elif device == "mps":
                is_device_oom = _is_mps_oom_error(exc)
            if is_device_oom and not saw_training_batch:
                raise RuntimeError(
                    "NON_RETRYABLE_OOM: OOM occurred before first training batch. "
                    "Model config is likely too large for the device."
                ) from exc
            should_retry = (
                is_device_oom
                and oom_retries < max_oom_retries
                and effective_batch_size > min_batch_size
            )
            if not should_retry:
                raise
            next_batch_size = max(min_batch_size, effective_batch_size // 2)
            if next_batch_size >= effective_batch_size:
                raise
            oom_retries += 1
            print(
                f"[{task}] {device.upper()} OOM detected. "
                "Retry with smaller batch size: "
                f"{effective_batch_size} -> {next_batch_size} "
                f"(retry {oom_retries}/{max_oom_retries})"
            )
            effective_batch_size = next_batch_size
            _empty_device_cache(device)


def load_task_model(checkpoint_path: str, device: str) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _normalize_checkpoint_state_dict(ckpt["model_state"])

    model_config_obj = ckpt.get("model_config", {})
    model_config = model_config_obj if isinstance(model_config_obj, dict) else {}
    conv_channels = model_config.get("conv_channels")
    kernel_size = int(model_config.get("kernel_size", 7))
    tcn_block_repeats = int(model_config.get("tcn_block_repeats", 2))
    tcn_causal = bool(model_config.get("tcn_causal", False))
    dropout = float(model_config.get("dropout", 0.3))
    fc_hidden = int(model_config.get("fc_hidden", 128))

    if conv_channels is None:
        conv_channels = [64, 128, 256]

    model = TCNSpliceCNN(
        in_channels=4,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        block_repeats=tcn_block_repeats,
        causal=tcn_causal,
        dropout=dropout,
        fc_hidden=fc_hidden,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_sequences(
    model: nn.Module,
    sequences: Sequence[str],
    window_len: int,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    if not sequences:
        return np.array([])

    model.eval()
    encoded = [one_hot_encode_dna(seq.upper(), window_len) for seq in sequences]
    x = torch.from_numpy(np.stack(encoded)).to(device)

    all_probs = []
    for i in range(0, len(x), batch_size):
        batch_x = x[i : i + batch_size]
        logits = model(batch_x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)


def infer_site_scores(
    site_rows: List[Dict[str, object]],
    donor_model_path: str,
    acceptor_model_path: str,
    device: str = "auto",
    batch_size: int = 512,
) -> List[Dict[str, object]]:
    device = pick_device(device)

    donor_model, donor_ckpt = load_task_model(donor_model_path, device)
    acceptor_model, acceptor_ckpt = load_task_model(acceptor_model_path, device)

    donor_window_len = int(donor_ckpt.get("window_len", 50))
    acceptor_window_len = int(acceptor_ckpt.get("window_len", 50))

    donor_seqs = [str(r["seq"]) for r in site_rows if r["site_type"] == "donor"]
    acceptor_seqs = [str(r["seq"]) for r in site_rows if r["site_type"] == "acceptor"]

    donor_scores = score_sequences(
        donor_model,
        donor_seqs,
        donor_window_len,
        device,
        batch_size=batch_size,
    )
    acceptor_scores = score_sequences(
        acceptor_model,
        acceptor_seqs,
        acceptor_window_len,
        device,
        batch_size=batch_size,
    )

    out_rows: List[Dict[str, object]] = []
    donor_idx = 0
    acceptor_idx = 0
    for row in site_rows:
        site_type = str(row["site_type"])
        if site_type == "donor":
            if donor_idx < len(donor_scores):
                score = float(donor_scores[donor_idx])
            else:
                score = 0.0
            donor_idx += 1
        else:
            score = (
                float(acceptor_scores[acceptor_idx])
                if acceptor_idx < len(acceptor_scores)
                else 0.0
            )
            acceptor_idx += 1

        out_rows.append(
            {
                "transcript_id": row["transcript_id"],
                "intron_index": int(row["intron_index"]),
                "site_type": site_type,
                "score": score,
            }
        )

    return out_rows


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register CNN-specific training arguments."""
    parser.add_argument(
        "--epochs",
        type=str,
        default="20",
        help="Epoch count (positive integer) or auto for early-stop mode.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Upper epoch limit used when --epochs=auto.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=12,
        help="Early-stop patience (epochs without improvement).",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation-metric improvement to reset patience.",
    )
    parser.add_argument(
        "--train_target",
        choices=["both", "donor", "acceptor"],
        default="both",
        help=(
            "Training target. 'both' trains donor and acceptor. "
            "'donor'/'acceptor' train one task only (for tuning)."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument(
        "--conv_channels",
        type=str,
        default=None,
        help=(
            "Comma-separated convolution channels, e.g. 64,128,256. "
            "If omitted, default architecture is used."
        ),
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=7,
        help="Convolution kernel size.",
    )
    parser.add_argument(
        "--tcn_block_repeats",
        type=int,
        default=2,
        help="How many times to repeat the dilated TCN channel schedule.",
    )
    parser.add_argument(
        "--tcn_causal",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use causal dilated convolutions when set to 1.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate used in convolution and fully-connected blocks.",
    )
    parser.add_argument(
        "--fc_hidden",
        type=int,
        default=128,
        help="Hidden units in the fully-connected block.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--eta_min_ratio",
        type=float,
        default=0.01,
        help="CosineAnnealingLR eta_min as lr * eta_min_ratio.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Validation split fraction for stratified split.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="Gradient clipping max norm. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--donor_batch_size",
        type=int,
        default=None,
        help="Donor-only override for --batch_size.",
    )
    parser.add_argument(
        "--acceptor_batch_size",
        type=int,
        default=None,
        help="Acceptor-only override for --batch_size.",
    )
    parser.add_argument(
        "--donor_lr",
        type=float,
        default=None,
        help="Donor-only override for --lr.",
    )
    parser.add_argument(
        "--acceptor_lr",
        type=float,
        default=None,
        help="Acceptor-only override for --lr.",
    )
    parser.add_argument(
        "--donor_conv_channels",
        type=str,
        default=None,
        help="Donor-only override for --conv_channels.",
    )
    parser.add_argument(
        "--acceptor_conv_channels",
        type=str,
        default=None,
        help="Acceptor-only override for --conv_channels.",
    )
    parser.add_argument(
        "--donor_kernel_size",
        type=int,
        default=None,
        help="Donor-only override for --kernel_size.",
    )
    parser.add_argument(
        "--acceptor_kernel_size",
        type=int,
        default=None,
        help="Acceptor-only override for --kernel_size.",
    )
    parser.add_argument(
        "--donor_tcn_block_repeats",
        type=int,
        default=None,
        help="Donor-only override for --tcn_block_repeats.",
    )
    parser.add_argument(
        "--acceptor_tcn_block_repeats",
        type=int,
        default=None,
        help="Acceptor-only override for --tcn_block_repeats.",
    )
    parser.add_argument(
        "--donor_tcn_causal",
        type=int,
        choices=[0, 1],
        default=None,
        help="Donor-only override for --tcn_causal.",
    )
    parser.add_argument(
        "--acceptor_tcn_causal",
        type=int,
        choices=[0, 1],
        default=None,
        help="Acceptor-only override for --tcn_causal.",
    )
    parser.add_argument(
        "--donor_dropout",
        type=float,
        default=None,
        help="Donor-only override for --dropout.",
    )
    parser.add_argument(
        "--acceptor_dropout",
        type=float,
        default=None,
        help="Acceptor-only override for --dropout.",
    )
    parser.add_argument(
        "--donor_fc_hidden",
        type=int,
        default=None,
        help="Donor-only override for --fc_hidden.",
    )
    parser.add_argument(
        "--acceptor_fc_hidden",
        type=int,
        default=None,
        help="Acceptor-only override for --fc_hidden.",
    )
    parser.add_argument(
        "--donor_weight_decay",
        type=float,
        default=None,
        help="Donor-only override for --weight_decay.",
    )
    parser.add_argument(
        "--acceptor_weight_decay",
        type=float,
        default=None,
        help="Acceptor-only override for --weight_decay.",
    )
    parser.add_argument(
        "--donor_eta_min_ratio",
        type=float,
        default=None,
        help="Donor-only override for --eta_min_ratio.",
    )
    parser.add_argument(
        "--acceptor_eta_min_ratio",
        type=float,
        default=None,
        help="Acceptor-only override for --eta_min_ratio.",
    )
    parser.add_argument(
        "--donor_val_frac",
        type=float,
        default=None,
        help="Donor-only override for --val_frac.",
    )
    parser.add_argument(
        "--acceptor_val_frac",
        type=float,
        default=None,
        help="Acceptor-only override for --val_frac.",
    )
    parser.add_argument(
        "--donor_grad_clip",
        type=float,
        default=None,
        help="Donor-only override for --grad_clip.",
    )
    parser.add_argument(
        "--acceptor_grad_clip",
        type=float,
        default=None,
        help="Acceptor-only override for --grad_clip.",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=["off", "on", "auto"],
        default="auto",
        help="Compilation mode for torch.compile.",
    )
    parser.add_argument(
        "--use_amp",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable CUDA automatic mixed precision when set to 1.",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="AMP dtype for CUDA autocast.",
    )
    parser.add_argument(
        "--allow_tf32",
        type=int,
        choices=[0, 1],
        default=1,
        help="Allow TF32 on CUDA matmul and cuDNN when set to 1.",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable cuDNN benchmark autotuning when set to 1.",
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable deterministic algorithms when set to 1.",
    )
    parser.add_argument(
        "--num_workers",
        default="auto",
        help="DataLoader worker count. Use integer or auto.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch factor (effective when num_workers > 0).",
    )
    parser.add_argument(
        "--persistent_workers",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable DataLoader persistent workers when set to 1.",
    )
    parser.add_argument(
        "--pin_memory",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable DataLoader pin_memory when set to 1.",
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=64,
        help="Minimum batch size for CUDA OOM backoff retries.",
    )
    parser.add_argument(
        "--max_oom_retries",
        type=int,
        default=8,
        help="Maximum retries when reducing batch size after CUDA OOM.",
    )
    parser.add_argument(
        "--loss",
        choices=list(LOSS_NAME_CHOICES),
        default="weighted_bce",
        help="Training loss type for donor/acceptor models.",
    )
    parser.add_argument(
        "--donor_loss",
        choices=list(LOSS_NAME_CHOICES),
        default=None,
        help="Donor-only override for --loss.",
    )
    parser.add_argument(
        "--acceptor_loss",
        choices=list(LOSS_NAME_CHOICES),
        default=None,
        help="Acceptor-only override for --loss.",
    )
    parser.add_argument(
        "--pos_weight_cap",
        type=float,
        default=20.0,
        help="Upper bound of positive-class weight for weighted_bce.",
    )
    parser.add_argument(
        "--donor_pos_weight_cap",
        type=float,
        default=None,
        help="Donor-only override for --pos_weight_cap.",
    )
    parser.add_argument(
        "--acceptor_pos_weight_cap",
        type=float,
        default=None,
        help="Acceptor-only override for --pos_weight_cap.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter used when --loss focal is selected.",
    )
    parser.add_argument(
        "--donor_focal_gamma",
        type=float,
        default=None,
        help="Donor-only override for --focal_gamma.",
    )
    parser.add_argument(
        "--acceptor_focal_gamma",
        type=float,
        default=None,
        help="Acceptor-only override for --focal_gamma.",
    )
    parser.add_argument(
        "--focal_alpha_pos",
        type=float,
        default=None,
        help=(
            "Positive-class alpha for focal loss (0 < alpha < 1). "
            "If omitted, it is inferred from class imbalance."
        ),
    )
    parser.add_argument(
        "--asym_gamma_pos",
        type=float,
        default=0.0,
        help="Positive-class gamma for --loss asymmetric_focal.",
    )
    parser.add_argument(
        "--donor_asym_gamma_pos",
        type=float,
        default=None,
        help="Donor-only override for --asym_gamma_pos.",
    )
    parser.add_argument(
        "--acceptor_asym_gamma_pos",
        type=float,
        default=None,
        help="Acceptor-only override for --asym_gamma_pos.",
    )
    parser.add_argument(
        "--asym_gamma_neg",
        type=float,
        default=4.0,
        help="Negative-class gamma for --loss asymmetric_focal.",
    )
    parser.add_argument(
        "--donor_asym_gamma_neg",
        type=float,
        default=None,
        help="Donor-only override for --asym_gamma_neg.",
    )
    parser.add_argument(
        "--acceptor_asym_gamma_neg",
        type=float,
        default=None,
        help="Acceptor-only override for --asym_gamma_neg.",
    )
    parser.add_argument(
        "--asym_alpha_pos",
        type=float,
        default=None,
        help=(
            "Positive-class alpha for --loss asymmetric_focal "
            "(0 < alpha < 1). If omitted, inferred from class imbalance."
        ),
    )
    parser.add_argument(
        "--donor_focal_alpha_pos",
        type=float,
        default=None,
        help="Donor-only override for --focal_alpha_pos.",
    )
    parser.add_argument(
        "--acceptor_focal_alpha_pos",
        type=float,
        default=None,
        help="Acceptor-only override for --focal_alpha_pos.",
    )
    parser.add_argument(
        "--donor_asym_alpha_pos",
        type=float,
        default=None,
        help="Donor-only override for --asym_alpha_pos.",
    )
    parser.add_argument(
        "--acceptor_asym_alpha_pos",
        type=float,
        default=None,
        help="Acceptor-only override for --asym_alpha_pos.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional run-name suffix for training summary.",
    )


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register CNN-specific inference arguments."""
    parser.add_argument("--batch_size", type=int, default=512)


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train donor/acceptor CNN models with unified argument interface."""
    shared_conv_channels = parse_conv_channels(model_args.conv_channels)
    donor_conv_channels = parse_conv_channels(
        getattr(model_args, "donor_conv_channels", None),
        arg_name="--donor_conv_channels",
    )
    acceptor_conv_channels = parse_conv_channels(
        getattr(model_args, "acceptor_conv_channels", None),
        arg_name="--acceptor_conv_channels",
    )

    train_pos_path, train_neg_path, inferred_train_len = resolve_train_paths(
        species=common_args.species,
        train_pos_path=common_args.train_pos_path,
        train_neg_path=common_args.train_neg_path,
        donor_len=common_args.donor_len,
        acceptor_len=common_args.acceptor_len,
    )

    donor_len, acceptor_len = resolve_effective_window_lengths(
        donor_len=common_args.donor_len,
        acceptor_len=common_args.acceptor_len,
        inferred_train_len=inferred_train_len,
    )
    validate_window_args(
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )

    donor_window_len = donor_len if donor_len is not None else 50
    acceptor_window_len = acceptor_len if acceptor_len is not None else 50

    donor_checkpoint_path = str(
        getattr(common_args, "donor_checkpoint_path", "")
    ).strip()
    acceptor_checkpoint_path = str(
        getattr(common_args, "acceptor_checkpoint_path", "")
    ).strip()
    if not donor_checkpoint_path:
        raise ValueError("Missing donor checkpoint path in common_args.")
    if not acceptor_checkpoint_path:
        raise ValueError("Missing acceptor checkpoint path in common_args.")

    train_target = str(getattr(model_args, "train_target", "both")).strip().lower()
    if train_target not in {"both", "donor", "acceptor"}:
        raise ValueError("--train_target must be one of: both, donor, acceptor.")

    resolved_epochs, epochs_auto = resolve_training_epoch_budget(
        epochs_arg=model_args.epochs,
        max_epochs=int(model_args.max_epochs),
    )
    early_stop_patience, early_stop_min_delta = resolve_early_stopping_params(
        patience_arg=model_args.early_stop_patience,
        min_delta_arg=model_args.early_stop_min_delta,
    )
    effective_early_stop_patience = early_stop_patience if epochs_auto else 0

    tasks_to_train = (
        ["donor", "acceptor"] if train_target == "both" else [train_target]
    )

    task_checkpoint_paths = {
        "donor": donor_checkpoint_path,
        "acceptor": acceptor_checkpoint_path,
    }
    task_window_len = {
        "donor": donor_window_len,
        "acceptor": acceptor_window_len,
    }

    task_hparams: dict[str, TaskTrainParams] = {}
    task_metrics: dict[str, Dict[str, object]] = {}
    for task in tasks_to_train:
        resolved = _resolve_task_train_params(
            task=task,
            model_args=model_args,
            shared_conv_channels=shared_conv_channels,
            donor_conv_channels=donor_conv_channels,
            acceptor_conv_channels=acceptor_conv_channels,
        )
        task_hparams[task] = resolved
        task_metrics[task] = train_task_model(
            task=task,
            pos_path=train_pos_path,
            neg_path=train_neg_path,
            checkpoint_path=task_checkpoint_paths[task],
            window_len=task_window_len[task],
            donor_len=donor_len,
            acceptor_len=acceptor_len,
            epochs=resolved_epochs,
            early_stop_patience=effective_early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            batch_size=resolved.batch_size,
            lr=resolved.lr,
            seed=common_args.seed,
            lightweight=model_args.lightweight,
            conv_channels=resolved.conv_channels,
            kernel_size=resolved.kernel_size,
            tcn_block_repeats=resolved.tcn_block_repeats,
            tcn_causal=resolved.tcn_causal,
            dropout=resolved.dropout,
            fc_hidden=resolved.fc_hidden,
            weight_decay=resolved.weight_decay,
            eta_min_ratio=resolved.eta_min_ratio,
            val_frac=resolved.val_frac,
            grad_clip=resolved.grad_clip,
            compile_model=model_args.compile,
            compile_mode=model_args.compile_mode,
            device=common_args.device,
            loss_name=resolved.loss_name,
            pos_weight_cap=resolved.pos_weight_cap,
            focal_gamma=resolved.focal_gamma,
            focal_alpha_pos=resolved.focal_alpha_pos,
            asym_gamma_pos=resolved.asym_gamma_pos,
            asym_gamma_neg=resolved.asym_gamma_neg,
            asym_alpha_pos=resolved.asym_alpha_pos,
            use_amp=model_args.use_amp,
            amp_dtype=model_args.amp_dtype,
            allow_tf32=model_args.allow_tf32,
            cudnn_benchmark=model_args.cudnn_benchmark,
            deterministic=model_args.deterministic,
            num_workers=model_args.num_workers,
            prefetch_factor=model_args.prefetch_factor,
            persistent_workers=model_args.persistent_workers,
            pin_memory=model_args.pin_memory,
            min_batch_size=model_args.min_batch_size,
            max_oom_retries=model_args.max_oom_retries,
            quick_phase=bool(getattr(common_args, "quick_phase", False)),
            gpu_id=getattr(common_args, "gpu_id", None),
        )

    run_name_lr = model_args.lr
    run_name_batch_size = model_args.batch_size
    if train_target != "both":
        selected_params = task_hparams[tasks_to_train[0]]
        run_name_lr = selected_params.lr
        run_name_batch_size = selected_params.batch_size

    run_name = build_run_name(
        model_name="tcn",
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        lr=run_name_lr,
        batch_size=run_name_batch_size,
        epochs=resolved_epochs,
        tag=model_args.tag,
    )
    task_hparams_summary: dict[str, Dict[str, object]] = {}
    for task, params in task_hparams.items():
        task_hparams_summary[task] = {
            "batch_size": params.batch_size,
            "lr": params.lr,
            "loss": params.loss_name,
            "conv_channels": (
                None
                if params.conv_channels is None
                else list(params.conv_channels)
            ),
            "kernel_size": params.kernel_size,
            "tcn_block_repeats": params.tcn_block_repeats,
            "tcn_causal": params.tcn_causal,
            "dropout": params.dropout,
            "fc_hidden": params.fc_hidden,
            "weight_decay": params.weight_decay,
            "eta_min_ratio": params.eta_min_ratio,
            "val_frac": params.val_frac,
            "grad_clip": params.grad_clip,
            "pos_weight_cap": params.pos_weight_cap,
            "focal_gamma": params.focal_gamma,
            "focal_alpha_pos": params.focal_alpha_pos,
            "asym_gamma_pos": params.asym_gamma_pos,
            "asym_gamma_neg": params.asym_gamma_neg,
            "asym_alpha_pos": params.asym_alpha_pos,
        }

    summary: Dict[str, object] = {
        "model": "tcn",
        "species": common_args.species,
        "train_pos_path": train_pos_path,
        "train_neg_path": train_neg_path,
        "donor_len": donor_len,
        "acceptor_len": acceptor_len,
        "epochs": resolved_epochs,
        "epochs_config": str(model_args.epochs),
        "epochs_auto": epochs_auto,
        "max_epochs": model_args.max_epochs,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "batch_size": model_args.batch_size,
        "lr": model_args.lr,
        "train_target": train_target,
        "seed": common_args.seed,
        "device": common_args.device,
        "checkpoint_name": os.path.basename(donor_checkpoint_path),
        "donor_checkpoint_path": donor_checkpoint_path,
        "acceptor_checkpoint_path": acceptor_checkpoint_path,
        "lightweight": model_args.lightweight,
        "conv_channels": (
            None if shared_conv_channels is None else list(shared_conv_channels)
        ),
        "kernel_size": model_args.kernel_size,
        "tcn_block_repeats": model_args.tcn_block_repeats,
        "tcn_causal": model_args.tcn_causal,
        "dropout": model_args.dropout,
        "fc_hidden": model_args.fc_hidden,
        "weight_decay": model_args.weight_decay,
        "eta_min_ratio": model_args.eta_min_ratio,
        "val_frac": model_args.val_frac,
        "grad_clip": model_args.grad_clip,
        "compile": model_args.compile,
        "compile_mode": model_args.compile_mode,
        "use_amp": bool(model_args.use_amp),
        "amp_dtype": model_args.amp_dtype,
        "allow_tf32": bool(model_args.allow_tf32),
        "cudnn_benchmark": bool(model_args.cudnn_benchmark),
        "deterministic": bool(model_args.deterministic),
        "num_workers": model_args.num_workers,
        "prefetch_factor": model_args.prefetch_factor,
        "persistent_workers": bool(model_args.persistent_workers),
        "pin_memory": bool(model_args.pin_memory),
        "min_batch_size": model_args.min_batch_size,
        "max_oom_retries": model_args.max_oom_retries,
        "loss": model_args.loss,
        "focal_gamma": model_args.focal_gamma,
        "focal_alpha_pos": model_args.focal_alpha_pos,
        "asym_gamma_pos": model_args.asym_gamma_pos,
        "asym_gamma_neg": model_args.asym_gamma_neg,
        "asym_alpha_pos": model_args.asym_alpha_pos,
        "run_name": run_name,
        "inferred_train_len": inferred_train_len,
        "task_hyperparameters": task_hparams_summary,
    }
    summary.update(task_metrics)
    return summary


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Run site-level inference and return rows with fixed schema."""
    dirs = species_data_dirs(common_args.species)
    inferred_train_len: Optional[int] = None
    if common_args.donor_len is None and common_args.acceptor_len is None:
        try:
            _, _, inferred_train_len = infer_default_train_paths(
                train_dir=dirs["train"],
                donor_len=None,
                acceptor_len=None,
            )
        except ValueError:
            inferred_train_len = None

    donor_len, acceptor_len = resolve_effective_window_lengths(
        donor_len=common_args.donor_len,
        acceptor_len=common_args.acceptor_len,
        inferred_train_len=inferred_train_len,
    )
    validate_window_args(
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )

    test_tsv = resolve_test_tsv(common_args.species, common_args.test_tsv)
    donor_model_path = str(getattr(common_args, "donor_checkpoint_path", "")).strip()
    acceptor_model_path = str(
        getattr(common_args, "acceptor_checkpoint_path", "")
    ).strip()
    if not donor_model_path:
        raise ValueError("Missing donor checkpoint path in common_args.")
    if not acceptor_model_path:
        raise ValueError("Missing acceptor checkpoint path in common_args.")
    if not os.path.exists(donor_model_path):
        raise FileNotFoundError(f"Donor checkpoint not found: {donor_model_path}")
    if not os.path.exists(acceptor_model_path):
        raise FileNotFoundError(f"Acceptor checkpoint not found: {acceptor_model_path}")

    site_rows, skipped_short = read_test_site_rows(
        test_tsv=test_tsv,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )
    print(f"Loaded test sites: {len(site_rows)}")
    if skipped_short:
        print(f"Skipped short sites: {skipped_short}")

    return infer_site_scores(
        site_rows=site_rows,
        donor_model_path=donor_model_path,
        acceptor_model_path=acceptor_model_path,
        device=common_args.device,
        batch_size=model_args.batch_size,
    )
