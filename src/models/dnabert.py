"""DNABERT-2 fine-tuning model for site-level splice scoring.

This module integrates Hugging Face DNABERT-2 checkpoints into the unified
pipeline contract used by ``run_model.py``.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import (
    ContextManager,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
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
    read_examples_pair_task,
    read_examples_single_task,
    read_test_pair_rows,
    read_test_site_rows,
    resolve_effective_window_lengths,
    resolve_test_tsv,
    resolve_train_paths,
    species_data_dirs,
    validate_window_args,
)
from util.losses import LOSS_NAME_CHOICES, build_binary_classification_loss
from util.model_task_paths import (
    attach_init_checkpoint_summary,
    checkpoint_tasks_for_model,
    resolve_required_checkpoint_paths,
    resolve_task_init_checkpoint_paths,
    resolve_tasks_to_train,
    resolve_train_target,
)
from util.model_runtime import (
    bool_from_flag as _bool_from_flag,
    compile_model_with_fallback as _compile_model_with_fallback,
    configure_torch_compile_runtime as _configure_torch_compile_runtime,
    configure_triton_tool_paths as _configure_triton_tool_paths,
    empty_device_cache as _empty_device_cache,
    export_model_state_dict as _export_model_state_dict,
    fallback_average_precision as _fallback_average_precision,
    fallback_max_f1 as _fallback_max_f1,
    fallback_roc_auc as _fallback_roc_auc,
    is_compile_runtime_error as _is_compile_runtime_error,
    is_cuda_oom_error as _is_cuda_oom_error,
    is_mps_oom_error as _is_mps_oom_error,
    log10_sigmoid_np,
    normalize_checkpoint_state_dict as _normalize_checkpoint_state_dict,
    pick_device,
    resolve_amp_dtype as _resolve_amp_dtype,
    resolve_compile_enabled as _resolve_compile_enabled,
    resolve_mps_max_batch_size,
    resolve_num_workers as _resolve_num_workers,
    record_compile_runtime_failure as _record_compile_runtime_failure,
    seed_worker as _seed_worker,
    set_seed,
    sigmoid_np,
    warm_start_model as _warm_start_model,
)
from util.process_title import (
    apply_eta_process_title_from_epoch_progress,
    apply_eta_process_title_placeholder,
)
from util.training_control import (
    resolve_training_schedule,
)
from util.transcript_eval import SCORE_SPACE_FIELD, SCORE_SPACE_LOG10

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoConfig = None
    AutoModel = None
    AutoTokenizer = None


@contextmanager
def _quiet_transformers_loading() -> Iterator[None]:
    """Suppress Hugging Face weight-loading INFO messages temporarily.

    Raises the ``transformers`` logger level to ERROR for the duration of
    the block, then restores the previous level.  This suppresses the
    verbose "loading weights", "unexpected keys", and "missing keys"
    lines that ``from_pretrained`` emits at INFO level.
    """
    hf_logger = logging.getLogger("transformers")
    prev_level = hf_logger.level
    hf_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        hf_logger.setLevel(prev_level)


DEFAULT_MPS_MAX_BATCH_SIZE: int = 1024
DEFAULT_PRETRAINED_MODEL_NAME: str = "zhihan1996/DNABERT-2-117M"
DNA_BASE_SET: frozenset[str] = frozenset({"A", "C", "G", "T"})
SPECIAL_TOKENS: frozenset[str] = frozenset(
    {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
)
READOUT_TYPE_CHOICES: tuple[str, ...] = ("cnn", "linear", "mlp")
DEFAULT_READOUT_TYPE: str = "cnn"
DEFAULT_READOUT_CNN_KERNEL_SIZE: int = 3
DEFAULT_READOUT_MLP_HIDDEN_DIM: int = 256
DEFAULT_READOUT_MLP_LAYERS: int = 1
LR_SCHEDULE_CHOICES: tuple[str, ...] = ("cosine", "linear")
DEFAULT_LR_SCHEDULE: str = "cosine"
DEFAULT_WARMUP_RATIO: float = 0.01
DEFAULT_ADAM_BETA1: float = 0.9
DEFAULT_ADAM_BETA2: float = 0.98
DEFAULT_ADAM_EPS: float = 1e-8


@dataclass(frozen=True)
class _TokenizerCacheKey:
    """Stable cache key for one pretrained tokenizer resource."""

    pretrained_model_name: str
    pretrained_revision: Optional[str]
    trust_remote_code: bool


@dataclass(frozen=True)
class _BackboneCacheKey:
    """Stable cache key for one pretrained DNABERT backbone template."""

    pretrained_model_name: str
    pretrained_revision: Optional[str]
    trust_remote_code: bool


@dataclass(frozen=True)
class _CachedBackboneTemplate:
    """CPU-resident DNABERT backbone template reused within one process."""

    backbone: nn.Module
    hidden_size: int


_TOKENIZER_CACHE: dict[_TokenizerCacheKey, object] = {}
_BACKBONE_TEMPLATE_CACHE: dict[_BackboneCacheKey, _CachedBackboneTemplate] = {}
SequenceT = TypeVar("SequenceT")


def _clear_pretrained_resource_caches() -> None:
    """Clear tokenizer and backbone template caches used in one process."""
    _TOKENIZER_CACHE.clear()
    _BACKBONE_TEMPLATE_CACHE.clear()


def _require_transformers() -> None:
    """Raise a clear error if transformers is unavailable."""
    if AutoConfig is None or AutoModel is None or AutoTokenizer is None:
        raise RuntimeError(
            "The 'transformers' package is required for model=dnabert. "
            "Install it in the intronmodel environment."
        )


def _resolve_mps_max_batch_size() -> int:
    """Resolve MPS batch-size cap from env with a safe default."""
    return resolve_mps_max_batch_size(
        model_tag="dnabert",
        default_batch_size=DEFAULT_MPS_MAX_BATCH_SIZE,
    )


def _normalize_revision(revision: str) -> Optional[str]:
    """Normalize revision text from CLI into ``None`` or non-empty string."""
    text = revision.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    return text


def _without_none_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    """Return a shallow kwargs copy with ``None`` values removed."""
    return {key: value for key, value in kwargs.items() if value is not None}


def _normalize_readout_type(readout_type: str, *, arg_name: str) -> str:
    """Normalize and validate one DNABERT readout type name."""
    normalized = readout_type.strip().lower()
    if normalized not in READOUT_TYPE_CHOICES:
        choices_text = ", ".join(READOUT_TYPE_CHOICES)
        raise ValueError(f"{arg_name} must be one of: {choices_text}.")
    return normalized


def _normalize_lr_schedule(lr_schedule: str, *, arg_name: str) -> str:
    """Normalize and validate one learning-rate schedule name."""
    normalized = lr_schedule.strip().lower()
    if normalized not in LR_SCHEDULE_CHOICES:
        choices_text = ", ".join(LR_SCHEDULE_CHOICES)
        raise ValueError(f"{arg_name} must be one of: {choices_text}.")
    return normalized


def _lr_schedule_multiplier(
    *,
    step_index: int,
    total_steps: int,
    warmup_steps: int,
    eta_min_ratio: float,
    lr_schedule: str,
) -> float:
    """Compute one LR multiplier with warmup + decay in O(1) time.

    Parameters
    ----------
    step_index : int
        Zero-based optimizer step index.
    total_steps : int
        Total optimizer steps in the run.
    warmup_steps : int
        Number of linear warmup steps.
    eta_min_ratio : float
        Final learning-rate ratio relative to the base learning rate.
    lr_schedule : str
        Decay family name (``cosine`` or ``linear``).

    Returns
    -------
    float
        Learning-rate multiplier in ``(0, 1]`` during warmup and
        ``[eta_min_ratio, 1]`` during decay.

    Raises
    ------
    ValueError
        If any argument is outside the supported range.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps <= total_steps.")
    if step_index < 0:
        raise ValueError("step_index must be non-negative.")
    if not (0.0 <= eta_min_ratio <= 1.0):
        raise ValueError("eta_min_ratio must satisfy 0 <= eta_min_ratio <= 1.")
    normalized_schedule = _normalize_lr_schedule(
        lr_schedule,
        arg_name="lr_schedule",
    )

    bounded_step = min(step_index, total_steps - 1)
    if warmup_steps > 0 and bounded_step < warmup_steps:
        return float(bounded_step + 1) / float(warmup_steps)

    decay_start = warmup_steps
    decay_steps = max(1, total_steps - decay_start)
    if decay_steps == 1:
        progress = 1.0
    else:
        progress = float(bounded_step - decay_start) / float(decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)

    if normalized_schedule == "cosine":
        decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * decay_factor
    return eta_min_ratio + (1.0 - eta_min_ratio) * (1.0 - progress)


def _format_duration_hms(total_seconds: float) -> str:
    """Format duration seconds as ``HH:MM:SS`` text."""
    clamped_seconds = max(0, int(total_seconds))
    hours, rem = divmod(clamped_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_eta_utc_from_now(remaining_seconds: float) -> str:
    """Return UTC ETA timestamp ``YYYY-mm-ddTHH:MM:SSZ`` from now."""
    eta_utc = datetime.now(timezone.utc) + timedelta(
        seconds=max(0.0, float(remaining_seconds))
    )
    return eta_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_pad_token_id_from_config(config: object) -> int:
    """Resolve a safe pad token id from a pretrained config object."""
    direct_pad = getattr(config, "pad_token_id", None)
    if isinstance(direct_pad, int) and direct_pad >= 0:
        return direct_pad

    for field_name in ("eos_token_id", "sep_token_id", "cls_token_id"):
        candidate = getattr(config, field_name, None)
        if isinstance(candidate, int) and candidate >= 0:
            return candidate

    return 0


def _patch_dnabert_alibi_meta_compat(
    *,
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
    config: object,
) -> None:
    """Patch DNABERT remote ALiBi build to avoid meta/cpu tensor mismatch.

    This patch is required for newer ``torch/transformers`` combinations where
    model initialization may run under a meta-device context.
    """
    if not trust_remote_code:
        return

    auto_map_obj = getattr(config, "auto_map", None)
    if not isinstance(auto_map_obj, Mapping):
        return
    auto_model_ref = auto_map_obj.get("AutoModel")
    if not isinstance(auto_model_ref, str) or auto_model_ref.strip() == "":
        return

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        return

    try:
        model_class = get_class_from_dynamic_module(
            auto_model_ref,
            pretrained_model_name,
            revision=pretrained_revision,
            code_revision=pretrained_revision,
        )
    except Exception:
        return

    module_name_obj = getattr(model_class, "__module__", None)
    if not isinstance(module_name_obj, str) or module_name_obj.strip() == "":
        return
    remote_module = sys.modules.get(module_name_obj)
    if remote_module is None:
        return

    encoder_cls_obj = getattr(remote_module, "BertEncoder", None)
    if encoder_cls_obj is None:
        return
    if bool(getattr(encoder_cls_obj, "_intronmodel_meta_patch_applied", False)):
        return

    original_method_obj = getattr(encoder_cls_obj, "rebuild_alibi_tensor", None)
    if not callable(original_method_obj):
        return

    def _patched_rebuild_alibi_tensor(
        self: object,
        size: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> object:
        resolved_device = device
        if resolved_device is None:
            existing_alibi = getattr(self, "alibi", None)
            if isinstance(existing_alibi, torch.Tensor):
                resolved_device = existing_alibi.device
            else:
                resolved_device = "cpu"
        return original_method_obj(self, size, device=resolved_device)

    setattr(encoder_cls_obj, "rebuild_alibi_tensor", _patched_rebuild_alibi_tensor)
    setattr(encoder_cls_obj, "_intronmodel_meta_patch_applied", True)


def _resolve_tokenizer_max_length(tokenizer: object) -> Optional[int]:
    """Resolve usable tokenizer max length from tokenizer metadata."""
    raw = getattr(tokenizer, "model_max_length", None)
    if isinstance(raw, int):
        if raw <= 0 or raw >= 100_000:
            return None
        return raw
    if isinstance(raw, float) and raw.is_integer():
        parsed = int(raw)
        if parsed <= 0 or parsed >= 100_000:
            return None
        return parsed
    return None


def _extract_tokenizer_vocab(tokenizer: object) -> dict[str, int]:
    """Extract tokenizer vocabulary as a ``dict[str, int]``."""
    get_vocab_fn = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab_fn):
        raw_vocab = get_vocab_fn()
        if isinstance(raw_vocab, Mapping):
            normalized: dict[str, int] = {}
            for key, value in raw_vocab.items():
                if isinstance(key, str) and isinstance(value, int):
                    normalized[key] = value
            return normalized

    raw_vocab_obj = getattr(tokenizer, "vocab", None)
    if isinstance(raw_vocab_obj, Mapping):
        normalized = {}
        for key, value in raw_vocab_obj.items():
            if isinstance(key, str) and isinstance(value, int):
                normalized[key] = value
        return normalized
    return {}


def _infer_fixed_kmer_from_vocab(vocab: Mapping[str, int]) -> Optional[int]:
    """Infer fixed k-mer length from vocabulary if it exists."""
    dna_tokens = [
        token
        for token in vocab.keys()
        if token not in SPECIAL_TOKENS and set(token).issubset(DNA_BASE_SET)
    ]
    if not dna_tokens:
        return None
    length_set = {len(token) for token in dna_tokens}
    if len(length_set) != 1:
        return None
    kmer_k = next(iter(length_set))
    if kmer_k <= 1:
        return None
    if len(dna_tokens) < (4**kmer_k):
        return None
    return kmer_k


def _resolve_tokenizer_input_kmer(tokenizer: object) -> Optional[int]:
    """Resolve sequence preprocessing k-mer from tokenizer metadata."""
    vocab = _extract_tokenizer_vocab(tokenizer)
    return _infer_fixed_kmer_from_vocab(vocab)


def _to_kmer_text(sequence: str, kmer_k: int) -> str:
    """Convert one DNA sequence to overlapping k-mer text."""
    if kmer_k <= 0:
        raise ValueError("kmer_k must be positive.")
    normalized = sequence.strip().upper()
    if len(normalized) < kmer_k:
        return normalized
    kmers = (
        normalized[start : start + kmer_k]
        for start in range(0, len(normalized) - kmer_k + 1)
    )
    return " ".join(kmers)


def _prepare_sequences_for_tokenizer(
    sequences: Sequence[str],
    input_kmer: Optional[int],
) -> list[str]:
    """Prepare sequences for tokenizer input based on k-mer strategy."""
    if input_kmer is None:
        return [sequence.upper() for sequence in sequences]
    return [_to_kmer_text(sequence, input_kmer) for sequence in sequences]


def _resolve_max_tokens(
    raw: Union[str, int],
    window_len: int,
    tokenizer_limit: Optional[int],
    input_kmer: Optional[int],
) -> int:
    """Resolve max token length from ``auto`` or integer input.

    ``auto`` uses ``window_len + 2`` for raw sequence input and
    ``(window_len - k + 1) + 2`` for fixed k-mer preprocessing.
    """
    if window_len <= 0:
        raise ValueError("window_len must be positive.")
    if input_kmer is not None and input_kmer <= 0:
        raise ValueError("input_kmer must be positive when provided.")

    if input_kmer is None:
        auto_tokens = max(8, window_len + 2)
    else:
        effective_token_count = max(1, window_len - input_kmer + 1)
        auto_tokens = max(8, effective_token_count + 2)
    if isinstance(raw, int):
        resolved = raw
    else:
        text = str(raw).strip().lower()
        if text == "auto":
            resolved = auto_tokens
        else:
            try:
                resolved = int(text)
            except ValueError as exc:
                raise ValueError(
                    "--max_tokens must be 'auto' or a positive integer."
                ) from exc
    if resolved < 2:
        raise ValueError("--max_tokens must be >= 2.")

    if tokenizer_limit is not None and resolved > tokenizer_limit:
        print(
            "[dnabert] max_tokens exceeds tokenizer limit. "
            f"Clamped: {resolved} -> {tokenizer_limit}"
        )
        resolved = tokenizer_limit
    return resolved


def _tokenize_sequences(
    tokenizer: object,
    sequences: Sequence[str],
    max_tokens: int,
    input_kmer: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a batch of DNA sequences to input ids and attention masks."""
    call_fn = getattr(tokenizer, "__call__", None)
    if not callable(call_fn):
        raise TypeError("Tokenizer object is not callable.")

    prepared_sequences = _prepare_sequences_for_tokenizer(
        sequences=sequences,
        input_kmer=input_kmer,
    )
    encoded_obj = call_fn(
        prepared_sequences,
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    if not isinstance(encoded_obj, Mapping):
        raise TypeError("Tokenizer output must be a mapping.")

    input_ids_obj = encoded_obj.get("input_ids")
    attention_mask_obj = encoded_obj.get("attention_mask")

    if isinstance(input_ids_obj, torch.Tensor):
        input_ids = input_ids_obj.long()
    else:
        input_ids = torch.as_tensor(input_ids_obj, dtype=torch.long)

    if attention_mask_obj is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    elif isinstance(attention_mask_obj, torch.Tensor):
        attention_mask = attention_mask_obj.long()
    else:
        attention_mask = torch.as_tensor(attention_mask_obj, dtype=torch.long)

    return input_ids, attention_mask


def _tokenize_sequence_pairs(
    tokenizer: object,
    donor_sequences: Sequence[str],
    acceptor_sequences: Sequence[str],
    max_tokens: int,
    input_kmer: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize donor/acceptor sequence pairs for one DNABERT forward pass."""
    if len(donor_sequences) != len(acceptor_sequences):
        raise ValueError(
            "donor_sequences and acceptor_sequences must have the same length."
        )
    call_fn = getattr(tokenizer, "__call__", None)
    if not callable(call_fn):
        raise TypeError("Tokenizer object is not callable.")

    prepared_donor_sequences = _prepare_sequences_for_tokenizer(
        sequences=donor_sequences,
        input_kmer=input_kmer,
    )
    prepared_acceptor_sequences = _prepare_sequences_for_tokenizer(
        sequences=acceptor_sequences,
        input_kmer=input_kmer,
    )
    encoded_obj = call_fn(
        prepared_donor_sequences,
        prepared_acceptor_sequences,
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    if not isinstance(encoded_obj, Mapping):
        raise TypeError("Tokenizer output must be a mapping.")

    input_ids_obj = encoded_obj.get("input_ids")
    attention_mask_obj = encoded_obj.get("attention_mask")

    if isinstance(input_ids_obj, torch.Tensor):
        input_ids = input_ids_obj.long()
    else:
        input_ids = torch.as_tensor(input_ids_obj, dtype=torch.long)

    if attention_mask_obj is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    elif isinstance(attention_mask_obj, torch.Tensor):
        attention_mask = attention_mask_obj.long()
    else:
        attention_mask = torch.as_tensor(attention_mask_obj, dtype=torch.long)
    return input_ids, attention_mask


def _build_tokenizer_cache_key(
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
) -> _TokenizerCacheKey:
    """Build a stable tokenizer cache key from pretrained arguments."""
    resolved_pretrained_model_name = _resolve_pretrained_model_name(
        pretrained_model_name
    )
    return _TokenizerCacheKey(
        pretrained_model_name=resolved_pretrained_model_name,
        pretrained_revision=pretrained_revision,
        trust_remote_code=trust_remote_code,
    )


def _load_tokenizer_uncached(
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
) -> object:
    """Load a cached-local DNABERT tokenizer without process-local reuse."""
    _require_transformers()
    assert AutoTokenizer is not None
    tokenizer_kwargs = _without_none_kwargs(
        {
            "local_files_only": True,
            "revision": pretrained_revision,
            "trust_remote_code": trust_remote_code,
        }
    )
    resolved_pretrained_model_name = _resolve_pretrained_model_name(
        pretrained_model_name
    )
    return AutoTokenizer.from_pretrained(
        resolved_pretrained_model_name,
        **tokenizer_kwargs,
    )


def _load_tokenizer(
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
) -> object:
    """Load and cache DNABERT tokenizer once per process."""
    cache_key = _build_tokenizer_cache_key(
        pretrained_model_name=pretrained_model_name,
        pretrained_revision=pretrained_revision,
        trust_remote_code=trust_remote_code,
    )
    cached_tokenizer = _TOKENIZER_CACHE.get(cache_key)
    if cached_tokenizer is None:
        cached_tokenizer = _load_tokenizer_uncached(
            pretrained_model_name=cache_key.pretrained_model_name,
            pretrained_revision=cache_key.pretrained_revision,
            trust_remote_code=cache_key.trust_remote_code,
        )
        _TOKENIZER_CACHE[cache_key] = cached_tokenizer
    return cached_tokenizer


def _resolve_pretrained_model_name(pretrained_model_name: str) -> str:
    """Resolve pretrained source string and validate explicit local paths.

    Parameters
    ----------
    pretrained_model_name : str
        Hugging Face repo id or local checkpoint directory path.

    Returns
    -------
    str
        Resolved model source string passed to ``from_pretrained``.

    Raises
    ------
    FileNotFoundError
        If an explicit local path is provided but does not exist.
    """
    explicit_local_path = (
        os.path.isabs(pretrained_model_name)
        or pretrained_model_name.startswith("./")
        or pretrained_model_name.startswith("../")
        or pretrained_model_name.startswith("~")
    )
    if not explicit_local_path:
        return pretrained_model_name

    expanded_path = Path(pretrained_model_name).expanduser()
    if not expanded_path.exists():
        cwd = Path.cwd()
        raise FileNotFoundError(
            "Explicit local --pretrained_model_name does not exist: "
            f"{expanded_path}. Current working directory: {cwd}."
        )
    return str(expanded_path)


class DnaBertTokenDataset(Dataset):
    """Dataset returning DNABERT token ids, masks, and binary labels.

    Parameters
    ----------
    examples : Sequence[tuple[str, int]]
        Sequence/label pairs.
    tokenizer : object
        Hugging Face tokenizer instance.
    max_tokens : int
        Padded token length.
    input_kmer : int | None
        Fixed k-mer preprocessing length.
    pretokenize : bool, default=False
        If ``True``, pre-encode all records at initialization.
    """

    def __init__(
        self,
        examples: Sequence[Tuple[str, int]],
        tokenizer: object,
        max_tokens: int,
        input_kmer: Optional[int],
        pretokenize: bool = False,
    ) -> None:
        self.examples: list[Tuple[str, int]] = list(examples)
        self.tokenizer: object = tokenizer
        self.max_tokens: int = max_tokens
        self.input_kmer: Optional[int] = input_kmer
        self.pretokenize: bool = pretokenize
        self._cached_ids: Optional[torch.Tensor]
        self._cached_masks: Optional[torch.Tensor]
        self._cached_labels: Optional[torch.Tensor]

        if pretokenize:
            sequences = [sequence for sequence, _ in self.examples]
            labels = [float(label) for _, label in self.examples]
            input_ids, attention_masks = _tokenize_sequences(
                tokenizer=self.tokenizer,
                sequences=sequences,
                max_tokens=self.max_tokens,
                input_kmer=self.input_kmer,
            )
            self._cached_ids = input_ids
            self._cached_masks = attention_masks
            self._cached_labels = torch.as_tensor(labels, dtype=torch.float32)
        else:
            self._cached_ids = None
            self._cached_masks = None
            self._cached_labels = None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._cached_ids is not None
            and self._cached_masks is not None
            and self._cached_labels is not None
        ):
            return (
                self._cached_ids[idx],
                self._cached_masks[idx],
                self._cached_labels[idx],
            )

        sequence, label = self.examples[idx]
        input_ids, attention_mask = _tokenize_sequences(
            tokenizer=self.tokenizer,
            sequences=[sequence],
            max_tokens=self.max_tokens,
            input_kmer=self.input_kmer,
        )
        return (
            input_ids.squeeze(0),
            attention_mask.squeeze(0),
            torch.tensor(float(label), dtype=torch.float32),
        )


class DnaBertPairTokenDataset(Dataset):
    """Dataset returning pair-tokenized DNABERT inputs and labels."""

    def __init__(
        self,
        examples: Sequence[Tuple[Tuple[str, str], int]],
        tokenizer: object,
        max_tokens: int,
        input_kmer: Optional[int],
        pretokenize: bool = False,
    ) -> None:
        self.examples: list[Tuple[Tuple[str, str], int]] = list(examples)
        self.tokenizer: object = tokenizer
        self.max_tokens: int = max_tokens
        self.input_kmer: Optional[int] = input_kmer
        self.pretokenize: bool = pretokenize
        self._cached_ids: Optional[torch.Tensor]
        self._cached_masks: Optional[torch.Tensor]
        self._cached_labels: Optional[torch.Tensor]

        if pretokenize:
            donor_sequences = [pair[0] for pair, _ in self.examples]
            acceptor_sequences = [pair[1] for pair, _ in self.examples]
            labels = [float(label) for _, label in self.examples]
            input_ids, attention_masks = _tokenize_sequence_pairs(
                tokenizer=self.tokenizer,
                donor_sequences=donor_sequences,
                acceptor_sequences=acceptor_sequences,
                max_tokens=self.max_tokens,
                input_kmer=self.input_kmer,
            )
            self._cached_ids = input_ids
            self._cached_masks = attention_masks
            self._cached_labels = torch.as_tensor(labels, dtype=torch.float32)
        else:
            self._cached_ids = None
            self._cached_masks = None
            self._cached_labels = None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._cached_ids is not None
            and self._cached_masks is not None
            and self._cached_labels is not None
        ):
            return (
                self._cached_ids[idx],
                self._cached_masks[idx],
                self._cached_labels[idx],
            )

        (donor_sequence, acceptor_sequence), label = self.examples[idx]
        input_ids, attention_mask = _tokenize_sequence_pairs(
            tokenizer=self.tokenizer,
            donor_sequences=[donor_sequence],
            acceptor_sequences=[acceptor_sequence],
            max_tokens=self.max_tokens,
            input_kmer=self.input_kmer,
        )
        return (
            input_ids.squeeze(0),
            attention_mask.squeeze(0),
            torch.tensor(float(label), dtype=torch.float32),
        )


def _run_backbone_forward_eager(
    backbone: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> object:
    """Run DNABERT backbone in eager mode to avoid dynamic-shape recompiles."""
    return backbone(input_ids=input_ids, attention_mask=attention_mask)


_dynamo_module = getattr(torch, "_dynamo", None)
_dynamo_disable = getattr(_dynamo_module, "disable", None)
if callable(_dynamo_disable):
    _run_backbone_forward_eager = _dynamo_disable(_run_backbone_forward_eager)


class DnaBertBinaryClassifier(nn.Module):
    """Binary classifier with selectable readout on top of a DNABERT backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        dropout: float,
        head_layer_norm: bool,
        readout_type: str = DEFAULT_READOUT_TYPE,
        readout_cnn_kernel_size: int = DEFAULT_READOUT_CNN_KERNEL_SIZE,
        readout_mlp_hidden_dim: int = DEFAULT_READOUT_MLP_HIDDEN_DIM,
        readout_mlp_layers: int = DEFAULT_READOUT_MLP_LAYERS,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        resolved_readout_type = _normalize_readout_type(
            readout_type,
            arg_name="readout_type",
        )
        if resolved_readout_type == "cnn":
            if readout_cnn_kernel_size <= 0:
                raise ValueError("readout_cnn_kernel_size must be positive.")
            if readout_cnn_kernel_size % 2 == 0:
                raise ValueError("readout_cnn_kernel_size must be odd.")
        if resolved_readout_type == "mlp":
            if readout_mlp_hidden_dim <= 0:
                raise ValueError("readout_mlp_hidden_dim must be positive.")
            if readout_mlp_layers <= 0:
                raise ValueError("readout_mlp_layers must be positive.")
        self.backbone = backbone
        self.readout_type: str = resolved_readout_type
        self.head_norm = nn.LayerNorm(hidden_size) if head_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.readout_cnn_kernel_size: int = int(readout_cnn_kernel_size)
        self.readout_cnn_padding: int = self.readout_cnn_kernel_size // 2
        self.readout_mlp_hidden_dim: int = int(readout_mlp_hidden_dim)
        self.readout_mlp_layers: int = int(readout_mlp_layers)
        self.mlp_hidden: nn.Module

        classifier_input_dim = hidden_size
        if self.readout_type == "mlp":
            mlp_layers: list[nn.Module] = []
            mlp_input_dim = hidden_size
            for _ in range(self.readout_mlp_layers):
                mlp_layers.append(
                    nn.Linear(
                        mlp_input_dim,
                        self.readout_mlp_hidden_dim,
                    )
                )
                mlp_layers.append(nn.GELU())
                mlp_layers.append(nn.Dropout(dropout))
                mlp_input_dim = self.readout_mlp_hidden_dim
            self.mlp_hidden = nn.Sequential(*mlp_layers)
            classifier_input_dim = self.readout_mlp_hidden_dim
        else:
            self.mlp_hidden = nn.Identity()

        # Keep the historical parameter name "classifier" for checkpoint
        # compatibility across readout variants.
        self.classifier = nn.Linear(classifier_input_dim, 1)

        if self.readout_type == "cnn":
            temporal_kernel = torch.arange(
                1,
                self.readout_cnn_kernel_size + 1,
                dtype=torch.float32,
            )
            center = float(self.readout_cnn_padding + 1)
            temporal_kernel = center - torch.abs(temporal_kernel - center)
            temporal_kernel = temporal_kernel / temporal_kernel.sum()
            self.register_buffer(
                "_frozen_temporal_kernel",
                temporal_kernel,
                persistent=False,
            )
        else:
            self.register_buffer(
                "_frozen_temporal_kernel",
                torch.zeros((1,), dtype=torch.float32),
                persistent=False,
            )

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters to train only the task head."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def assert_backbone_frozen(self) -> None:
        """Raise when any backbone parameter unexpectedly requires gradients."""
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise RuntimeError("DNABERT backbone must remain non-trainable.")

    @staticmethod
    def _masked_mean(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked token average with shape ``(batch, hidden)``."""
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        masked_sum = (hidden * mask).sum(dim=1)
        mask_denom = mask.sum(dim=1).clamp_min(1.0)
        return masked_sum / mask_denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return binary logits of shape ``(batch,)``."""
        outputs = _run_backbone_forward_eager(
            self.backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden: torch.Tensor
        if hasattr(outputs, "last_hidden_state"):
            candidate = getattr(outputs, "last_hidden_state")
            if not isinstance(candidate, torch.Tensor):
                raise RuntimeError("backbone output.last_hidden_state must be Tensor.")
            hidden = candidate
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            first = outputs[0]
            if not isinstance(first, torch.Tensor):
                raise RuntimeError("backbone output tuple[0] must be Tensor.")
            hidden = first
        else:
            raise RuntimeError("Unsupported backbone output format.")

        if attention_mask.dim() != 2:
            raise RuntimeError("attention_mask must have shape (batch, tokens).")
        if hidden.shape[:2] != attention_mask.shape:
            raise RuntimeError(
                "backbone hidden and attention_mask token lengths must match."
            )

        token_hidden = self.dropout(self.head_norm(hidden))
        if self.readout_type == "cnn":
            token_features = token_hidden.transpose(1, 2).contiguous()
            frozen_kernel = self.classifier.weight.view(
                1, -1, 1
            ) * self._frozen_temporal_kernel.view(1, 1, -1)
            logits_map = F.conv1d(
                token_features,
                weight=frozen_kernel,
                bias=self.classifier.bias,
                padding=self.readout_cnn_padding,
            )
            mask = attention_mask.to(dtype=logits_map.dtype).unsqueeze(1)
            masked_sum = (logits_map * mask).sum(dim=2)
            mask_denom = mask.sum(dim=2).clamp_min(1.0)
            logits = masked_sum / mask_denom
            return logits.squeeze(1)
        pooled_hidden = self._masked_mean(token_hidden, attention_mask)
        if self.readout_type == "linear":
            return self.classifier(pooled_hidden).squeeze(1)
        hidden_features = self.mlp_hidden(pooled_hidden)
        return self.classifier(hidden_features).squeeze(1)


def _resolve_hidden_size(backbone: nn.Module) -> int:
    """Resolve hidden size from backbone config fields."""
    config = getattr(backbone, "config", None)
    candidates = ("hidden_size", "dim", "d_model", "n_embd")
    for key in candidates:
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("Failed to resolve DNABERT hidden size from config.")


def _materialize_dnabert_meta_buffers(backbone: nn.Module) -> None:
    """Materialize DNABERT buffers that may remain on meta device.

    Some remote DNABERT implementations can leave ``encoder.alibi`` as a meta
    tensor under newer ``torch/transformers`` loading paths. This function
    rebuilds that tensor on CPU so later ``.to(cuda)`` works.
    """
    encoder_obj = getattr(backbone, "encoder", None)
    if encoder_obj is None:
        return

    alibi_obj = getattr(encoder_obj, "alibi", None)
    if not isinstance(alibi_obj, torch.Tensor):
        return
    if alibi_obj.device.type != "meta":
        return

    size_obj = getattr(encoder_obj, "_current_alibi_size", None)
    if not isinstance(size_obj, int) or size_obj <= 0:
        size_obj = int(alibi_obj.shape[-1])

    rebuild_fn = getattr(encoder_obj, "rebuild_alibi_tensor", None)
    if not callable(rebuild_fn):
        raise RuntimeError(
            "DNABERT encoder has meta alibi buffer but no rebuild function."
        )
    rebuild_fn(size=int(size_obj), device="cpu")


def _disable_dnabert_triton_flash_attention(backbone: nn.Module) -> None:
    """Disable DNABERT Triton FlashAttention hooks when present.

    DNABERT remote modules can provide a Triton-only FlashAttention path that is
    incompatible with newer Triton APIs on some systems. Setting the exported
    flash-attention callables to ``None`` forces the model to use the fallback
    non-Triton path in ``bert_layers.py``.
    """
    module_name = getattr(backbone.__class__, "__module__", "")
    if not isinstance(module_name, str) or not module_name:
        return
    module_obj = sys.modules.get(module_name)
    if module_obj is None:
        return
    for attr_name in ("flash_attn_qkvpacked_func", "flash_attn_func"):
        if hasattr(module_obj, attr_name):
            setattr(module_obj, attr_name, None)


def _build_backbone_cache_key(
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
) -> _BackboneCacheKey:
    """Build a stable backbone cache key from pretrained arguments."""
    resolved_pretrained_model_name = _resolve_pretrained_model_name(
        pretrained_model_name
    )
    return _BackboneCacheKey(
        pretrained_model_name=resolved_pretrained_model_name,
        pretrained_revision=pretrained_revision,
        trust_remote_code=trust_remote_code,
    )


def _load_backbone_template_uncached(
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
) -> _CachedBackboneTemplate:
    """Load one cached-local DNABERT backbone template without cache reuse."""
    _require_transformers()
    assert AutoConfig is not None
    assert AutoModel is not None
    resolved_pretrained_model_name = _resolve_pretrained_model_name(
        pretrained_model_name
    )
    config_kwargs = _without_none_kwargs(
        {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
            "revision": pretrained_revision,
        }
    )
    config = AutoConfig.from_pretrained(
        resolved_pretrained_model_name,
        **config_kwargs,
    )
    resolved_pad_token_id = _resolve_pad_token_id_from_config(config)
    if getattr(config, "pad_token_id", None) != resolved_pad_token_id:
        setattr(config, "pad_token_id", resolved_pad_token_id)
    _patch_dnabert_alibi_meta_compat(
        pretrained_model_name=resolved_pretrained_model_name,
        pretrained_revision=pretrained_revision,
        trust_remote_code=trust_remote_code,
        config=config,
    )
    model_kwargs = _without_none_kwargs(
        {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
            "config": config,
            "low_cpu_mem_usage": False,
            "revision": pretrained_revision,
        }
    )
    try:
        with _quiet_transformers_loading():
            backbone = AutoModel.from_pretrained(
                resolved_pretrained_model_name,
                **model_kwargs,
            )
    except ImportError as exc:
        message = str(exc)
        if "einops" in message:
            raise RuntimeError(
                "DNABERT remote module dependency is missing: einops. "
                "Install it in the intronmodel environment "
                "(e.g., `conda env update -f environment.yml --prune` "
                "or `pip install einops`)."
            ) from exc
        raise
    _disable_dnabert_triton_flash_attention(backbone)
    _materialize_dnabert_meta_buffers(backbone)
    hidden_size = _resolve_hidden_size(backbone)
    backbone_cpu = backbone.to("cpu")
    return _CachedBackboneTemplate(
        backbone=backbone_cpu,
        hidden_size=hidden_size,
    )


def _get_cached_backbone_template(
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
) -> _CachedBackboneTemplate:
    """Return one cached DNABERT backbone template for this process."""
    cache_key = _build_backbone_cache_key(
        pretrained_model_name=pretrained_model_name,
        pretrained_revision=pretrained_revision,
        trust_remote_code=trust_remote_code,
    )
    cached_template = _BACKBONE_TEMPLATE_CACHE.get(cache_key)
    if cached_template is None:
        cached_template = _load_backbone_template_uncached(
            pretrained_model_name=cache_key.pretrained_model_name,
            pretrained_revision=cache_key.pretrained_revision,
            trust_remote_code=cache_key.trust_remote_code,
        )
        _BACKBONE_TEMPLATE_CACHE[cache_key] = cached_template
    return cached_template


def _build_dnabert_model(
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
    dropout: float,
    head_layer_norm: bool,
    readout_type: str = DEFAULT_READOUT_TYPE,
    readout_cnn_kernel_size: int = DEFAULT_READOUT_CNN_KERNEL_SIZE,
    readout_mlp_hidden_dim: int = DEFAULT_READOUT_MLP_HIDDEN_DIM,
    readout_mlp_layers: int = DEFAULT_READOUT_MLP_LAYERS,
) -> DnaBertBinaryClassifier:
    """Build DNABERT classifier from a cached pretrained backbone template."""
    cached_template = _get_cached_backbone_template(
        pretrained_model_name=pretrained_model_name,
        pretrained_revision=pretrained_revision,
        trust_remote_code=trust_remote_code,
    )
    backbone = copy.deepcopy(cached_template.backbone)
    return DnaBertBinaryClassifier(
        backbone=backbone,
        hidden_size=cached_template.hidden_size,
        dropout=dropout,
        head_layer_norm=head_layer_norm,
        readout_type=readout_type,
        readout_cnn_kernel_size=readout_cnn_kernel_size,
        readout_mlp_hidden_dim=readout_mlp_hidden_dim,
        readout_mlp_layers=readout_mlp_layers,
    )


def prewarm_persistent_worker(
    base_args: Mapping[str, object],
    assigned_gpu_id: Optional[str] = None,
) -> None:
    """Preload reusable DNABERT resources inside one persistent worker.

    Parameters
    ----------
    base_args : Mapping[str, object]
        Base hyperparameter-search arguments for the worker.
    assigned_gpu_id : str | None, default=None
        GPU slot identifier associated with this worker, used for logging only.
    """
    raw_pretrained_model_name = base_args.get(
        "pretrained_model_name",
        DEFAULT_PRETRAINED_MODEL_NAME,
    )
    pretrained_model_name = str(raw_pretrained_model_name).strip()
    if pretrained_model_name == "":
        pretrained_model_name = DEFAULT_PRETRAINED_MODEL_NAME
    raw_revision = base_args.get("pretrained_revision", "")
    revision = _normalize_revision("" if raw_revision is None else str(raw_revision))
    trust_remote_code = _bool_from_flag(base_args.get("trust_remote_code", 1))
    _ = _load_tokenizer(
        pretrained_model_name=pretrained_model_name,
        pretrained_revision=revision,
        trust_remote_code=trust_remote_code,
    )
    _ = _get_cached_backbone_template(
        pretrained_model_name=pretrained_model_name,
        pretrained_revision=revision,
        trust_remote_code=trust_remote_code,
    )
    gpu_label = "cpu" if assigned_gpu_id is None else assigned_gpu_id
    print(
        f"[dnabert] prewarmed persistent worker cache "
        f"(gpu={gpu_label}, model={pretrained_model_name})"
    )


@dataclass(frozen=True)
class TaskTrainParams:
    """Resolved train-time hyperparameters for one task."""

    batch_size: int
    lr: float
    loss_name: str
    max_tokens: str
    dropout: float
    head_layer_norm: int
    readout_type: str
    readout_cnn_kernel_size: int
    readout_mlp_hidden_dim: int
    readout_mlp_layers: int
    weight_decay: float
    eta_min_ratio: float
    lr_schedule: str
    warmup_ratio: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    val_frac: float
    grad_clip: float
    pos_weight_cap: float
    focal_gamma: float
    focal_alpha_pos: Optional[float]
    asym_gamma_pos: float
    asym_gamma_neg: float
    asym_alpha_pos: Optional[float]


@dataclass(frozen=True)
class InferRuntimeConfig:
    """Resolved runtime controls for DNABERT inference."""

    batch_size: int
    use_amp: bool
    amp_dtype: Optional[torch.dtype]
    compile_enabled: bool


def _resolve_task_train_params(
    *,
    task: str,
    model_args: argparse.Namespace,
) -> TaskTrainParams:
    """Resolve task-specific train parameters with fallback to shared values."""
    if task not in {"donor", "acceptor", "pair"}:
        raise ValueError(f"Unsupported task: {task}")

    prefix = "" if task == "pair" else f"{task}_"

    def _override_or_default(name: str, default: object) -> object:
        override_name = f"{prefix}{name}" if prefix != "" else name
        override = getattr(model_args, override_name, None)
        return default if override is None else override

    return TaskTrainParams(
        batch_size=int(_override_or_default("batch_size", model_args.batch_size)),
        lr=float(_override_or_default("lr", model_args.lr)),
        loss_name=str(_override_or_default("loss", model_args.loss)),
        max_tokens=str(_override_or_default("max_tokens", model_args.max_tokens)),
        dropout=float(_override_or_default("dropout", model_args.dropout)),
        head_layer_norm=int(
            _override_or_default("head_layer_norm", model_args.head_layer_norm)
        ),
        readout_type=str(_override_or_default("readout_type", model_args.readout_type)),
        readout_cnn_kernel_size=int(
            _override_or_default(
                "readout_cnn_kernel_size",
                model_args.readout_cnn_kernel_size,
            )
        ),
        readout_mlp_hidden_dim=int(
            _override_or_default(
                "readout_mlp_hidden_dim",
                model_args.readout_mlp_hidden_dim,
            )
        ),
        readout_mlp_layers=int(
            _override_or_default(
                "readout_mlp_layers",
                model_args.readout_mlp_layers,
            )
        ),
        weight_decay=float(
            _override_or_default("weight_decay", model_args.weight_decay)
        ),
        eta_min_ratio=float(
            _override_or_default("eta_min_ratio", model_args.eta_min_ratio)
        ),
        lr_schedule=str(_override_or_default("lr_schedule", model_args.lr_schedule)),
        warmup_ratio=float(
            _override_or_default("warmup_ratio", model_args.warmup_ratio)
        ),
        adam_beta1=float(_override_or_default("adam_beta1", model_args.adam_beta1)),
        adam_beta2=float(_override_or_default("adam_beta2", model_args.adam_beta2)),
        adam_eps=float(_override_or_default("adam_eps", model_args.adam_eps)),
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


def _resolve_infer_runtime_config(
    *,
    device: str,
    batch_size: int,
    infer_use_amp: Union[bool, int],
    infer_amp_dtype: str,
    infer_compile: Union[bool, int],
    infer_compile_mode: str,
) -> InferRuntimeConfig:
    """Resolve inference runtime configuration from user-facing flags."""
    if batch_size <= 0:
        raise ValueError("inference batch_size must be positive.")

    effective_batch_size = batch_size
    if device == "mps":
        mps_max_batch_size = _resolve_mps_max_batch_size()
        if effective_batch_size > mps_max_batch_size:
            print(
                "[infer] mps batch clamp: "
                f"{effective_batch_size} -> {mps_max_batch_size} "
                "(set INTRONMODEL_MPS_MAX_BATCH_SIZE to change)."
            )
            effective_batch_size = mps_max_batch_size

    use_amp = _bool_from_flag(infer_use_amp) and device == "cuda"
    amp_dtype = _resolve_amp_dtype(infer_amp_dtype, device)
    compile_enabled = _resolve_compile_enabled(
        compile_mode=infer_compile_mode,
        compile_flag=_bool_from_flag(infer_compile),
        quick_phase=False,
        device=device,
        epochs=2,
    )
    return InferRuntimeConfig(
        batch_size=effective_batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        compile_enabled=compile_enabled,
    )


def _prepare_infer_model(
    *,
    model: nn.Module,
    task_name: str,
    compile_enabled: bool,
    compile_mode: str,
) -> nn.Module:
    """Compile one inference model when requested and supported."""
    compile_enabled_attempt = compile_enabled and hasattr(torch, "compile")
    if not compile_enabled_attempt:
        return model

    _configure_triton_tool_paths()
    _configure_torch_compile_runtime()
    ptxas_path = os.environ.get("TRITON_PTXAS_PATH")
    ptxas_blackwell_path = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH")
    print(
        f"[{task_name}] infer torch.compile requested "
        f"(ptxas={ptxas_path}, ptxas_blackwell={ptxas_blackwell_path})."
    )
    (
        compiled_model,
        compile_enabled_attempt,
        _compile_selected_mode,
        compile_setup_error,
    ) = _compile_model_with_fallback(model, compile_mode=compile_mode)
    if (not compile_enabled_attempt) and compile_setup_error is not None:
        print(
            f"[{task_name}] infer torch.compile setup failed "
            f"({compile_setup_error.__class__.__name__}). Continue without compile."
        )
        return model
    return compiled_model


def stratified_split(
    examples: Sequence[Tuple[SequenceT, int]],
    val_frac: float = 0.1,
    seed: int = 1337,
) -> Tuple[List[Tuple[SequenceT, int]], List[Tuple[SequenceT, int]]]:
    """Split examples into train/validation subsets preserving labels."""
    rng = random.Random(seed)
    positives = [(seq, label) for seq, label in examples if label == 1]
    negatives = [(seq, label) for seq, label in examples if label == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_val_pos = max(1, int(len(positives) * val_frac))
    n_val_neg = max(1, int(len(negatives) * val_frac))
    n_val_pos = min(n_val_pos, len(positives) - 1)
    n_val_neg = min(n_val_neg, len(negatives) - 1)

    train = positives[n_val_pos:] + negatives[n_val_neg:]
    val = positives[:n_val_pos] + negatives[:n_val_neg]

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
    """Evaluate one model on a validation loader."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    use_non_blocking = device == "cuda"

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device, non_blocking=use_non_blocking)
        attention_mask = attention_mask.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)

        if use_amp and device == "cuda" and amp_dtype is not None:
            amp_context: ContextManager[object] = torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            )
        else:
            amp_context = nullcontext()

        with amp_context:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.float().cpu().numpy())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    probs = sigmoid_np(logits) if logits.size else np.array([])

    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    labels = labels.astype(np.int32)

    metrics: Dict[str, float] = {}
    if labels.size:
        metrics["acc@0.5"] = float(np.mean((probs >= 0.5) == (labels >= 0.5)))
        max_f1_value: Optional[float] = None
        try:
            max_f1_value = _fallback_max_f1(labels, probs)
        except ValueError:
            max_f1_value = None
        if max_f1_value is not None:
            metrics["max_f1"] = max_f1_value

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
    pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL_NAME,
    pretrained_revision: str = "",
    trust_remote_code: Union[bool, int] = 1,
    epochs: int = 20,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    batch_size: int = 64,
    lr: float = 2e-5,
    seed: int = 1337,
    max_tokens: Union[str, int] = "auto",
    dropout: float = 0.1,
    head_layer_norm: Union[bool, int] = 1,
    readout_type: str = DEFAULT_READOUT_TYPE,
    readout_cnn_kernel_size: int = DEFAULT_READOUT_CNN_KERNEL_SIZE,
    readout_mlp_hidden_dim: int = DEFAULT_READOUT_MLP_HIDDEN_DIM,
    readout_mlp_layers: int = DEFAULT_READOUT_MLP_LAYERS,
    weight_decay: float = 0.01,
    eta_min_ratio: float = 0.01,
    lr_schedule: str = DEFAULT_LR_SCHEDULE,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    adam_beta1: float = DEFAULT_ADAM_BETA1,
    adam_beta2: float = DEFAULT_ADAM_BETA2,
    adam_eps: float = DEFAULT_ADAM_EPS,
    val_frac: float = 0.1,
    grad_clip: float = 1.0,
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
    init_checkpoint_path: Optional[str] = None,
) -> Dict[str, object]:
    """Fine-tune one DNABERT task model with GPU-aware runtime options.

    Parameters
    ----------
    task : str
        Task name (``donor``, ``acceptor``, or ``pair``).
    pos_path : str
        Positive training examples path.
    neg_path : str
        Negative training examples path.
    checkpoint_path : str
        Output checkpoint path.
    window_len : int
        Sequence window length in nucleotides.
    donor_len : int | None
        Donor window length.
    acceptor_len : int | None
        Acceptor window length.
    pretrained_model_name : str, default=DEFAULT_PRETRAINED_MODEL_NAME
        Hugging Face model id or local path for DNABERT backbone.
    pretrained_revision : str, default=""
        Optional model revision/tag.
    trust_remote_code : bool | int, default=1
        Forwarded to ``from_pretrained``.
    epochs : int, default=20
        Number of epochs.
    batch_size : int, default=64
        Initial batch size.
    lr : float, default=2e-5
        Learning rate.
    seed : int, default=1337
        Random seed.
    max_tokens : str | int, default="auto"
        Maximum token length (``auto`` or integer >= 2).
    dropout : float, default=0.1
        Dropout rate of classification head.
    head_layer_norm : bool | int, default=1
        Whether to apply LayerNorm before the classification head.
    readout_type : str, default="cnn"
        Readout head variant: ``cnn``, ``linear``, or ``mlp``.
    readout_cnn_kernel_size : int, default=3
        Odd kernel size used when ``readout_type="cnn"``.
    readout_mlp_hidden_dim : int, default=256
        Hidden width used when ``readout_type="mlp"``.
    readout_mlp_layers : int, default=1
        Number of hidden MLP layers used when ``readout_type="mlp"``.
    weight_decay : float, default=0.01
        AdamW weight decay.
    eta_min_ratio : float, default=0.01
        Scheduler eta_min ratio.
    lr_schedule : str, default="cosine"
        Learning-rate decay family: ``cosine`` or ``linear``.
    warmup_ratio : float, default=0.01
        Fraction of optimizer steps used for linear warmup.
    adam_beta1 : float, default=0.9
        AdamW first-moment decay.
    adam_beta2 : float, default=0.98
        AdamW second-moment decay.
    adam_eps : float, default=1e-8
        AdamW epsilon for numerical stability.
    val_frac : float, default=0.1
        Validation fraction.
    grad_clip : float, default=1.0
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
        Maximum positive class weight.
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
        Minimum batch size when retrying after OOM.
    max_oom_retries : int, default=8
        Maximum OOM retries.
    quick_phase : bool, default=False
        Whether this run is a quick-phase trial.
    gpu_id : int | None, default=None
        Assigned GPU id for sweep logs.
    init_checkpoint_path : str | None, default=None
        Optional checkpoint path used as initialization before training.

    Returns
    -------
    dict[str, object]
        Task training summary with metrics and runtime metadata.

    Raises
    ------
    ValueError
        If public arguments are invalid.
    RuntimeError
        If training fails for non-recoverable runtime errors.
    """
    _require_transformers()

    if window_len <= 0:
        raise ValueError("window_len must be positive.")
    if pretrained_model_name.strip() == "":
        raise ValueError("--pretrained_model_name must be non-empty.")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("--dropout must satisfy 0 <= dropout < 1.")
    if isinstance(head_layer_norm, int) and head_layer_norm not in (0, 1):
        raise ValueError("--head_layer_norm must be 0 or 1.")
    normalized_readout_type = _normalize_readout_type(
        readout_type,
        arg_name="--readout_type",
    )
    if normalized_readout_type == "cnn":
        if readout_cnn_kernel_size <= 0:
            raise ValueError("--readout_cnn_kernel_size must be positive.")
        if readout_cnn_kernel_size % 2 == 0:
            raise ValueError("--readout_cnn_kernel_size must be odd.")
    if normalized_readout_type == "mlp":
        if readout_mlp_hidden_dim <= 0:
            raise ValueError("--readout_mlp_hidden_dim must be positive.")
        if readout_mlp_layers <= 0:
            raise ValueError("--readout_mlp_layers must be positive.")
    if weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative.")
    if eta_min_ratio < 0.0 or eta_min_ratio > 1.0:
        raise ValueError("--eta_min_ratio must satisfy 0 <= eta_min_ratio <= 1.")
    normalized_lr_schedule = _normalize_lr_schedule(
        lr_schedule,
        arg_name="--lr_schedule",
    )
    if warmup_ratio < 0.0 or warmup_ratio >= 1.0:
        raise ValueError("--warmup_ratio must satisfy 0 <= warmup_ratio < 1.")
    if adam_beta1 <= 0.0 or adam_beta1 >= 1.0:
        raise ValueError("--adam_beta1 must satisfy 0 < adam_beta1 < 1.")
    if adam_beta2 <= 0.0 or adam_beta2 >= 1.0:
        raise ValueError("--adam_beta2 must satisfy 0 < adam_beta2 < 1.")
    if adam_beta1 >= adam_beta2:
        raise ValueError("--adam_beta1 must be smaller than --adam_beta2.")
    if adam_eps <= 0.0:
        raise ValueError("--adam_eps must be positive.")
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
    if init_checkpoint_path is not None and init_checkpoint_path.strip() == "":
        init_checkpoint_path = None

    device = pick_device(device)
    resolved_num_workers = _resolve_num_workers(num_workers, device=device)
    use_pin_memory = _bool_from_flag(pin_memory) and device == "cuda"
    use_persistent_workers = (
        _bool_from_flag(persistent_workers) and resolved_num_workers > 0
    )
    # DNABERT keeps train/val/train-eval loaders alive for the full training loop.
    # Persisting all three worker pools can exhaust file descriptors on large nodes.
    eval_persistent_workers = False
    use_amp_bool = _bool_from_flag(use_amp) and device == "cuda"
    allow_tf32_bool = _bool_from_flag(allow_tf32)
    deterministic_bool = _bool_from_flag(deterministic)
    cudnn_benchmark_bool = _bool_from_flag(cudnn_benchmark)
    trust_remote_code_bool = _bool_from_flag(trust_remote_code)
    head_layer_norm_bool = _bool_from_flag(head_layer_norm)
    amp_dtype_resolved = _resolve_amp_dtype(amp_dtype, device)
    compile_enabled = _resolve_compile_enabled(
        compile_mode=compile_mode,
        compile_flag=compile_model,
        quick_phase=quick_phase,
        device=device,
        epochs=epochs,
    )
    revision = _normalize_revision(pretrained_revision)

    set_seed(
        seed=seed,
        deterministic=deterministic_bool,
        cudnn_benchmark=cudnn_benchmark_bool,
        allow_tf32=allow_tf32_bool,
    )
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if task == "pair":
        examples = read_examples_pair_task(
            pos_path,
            neg_path,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
            negative_pair_only=True,
        )
    else:
        examples = read_examples_single_task(
            pos_path,
            neg_path,
            task,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
        )

    n_pos = sum(label for _, label in examples)
    n_neg = len(examples) - n_pos
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Insufficient training examples for {task}: pos={n_pos}, neg={n_neg}."
        )

    train_ex, val_ex = stratified_split(examples, val_frac=val_frac, seed=seed)
    _ = apply_eta_process_title_placeholder()
    print(
        f"[{task}] device={device} total={len(examples)} "
        f"(pos={n_pos}, neg={n_neg}) train={len(train_ex)} val={len(val_ex)}"
    )

    tokenizer = _load_tokenizer(
        pretrained_model_name=pretrained_model_name,
        pretrained_revision=revision,
        trust_remote_code=trust_remote_code_bool,
    )
    resolved_input_kmer = _resolve_tokenizer_input_kmer(tokenizer)
    input_mode_text = (
        f"{resolved_input_kmer}-mer" if resolved_input_kmer is not None else "raw"
    )
    print(f"[{task}] tokenizer_input_mode={input_mode_text}")
    tokenizer_limit = _resolve_tokenizer_max_length(tokenizer)
    max_tokens_effective = _resolve_max_tokens(
        raw=max_tokens,
        window_len=window_len,
        tokenizer_limit=tokenizer_limit,
        input_kmer=resolved_input_kmer,
    )

    pretokenize_dataset = True
    if task == "pair":
        train_ds = DnaBertPairTokenDataset(
            examples=train_ex,
            tokenizer=tokenizer,
            max_tokens=max_tokens_effective,
            input_kmer=resolved_input_kmer,
            pretokenize=pretokenize_dataset,
        )
        val_ds = DnaBertPairTokenDataset(
            examples=val_ex,
            tokenizer=tokenizer,
            max_tokens=max_tokens_effective,
            input_kmer=resolved_input_kmer,
            pretokenize=pretokenize_dataset,
        )
    else:
        train_ds = DnaBertTokenDataset(
            examples=train_ex,
            tokenizer=tokenizer,
            max_tokens=max_tokens_effective,
            input_kmer=resolved_input_kmer,
            pretokenize=pretokenize_dataset,
        )
        val_ds = DnaBertTokenDataset(
            examples=val_ex,
            tokenizer=tokenizer,
            max_tokens=max_tokens_effective,
            input_kmer=resolved_input_kmer,
            pretokenize=pretokenize_dataset,
        )

    train_pos = sum(label for _, label in train_ex)
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
        compile_selected_mode: str | None = None

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
            val_loader_kwargs["persistent_workers"] = eval_persistent_workers
        val_loader = DataLoader(**val_loader_kwargs)
        train_eval_loader_kwargs: dict[str, object] = {
            "dataset": train_ds,
            "batch_size": effective_batch_size,
            "shuffle": False,
            "num_workers": resolved_num_workers,
            "pin_memory": use_pin_memory,
        }
        if resolved_num_workers > 0:
            train_eval_loader_kwargs["prefetch_factor"] = prefetch_factor
            train_eval_loader_kwargs["persistent_workers"] = eval_persistent_workers
        train_eval_loader = DataLoader(**train_eval_loader_kwargs)

        print(
            f"[{task}] loader train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} batch_size={effective_batch_size} "
            f"workers={resolved_num_workers} train_eval=on "
            f"persistent(train={int(use_persistent_workers)},eval=0)"
        )

        try:
            model = _build_dnabert_model(
                pretrained_model_name=pretrained_model_name,
                pretrained_revision=revision,
                trust_remote_code=trust_remote_code_bool,
                dropout=dropout,
                head_layer_norm=head_layer_norm_bool,
                readout_type=normalized_readout_type,
                readout_cnn_kernel_size=readout_cnn_kernel_size,
                readout_mlp_hidden_dim=readout_mlp_hidden_dim,
                readout_mlp_layers=readout_mlp_layers,
            ).to(device)
            warm_start_result = _warm_start_model(
                model,
                init_checkpoint_path=init_checkpoint_path,
                device=device,
                log_prefix=task,
            )
            initialized_from_checkpoint = warm_start_result.initialized_from_checkpoint
            init_checkpoint_path = warm_start_result.init_checkpoint_path

            if compile_enabled_attempt:
                _configure_triton_tool_paths()
                _configure_torch_compile_runtime()
                ptxas_path = os.environ.get("TRITON_PTXAS_PATH")
                ptxas_blackwell_path = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH")
                print(
                    f"[{task}] torch.compile requested "
                    f"(ptxas={ptxas_path}, "
                    f"ptxas_blackwell={ptxas_blackwell_path})."
                )
                (
                    model,
                    compile_enabled_attempt,
                    compile_selected_mode,
                    compile_setup_error,
                ) = _compile_model_with_fallback(model, compile_mode=compile_mode)
                compile_enabled = compile_enabled_attempt
                if (not compile_enabled_attempt) and compile_setup_error is not None:
                    print(
                        f"[{task}] torch.compile setup failed "
                        f"({compile_setup_error.__class__.__name__}). "
                        "Continue without compile."
                    )

            trainable_params = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            if not trainable_params:
                raise RuntimeError(
                    "No trainable parameters remain. Check frozen-head configuration."
                )

            optimizer_impl = "adamw"
            adamw_kwargs: dict[str, object] = {
                "params": trainable_params,
                "lr": lr,
                "betas": (adam_beta1, adam_beta2),
                "eps": adam_eps,
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
            total_steps = max(1, epochs * max(1, len(train_loader)))
            if total_steps <= 1:
                warmup_steps = 0
            else:
                warmup_steps = int(total_steps * warmup_ratio)
                warmup_steps = min(warmup_steps, total_steps - 1)

            def _lr_lambda(step_index: int) -> float:
                return _lr_schedule_multiplier(
                    step_index=step_index,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    eta_min_ratio=eta_min_ratio,
                    lr_schedule=normalized_lr_schedule,
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=_lr_lambda,
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
            best_max_f1: Optional[float] = None
            best_acc_at_0_5: Optional[float] = None
            epoch_history: list[dict[str, object]] = []
            epochs_completed = 0
            epochs_since_improvement = 0
            stopped_early = False
            task_started_at = time.perf_counter()

            for epoch in range(1, epochs + 1):
                epochs_completed = epoch
                epoch_started_at = time.perf_counter()
                model.train()
                running_loss = torch.zeros((), dtype=torch.float64)
                for input_ids, attention_mask, labels in train_loader:
                    saw_training_batch = True
                    input_ids = input_ids.to(device, non_blocking=use_non_blocking)
                    attention_mask = attention_mask.to(
                        device,
                        non_blocking=use_non_blocking,
                    )
                    labels = labels.to(device, non_blocking=use_non_blocking)

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
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        loss = criterion(logits, labels)

                    if scaler_enabled:
                        scaler.scale(loss).backward()
                        if grad_clip > 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                trainable_params,
                                grad_clip,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                    else:
                        loss.backward()
                        if grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                trainable_params,
                                grad_clip,
                            )
                        optimizer.step()
                        scheduler.step()

                    running_loss = running_loss + loss.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )

                train_loss = float(running_loss / max(1, len(train_loader)))

                val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    use_amp=use_amp_bool,
                    amp_dtype=amp_dtype_resolved,
                )
                train_metrics = evaluate(
                    model=model,
                    loader=train_eval_loader,
                    device=device,
                    use_amp=use_amp_bool,
                    amp_dtype=amp_dtype_resolved,
                )
                train_pr_auc = train_metrics.get("pr_auc")
                pr_auc = val_metrics.get("pr_auc")
                roc_auc = val_metrics.get("roc_auc")
                max_f1 = val_metrics.get("max_f1")
                acc_at_0_5 = val_metrics.get("acc@0.5")
                epoch_elapsed_sec = time.perf_counter() - epoch_started_at
                if pr_auc is not None:
                    best_pr_auc = (
                        pr_auc if best_pr_auc is None else max(best_pr_auc, pr_auc)
                    )
                if roc_auc is not None:
                    best_roc_auc = (
                        roc_auc if best_roc_auc is None else max(best_roc_auc, roc_auc)
                    )
                if max_f1 is not None:
                    best_max_f1 = (
                        max_f1 if best_max_f1 is None else max(best_max_f1, max_f1)
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
                                "pretrained_model_name": pretrained_model_name,
                                "pretrained_revision": revision,
                                "trust_remote_code": trust_remote_code_bool,
                                "max_tokens": max_tokens_effective,
                                "input_kmer": resolved_input_kmer,
                                "dropout": dropout,
                                "head_layer_norm": head_layer_norm_bool,
                                "readout_type": normalized_readout_type,
                                "readout_cnn_kernel_size": readout_cnn_kernel_size,
                                "readout_mlp_hidden_dim": readout_mlp_hidden_dim,
                                "readout_mlp_layers": readout_mlp_layers,
                                "lr_schedule": normalized_lr_schedule,
                                "warmup_ratio": warmup_ratio,
                                "adam_beta1": adam_beta1,
                                "adam_beta2": adam_beta2,
                                "adam_eps": adam_eps,
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
                        "train_pr_auc": train_pr_auc,
                        "test_pr_auc": pr_auc,
                        "pr_auc": pr_auc,
                        "roc_auc": roc_auc,
                        "max_f1": max_f1,
                        "acc@0.5": acc_at_0_5,
                        "elapsed_sec": epoch_elapsed_sec,
                        "objective_metric": score_name,
                        "objective_score": score,
                        "improved": improved,
                        "best_metric": best_metric_name,
                        "best_score": float(best_score),
                        "best_epoch": best_epoch,
                    }
                )

                mark = "*" if improved else "-"
                train_pr_auc_text = (
                    "nan" if train_pr_auc is None else f"{train_pr_auc:.4f}"
                )
                test_pr_auc_text = "nan" if pr_auc is None else f"{pr_auc:.4f}"
                objective_text = (
                    "" if score_name == "pr_auc" else f"{score_name}={score:.4f} "
                )
                task_elapsed_sec = time.perf_counter() - task_started_at
                avg_epoch_sec = task_elapsed_sec / max(1, epoch)
                epochs_remaining = max(0, epochs - epoch)
                eta_remaining_sec = avg_epoch_sec * epochs_remaining
                eta_remaining_text = _format_duration_hms(eta_remaining_sec)
                eta_utc_text = _format_eta_utc_from_now(eta_remaining_sec)
                _ = apply_eta_process_title_from_epoch_progress(
                    task_started_at=task_started_at,
                    completed_epochs=epoch,
                    total_epochs=epochs,
                )
                print(
                    f"[{task}] {mark} epoch {epoch}/{epochs} "
                    f"loss={train_loss:.4f} train_pr_auc={train_pr_auc_text} "
                    f"test_pr_auc={test_pr_auc_text} "
                    f"elapsed={epoch_elapsed_sec:.2f}s "
                    f"eta_to_max={eta_remaining_text} "
                    f"eta_max_utc={eta_utc_text} "
                    f"{objective_text}best={best_score:.4f} "
                    f"(ep {best_epoch})"
                )

                if (
                    early_stop_patience > 0
                    and epochs_since_improvement >= early_stop_patience
                ):
                    stopped_early = True
                    print(
                        f"[{task}] early stop at epoch {epoch} "
                        f"(patience={early_stop_patience}, "
                        f"min_delta={early_stop_min_delta:g})"
                    )
                    break

            print(
                f"[{task}] done best_{best_metric_name}={best_score:.4f} "
                f"at epoch {best_epoch}"
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
                "best_max_f1": best_max_f1,
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
                "pretrained_model_name": pretrained_model_name,
                "pretrained_revision": revision,
                "trust_remote_code": trust_remote_code_bool,
                "max_tokens": max_tokens_effective,
                "input_kmer": resolved_input_kmer,
                "dropout": dropout,
                "head_layer_norm": head_layer_norm_bool,
                "readout_type": normalized_readout_type,
                "readout_cnn_kernel_size": readout_cnn_kernel_size,
                "readout_mlp_hidden_dim": readout_mlp_hidden_dim,
                "readout_mlp_layers": readout_mlp_layers,
                "weight_decay": weight_decay,
                "eta_min_ratio": eta_min_ratio,
                "lr_schedule": normalized_lr_schedule,
                "warmup_ratio": warmup_ratio,
                "adam_beta1": adam_beta1,
                "adam_beta2": adam_beta2,
                "adam_eps": adam_eps,
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
                "eval_persistent_workers": eval_persistent_workers,
                "pin_memory": use_pin_memory,
                "effective_batch_size": effective_batch_size,
                "oom_retries": oom_retries,
                "gpu_id": gpu_id,
                "quick_phase": quick_phase,
                "initialized_from_checkpoint": initialized_from_checkpoint,
                "init_checkpoint_path": init_checkpoint_path,
                "optimizer_impl": optimizer_impl,
            }
        except (RuntimeError, NotImplementedError) as exc:
            is_compile_failure = compile_enabled_attempt and (
                isinstance(exc, NotImplementedError) or _is_compile_runtime_error(exc)
            )
            if is_compile_failure:
                compile_enabled = False
                _record_compile_runtime_failure(compile_selected_mode)
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


def _int_from_checkpoint(
    mapping: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    """Read one integer config field from a mapping with fallback."""
    raw = mapping.get(key, default)
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        return default
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return default
    return default


def _float_from_checkpoint(
    mapping: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    """Read one float config field from a mapping with fallback."""
    raw = mapping.get(key, default)
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return default
    return default


def load_task_model(
    checkpoint_path: str,
    device: str,
) -> Tuple[nn.Module, Dict[str, object], object]:
    """Load task model and tokenizer metadata from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")

    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint missing model_state: {checkpoint_path}")
    normalized_state = _normalize_checkpoint_state_dict(model_state)

    model_config_obj = ckpt.get("model_config", {})
    model_config: Dict[str, object]
    if isinstance(model_config_obj, dict):
        model_config = dict(model_config_obj)
    else:
        model_config = {}

    pretrained_model_name_obj = model_config.get(
        "pretrained_model_name", DEFAULT_PRETRAINED_MODEL_NAME
    )
    pretrained_model_name = str(pretrained_model_name_obj)
    pretrained_revision_obj = model_config.get("pretrained_revision", "")
    pretrained_revision_raw = (
        "" if pretrained_revision_obj is None else str(pretrained_revision_obj)
    )
    pretrained_revision = _normalize_revision(pretrained_revision_raw)
    trust_remote_code_raw = model_config.get("trust_remote_code", True)
    trust_remote_code = _bool_from_flag(
        trust_remote_code_raw
        if isinstance(trust_remote_code_raw, (bool, int))
        else True
    )
    max_tokens = _int_from_checkpoint(model_config, "max_tokens", 128)
    input_kmer_raw = model_config.get("input_kmer")
    if isinstance(input_kmer_raw, bool):
        input_kmer = None
    elif isinstance(input_kmer_raw, int):
        input_kmer = input_kmer_raw if input_kmer_raw > 0 else None
    elif isinstance(input_kmer_raw, float) and input_kmer_raw.is_integer():
        parsed_kmer = int(input_kmer_raw)
        input_kmer = parsed_kmer if parsed_kmer > 0 else None
    elif isinstance(input_kmer_raw, str) and input_kmer_raw.strip():
        try:
            parsed_kmer = int(input_kmer_raw.strip())
        except ValueError:
            input_kmer = None
        else:
            input_kmer = parsed_kmer if parsed_kmer > 0 else None
    else:
        input_kmer = None
    dropout = _float_from_checkpoint(model_config, "dropout", 0.1)
    if "head_layer_norm" not in model_config:
        raise ValueError(
            "Checkpoint model_config is missing 'head_layer_norm'. "
            "Retrain DNABERT checkpoints with the current code."
        )
    head_layer_norm_raw = model_config["head_layer_norm"]
    if isinstance(head_layer_norm_raw, bool):
        head_layer_norm = head_layer_norm_raw
    elif isinstance(head_layer_norm_raw, int) and head_layer_norm_raw in (0, 1):
        head_layer_norm = bool(head_layer_norm_raw)
    else:
        raise ValueError(
            "Checkpoint model_config.head_layer_norm must be bool or 0/1 int."
        )
    readout_type_raw = model_config.get("readout_type", DEFAULT_READOUT_TYPE)
    readout_type = _normalize_readout_type(
        str(readout_type_raw),
        arg_name="checkpoint readout_type",
    )
    readout_cnn_kernel_size = _int_from_checkpoint(
        model_config,
        "readout_cnn_kernel_size",
        DEFAULT_READOUT_CNN_KERNEL_SIZE,
    )
    readout_mlp_hidden_dim = _int_from_checkpoint(
        model_config,
        "readout_mlp_hidden_dim",
        DEFAULT_READOUT_MLP_HIDDEN_DIM,
    )
    readout_mlp_layers = _int_from_checkpoint(
        model_config,
        "readout_mlp_layers",
        DEFAULT_READOUT_MLP_LAYERS,
    )
    if readout_type == "cnn":
        if readout_cnn_kernel_size <= 0:
            raise ValueError(
                "Checkpoint model_config.readout_cnn_kernel_size must be positive."
            )
        if readout_cnn_kernel_size % 2 == 0:
            raise ValueError(
                "Checkpoint model_config.readout_cnn_kernel_size must be odd."
            )
    if readout_type == "mlp":
        if readout_mlp_hidden_dim <= 0:
            raise ValueError(
                "Checkpoint model_config.readout_mlp_hidden_dim must be positive."
            )
        if readout_mlp_layers <= 0:
            raise ValueError(
                "Checkpoint model_config.readout_mlp_layers must be positive."
            )

    model = _build_dnabert_model(
        pretrained_model_name=pretrained_model_name,
        pretrained_revision=pretrained_revision,
        trust_remote_code=trust_remote_code,
        dropout=dropout,
        head_layer_norm=head_layer_norm,
        readout_type=readout_type,
        readout_cnn_kernel_size=readout_cnn_kernel_size,
        readout_mlp_hidden_dim=readout_mlp_hidden_dim,
        readout_mlp_layers=readout_mlp_layers,
    ).to(device)
    model.load_state_dict(normalized_state)
    model.eval()

    tokenizer = _load_tokenizer(
        pretrained_model_name=pretrained_model_name,
        pretrained_revision=pretrained_revision,
        trust_remote_code=trust_remote_code,
    )

    resolved_config: Dict[str, object] = {
        "pretrained_model_name": pretrained_model_name,
        "pretrained_revision": pretrained_revision,
        "trust_remote_code": trust_remote_code,
        "max_tokens": max_tokens,
        "input_kmer": input_kmer,
        "dropout": dropout,
        "head_layer_norm": head_layer_norm,
        "readout_type": readout_type,
        "readout_cnn_kernel_size": readout_cnn_kernel_size,
        "readout_mlp_hidden_dim": readout_mlp_hidden_dim,
        "readout_mlp_layers": readout_mlp_layers,
    }
    return model, resolved_config, tokenizer


@torch.no_grad()
def score_sequences(
    model: nn.Module,
    sequences: Sequence[str],
    tokenizer: object,
    max_tokens: int,
    device: str,
    batch_size: int = 256,
    task_name: str = "task",
    input_kmer: Optional[int] = None,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> np.ndarray:
    """Score input sequences with one trained task model."""
    if not sequences:
        return np.array([])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()
    all_log_scores: list[np.ndarray] = []
    total = len(sequences)
    log_every_batches = 200
    use_non_blocking = device == "cuda"

    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        batch_sequences = sequences[start : start + batch_size]
        ids_tensor, mask_tensor = _tokenize_sequences(
            tokenizer=tokenizer,
            sequences=batch_sequences,
            max_tokens=max_tokens,
            input_kmer=input_kmer,
        )
        ids_tensor = ids_tensor.to(device, non_blocking=use_non_blocking)
        mask_tensor = mask_tensor.to(device, non_blocking=use_non_blocking)

        if use_amp and device == "cuda" and amp_dtype is not None:
            amp_context: ContextManager[object] = torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            )
        else:
            amp_context = nullcontext()
        with amp_context:
            logits = model(input_ids=ids_tensor, attention_mask=mask_tensor)
        log_scores = log10_sigmoid_np(logits.float().cpu().numpy())
        all_log_scores.append(log_scores)
        should_log = (
            batch_idx == 1
            or (batch_idx % log_every_batches == 0)
            or (start + len(batch_sequences) >= total)
        )
        if should_log:
            done = start + len(batch_sequences)
            pct = (100.0 * float(done)) / float(total)
            print(
                f"[{task_name}] infer progress: {done}/{total} "
                f"({pct:.1f}%) batches={batch_idx}"
            )

    return np.concatenate(all_log_scores)


@torch.no_grad()
def score_sequence_pairs(
    model: nn.Module,
    donor_sequences: Sequence[str],
    acceptor_sequences: Sequence[str],
    tokenizer: object,
    max_tokens: int,
    device: str,
    batch_size: int = 256,
    task_name: str = "pair",
    input_kmer: Optional[int] = None,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> np.ndarray:
    """Score donor/acceptor pairs with one DNABERT pair model."""
    if len(donor_sequences) != len(acceptor_sequences):
        raise ValueError(
            "donor_sequences and acceptor_sequences must have the same length."
        )
    if not donor_sequences:
        return np.array([])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()
    all_log_scores: list[np.ndarray] = []
    total = len(donor_sequences)
    log_every_batches = 200
    use_non_blocking = device == "cuda"

    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        batch_donor_sequences = donor_sequences[start : start + batch_size]
        batch_acceptor_sequences = acceptor_sequences[start : start + batch_size]
        ids_tensor, mask_tensor = _tokenize_sequence_pairs(
            tokenizer=tokenizer,
            donor_sequences=batch_donor_sequences,
            acceptor_sequences=batch_acceptor_sequences,
            max_tokens=max_tokens,
            input_kmer=input_kmer,
        )
        ids_tensor = ids_tensor.to(device, non_blocking=use_non_blocking)
        mask_tensor = mask_tensor.to(device, non_blocking=use_non_blocking)

        if use_amp and device == "cuda" and amp_dtype is not None:
            amp_context: ContextManager[object] = torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            )
        else:
            amp_context = nullcontext()
        with amp_context:
            logits = model(input_ids=ids_tensor, attention_mask=mask_tensor)
        log_scores = log10_sigmoid_np(logits.float().cpu().numpy())
        all_log_scores.append(log_scores)
        should_log = (
            batch_idx == 1
            or (batch_idx % log_every_batches == 0)
            or (start + len(batch_donor_sequences) >= total)
        )
        if should_log:
            done = start + len(batch_donor_sequences)
            pct = (100.0 * float(done)) / float(total)
            print(
                f"[{task_name}] infer progress: {done}/{total} "
                f"({pct:.1f}%) batches={batch_idx}"
            )

    return np.concatenate(all_log_scores)


def infer_site_scores(
    site_rows: List[Dict[str, object]],
    donor_model_path: str,
    acceptor_model_path: str,
    device: str = "auto",
    batch_size: int = 256,
    infer_use_amp: Union[bool, int] = 0,
    infer_amp_dtype: str = "auto",
    infer_compile: Union[bool, int] = 0,
    infer_compile_mode: str = "off",
) -> List[Dict[str, object]]:
    """Run donor/acceptor inference and return normalized site rows."""
    device = pick_device(device)
    infer_runtime = _resolve_infer_runtime_config(
        device=device,
        batch_size=batch_size,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
        infer_compile=infer_compile,
        infer_compile_mode=infer_compile_mode,
    )

    donor_model, donor_config, donor_tokenizer = load_task_model(
        donor_model_path,
        device,
    )
    acceptor_model, acceptor_config, acceptor_tokenizer = load_task_model(
        acceptor_model_path,
        device,
    )
    donor_model = _prepare_infer_model(
        model=donor_model,
        task_name="donor",
        compile_enabled=infer_runtime.compile_enabled,
        compile_mode=infer_compile_mode,
    )
    acceptor_model = _prepare_infer_model(
        model=acceptor_model,
        task_name="acceptor",
        compile_enabled=infer_runtime.compile_enabled,
        compile_mode=infer_compile_mode,
    )

    donor_max_tokens = _int_from_checkpoint(donor_config, "max_tokens", 128)
    acceptor_max_tokens = _int_from_checkpoint(acceptor_config, "max_tokens", 128)
    donor_input_kmer_obj = donor_config.get("input_kmer")
    donor_input_kmer = (
        int(donor_input_kmer_obj)
        if isinstance(donor_input_kmer_obj, int) and donor_input_kmer_obj > 0
        else None
    )
    acceptor_input_kmer_obj = acceptor_config.get("input_kmer")
    acceptor_input_kmer = (
        int(acceptor_input_kmer_obj)
        if isinstance(acceptor_input_kmer_obj, int) and acceptor_input_kmer_obj > 0
        else None
    )

    donor_seqs = [str(row["seq"]) for row in site_rows if row["site_type"] == "donor"]
    acceptor_seqs = [
        str(row["seq"]) for row in site_rows if row["site_type"] == "acceptor"
    ]

    donor_scores = score_sequences(
        model=donor_model,
        sequences=donor_seqs,
        tokenizer=donor_tokenizer,
        max_tokens=donor_max_tokens,
        device=device,
        batch_size=infer_runtime.batch_size,
        task_name="donor",
        input_kmer=donor_input_kmer,
        use_amp=infer_runtime.use_amp,
        amp_dtype=infer_runtime.amp_dtype,
    )
    acceptor_scores = score_sequences(
        model=acceptor_model,
        sequences=acceptor_seqs,
        tokenizer=acceptor_tokenizer,
        max_tokens=acceptor_max_tokens,
        device=device,
        batch_size=infer_runtime.batch_size,
        task_name="acceptor",
        input_kmer=acceptor_input_kmer,
        use_amp=infer_runtime.use_amp,
        amp_dtype=infer_runtime.amp_dtype,
    )
    if len(donor_scores) != len(donor_seqs):
        raise ValueError(
            "Donor score count does not match donor site count: "
            f"{len(donor_scores)} != {len(donor_seqs)}"
        )
    if len(acceptor_scores) != len(acceptor_seqs):
        raise ValueError(
            "Acceptor score count does not match acceptor site count: "
            f"{len(acceptor_scores)} != {len(acceptor_seqs)}"
        )

    out_rows: List[Dict[str, object]] = []
    donor_idx = 0
    acceptor_idx = 0

    for row in site_rows:
        site_type = str(row["site_type"])
        if site_type == "donor":
            score = float(donor_scores[donor_idx])
            donor_idx += 1
        else:
            score = float(acceptor_scores[acceptor_idx])
            acceptor_idx += 1

        out_rows.append(
            {
                "transcript_id": row["transcript_id"],
                "intron_index": int(row["intron_index"]),
                "site_type": site_type,
                "score": score,
                SCORE_SPACE_FIELD: SCORE_SPACE_LOG10,
            }
        )

    return out_rows


def infer_pair_site_scores(
    pair_rows: List[Dict[str, object]],
    pair_model_path: str,
    device: str = "auto",
    batch_size: int = 256,
    infer_use_amp: Union[bool, int] = 0,
    infer_amp_dtype: str = "auto",
    infer_compile: Union[bool, int] = 0,
    infer_compile_mode: str = "off",
) -> List[Dict[str, object]]:
    """Run pair inference and return normalized site rows."""
    device = pick_device(device)
    infer_runtime = _resolve_infer_runtime_config(
        device=device,
        batch_size=batch_size,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
        infer_compile=infer_compile,
        infer_compile_mode=infer_compile_mode,
    )

    pair_model, pair_config, pair_tokenizer = load_task_model(
        pair_model_path,
        device,
    )
    pair_model = _prepare_infer_model(
        model=pair_model,
        task_name="pair",
        compile_enabled=infer_runtime.compile_enabled,
        compile_mode=infer_compile_mode,
    )
    pair_max_tokens = _int_from_checkpoint(pair_config, "max_tokens", 128)
    pair_input_kmer_obj = pair_config.get("input_kmer")
    pair_input_kmer = (
        int(pair_input_kmer_obj)
        if isinstance(pair_input_kmer_obj, int) and pair_input_kmer_obj > 0
        else None
    )

    donor_sequences = [str(row["donor_seq"]) for row in pair_rows]
    acceptor_sequences = [str(row["acceptor_seq"]) for row in pair_rows]
    pair_scores = score_sequence_pairs(
        model=pair_model,
        donor_sequences=donor_sequences,
        acceptor_sequences=acceptor_sequences,
        tokenizer=pair_tokenizer,
        max_tokens=pair_max_tokens,
        device=device,
        batch_size=infer_runtime.batch_size,
        task_name="pair",
        input_kmer=pair_input_kmer,
        use_amp=infer_runtime.use_amp,
        amp_dtype=infer_runtime.amp_dtype,
    )
    if len(pair_scores) != len(pair_rows):
        raise ValueError(
            "Pair score count does not match pair row count: "
            f"{len(pair_scores)} != {len(pair_rows)}"
        )

    out_rows: List[Dict[str, object]] = []
    for index, row in enumerate(pair_rows):
        score = float(pair_scores[index])
        out_rows.append(
            {
                "transcript_id": row["transcript_id"],
                "intron_index": int(row["intron_index"]),
                "site_type": "pair",
                "score": score,
                SCORE_SPACE_FIELD: SCORE_SPACE_LOG10,
            }
        )
    return out_rows


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register DNABERT-specific training arguments."""
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
        choices=["both", "donor", "acceptor", "pair"],
        default="both",
        help=(
            "Training target. 'both' trains donor and acceptor. "
            "'donor'/'acceptor' train one task only (for tuning). "
            "'pair' trains one pair-model readout."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name",
        default=DEFAULT_PRETRAINED_MODEL_NAME,
        help="Hugging Face model id or local path for DNABERT backbone.",
    )
    parser.add_argument(
        "--pretrained_revision",
        default="",
        help="Optional Hugging Face revision (tag/branch/commit).",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=int,
        choices=[0, 1],
        default=1,
        help="Forwarded to from_pretrained(..., trust_remote_code=...).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--max_tokens",
        default="auto",
        help=(
            "Max token length. Use integer or auto. Auto follows tokenizer input "
            "mode (raw: window_len + 2, fixed k-mer: window_len - k + 3)."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--head_layer_norm",
        type=int,
        choices=[0, 1],
        default=1,
        help="Apply LayerNorm before the DNABERT classification head.",
    )
    parser.add_argument(
        "--readout_type",
        choices=list(READOUT_TYPE_CHOICES),
        default=DEFAULT_READOUT_TYPE,
        help="Readout type on top of DNABERT token features.",
    )
    parser.add_argument(
        "--readout_cnn_kernel_size",
        type=int,
        default=DEFAULT_READOUT_CNN_KERNEL_SIZE,
        help="Odd kernel size used when --readout_type=cnn.",
    )
    parser.add_argument(
        "--readout_mlp_hidden_dim",
        type=int,
        default=DEFAULT_READOUT_MLP_HIDDEN_DIM,
        help="MLP hidden width used when --readout_type=mlp.",
    )
    parser.add_argument(
        "--readout_mlp_layers",
        type=int,
        default=DEFAULT_READOUT_MLP_LAYERS,
        help="Number of hidden MLP layers used when --readout_type=mlp.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eta_min_ratio", type=float, default=0.01)
    parser.add_argument(
        "--lr_schedule",
        choices=list(LR_SCHEDULE_CHOICES),
        default=DEFAULT_LR_SCHEDULE,
        help="Learning-rate schedule after warmup.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help="Linear warmup ratio over optimizer steps (0 <= r < 1).",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=DEFAULT_ADAM_BETA1,
        help="AdamW beta1 coefficient.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=DEFAULT_ADAM_BETA2,
        help="AdamW beta2 coefficient.",
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=DEFAULT_ADAM_EPS,
        help="AdamW epsilon.",
    )
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

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
    parser.add_argument("--donor_max_tokens", default=None)
    parser.add_argument("--acceptor_max_tokens", default=None)
    parser.add_argument("--donor_dropout", type=float, default=None)
    parser.add_argument("--acceptor_dropout", type=float, default=None)
    parser.add_argument("--donor_head_layer_norm", type=int, default=None)
    parser.add_argument("--acceptor_head_layer_norm", type=int, default=None)
    parser.add_argument(
        "--donor_readout_type",
        choices=list(READOUT_TYPE_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--acceptor_readout_type",
        choices=list(READOUT_TYPE_CHOICES),
        default=None,
    )
    parser.add_argument("--donor_readout_cnn_kernel_size", type=int, default=None)
    parser.add_argument("--acceptor_readout_cnn_kernel_size", type=int, default=None)
    parser.add_argument("--donor_readout_mlp_hidden_dim", type=int, default=None)
    parser.add_argument("--acceptor_readout_mlp_hidden_dim", type=int, default=None)
    parser.add_argument("--donor_readout_mlp_layers", type=int, default=None)
    parser.add_argument("--acceptor_readout_mlp_layers", type=int, default=None)
    parser.add_argument("--donor_weight_decay", type=float, default=None)
    parser.add_argument("--acceptor_weight_decay", type=float, default=None)
    parser.add_argument("--donor_eta_min_ratio", type=float, default=None)
    parser.add_argument("--acceptor_eta_min_ratio", type=float, default=None)
    parser.add_argument(
        "--donor_lr_schedule",
        choices=list(LR_SCHEDULE_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--acceptor_lr_schedule",
        choices=list(LR_SCHEDULE_CHOICES),
        default=None,
    )
    parser.add_argument("--donor_warmup_ratio", type=float, default=None)
    parser.add_argument("--acceptor_warmup_ratio", type=float, default=None)
    parser.add_argument("--donor_adam_beta1", type=float, default=None)
    parser.add_argument("--acceptor_adam_beta1", type=float, default=None)
    parser.add_argument("--donor_adam_beta2", type=float, default=None)
    parser.add_argument("--acceptor_adam_beta2", type=float, default=None)
    parser.add_argument("--donor_adam_eps", type=float, default=None)
    parser.add_argument("--acceptor_adam_eps", type=float, default=None)
    parser.add_argument("--donor_val_frac", type=float, default=None)
    parser.add_argument("--acceptor_val_frac", type=float, default=None)
    parser.add_argument("--donor_grad_clip", type=float, default=None)
    parser.add_argument("--acceptor_grad_clip", type=float, default=None)

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
        help="Minimum batch size for OOM backoff retries.",
    )
    parser.add_argument(
        "--max_oom_retries",
        type=int,
        default=8,
        help="Maximum retries when reducing batch size after OOM.",
    )

    parser.add_argument(
        "--loss",
        choices=list(LOSS_NAME_CHOICES),
        default="weighted_bce",
        help="Training loss type for donor/acceptor/pair models.",
    )
    parser.add_argument(
        "--pos_weight_cap",
        type=float,
        default=20.0,
        help="Upper bound of positive-class weight for weighted_bce.",
    )
    parser.add_argument("--donor_pos_weight_cap", type=float, default=None)
    parser.add_argument("--acceptor_pos_weight_cap", type=float, default=None)
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter used when --loss focal is selected.",
    )
    parser.add_argument("--donor_focal_gamma", type=float, default=None)
    parser.add_argument("--acceptor_focal_gamma", type=float, default=None)
    parser.add_argument(
        "--focal_alpha_pos",
        type=float,
        default=None,
        help=(
            "Positive-class alpha for focal loss (0 < alpha < 1). "
            "If omitted, it is inferred from class imbalance."
        ),
    )
    parser.add_argument("--donor_focal_alpha_pos", type=float, default=None)
    parser.add_argument("--acceptor_focal_alpha_pos", type=float, default=None)
    parser.add_argument(
        "--asym_gamma_pos",
        type=float,
        default=0.0,
        help="Positive-class gamma for --loss asymmetric_focal.",
    )
    parser.add_argument("--donor_asym_gamma_pos", type=float, default=None)
    parser.add_argument("--acceptor_asym_gamma_pos", type=float, default=None)
    parser.add_argument(
        "--asym_gamma_neg",
        type=float,
        default=4.0,
        help="Negative-class gamma for --loss asymmetric_focal.",
    )
    parser.add_argument("--donor_asym_gamma_neg", type=float, default=None)
    parser.add_argument("--acceptor_asym_gamma_neg", type=float, default=None)
    parser.add_argument(
        "--asym_alpha_pos",
        type=float,
        default=None,
        help=(
            "Positive-class alpha for --loss asymmetric_focal "
            "(0 < alpha < 1). If omitted, inferred from class imbalance."
        ),
    )
    parser.add_argument("--donor_asym_alpha_pos", type=float, default=None)
    parser.add_argument("--acceptor_asym_alpha_pos", type=float, default=None)
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional run-name suffix for training summary.",
    )


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register DNABERT-specific inference arguments."""
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=None,
        help="Inference-only override for --batch_size.",
    )
    parser.add_argument(
        "--infer_use_amp",
        type=int,
        choices=[0, 1],
        default=None,
        help="Inference-only AMP override. Default follows --use_amp.",
    )
    parser.add_argument(
        "--infer_amp_dtype",
        choices=["auto", "bf16", "fp16"],
        default=None,
        help="Inference-only AMP dtype override. Default follows --amp_dtype.",
    )
    parser.add_argument(
        "--infer_compile",
        type=int,
        choices=[0, 1],
        default=None,
        help="Inference-only compile override. Default follows --compile.",
    )
    parser.add_argument(
        "--infer_compile_mode",
        choices=["off", "on", "auto"],
        default=None,
        help=("Inference-only compile mode override. Default follows --compile_mode."),
    )


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train DNABERT donor/acceptor or pair models via one unified interface."""
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
    pair_window_len = donor_window_len + acceptor_window_len + 1
    model_name = str(getattr(common_args, "model", "dnabert"))
    model_tasks = checkpoint_tasks_for_model(model_name)

    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=False,
        tasks=model_tasks,
    )
    allowed_train_targets = (
        ("both", *model_tasks) if len(model_tasks) > 1 else model_tasks
    )
    train_target = resolve_train_target(
        model_args,
        allowed_targets=allowed_train_targets,
    )

    schedule = resolve_training_schedule(
        epochs_arg=model_args.epochs,
        max_epochs=int(model_args.max_epochs),
        patience_arg=model_args.early_stop_patience,
        min_delta_arg=model_args.early_stop_min_delta,
    )

    tasks_to_train = resolve_tasks_to_train(
        train_target,
        both_tasks=model_tasks,
    )
    task_init_checkpoint_paths = resolve_task_init_checkpoint_paths(
        common_args,
        tasks=model_tasks,
    )
    task_window_len = {
        "donor": donor_window_len,
        "acceptor": acceptor_window_len,
        "pair": pair_window_len,
    }

    task_hparams: dict[str, TaskTrainParams] = {}
    task_metrics: dict[str, Dict[str, object]] = {}
    for task in tasks_to_train:
        resolved = _resolve_task_train_params(task=task, model_args=model_args)
        task_hparams[task] = resolved
        task_metrics[task] = train_task_model(
            task=task,
            pos_path=train_pos_path,
            neg_path=train_neg_path,
            checkpoint_path=task_checkpoint_paths[task],
            window_len=task_window_len[task],
            donor_len=donor_len,
            acceptor_len=acceptor_len,
            pretrained_model_name=model_args.pretrained_model_name,
            pretrained_revision=model_args.pretrained_revision,
            trust_remote_code=model_args.trust_remote_code,
            epochs=schedule.resolved_epochs,
            early_stop_patience=schedule.effective_early_stop_patience,
            early_stop_min_delta=schedule.early_stop_min_delta,
            batch_size=resolved.batch_size,
            lr=resolved.lr,
            seed=common_args.seed,
            max_tokens=resolved.max_tokens,
            dropout=resolved.dropout,
            head_layer_norm=resolved.head_layer_norm,
            readout_type=resolved.readout_type,
            readout_cnn_kernel_size=resolved.readout_cnn_kernel_size,
            readout_mlp_hidden_dim=resolved.readout_mlp_hidden_dim,
            readout_mlp_layers=resolved.readout_mlp_layers,
            weight_decay=resolved.weight_decay,
            eta_min_ratio=resolved.eta_min_ratio,
            lr_schedule=resolved.lr_schedule,
            warmup_ratio=resolved.warmup_ratio,
            adam_beta1=resolved.adam_beta1,
            adam_beta2=resolved.adam_beta2,
            adam_eps=resolved.adam_eps,
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
            init_checkpoint_path=task_init_checkpoint_paths[task],
        )

    run_name_lr = model_args.lr
    run_name_batch_size = model_args.batch_size
    if train_target != "both":
        selected_params = task_hparams[tasks_to_train[0]]
        run_name_lr = selected_params.lr
        run_name_batch_size = selected_params.batch_size

    run_name = build_run_name(
        model_name=model_name,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        lr=run_name_lr,
        batch_size=run_name_batch_size,
        epochs=schedule.resolved_epochs,
        tag=model_args.tag,
    )

    task_hparams_summary: dict[str, Dict[str, object]] = {}
    for task, params in task_hparams.items():
        task_hparams_summary[task] = {
            "batch_size": params.batch_size,
            "lr": params.lr,
            "loss": params.loss_name,
            "max_tokens": params.max_tokens,
            "dropout": params.dropout,
            "head_layer_norm": bool(params.head_layer_norm),
            "readout_type": params.readout_type,
            "readout_cnn_kernel_size": params.readout_cnn_kernel_size,
            "readout_mlp_hidden_dim": params.readout_mlp_hidden_dim,
            "readout_mlp_layers": params.readout_mlp_layers,
            "weight_decay": params.weight_decay,
            "eta_min_ratio": params.eta_min_ratio,
            "lr_schedule": params.lr_schedule,
            "warmup_ratio": params.warmup_ratio,
            "adam_beta1": params.adam_beta1,
            "adam_beta2": params.adam_beta2,
            "adam_eps": params.adam_eps,
            "val_frac": params.val_frac,
            "grad_clip": params.grad_clip,
            "pos_weight_cap": params.pos_weight_cap,
            "focal_gamma": params.focal_gamma,
            "focal_alpha_pos": params.focal_alpha_pos,
            "asym_gamma_pos": params.asym_gamma_pos,
            "asym_gamma_neg": params.asym_gamma_neg,
            "asym_alpha_pos": params.asym_alpha_pos,
        }

    primary_task = model_tasks[0]
    primary_checkpoint_path = task_checkpoint_paths[primary_task]

    summary: Dict[str, object] = {
        "model": model_name,
        "species": common_args.species,
        "train_pos_path": train_pos_path,
        "train_neg_path": train_neg_path,
        "donor_len": donor_len,
        "acceptor_len": acceptor_len,
        "epochs": schedule.resolved_epochs,
        "epochs_config": str(model_args.epochs),
        "epochs_auto": schedule.epochs_auto,
        "max_epochs": model_args.max_epochs,
        "early_stop_patience": schedule.early_stop_patience,
        "early_stop_min_delta": schedule.early_stop_min_delta,
        "batch_size": model_args.batch_size,
        "lr": model_args.lr,
        "train_target": train_target,
        "seed": common_args.seed,
        "device": common_args.device,
        "checkpoint_name": os.path.basename(primary_checkpoint_path),
        "pretrained_model_name": model_args.pretrained_model_name,
        "pretrained_revision": model_args.pretrained_revision,
        "trust_remote_code": bool(model_args.trust_remote_code),
        "max_tokens": model_args.max_tokens,
        "dropout": model_args.dropout,
        "head_layer_norm": bool(model_args.head_layer_norm),
        "readout_type": model_args.readout_type,
        "readout_cnn_kernel_size": model_args.readout_cnn_kernel_size,
        "readout_mlp_hidden_dim": model_args.readout_mlp_hidden_dim,
        "readout_mlp_layers": model_args.readout_mlp_layers,
        "weight_decay": model_args.weight_decay,
        "eta_min_ratio": model_args.eta_min_ratio,
        "lr_schedule": model_args.lr_schedule,
        "warmup_ratio": model_args.warmup_ratio,
        "adam_beta1": model_args.adam_beta1,
        "adam_beta2": model_args.adam_beta2,
        "adam_eps": model_args.adam_eps,
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
    for task in model_tasks:
        summary[f"{task}_checkpoint_path"] = task_checkpoint_paths[task]
    attach_init_checkpoint_summary(
        summary,
        task_init_checkpoint_paths=task_init_checkpoint_paths,
    )
    summary.update(task_metrics)
    return summary


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Run site-level inference and return rows with fixed schema."""
    model_name = str(getattr(common_args, "model", "dnabert"))
    model_tasks = checkpoint_tasks_for_model(model_name)
    dirs = species_data_dirs(common_args.species)
    inferred_train_len: Optional[int] = None
    if common_args.donor_len is None and common_args.acceptor_len is None:
        try:
            _, _, inferred_train_len = infer_default_train_paths(
                train_dir=dirs["raw"],
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
    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=True,
        tasks=model_tasks,
    )
    infer_batch_size = (
        int(model_args.infer_batch_size)
        if model_args.infer_batch_size is not None
        else int(model_args.batch_size)
    )
    infer_use_amp = (
        int(model_args.infer_use_amp)
        if model_args.infer_use_amp is not None
        else int(model_args.use_amp)
    )
    infer_amp_dtype = (
        str(model_args.infer_amp_dtype)
        if model_args.infer_amp_dtype is not None
        else str(model_args.amp_dtype)
    )
    infer_compile = (
        int(model_args.infer_compile)
        if model_args.infer_compile is not None
        else int(bool(model_args.compile))
    )
    infer_compile_mode = (
        str(model_args.infer_compile_mode)
        if model_args.infer_compile_mode is not None
        else str(model_args.compile_mode)
    )
    if model_tasks == ("pair",):
        pair_model_path = task_checkpoint_paths["pair"]
        pair_rows, skipped_short, skipped_unpaired = read_test_pair_rows(
            test_tsv=test_tsv,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
        )
        print(f"Loaded test pairs: {len(pair_rows)}")
        if skipped_short:
            print(f"Skipped short sites: {skipped_short}")
        if skipped_unpaired:
            print(f"Skipped unpaired introns: {skipped_unpaired}")
        return infer_pair_site_scores(
            pair_rows=pair_rows,
            pair_model_path=pair_model_path,
            device=common_args.device,
            batch_size=infer_batch_size,
            infer_use_amp=infer_use_amp,
            infer_amp_dtype=infer_amp_dtype,
            infer_compile=infer_compile,
            infer_compile_mode=infer_compile_mode,
        )

    donor_model_path = task_checkpoint_paths["donor"]
    acceptor_model_path = task_checkpoint_paths["acceptor"]
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
        batch_size=infer_batch_size,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
        infer_compile=infer_compile,
        infer_compile_mode=infer_compile_mode,
    )
