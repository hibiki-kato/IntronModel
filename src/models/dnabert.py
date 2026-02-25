"""DNABERT-2 fine-tuning model for site-level splice scoring.

This module integrates Hugging Face DNABERT-2 checkpoints into the unified
pipeline contract used by ``run_model.py``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import random
import shutil
import sys
from typing import (
    ContextManager,
    Dict,
    Iterator,
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
from util.model_task_paths import (
    resolve_required_checkpoint_paths,
    resolve_tasks_to_train,
    resolve_train_target,
)
from util.model_runtime import (
    bool_from_flag as _bool_from_flag,
    configure_torch_compile_runtime as _configure_torch_compile_runtime,
    configure_triton_tool_paths as _configure_triton_tool_paths,
    empty_device_cache as _empty_device_cache,
    export_model_state_dict as _export_model_state_dict,
    fallback_average_precision as _fallback_average_precision,
    fallback_roc_auc as _fallback_roc_auc,
    is_compile_runtime_error as _is_compile_runtime_error,
    is_cuda_oom_error as _is_cuda_oom_error,
    is_mps_oom_error as _is_mps_oom_error,
    normalize_checkpoint_state_dict as _normalize_checkpoint_state_dict,
    pick_device,
    resolve_amp_dtype as _resolve_amp_dtype,
    resolve_compile_enabled as _resolve_compile_enabled,
    resolve_mps_max_batch_size,
    resolve_num_workers as _resolve_num_workers,
    seed_worker as _seed_worker,
    set_seed,
    sigmoid_np,
)
from util.training_control import (
    resolve_early_stopping_params,
    resolve_training_epoch_budget,
)

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


def _load_tokenizer(
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
) -> object:
    """Load DNABERT tokenizer from Hugging Face."""
    _require_transformers()
    assert AutoTokenizer is not None
    tokenizer_kwargs = _without_none_kwargs(
        {
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


class DnaBertBinaryClassifier(nn.Module):
    """Binary classifier head on top of a pretrained DNABERT backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        dropout: float,
        head_layer_norm: bool,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        self.backbone = backbone
        self.head_norm = nn.LayerNorm(hidden_size) if head_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return binary logits of shape ``(batch,)``."""
        outputs = self.backbone(
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

        cls_hidden = hidden[:, 0, :]
        head_input = self.head_norm(cls_hidden)
        logits = self.classifier(self.dropout(head_input)).squeeze(-1)
        return logits


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


def _build_dnabert_model(
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
    dropout: float,
    head_layer_norm: bool,
) -> DnaBertBinaryClassifier:
    """Build DNABERT classifier from a pretrained checkpoint."""
    _require_transformers()
    assert AutoConfig is not None
    assert AutoModel is not None
    resolved_pretrained_model_name = _resolve_pretrained_model_name(
        pretrained_model_name
    )
    config_kwargs = _without_none_kwargs(
        {
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
    return DnaBertBinaryClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        dropout=dropout,
        head_layer_norm=head_layer_norm,
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
) -> TaskTrainParams:
    """Resolve task-specific train parameters with fallback to shared values."""
    if task not in {"donor", "acceptor"}:
        raise ValueError(f"Unsupported task: {task}")

    prefix = f"{task}_"

    def _override_or_default(name: str, default: object) -> object:
        override = getattr(model_args, f"{prefix}{name}", None)
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


def stratified_split(
    examples: Sequence[Tuple[str, int]],
    val_frac: float = 0.1,
    seed: int = 1337,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
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
    weight_decay: float = 0.01,
    eta_min_ratio: float = 0.01,
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
        Task name (``donor`` or ``acceptor``).
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
    weight_decay : float, default=0.01
        AdamW weight decay.
    eta_min_ratio : float, default=0.01
        Scheduler eta_min ratio.
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
    if init_checkpoint_path is not None and init_checkpoint_path.strip() == "":
        init_checkpoint_path = None

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

        print(
            f"[{task}] loader train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} batch_size={effective_batch_size} "
            f"workers={resolved_num_workers}"
        )

        try:
            model = _build_dnabert_model(
                pretrained_model_name=pretrained_model_name,
                pretrained_revision=revision,
                trust_remote_code=trust_remote_code_bool,
                dropout=dropout,
                head_layer_norm=head_layer_norm_bool,
            ).to(device)
            initialized_from_checkpoint = False
            if init_checkpoint_path is not None:
                ckpt = torch.load(
                    init_checkpoint_path,
                    map_location=device,
                    weights_only=False,
                )
                if not isinstance(ckpt, dict):
                    raise ValueError(
                        f"Invalid init checkpoint payload: {init_checkpoint_path}"
                    )
                model_state_obj = ckpt.get("model_state")
                if not isinstance(model_state_obj, dict):
                    raise ValueError(
                        f"Init checkpoint missing model_state: {init_checkpoint_path}"
                    )
                normalized_state = _normalize_checkpoint_state_dict(model_state_obj)
                model.load_state_dict(normalized_state)
                initialized_from_checkpoint = True
                print(f"[{task}] initialized from checkpoint: {init_checkpoint_path}")

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

                    running_loss = running_loss + loss.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )

                scheduler.step()
                train_loss = float(running_loss / max(1, len(train_loader)))

                val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    use_amp=use_amp_bool,
                    amp_dtype=amp_dtype_resolved,
                )
                pr_auc = val_metrics.get("pr_auc")
                roc_auc = val_metrics.get("roc_auc")
                acc_at_0_5 = val_metrics.get("acc@0.5")
                if pr_auc is not None:
                    best_pr_auc = (
                        pr_auc if best_pr_auc is None else max(best_pr_auc, pr_auc)
                    )
                if roc_auc is not None:
                    best_roc_auc = (
                        roc_auc if best_roc_auc is None else max(best_roc_auc, roc_auc)
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
                    epoch == 1 or epoch == epochs or epoch % log_every == 0 or improved
                )
                if should_log:
                    mark = "*" if improved else "-"
                    print(
                        f"[{task}] {mark} epoch {epoch}/{epochs} "
                        f"loss={train_loss:.4f} {score_name}={score:.4f} "
                        f"best={best_score:.4f} (ep {best_epoch})"
                    )

                if (
                    early_stop_patience > 0
                    and epochs_since_improvement >= early_stop_patience
                ):
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
                "pretrained_model_name": pretrained_model_name,
                "pretrained_revision": revision,
                "trust_remote_code": trust_remote_code_bool,
                "max_tokens": max_tokens_effective,
                "input_kmer": resolved_input_kmer,
                "dropout": dropout,
                "head_layer_norm": head_layer_norm_bool,
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

    model = _build_dnabert_model(
        pretrained_model_name=pretrained_model_name,
        pretrained_revision=pretrained_revision,
        trust_remote_code=trust_remote_code,
        dropout=dropout,
        head_layer_norm=head_layer_norm,
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
) -> np.ndarray:
    """Score input sequences with one trained task model."""
    if not sequences:
        return np.array([])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()
    all_probs: list[np.ndarray] = []
    total = len(sequences)
    log_every_batches = 200

    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        batch_sequences = sequences[start : start + batch_size]
        ids_tensor, mask_tensor = _tokenize_sequences(
            tokenizer=tokenizer,
            sequences=batch_sequences,
            max_tokens=max_tokens,
            input_kmer=input_kmer,
        )
        ids_tensor = ids_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        logits = model(input_ids=ids_tensor, attention_mask=mask_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
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

    return np.concatenate(all_probs)


def infer_site_scores(
    site_rows: List[Dict[str, object]],
    donor_model_path: str,
    acceptor_model_path: str,
    device: str = "auto",
    batch_size: int = 256,
) -> List[Dict[str, object]]:
    """Run donor/acceptor inference and return normalized site rows."""
    device = pick_device(device)

    donor_model, donor_config, donor_tokenizer = load_task_model(
        donor_model_path,
        device,
    )
    acceptor_model, acceptor_config, acceptor_tokenizer = load_task_model(
        acceptor_model_path,
        device,
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
        batch_size=batch_size,
        task_name="donor",
        input_kmer=donor_input_kmer,
    )
    acceptor_scores = score_sequences(
        model=acceptor_model,
        sequences=acceptor_seqs,
        tokenizer=acceptor_tokenizer,
        max_tokens=acceptor_max_tokens,
        device=device,
        batch_size=batch_size,
        task_name="acceptor",
        input_kmer=acceptor_input_kmer,
    )

    out_rows: List[Dict[str, object]] = []
    donor_idx = 0
    acceptor_idx = 0

    for row in site_rows:
        site_type = str(row["site_type"])
        if site_type == "donor":
            score = (
                float(donor_scores[donor_idx]) if donor_idx < len(donor_scores) else 0.0
            )
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
        choices=["both", "donor", "acceptor"],
        default="both",
        help=(
            "Training target. 'both' trains donor and acceptor. "
            "'donor'/'acceptor' train one task only (for tuning)."
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
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eta_min_ratio", type=float, default=0.01)
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
    parser.add_argument("--donor_weight_decay", type=float, default=None)
    parser.add_argument("--acceptor_weight_decay", type=float, default=None)
    parser.add_argument("--donor_eta_min_ratio", type=float, default=None)
    parser.add_argument("--acceptor_eta_min_ratio", type=float, default=None)
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
        help="Training loss type for donor/acceptor models.",
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


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train donor/acceptor DNABERT models with unified argument interface."""
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

    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=False,
    )
    donor_checkpoint_path = task_checkpoint_paths["donor"]
    acceptor_checkpoint_path = task_checkpoint_paths["acceptor"]
    train_target = resolve_train_target(model_args)

    resolved_epochs, epochs_auto = resolve_training_epoch_budget(
        epochs_arg=model_args.epochs,
        max_epochs=int(model_args.max_epochs),
    )
    early_stop_patience, early_stop_min_delta = resolve_early_stopping_params(
        patience_arg=model_args.early_stop_patience,
        min_delta_arg=model_args.early_stop_min_delta,
    )
    effective_early_stop_patience = early_stop_patience if epochs_auto else 0

    tasks_to_train = resolve_tasks_to_train(train_target)
    donor_init_checkpoint_path = str(
        getattr(common_args, "donor_init_checkpoint_path", "")
    ).strip()
    acceptor_init_checkpoint_path = str(
        getattr(common_args, "acceptor_init_checkpoint_path", "")
    ).strip()
    task_init_checkpoint_paths = {
        "donor": donor_init_checkpoint_path or None,
        "acceptor": acceptor_init_checkpoint_path or None,
    }
    task_window_len = {
        "donor": donor_window_len,
        "acceptor": acceptor_window_len,
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
            epochs=resolved_epochs,
            early_stop_patience=effective_early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            batch_size=resolved.batch_size,
            lr=resolved.lr,
            seed=common_args.seed,
            max_tokens=resolved.max_tokens,
            dropout=resolved.dropout,
            head_layer_norm=resolved.head_layer_norm,
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
            init_checkpoint_path=task_init_checkpoint_paths[task],
        )

    run_name_lr = model_args.lr
    run_name_batch_size = model_args.batch_size
    model_name = str(getattr(common_args, "model", "dnabert"))
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
        epochs=resolved_epochs,
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
        "model": model_name,
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
        "donor_init_checkpoint_path": donor_init_checkpoint_path,
        "acceptor_init_checkpoint_path": acceptor_init_checkpoint_path,
        "pretrained_model_name": model_args.pretrained_model_name,
        "pretrained_revision": model_args.pretrained_revision,
        "trust_remote_code": bool(model_args.trust_remote_code),
        "max_tokens": model_args.max_tokens,
        "dropout": model_args.dropout,
        "head_layer_norm": bool(model_args.head_layer_norm),
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
    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=True,
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
        batch_size=model_args.batch_size,
    )
