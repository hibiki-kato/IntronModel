"""BERT model implementation for site-level splice scoring.

This module provides a unified model API compatible with ``run_model.py``:
- argument registration
- donor/acceptor training
- site-level inference

The architecture is a compact encoder-only Transformer over DNA k-mer tokens.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import product
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
    compile_model_with_fallback as _compile_model_with_fallback,
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
    record_compile_runtime_failure as _record_compile_runtime_failure,
    seed_worker as _seed_worker,
    set_seed,
    sigmoid_np,
)
from util.process_title import (
    apply_eta_process_title_from_epoch_progress,
    apply_eta_process_title_placeholder,
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

DEFAULT_MPS_MAX_BATCH_SIZE: int = 1024
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]")
DNA_BASES: tuple[str, ...] = ("A", "C", "G", "T")


def _resolve_mps_max_batch_size() -> int:
    """Resolve MPS batch-size cap from env with a safe default."""
    return resolve_mps_max_batch_size(
        model_tag="bert",
        default_batch_size=DEFAULT_MPS_MAX_BATCH_SIZE,
    )


def build_kmer_vocab(kmer_k: int) -> dict[str, int]:
    """Build deterministic k-mer vocabulary with special tokens.

    Parameters
    ----------
    kmer_k : int
        K-mer size.

    Returns
    -------
    dict[str, int]
        Mapping from token to integer id.

    Raises
    ------
    ValueError
        If ``kmer_k`` is not positive.
    """
    if kmer_k <= 0:
        raise ValueError("--kmer_k must be positive.")

    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    next_index = len(vocab)
    for token_parts in product(DNA_BASES, repeat=kmer_k):
        token = "".join(token_parts)
        vocab[token] = next_index
        next_index += 1
    return vocab


def kmerize(seq: str, kmer_k: int) -> list[str]:
    """Convert one DNA sequence into overlapping k-mer tokens."""
    if kmer_k <= 0:
        raise ValueError("kmer_k must be positive.")
    upper = seq.upper()
    if len(upper) < kmer_k:
        return []
    return [upper[i : i + kmer_k] for i in range(0, len(upper) - kmer_k + 1)]


def encode_kmers(kmers: Sequence[str], vocab: Mapping[str, int]) -> list[int]:
    """Map k-mer strings into token ids using ``[UNK]`` fallback."""
    unk_id = vocab["[UNK]"]
    return [vocab.get(token, unk_id) for token in kmers]


def _resolve_max_tokens(raw: Union[str, int], window_len: int, kmer_k: int) -> int:
    """Resolve max token length from ``auto`` or integer input.

    ``auto`` uses ``max(2, window_len - kmer_k + 3)`` corresponding to:
    ``[CLS] + kmers + [SEP]``.
    """
    if window_len <= 0:
        raise ValueError("window_len must be positive.")
    if kmer_k <= 0:
        raise ValueError("kmer_k must be positive.")

    auto_tokens = max(2, window_len - kmer_k + 3)
    if isinstance(raw, int):
        resolved = raw
    else:
        text = str(raw).strip().lower()
        if text == "auto":
            return auto_tokens
        try:
            resolved = int(text)
        except ValueError as exc:
            raise ValueError(
                "--max_tokens must be 'auto' or a positive integer."
            ) from exc
    if resolved < 2:
        raise ValueError("--max_tokens must be >= 2.")
    return resolved


def encode_sequence(
    seq: str,
    vocab: Mapping[str, int],
    kmer_k: int,
    max_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one DNA sequence into padded token ids and attention mask.

    Parameters
    ----------
    seq : str
        Input DNA sequence.
    vocab : Mapping[str, int]
        Token vocabulary.
    kmer_k : int
        K-mer size.
    max_tokens : int
        Final fixed token length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(token_ids, attention_mask)`` each shaped ``(max_tokens,)``.
    """
    if max_tokens < 2:
        raise ValueError("max_tokens must be >= 2.")

    pad_id = vocab["[PAD]"]
    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]

    kmer_tokens = kmerize(seq, kmer_k)
    token_ids: list[int] = [cls_id] + encode_kmers(kmer_tokens, vocab) + [sep_id]

    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
        token_ids[-1] = sep_id

    attention_mask: list[int] = [1] * len(token_ids)

    padding = max_tokens - len(token_ids)
    if padding > 0:
        token_ids.extend([pad_id] * padding)
        attention_mask.extend([0] * padding)

    return (
        np.asarray(token_ids, dtype=np.int64),
        np.asarray(attention_mask, dtype=np.int64),
    )


class SpliceTokenDataset(Dataset):
    """Dataset returning token ids, attention mask, and binary labels.

    Parameters
    ----------
    examples : Sequence[tuple[str, int]]
        Sequence/label pairs.
    vocab : Mapping[str, int]
        K-mer vocabulary.
    kmer_k : int
        K-mer size.
    max_tokens : int
        Padded token length.
    pretokenize : bool, default=False
        If ``True``, pre-encode all records at initialization.
    """

    def __init__(
        self,
        examples: Sequence[Tuple[str, int]],
        vocab: Mapping[str, int],
        kmer_k: int,
        max_tokens: int,
        pretokenize: bool = False,
    ) -> None:
        self.examples: list[Tuple[str, int]] = list(examples)
        self.vocab: Mapping[str, int] = vocab
        self.kmer_k: int = kmer_k
        self.max_tokens: int = max_tokens
        self.pretokenize: bool = pretokenize
        self._cached_ids: Optional[torch.Tensor]
        self._cached_masks: Optional[torch.Tensor]
        self._cached_labels: Optional[torch.Tensor]

        if pretokenize:
            all_ids: list[np.ndarray] = []
            all_masks: list[np.ndarray] = []
            labels: list[float] = []
            for seq, label in self.examples:
                token_ids, mask = encode_sequence(
                    seq=seq,
                    vocab=self.vocab,
                    kmer_k=self.kmer_k,
                    max_tokens=self.max_tokens,
                )
                all_ids.append(token_ids)
                all_masks.append(mask)
                labels.append(float(label))
            self._cached_ids = torch.from_numpy(np.stack(all_ids))
            self._cached_masks = torch.from_numpy(np.stack(all_masks))
            self._cached_labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
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

        seq, label = self.examples[idx]
        token_ids, mask = encode_sequence(
            seq=seq,
            vocab=self.vocab,
            kmer_k=self.kmer_k,
            max_tokens=self.max_tokens,
        )
        return (
            torch.from_numpy(token_ids),
            torch.from_numpy(mask),
            torch.tensor(float(label), dtype=torch.float32),
        )


class SmallBertEncoder(nn.Module):
    """Compact encoder-only Transformer for tokenized splice sequences."""

    def __init__(
        self,
        vocab_size: int,
        max_tokens: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if max_tokens < 2:
            raise ValueError("max_tokens must be >= 2.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if ff_mult <= 0:
            raise ValueError("ff_mult must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_tokens, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized sequences.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape ``(batch, tokens)`` with dtype ``torch.long``.
        attention_mask : torch.Tensor
            Shape ``(batch, tokens)`` where 1 marks real token and 0 pad.

        Returns
        -------
        torch.Tensor
            Hidden states with shape ``(batch, tokens, d_model)``.
        """
        batch_size, seq_len = input_ids.shape
        pos_ids = (
            torch.arange(
                seq_len,
                device=input_ids.device,
            )
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        hidden = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        hidden = self.dropout(hidden)

        key_padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return self.norm(hidden)


class SingleTaskSpliceBert(nn.Module):
    """Single-task binary classifier over encoder ``[CLS]`` representation."""

    def __init__(
        self,
        encoder: SmallBertEncoder,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = hidden[:, 0, :]
        logits = self.classifier(cls_hidden).squeeze(-1)
        return logits


@dataclass(frozen=True)
class TaskTrainParams:
    """Resolved train-time hyperparameters for one task."""

    batch_size: int
    lr: float
    loss_name: str
    kmer_k: int
    max_tokens: str
    d_model: int
    n_heads: int
    n_layers: int
    ff_mult: int
    dropout: float
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
        kmer_k=int(_override_or_default("kmer_k", model_args.kmer_k)),
        max_tokens=str(_override_or_default("max_tokens", model_args.max_tokens)),
        d_model=int(_override_or_default("d_model", model_args.d_model)),
        n_heads=int(_override_or_default("n_heads", model_args.n_heads)),
        n_layers=int(_override_or_default("n_layers", model_args.n_layers)),
        ff_mult=int(_override_or_default("ff_mult", model_args.ff_mult)),
        dropout=float(_override_or_default("dropout", model_args.dropout)),
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


def _build_single_task_model(
    *,
    vocab_size: int,
    max_tokens: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ff_mult: int,
    dropout: float,
) -> SingleTaskSpliceBert:
    """Build one BERT classifier model from explicit configuration."""
    encoder = SmallBertEncoder(
        vocab_size=vocab_size,
        max_tokens=max_tokens,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=ff_mult,
        dropout=dropout,
    )
    return SingleTaskSpliceBert(
        encoder=encoder,
        d_model=d_model,
        dropout=dropout,
    )


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
    batch_size: int = 256,
    lr: float = 2e-4,
    seed: int = 1337,
    kmer_k: int = 3,
    max_tokens: Union[str, int] = "auto",
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    ff_mult: int = 4,
    dropout: float = 0.1,
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
) -> Dict[str, object]:
    """Train one task model with GPU-aware runtime options.

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
    epochs : int, default=20
        Number of epochs.
    batch_size : int, default=256
        Initial batch size.
    lr : float, default=2e-4
        Learning rate.
    seed : int, default=1337
        Random seed.
    kmer_k : int, default=3
        K-mer tokenizer size.
    max_tokens : str | int, default="auto"
        Max token length (``auto`` or integer >= 2).
    d_model : int, default=128
        Transformer hidden dimension.
    n_heads : int, default=4
        Number of attention heads.
    n_layers : int, default=4
        Number of encoder layers.
    ff_mult : int, default=4
        Feed-forward multiplier.
    dropout : float, default=0.1
        Dropout rate.
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
        Minimum batch size when retrying after OOM.
    max_oom_retries : int, default=8
        Maximum OOM retries.
    quick_phase : bool, default=False
        Whether this run is a quick-phase trial.
    gpu_id : int | None, default=None
        Assigned GPU id for sweep logs.

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
    if window_len <= 0:
        raise ValueError("window_len must be positive.")
    if kmer_k <= 0:
        raise ValueError("--kmer_k must be positive.")
    if d_model <= 0:
        raise ValueError("--d_model must be positive.")
    if n_heads <= 0:
        raise ValueError("--n_heads must be positive.")
    if n_layers <= 0:
        raise ValueError("--n_layers must be positive.")
    if ff_mult <= 0:
        raise ValueError("--ff_mult must be positive.")
    if d_model % n_heads != 0:
        raise ValueError("--d_model must be divisible by --n_heads.")
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
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Insufficient training examples for {task}: pos={n_pos}, neg={n_neg}."
        )

    train_ex, val_ex = stratified_split(examples, val_frac=val_frac, seed=seed)
    print(
        f"[{task}] device={device} total={len(examples)} "
        f"(pos={n_pos}, neg={n_neg}) train={len(train_ex)} val={len(val_ex)}"
    )

    max_tokens_effective = _resolve_max_tokens(
        raw=max_tokens,
        window_len=window_len,
        kmer_k=kmer_k,
    )
    vocab = build_kmer_vocab(kmer_k=kmer_k)
    pretokenize_dataset = device == "mps"
    if pretokenize_dataset:
        print(f"[{task}] dataset pre-tokenization enabled for mps.")

    train_ds = SpliceTokenDataset(
        examples=train_ex,
        vocab=vocab,
        kmer_k=kmer_k,
        max_tokens=max_tokens_effective,
        pretokenize=pretokenize_dataset,
    )
    val_ds = SpliceTokenDataset(
        examples=val_ex,
        vocab=vocab,
        kmer_k=kmer_k,
        max_tokens=max_tokens_effective,
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
            val_loader_kwargs["persistent_workers"] = use_persistent_workers
        val_loader = DataLoader(**val_loader_kwargs)

        print(
            f"[{task}] loader train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} batch_size={effective_batch_size} "
            f"workers={resolved_num_workers}"
        )

        try:
            model = _build_single_task_model(
                vocab_size=len(vocab),
                max_tokens=max_tokens_effective,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                ff_mult=ff_mult,
                dropout=dropout,
            ).to(device)

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
            _ = apply_eta_process_title_placeholder()
            task_started_at = time.perf_counter()

            for epoch in range(1, epochs + 1):
                epochs_completed = epoch
                if device == "mps":
                    print(f"[{task}] epoch {epoch}/{epochs} start")

                model.train()
                running_loss = 0.0
                for batch_idx, (input_ids, attention_mask, labels) in enumerate(
                    train_loader,
                    start=1,
                ):
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

                    if device == "mps" and batch_idx == 1:
                        print(f"[{task}] epoch {epoch}/{epochs} first batch done")
                    running_loss += float(loss.detach().item())

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
                                "kmer_k": kmer_k,
                                "max_tokens": max_tokens_effective,
                                "d_model": d_model,
                                "n_heads": n_heads,
                                "n_layers": n_layers,
                                "ff_mult": ff_mult,
                                "dropout": dropout,
                            },
                            "vocab": dict(vocab),
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
                _ = apply_eta_process_title_from_epoch_progress(
                    task_started_at=task_started_at,
                    completed_epochs=epoch,
                    total_epochs=epochs,
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
                "kmer_k": kmer_k,
                "max_tokens": max_tokens_effective,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "ff_mult": ff_mult,
                "dropout": dropout,
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
            }
        except RuntimeError as exc:
            is_compile_failure = compile_enabled_attempt and _is_compile_runtime_error(
                exc
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
) -> Tuple[nn.Module, Dict[str, object], Dict[str, int]]:
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

    legacy_args_obj = ckpt.get("args")
    legacy_args: dict[str, object]
    if isinstance(legacy_args_obj, dict):
        legacy_args = dict(legacy_args_obj)
    else:
        legacy_args = {}

    if "kmer_k" not in model_config:
        if "k" in legacy_args:
            model_config["kmer_k"] = legacy_args["k"]
        else:
            model_config["kmer_k"] = 3
    if "max_tokens" not in model_config:
        if "max_len" in legacy_args:
            model_config["max_tokens"] = legacy_args["max_len"]
        else:
            model_config["max_tokens"] = 32
    if "d_model" not in model_config and "d_model" in legacy_args:
        model_config["d_model"] = legacy_args["d_model"]
    if "n_heads" not in model_config and "n_heads" in legacy_args:
        model_config["n_heads"] = legacy_args["n_heads"]
    if "n_layers" not in model_config and "n_layers" in legacy_args:
        model_config["n_layers"] = legacy_args["n_layers"]
    if "dropout" not in model_config and "dropout" in legacy_args:
        model_config["dropout"] = legacy_args["dropout"]

    kmer_k = _int_from_checkpoint(model_config, "kmer_k", 3)
    max_tokens = _int_from_checkpoint(model_config, "max_tokens", 32)
    d_model = _int_from_checkpoint(model_config, "d_model", 128)
    n_heads = _int_from_checkpoint(model_config, "n_heads", 4)
    n_layers = _int_from_checkpoint(model_config, "n_layers", 4)
    ff_mult = _int_from_checkpoint(model_config, "ff_mult", 4)
    dropout = _float_from_checkpoint(model_config, "dropout", 0.1)

    vocab_obj = ckpt.get("vocab")
    if not isinstance(vocab_obj, dict):
        raise RuntimeError("Checkpoint missing vocab. Re-train the model.")
    vocab: Dict[str, int] = {}
    for key, value in vocab_obj.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            vocab[key] = value
        elif isinstance(value, float) and value.is_integer():
            vocab[key] = int(value)
        elif isinstance(value, str):
            try:
                vocab[key] = int(value)
            except ValueError:
                continue
    if not vocab:
        raise RuntimeError("Checkpoint vocab is empty or invalid.")

    model = _build_single_task_model(
        vocab_size=len(vocab),
        max_tokens=max_tokens,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=ff_mult,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(normalized_state)
    model.eval()

    resolved_config: Dict[str, object] = {
        "kmer_k": kmer_k,
        "max_tokens": max_tokens,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "ff_mult": ff_mult,
        "dropout": dropout,
    }
    return model, resolved_config, vocab


@torch.no_grad()
def score_sequences(
    model: nn.Module,
    sequences: Sequence[str],
    vocab: Mapping[str, int],
    kmer_k: int,
    max_tokens: int,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Score input sequences with one trained task model."""
    if not sequences:
        return np.array([])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()
    all_probs: list[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        encoded_ids: list[np.ndarray] = []
        encoded_masks: list[np.ndarray] = []
        for seq in batch_sequences:
            token_ids, attention_mask = encode_sequence(
                seq=seq,
                vocab=vocab,
                kmer_k=kmer_k,
                max_tokens=max_tokens,
            )
            encoded_ids.append(token_ids)
            encoded_masks.append(attention_mask)

        ids_tensor = torch.from_numpy(np.stack(encoded_ids)).to(device)
        mask_tensor = torch.from_numpy(np.stack(encoded_masks)).to(device)

        logits = model(input_ids=ids_tensor, attention_mask=mask_tensor)
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
    """Run donor/acceptor inference and return normalized site rows."""
    device = pick_device(device)

    donor_model, donor_config, donor_vocab = load_task_model(donor_model_path, device)
    acceptor_model, acceptor_config, acceptor_vocab = load_task_model(
        acceptor_model_path,
        device,
    )

    donor_k = _int_from_checkpoint(donor_config, "kmer_k", 3)
    donor_max_tokens = _int_from_checkpoint(donor_config, "max_tokens", 32)
    acceptor_k = _int_from_checkpoint(acceptor_config, "kmer_k", 3)
    acceptor_max_tokens = _int_from_checkpoint(acceptor_config, "max_tokens", 32)

    donor_seqs = [str(row["seq"]) for row in site_rows if row["site_type"] == "donor"]
    acceptor_seqs = [
        str(row["seq"]) for row in site_rows if row["site_type"] == "acceptor"
    ]

    donor_scores = score_sequences(
        model=donor_model,
        sequences=donor_seqs,
        vocab=donor_vocab,
        kmer_k=donor_k,
        max_tokens=donor_max_tokens,
        device=device,
        batch_size=batch_size,
    )
    acceptor_scores = score_sequences(
        model=acceptor_model,
        sequences=acceptor_seqs,
        vocab=acceptor_vocab,
        kmer_k=acceptor_k,
        max_tokens=acceptor_max_tokens,
        device=device,
        batch_size=batch_size,
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
    """Register BERT-specific training arguments."""
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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--kmer_k",
        type=int,
        default=3,
        help="K-mer size for tokenizer.",
    )
    parser.add_argument(
        "--max_tokens",
        default="auto",
        help=(
            "Max token length after [CLS]/[SEP]. "
            "Use integer or auto (derived from window length)."
        ),
    )
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
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
    parser.add_argument("--donor_kmer_k", type=int, default=None)
    parser.add_argument("--acceptor_kmer_k", type=int, default=None)
    parser.add_argument("--donor_max_tokens", default=None)
    parser.add_argument("--acceptor_max_tokens", default=None)
    parser.add_argument("--donor_d_model", type=int, default=None)
    parser.add_argument("--acceptor_d_model", type=int, default=None)
    parser.add_argument("--donor_n_heads", type=int, default=None)
    parser.add_argument("--acceptor_n_heads", type=int, default=None)
    parser.add_argument("--donor_n_layers", type=int, default=None)
    parser.add_argument("--acceptor_n_layers", type=int, default=None)
    parser.add_argument("--donor_ff_mult", type=int, default=None)
    parser.add_argument("--acceptor_ff_mult", type=int, default=None)
    parser.add_argument("--donor_dropout", type=float, default=None)
    parser.add_argument("--acceptor_dropout", type=float, default=None)
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
    """Register BERT-specific inference arguments."""
    parser.add_argument("--batch_size", type=int, default=512)


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train donor/acceptor BERT models with unified argument interface."""
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
            epochs=resolved_epochs,
            early_stop_patience=effective_early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            batch_size=resolved.batch_size,
            lr=resolved.lr,
            seed=common_args.seed,
            kmer_k=resolved.kmer_k,
            max_tokens=resolved.max_tokens,
            d_model=resolved.d_model,
            n_heads=resolved.n_heads,
            n_layers=resolved.n_layers,
            ff_mult=resolved.ff_mult,
            dropout=resolved.dropout,
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
        model_name="bert",
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
            "kmer_k": params.kmer_k,
            "max_tokens": params.max_tokens,
            "d_model": params.d_model,
            "n_heads": params.n_heads,
            "n_layers": params.n_layers,
            "ff_mult": params.ff_mult,
            "dropout": params.dropout,
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
        "model": "bert",
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
        "kmer_k": model_args.kmer_k,
        "max_tokens": model_args.max_tokens,
        "d_model": model_args.d_model,
        "n_heads": model_args.n_heads,
        "n_layers": model_args.n_layers,
        "ff_mult": model_args.ff_mult,
        "dropout": model_args.dropout,
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
