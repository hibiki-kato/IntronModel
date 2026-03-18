"""BiLSTM pair model for intron true/false classification.

This module provides two pair encoders over donor/acceptor sequences:
- ``separate``: independent donor/acceptor embedding+BiLSTM encoders.
- ``concat``: donor+acceptor concatenation (optional SEP token) into one encoder.

The unified public API matches ``run_model.py`` model contracts.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import product
import os
import random
import time
from typing import Callable, ContextManager, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from util.data_proc import (
    build_run_name,
    infer_default_train_paths,
    read_examples_pair_task_with_metadata,
    read_test_pair_rows,
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
    compile_model_with_fallback as _compile_model_with_fallback,
    fallback_average_precision as _fallback_average_precision,
    fallback_max_f1 as _fallback_max_f1,
    fallback_roc_auc as _fallback_roc_auc,
    is_compile_runtime_error as _is_compile_runtime_error,
    pick_device,
    record_compile_runtime_failure as _record_compile_runtime_failure,
    resolve_amp_dtype as _resolve_amp_dtype,
    resolve_compile_enabled as _resolve_compile_enabled,
    resolve_num_workers as _resolve_num_workers,
    seed_worker as _seed_worker,
    set_seed,
    sigmoid_np,
)
from util.sequence_transform import (
    SEQUENCE_TRANSFORM_CHOICES,
    PairSequenceRecord,
    apply_pair_sequence_transform,
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
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None

PAIR_ARCH_CHOICES: tuple[str, ...] = ("separate", "concat")
INPUT_MODE_CHOICES: tuple[str, ...] = ("dna", "kmer3", "bpe")
INPUT_MODE_ALIASES: dict[str, str] = {"onehot": "dna"}
INPUT_MODE_PARSE_CHOICES: tuple[str, ...] = (
    *INPUT_MODE_CHOICES,
    *tuple(INPUT_MODE_ALIASES.keys()),
)
_DNA_TO_TOKEN_ID: dict[str, int] = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "N": 5,
}
_PAD_TOKEN_ID: int = 0
_SEP_TOKEN_ID: int = 6
_VOCAB_SIZE: int = 7
BPE_DEFAULT_MODEL_NAME: str = "zhihan1996/DNABERT-2-117M"
_TOKENIZER_CACHE: dict[tuple[str, Optional[str], bool], object] = {}

PairBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass(frozen=True)
class PairTokenExample:
    """Tokenized donor/acceptor pair example."""

    donor_tokens: list[int]
    acceptor_tokens: list[int]
    label: int


@dataclass(frozen=True)
class PairTrainParams:
    """Resolved train-time hyperparameters for BiLSTM pair model."""

    batch_size: int
    lr: float
    loss_name: str
    input_mode: str
    bpe_pretrained_model_name: str
    bpe_pretrained_revision: Optional[str]
    bpe_trust_remote_code: bool
    pair_arch: str
    use_sep_token: bool
    embedding_dim: int
    hidden_size: int
    num_layers: int
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
    f1_lambda: float


@dataclass(frozen=True)
class InferRuntimeConfig:
    """Resolved inference runtime settings."""

    batch_size: int
    use_amp: bool
    amp_dtype: Optional[torch.dtype]


@dataclass(frozen=True)
class SequenceInputEncoder:
    """Encode DNA sequence into integer-token list for BiLSTM input."""

    mode: str
    kmer_vocab: Optional[Mapping[str, int]]
    bpe_tokenizer: Optional[object]

    @property
    def vocab_size(self) -> int:
        """Return embedding vocabulary size including PAD and SEP IDs."""
        if self.mode == "dna":
            return _VOCAB_SIZE
        if self.mode == "kmer3":
            assert self.kmer_vocab is not None
            return len(self.kmer_vocab) + 3
        assert self.mode == "bpe"
        if self.bpe_tokenizer is None:
            raise RuntimeError("BPE tokenizer is not initialized.")
        return int(getattr(self.bpe_tokenizer, "vocab_size")) + 2

    @property
    def pad_token_id(self) -> int:
        """Return reserved PAD token ID."""
        return _PAD_TOKEN_ID

    @property
    def sep_token_id(self) -> int:
        """Return reserved SEP token ID for concat architecture."""
        if self.mode == "dna":
            return _SEP_TOKEN_ID
        return self.vocab_size - 1

    def encode(self, sequence: str, *, window_len: Optional[int]) -> list[int]:
        """Encode one sequence according to configured input mode."""
        normalized = sequence.upper()
        if self.mode == "dna":
            return _encode_dna_sequence(normalized)

        if window_len is None or window_len <= 0:
            raise ValueError(
                "Positive --donor_len/--acceptor_len is required for token modes "
                "kmer3 and bpe."
            )

        if self.mode == "kmer3":
            assert self.kmer_vocab is not None
            clipped = normalized[:window_len]
            if len(clipped) < window_len:
                clipped = clipped + ("N" * (window_len - len(clipped)))
            token_count = max(1, window_len - 3 + 1)
            unknown_token_id = len(self.kmer_vocab) + 1
            tokens = [unknown_token_id] * token_count
            for index in range(token_count):
                token = clipped[index : index + 3]
                token_id = self.kmer_vocab.get(token)
                if token_id is not None:
                    tokens[index] = int(token_id) + 1
            return tokens

        assert self.mode == "bpe"
        if self.bpe_tokenizer is None:
            raise RuntimeError("BPE tokenizer is not initialized.")
        call_fn = getattr(self.bpe_tokenizer, "__call__", None)
        if not callable(call_fn):
            raise TypeError("Tokenizer object is not callable.")
        encoded_obj = call_fn(
            normalized,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=window_len,
            return_attention_mask=False,
        )
        if not isinstance(encoded_obj, Mapping):
            raise TypeError("Tokenizer output must be a mapping.")
        input_ids_obj = encoded_obj.get("input_ids")
        token_ids = np.asarray(input_ids_obj, dtype=np.int64).tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return [int(token_id) + 1 for token_id in token_ids]


def _normalize_input_mode(raw_mode: object, *, arg_name: str) -> str:
    """Normalize input mode and resolve backward-compatible aliases."""
    mode = str(raw_mode).strip().lower()
    if mode in INPUT_MODE_ALIASES:
        return INPUT_MODE_ALIASES[mode]
    if mode not in INPUT_MODE_CHOICES:
        choices_text = ", ".join(INPUT_MODE_CHOICES)
        raise ValueError(f"{arg_name} must be one of: {choices_text}.")
    return mode


def _build_kmer3_vocab() -> dict[str, int]:
    """Build fixed k=3 vocabulary over A/C/G/T."""
    return {
        "".join(chars): index
        for index, chars in enumerate(product(("A", "C", "G", "T"), repeat=3))
    }


def _load_bpe_tokenizer(
    *,
    pretrained_model_name: str,
    pretrained_revision: Optional[str],
    trust_remote_code: bool,
) -> object:
    """Load and cache BPE tokenizer from Hugging Face source."""
    if AutoTokenizer is None:
        raise RuntimeError(
            "transformers is required for --input_mode=bpe but is not installed."
        )
    cache_key = (pretrained_model_name, pretrained_revision, trust_remote_code)
    tokenizer = _TOKENIZER_CACHE.get(cache_key)
    if tokenizer is not None:
        return tokenizer
    tokenizer_kwargs: dict[str, object] = {
        "trust_remote_code": trust_remote_code,
    }
    if pretrained_revision is not None:
        tokenizer_kwargs["revision"] = pretrained_revision
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name,
        **tokenizer_kwargs,
    )
    _TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def _build_sequence_encoder(
    *,
    mode: str,
    bpe_pretrained_model_name: str,
    bpe_pretrained_revision: Optional[str],
    bpe_trust_remote_code: bool,
) -> SequenceInputEncoder:
    """Construct one sequence encoder from tokenizer-mode configuration."""
    normalized_mode = _normalize_input_mode(mode, arg_name="--input_mode")
    if normalized_mode == "dna":
        return SequenceInputEncoder(
            mode=normalized_mode,
            kmer_vocab=None,
            bpe_tokenizer=None,
        )
    if normalized_mode == "kmer3":
        return SequenceInputEncoder(
            mode=normalized_mode,
            kmer_vocab=_build_kmer3_vocab(),
            bpe_tokenizer=None,
        )
    tokenizer = _load_bpe_tokenizer(
        pretrained_model_name=bpe_pretrained_model_name,
        pretrained_revision=bpe_pretrained_revision,
        trust_remote_code=bpe_trust_remote_code,
    )
    return SequenceInputEncoder(
        mode=normalized_mode,
        kmer_vocab=None,
        bpe_tokenizer=tokenizer,
    )


def _normalize_pair_arch(raw_arch: object) -> str:
    """Normalize and validate pair architecture name."""
    arch = str(raw_arch).strip().lower()
    if arch not in PAIR_ARCH_CHOICES:
        choices_text = ", ".join(PAIR_ARCH_CHOICES)
        raise ValueError(f"--pair_arch must be one of: {choices_text}.")
    return arch


def _encode_dna_sequence(sequence: str) -> list[int]:
    """Encode one DNA sequence using A/C/G/T/N vocabulary.

    Unknown bases are mapped to ``N``.
    """
    tokens: list[int] = []
    for base in sequence.upper():
        tokens.append(_DNA_TO_TOKEN_ID.get(base, _DNA_TO_TOKEN_ID["N"]))
    return tokens


class PairTokenDataset(Dataset):
    """Pair token dataset for donor/acceptor sequence examples."""

    def __init__(self, examples: Sequence[PairTokenExample]) -> None:
        self.examples: list[PairTokenExample] = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PairTokenExample:
        return self.examples[index]


def _build_pair_collate(
    *,
    use_sep_token: bool,
    pad_token_id: int,
    sep_token_id: int,
) -> Callable[[Sequence[PairTokenExample]], PairBatch]:
    """Build one collate function for pair-token batches."""

    def _collate(batch: Sequence[PairTokenExample]) -> PairBatch:
        if not batch:
            raise ValueError("batch must not be empty.")

        donor_lengths = torch.tensor(
            [len(item.donor_tokens) for item in batch],
            dtype=torch.long,
        )
        acceptor_lengths = torch.tensor(
            [len(item.acceptor_tokens) for item in batch],
            dtype=torch.long,
        )

        donor_max_len = int(donor_lengths.max().item())
        acceptor_max_len = int(acceptor_lengths.max().item())
        donor_ids = torch.full(
            (len(batch), donor_max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        acceptor_ids = torch.full(
            (len(batch), acceptor_max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )

        concat_lengths_list: list[int] = []
        concat_tokens_list: list[list[int]] = []
        for row_index, item in enumerate(batch):
            donor_tokens = item.donor_tokens
            acceptor_tokens = item.acceptor_tokens
            donor_ids[row_index, : len(donor_tokens)] = torch.tensor(
                donor_tokens,
                dtype=torch.long,
            )
            acceptor_ids[row_index, : len(acceptor_tokens)] = torch.tensor(
                acceptor_tokens,
                dtype=torch.long,
            )
            if use_sep_token:
                concat_tokens = donor_tokens + [sep_token_id] + acceptor_tokens
            else:
                concat_tokens = donor_tokens + acceptor_tokens
            concat_tokens_list.append(concat_tokens)
            concat_lengths_list.append(len(concat_tokens))

        concat_lengths = torch.tensor(concat_lengths_list, dtype=torch.long)
        concat_max_len = int(concat_lengths.max().item())
        concat_ids = torch.full(
            (len(batch), concat_max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        for row_index, concat_tokens in enumerate(concat_tokens_list):
            concat_ids[row_index, : len(concat_tokens)] = torch.tensor(
                concat_tokens,
                dtype=torch.long,
            )

        labels = torch.tensor([item.label for item in batch], dtype=torch.float32)
        return (
            donor_ids,
            donor_lengths,
            acceptor_ids,
            acceptor_lengths,
            concat_ids,
            concat_lengths,
            labels,
        )

    return _collate


def _masked_mean_and_max_pool(
    sequence_outputs: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Apply masked mean/max pooling over padded LSTM outputs."""
    if sequence_outputs.ndim != 3:
        raise ValueError("sequence_outputs must have shape (batch, time, dim).")
    if lengths.ndim != 1:
        raise ValueError("lengths must have shape (batch,).")

    batch_size, max_len, _hidden_dim = sequence_outputs.shape
    mask = (
        torch.arange(max_len, device=sequence_outputs.device)
        .unsqueeze(0)
        .expand(batch_size, max_len)
    ) < lengths.unsqueeze(1)
    mask_3d = mask.unsqueeze(2)

    masked_sum = (sequence_outputs * mask_3d).sum(dim=1)
    denom = lengths.clamp_min(1).unsqueeze(1).to(sequence_outputs.dtype)
    mean_pool = masked_sum / denom

    masked_max_input = sequence_outputs.masked_fill(~mask_3d, float("-inf"))
    max_pool = masked_max_input.max(dim=1).values
    max_pool = torch.where(
        torch.isfinite(max_pool),
        max_pool,
        torch.zeros_like(max_pool),
    )
    return torch.cat([mean_pool, max_pool], dim=1)


class BiLSTMEncoder(nn.Module):
    """Embedding + BiLSTM encoder with masked mean/max pooling."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=_PAD_TOKEN_ID,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.output_dim: int = 4 * hidden_size

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode one padded token batch to one pooled feature vector."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, time).")
        if lengths.ndim != 1:
            raise ValueError("lengths must have shape (batch,).")

        embedded = self.embedding_dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths=lengths.to(device="cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=token_ids.shape[1],
        )
        return _masked_mean_and_max_pool(outputs, lengths)


class PairBiLSTMClassifier(nn.Module):
    """Pair BiLSTM classifier with separate or concatenated encoders."""

    def __init__(
        self,
        *,
        pair_arch: str,
        use_sep_token: bool,
        vocab_size: int = _VOCAB_SIZE,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        fc_hidden: int,
    ) -> None:
        super().__init__()
        normalized_arch = _normalize_pair_arch(pair_arch)
        if fc_hidden <= 0:
            raise ValueError("fc_hidden must be positive.")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        self.pair_arch = normalized_arch
        self.use_sep_token = bool(use_sep_token)
        if self.pair_arch == "separate":
            self.donor_encoder = BiLSTMEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.acceptor_encoder = BiLSTMEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            merged_dim = (
                self.donor_encoder.output_dim + self.acceptor_encoder.output_dim
            )
            self.concat_encoder = None
        else:
            self.concat_encoder = BiLSTMEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.donor_encoder = None
            self.acceptor_encoder = None
            merged_dim = self.concat_encoder.output_dim

        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(
        self,
        donor_ids: torch.Tensor,
        donor_lengths: torch.Tensor,
        acceptor_ids: torch.Tensor,
        acceptor_lengths: torch.Tensor,
        concat_ids: torch.Tensor,
        concat_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return one binary logit per donor/acceptor pair sample."""
        if self.pair_arch == "separate":
            if self.donor_encoder is None or self.acceptor_encoder is None:
                raise RuntimeError("separate encoders are not initialized.")
            donor_features = self.donor_encoder(donor_ids, donor_lengths)
            acceptor_features = self.acceptor_encoder(acceptor_ids, acceptor_lengths)
            features = torch.cat([donor_features, acceptor_features], dim=1)
        else:
            if self.concat_encoder is None:
                raise RuntimeError("concat encoder is not initialized.")
            features = self.concat_encoder(concat_ids, concat_lengths)
        return self.mlp(features).squeeze(-1)


def _resolve_pair_train_params(model_args: argparse.Namespace) -> PairTrainParams:
    """Resolve and validate pair BiLSTM train args."""
    hidden_size = int(model_args.hidden_size)
    if hidden_size <= 0:
        raise ValueError("--hidden_size must be positive.")

    input_mode = _normalize_input_mode(
        getattr(model_args, "input_mode", "dna"),
        arg_name="--input_mode",
    )
    bpe_pretrained_model_name = str(
        getattr(model_args, "bpe_pretrained_model_name", BPE_DEFAULT_MODEL_NAME)
    ).strip()
    if input_mode == "bpe" and bpe_pretrained_model_name == "":
        raise ValueError("--bpe_pretrained_model_name must not be empty.")
    bpe_pretrained_revision_raw = getattr(model_args, "bpe_pretrained_revision", None)
    bpe_pretrained_revision = (
        str(bpe_pretrained_revision_raw).strip()
        if bpe_pretrained_revision_raw is not None
        and str(bpe_pretrained_revision_raw).strip() != ""
        else None
    )

    pair_arch = _normalize_pair_arch(model_args.pair_arch)
    return PairTrainParams(
        batch_size=int(model_args.batch_size),
        lr=float(model_args.lr),
        loss_name=str(model_args.loss),
        input_mode=input_mode,
        bpe_pretrained_model_name=bpe_pretrained_model_name,
        bpe_pretrained_revision=bpe_pretrained_revision,
        bpe_trust_remote_code=bool(int(model_args.bpe_trust_remote_code)),
        pair_arch=pair_arch,
        use_sep_token=bool(int(model_args.use_sep_token)),
        embedding_dim=int(model_args.embedding_dim),
        hidden_size=hidden_size,
        num_layers=int(model_args.num_layers),
        dropout=float(model_args.dropout),
        fc_hidden=int(model_args.fc_hidden),
        weight_decay=float(model_args.weight_decay),
        eta_min_ratio=float(model_args.eta_min_ratio),
        val_frac=float(model_args.val_frac),
        grad_clip=float(model_args.grad_clip),
        pos_weight_cap=float(model_args.pos_weight_cap),
        focal_gamma=float(model_args.focal_gamma),
        focal_alpha_pos=model_args.focal_alpha_pos,
        asym_gamma_pos=float(model_args.asym_gamma_pos),
        asym_gamma_neg=float(model_args.asym_gamma_neg),
        asym_alpha_pos=model_args.asym_alpha_pos,
        f1_lambda=float(model_args.f1_lambda),
    )


def _resolve_infer_runtime_config(
    *,
    device: str,
    batch_size: int,
    infer_use_amp: int,
    infer_amp_dtype: str,
) -> InferRuntimeConfig:
    """Resolve inference runtime settings from user flags."""
    if batch_size <= 0:
        raise ValueError("inference batch_size must be positive.")
    use_amp = bool(infer_use_amp) and device == "cuda"
    amp_dtype = _resolve_amp_dtype(infer_amp_dtype, device)
    return InferRuntimeConfig(
        batch_size=batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )


def _stratified_split_pair(
    examples: Sequence[PairTokenExample],
    *,
    val_frac: float,
    seed: int,
) -> tuple[list[PairTokenExample], list[PairTokenExample]]:
    """Split pair examples into train/validation subsets with label stratify."""
    rng = random.Random(seed)
    pos = [item for item in examples if item.label == 1]
    neg = [item for item in examples if item.label == 0]
    if not pos or not neg:
        raise ValueError("Both positive and negative examples are required.")

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
def _evaluate_pair(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
) -> dict[str, float]:
    """Evaluate pair model and return common binary metrics."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    use_non_blocking = device == "cuda"

    for (
        donor_ids,
        donor_lengths,
        acceptor_ids,
        acceptor_lengths,
        concat_ids,
        concat_lengths,
        labels,
    ) in loader:
        donor_ids = donor_ids.to(device, non_blocking=use_non_blocking)
        donor_lengths = donor_lengths.to(device, non_blocking=use_non_blocking)
        acceptor_ids = acceptor_ids.to(device, non_blocking=use_non_blocking)
        acceptor_lengths = acceptor_lengths.to(device, non_blocking=use_non_blocking)
        concat_ids = concat_ids.to(device, non_blocking=use_non_blocking)
        concat_lengths = concat_lengths.to(device, non_blocking=use_non_blocking)
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
            logits = model(
                donor_ids,
                donor_lengths,
                acceptor_ids,
                acceptor_lengths,
                concat_ids,
                concat_lengths,
            )

        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.float().cpu().numpy())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels_np = np.concatenate(all_labels) if all_labels else np.array([])
    probs = sigmoid_np(logits) if logits.size else np.array([])
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    labels_int = labels_np.astype(np.int32)

    metrics: dict[str, float] = {}
    if labels_int.size:
        metrics["acc@0.5"] = float(np.mean((probs >= 0.5) == (labels_int >= 0.5)))
        try:
            metrics["max_f1"] = _fallback_max_f1(labels_int, probs)
        except ValueError:
            pass

        if len(np.unique(labels_int)) > 1:
            roc_auc_value: Optional[float] = None
            if roc_auc_score is not None:
                try:
                    roc_auc_value = float(roc_auc_score(labels_int, probs))
                except Exception:
                    roc_auc_value = None
            if roc_auc_value is None:
                try:
                    roc_auc_value = _fallback_roc_auc(labels_int, probs)
                except ValueError:
                    roc_auc_value = None
            if roc_auc_value is not None:
                metrics["roc_auc"] = roc_auc_value

            pr_auc_value: Optional[float] = None
            if average_precision_score is not None:
                try:
                    pr_auc_value = float(average_precision_score(labels_int, probs))
                except Exception:
                    pr_auc_value = None
            if pr_auc_value is None:
                try:
                    pr_auc_value = _fallback_average_precision(labels_int, probs)
                except ValueError:
                    pr_auc_value = None
            if pr_auc_value is not None:
                metrics["pr_auc"] = pr_auc_value

    return metrics


def train_pair_model(
    *,
    pos_path: str,
    neg_path: str,
    checkpoint_path: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    train_params: PairTrainParams,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    sequence_transform: str,
    seed: int,
    device: str,
    use_amp: int,
    amp_dtype: str,
    allow_tf32: int,
    cudnn_benchmark: int,
    deterministic: int,
    compile_model: bool,
    compile_mode: str,
    quick_phase: bool,
    num_workers: str | int,
    prefetch_factor: int,
    persistent_workers: int,
    pin_memory: int,
) -> dict[str, object]:
    """Train pair BiLSTM model and return training summary metrics."""
    if sequence_transform not in SEQUENCE_TRANSFORM_CHOICES:
        raise ValueError(
            "Unsupported --sequence_transform: "
            f"{sequence_transform}. Supported: {SEQUENCE_TRANSFORM_CHOICES}"
        )
    if train_params.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if train_params.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if train_params.dropout < 0.0 or train_params.dropout >= 1.0:
        raise ValueError("--dropout must satisfy 0 <= dropout < 1.")
    if train_params.fc_hidden <= 0:
        raise ValueError("--fc_hidden must be positive.")
    if train_params.weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative.")
    if train_params.val_frac <= 0.0 or train_params.val_frac >= 1.0:
        raise ValueError("--val_frac must satisfy 0 < val_frac < 1.")
    if train_params.grad_clip < 0.0:
        raise ValueError("--grad_clip must be non-negative.")
    if prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive.")

    device_name = pick_device(device)
    resolved_num_workers = _resolve_num_workers(num_workers, device=device_name)
    use_non_blocking = device_name == "cuda"
    use_pin_memory = bool(pin_memory) and device_name == "cuda"
    use_persistent_workers = bool(persistent_workers) and resolved_num_workers > 0
    use_amp_bool = bool(use_amp) and device_name == "cuda"
    amp_dtype_resolved = _resolve_amp_dtype(amp_dtype, device_name)
    compile_enabled = _resolve_compile_enabled(
        compile_mode=compile_mode,
        compile_flag=compile_model,
        quick_phase=quick_phase,
        device=device_name,
        epochs=epochs,
    )

    set_seed(
        seed=seed,
        deterministic=bool(deterministic),
        cudnn_benchmark=bool(cudnn_benchmark),
        allow_tf32=bool(allow_tf32),
    )

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    raw_examples = read_examples_pair_task_with_metadata(
        pos_path=pos_path,
        neg_path=neg_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        negative_pair_only=True,
    )
    sequence_encoder = _build_sequence_encoder(
        mode=train_params.input_mode,
        bpe_pretrained_model_name=train_params.bpe_pretrained_model_name,
        bpe_pretrained_revision=train_params.bpe_pretrained_revision,
        bpe_trust_remote_code=train_params.bpe_trust_remote_code,
    )
    token_examples: list[PairTokenExample] = []
    for item in raw_examples:
        transformed_pair = apply_pair_sequence_transform(
            PairSequenceRecord(
                donor_seq=item.donor_sequence,
                acceptor_seq=item.acceptor_sequence,
            ),
            transform_mode=sequence_transform,
            intron_half_length=item.intron_half_length,
        )
        token_examples.append(
            PairTokenExample(
                donor_tokens=sequence_encoder.encode(
                    transformed_pair.donor_seq,
                    window_len=donor_len,
                ),
                acceptor_tokens=sequence_encoder.encode(
                    transformed_pair.acceptor_seq,
                    window_len=acceptor_len,
                ),
                label=item.label,
            )
        )

    n_pos = sum(item.label for item in token_examples)
    n_neg = len(token_examples) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Insufficient training examples for pair: pos={n_pos}, neg={n_neg}."
        )

    train_examples, val_examples = _stratified_split_pair(
        token_examples,
        val_frac=train_params.val_frac,
        seed=seed,
    )
    print(
        f"[pair] device={device_name} total={len(token_examples)} "
        f"(pos={n_pos}, neg={n_neg}) "
        f"train={len(train_examples)} val={len(val_examples)}"
    )

    train_ds = PairTokenDataset(train_examples)
    val_ds = PairTokenDataset(val_examples)
    collate_fn = _build_pair_collate(
        use_sep_token=train_params.use_sep_token,
        pad_token_id=sequence_encoder.pad_token_id,
        sep_token_id=sequence_encoder.sep_token_id,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader_kwargs: dict[str, object] = {
        "dataset": train_ds,
        "batch_size": train_params.batch_size,
        "shuffle": True,
        "num_workers": resolved_num_workers,
        "pin_memory": use_pin_memory,
        "collate_fn": collate_fn,
        "worker_init_fn": _seed_worker if resolved_num_workers > 0 else None,
        "generator": loader_generator,
    }
    if resolved_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        train_loader_kwargs["persistent_workers"] = use_persistent_workers
    train_loader = DataLoader(**train_loader_kwargs)

    eval_loader_kwargs: dict[str, object] = {
        "batch_size": train_params.batch_size,
        "shuffle": False,
        "num_workers": resolved_num_workers,
        "pin_memory": use_pin_memory,
        "collate_fn": collate_fn,
    }
    if resolved_num_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = prefetch_factor
        eval_loader_kwargs["persistent_workers"] = use_persistent_workers
    val_loader = DataLoader(dataset=val_ds, **eval_loader_kwargs)
    train_eval_loader = DataLoader(dataset=train_ds, **eval_loader_kwargs)

    train_pos = sum(item.label for item in train_examples)
    train_neg = len(train_examples) - train_pos
    criterion, loss_meta = build_binary_classification_loss(
        loss_name=train_params.loss_name,
        train_pos=train_pos,
        train_neg=train_neg,
        device=device_name,
        pos_weight_cap=train_params.pos_weight_cap,
        focal_gamma=train_params.focal_gamma,
        focal_alpha_pos=train_params.focal_alpha_pos,
        asym_gamma_pos=train_params.asym_gamma_pos,
        asym_gamma_neg=train_params.asym_gamma_neg,
        asym_alpha_pos=train_params.asym_alpha_pos,
        f1_lambda=train_params.f1_lambda,
    )

    model = PairBiLSTMClassifier(
        pair_arch=train_params.pair_arch,
        use_sep_token=train_params.use_sep_token,
        vocab_size=sequence_encoder.vocab_size,
        embedding_dim=train_params.embedding_dim,
        hidden_size=train_params.hidden_size,
        num_layers=train_params.num_layers,
        dropout=train_params.dropout,
        fc_hidden=train_params.fc_hidden,
    ).to(device_name)

    compile_enabled_effective = False
    compile_selected_mode: Optional[str] = None
    compile_setup_error: Optional[Exception] = None
    if compile_enabled:
        print(f"[pair] torch.compile requested (mode={compile_mode}).")
        (
            model,
            compile_enabled_effective,
            compile_selected_mode,
            compile_setup_error,
        ) = _compile_model_with_fallback(model, compile_mode=compile_mode)
        if compile_setup_error is not None:
            print(
                "[pair] torch.compile setup failed; fallback to eager. "
                f"reason={compile_setup_error}"
            )
        elif compile_enabled_effective and compile_selected_mode is not None:
            print(f"[pair] torch.compile enabled (mode={compile_selected_mode}).")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_params.lr,
        weight_decay=train_params.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=train_params.lr * train_params.eta_min_ratio,
    )

    scaler_enabled = (
        use_amp_bool and device_name == "cuda" and amp_dtype_resolved == torch.float16
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
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
    epochs_since_improvement = 0
    stopped_early = False
    train_started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        model.train()
        running_loss = torch.zeros((), dtype=torch.float64)

        for (
            donor_ids,
            donor_lengths,
            acceptor_ids,
            acceptor_lengths,
            concat_ids,
            concat_lengths,
            labels,
        ) in train_loader:
            donor_ids = donor_ids.to(device_name, non_blocking=use_non_blocking)
            donor_lengths = donor_lengths.to(device_name, non_blocking=use_non_blocking)
            acceptor_ids = acceptor_ids.to(device_name, non_blocking=use_non_blocking)
            acceptor_lengths = acceptor_lengths.to(
                device_name,
                non_blocking=use_non_blocking,
            )
            concat_ids = concat_ids.to(device_name, non_blocking=use_non_blocking)
            concat_lengths = concat_lengths.to(
                device_name,
                non_blocking=use_non_blocking,
            )
            labels = labels.to(device_name, non_blocking=use_non_blocking)

            optimizer.zero_grad(set_to_none=True)
            if (
                use_amp_bool
                and device_name == "cuda"
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
                try:
                    logits = model(
                        donor_ids,
                        donor_lengths,
                        acceptor_ids,
                        acceptor_lengths,
                        concat_ids,
                        concat_lengths,
                    )
                except RuntimeError as exc:
                    if compile_enabled_effective and _is_compile_runtime_error(exc):
                        print(
                            "[pair] torch.compile runtime failed; fallback to "
                            f"eager. reason={exc}"
                        )
                        _record_compile_runtime_failure(compile_selected_mode)
                        original_model = getattr(model, "_orig_mod", None)
                        if isinstance(original_model, nn.Module):
                            model = original_model
                        compile_enabled_effective = False
                        compile_selected_mode = None
                        logits = model(
                            donor_ids,
                            donor_lengths,
                            acceptor_ids,
                            acceptor_lengths,
                            concat_ids,
                            concat_lengths,
                        )
                    else:
                        raise
                loss = criterion(logits, labels)

            if scaler_enabled:
                scaler.scale(loss).backward()
                if train_params.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_params.grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if train_params.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_params.grad_clip,
                    )
                optimizer.step()

            running_loss = running_loss + loss.detach().to(
                device="cpu",
                dtype=torch.float64,
            )

        scheduler.step()
        train_loss = float(running_loss / max(1, len(train_loader)))
        val_metrics = _evaluate_pair(
            model=model,
            loader=val_loader,
            device=device_name,
            use_amp=use_amp_bool,
            amp_dtype=amp_dtype_resolved,
        )
        train_metrics = _evaluate_pair(
            model=model,
            loader=train_eval_loader,
            device=device_name,
            use_amp=use_amp_bool,
            amp_dtype=amp_dtype_resolved,
        )

        pr_auc = val_metrics.get("pr_auc")
        roc_auc = val_metrics.get("roc_auc")
        max_f1 = val_metrics.get("max_f1")
        acc_at_0_5 = val_metrics.get("acc@0.5")
        train_pr_auc = train_metrics.get("pr_auc")

        if pr_auc is not None:
            best_pr_auc = pr_auc if best_pr_auc is None else max(best_pr_auc, pr_auc)
        if roc_auc is not None:
            best_roc_auc = (
                roc_auc if best_roc_auc is None else max(best_roc_auc, roc_auc)
            )
        if max_f1 is not None:
            best_max_f1 = max_f1 if best_max_f1 is None else max(best_max_f1, max_f1)
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
            best_score = score
            best_metric_name = score_name
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(
                {
                    "task": "pair",
                    "model_state": model.state_dict(),
                    "model_config": {
                        "input_mode": train_params.input_mode,
                        "vocab_size": sequence_encoder.vocab_size,
                        "bpe_pretrained_model_name": (
                            train_params.bpe_pretrained_model_name
                        ),
                        "bpe_pretrained_revision": train_params.bpe_pretrained_revision,
                        "bpe_trust_remote_code": train_params.bpe_trust_remote_code,
                        "pair_arch": train_params.pair_arch,
                        "use_sep_token": train_params.use_sep_token,
                        "embedding_dim": train_params.embedding_dim,
                        "hidden_size": train_params.hidden_size,
                        "num_layers": train_params.num_layers,
                        "dropout": train_params.dropout,
                        "fc_hidden": train_params.fc_hidden,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_since_improvement += 1

        epoch_elapsed_sec = time.perf_counter() - epoch_started_at
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
        train_pr_auc_text = "nan" if train_pr_auc is None else f"{train_pr_auc:.4f}"
        test_pr_auc_text = "nan" if pr_auc is None else f"{pr_auc:.4f}"
        print(
            f"[pair] {mark} epoch {epoch}/{epochs} "
            f"loss={train_loss:.4f} train_pr_auc={train_pr_auc_text} "
            f"test_pr_auc={test_pr_auc_text} best={best_score:.4f} "
            f"(ep {best_epoch})"
        )

        if early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience:
            stopped_early = True
            print(
                f"[pair] early stop at epoch {epoch} "
                f"(patience={early_stop_patience}, min_delta={early_stop_min_delta:g})"
            )
            break

    total_elapsed_sec = time.perf_counter() - train_started_at
    print(f"[pair] done best_{best_metric_name}={best_score:.4f} at epoch {best_epoch}")
    return {
        "task": "pair",
        "num_examples": len(token_examples),
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
        "stopped_early": stopped_early,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "checkpoint": checkpoint_path,
        "loss": train_params.loss_name,
        "input_mode": train_params.input_mode,
        "bpe_pretrained_model_name": train_params.bpe_pretrained_model_name,
        "bpe_pretrained_revision": train_params.bpe_pretrained_revision,
        "bpe_trust_remote_code": train_params.bpe_trust_remote_code,
        "pos_weight": loss_meta["pos_weight"],
        "focal_gamma": loss_meta["focal_gamma"],
        "focal_alpha_pos": loss_meta["focal_alpha_pos"],
        "asym_gamma_pos": loss_meta["asym_gamma_pos"],
        "asym_gamma_neg": loss_meta["asym_gamma_neg"],
        "asym_alpha_pos": loss_meta["asym_alpha_pos"],
        "f1_lambda": loss_meta["f1_lambda"],
        "pair_arch": train_params.pair_arch,
        "use_sep_token": train_params.use_sep_token,
        "embedding_dim": train_params.embedding_dim,
        "hidden_size": train_params.hidden_size,
        "num_layers": train_params.num_layers,
        "dropout": train_params.dropout,
        "fc_hidden": train_params.fc_hidden,
        "weight_decay": train_params.weight_decay,
        "eta_min_ratio": train_params.eta_min_ratio,
        "val_frac": train_params.val_frac,
        "grad_clip": train_params.grad_clip,
        "use_amp": use_amp_bool,
        "amp_dtype": (
            str(amp_dtype_resolved).replace("torch.", "")
            if amp_dtype_resolved is not None
            else None
        ),
        "num_workers": resolved_num_workers,
        "prefetch_factor": prefetch_factor if resolved_num_workers > 0 else None,
        "persistent_workers": use_persistent_workers,
        "pin_memory": use_pin_memory,
        "compile_requested": bool(compile_enabled),
        "compile_enabled": bool(compile_enabled_effective),
        "compile_mode": compile_mode,
        "compile_selected_mode": compile_selected_mode,
        "compile_setup_error": (
            str(compile_setup_error) if compile_setup_error is not None else None
        ),
        "effective_batch_size": train_params.batch_size,
        "optimizer_impl": "adamw",
        "sequence_transform": sequence_transform,
        "elapsed_sec": total_elapsed_sec,
    }


@torch.no_grad()
def infer_pair_site_scores(
    *,
    pair_rows: Sequence[dict[str, object]],
    pair_model_path: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    device: str,
    batch_size: int,
    sequence_transform: str,
    infer_use_amp: int,
    infer_amp_dtype: str,
) -> list[dict[str, object]]:
    """Run pair model inference and return site-score rows."""
    if not pair_rows:
        return []
    if sequence_transform not in SEQUENCE_TRANSFORM_CHOICES:
        raise ValueError(
            "Unsupported --sequence_transform: "
            f"{sequence_transform}. Supported: {SEQUENCE_TRANSFORM_CHOICES}"
        )

    device_name = pick_device(device)
    ckpt = torch.load(pair_model_path, map_location=device_name)
    model_config = ckpt.get("model_config", {})
    input_mode = _normalize_input_mode(
        model_config.get("input_mode", "dna"),
        arg_name="checkpoint input_mode",
    )
    bpe_pretrained_model_name = str(
        model_config.get("bpe_pretrained_model_name", BPE_DEFAULT_MODEL_NAME)
    )
    bpe_pretrained_revision_raw = model_config.get("bpe_pretrained_revision")
    bpe_pretrained_revision = (
        str(bpe_pretrained_revision_raw).strip()
        if bpe_pretrained_revision_raw is not None
        and str(bpe_pretrained_revision_raw).strip() != ""
        else None
    )
    bpe_trust_remote_code = bool(model_config.get("bpe_trust_remote_code", False))
    sequence_encoder = _build_sequence_encoder(
        mode=input_mode,
        bpe_pretrained_model_name=bpe_pretrained_model_name,
        bpe_pretrained_revision=bpe_pretrained_revision,
        bpe_trust_remote_code=bpe_trust_remote_code,
    )
    pair_arch = _normalize_pair_arch(model_config.get("pair_arch", "separate"))
    use_sep_token = bool(model_config.get("use_sep_token", True))
    vocab_size = int(model_config.get("vocab_size", sequence_encoder.vocab_size))
    model = PairBiLSTMClassifier(
        pair_arch=pair_arch,
        use_sep_token=use_sep_token,
        vocab_size=vocab_size,
        embedding_dim=int(model_config.get("embedding_dim", 16)),
        hidden_size=int(model_config.get("hidden_size", 64)),
        num_layers=int(model_config.get("num_layers", 1)),
        dropout=float(model_config.get("dropout", 0.3)),
        fc_hidden=int(model_config.get("fc_hidden", 128)),
    ).to(device_name)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    infer_runtime = _resolve_infer_runtime_config(
        device=device_name,
        batch_size=batch_size,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
    )

    token_examples: list[PairTokenExample] = []
    for row in pair_rows:
        transformed = apply_pair_sequence_transform(
            PairSequenceRecord(
                donor_seq=str(row["donor_seq"]),
                acceptor_seq=str(row["acceptor_seq"]),
            ),
            transform_mode=sequence_transform,
            intron_half_length=(
                int(row["intron_half_length"])
                if row.get("intron_half_length") is not None
                else None
            ),
        )
        token_examples.append(
            PairTokenExample(
                donor_tokens=sequence_encoder.encode(
                    transformed.donor_seq,
                    window_len=donor_len,
                ),
                acceptor_tokens=sequence_encoder.encode(
                    transformed.acceptor_seq,
                    window_len=acceptor_len,
                ),
                label=0,
            )
        )

    dataset = PairTokenDataset(token_examples)
    loader = DataLoader(
        dataset=dataset,
        batch_size=infer_runtime.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device_name == "cuda"),
        collate_fn=_build_pair_collate(
            use_sep_token=use_sep_token,
            pad_token_id=sequence_encoder.pad_token_id,
            sep_token_id=sequence_encoder.sep_token_id,
        ),
    )

    probs_list: list[np.ndarray] = []
    use_non_blocking = device_name == "cuda"
    for (
        donor_ids,
        donor_lengths,
        acceptor_ids,
        acceptor_lengths,
        concat_ids,
        concat_lengths,
        _labels,
    ) in loader:
        donor_ids = donor_ids.to(device_name, non_blocking=use_non_blocking)
        donor_lengths = donor_lengths.to(device_name, non_blocking=use_non_blocking)
        acceptor_ids = acceptor_ids.to(device_name, non_blocking=use_non_blocking)
        acceptor_lengths = acceptor_lengths.to(
            device_name,
            non_blocking=use_non_blocking,
        )
        concat_ids = concat_ids.to(device_name, non_blocking=use_non_blocking)
        concat_lengths = concat_lengths.to(device_name, non_blocking=use_non_blocking)

        if (
            infer_runtime.use_amp
            and device_name == "cuda"
            and infer_runtime.amp_dtype is not None
        ):
            amp_context: ContextManager[object] = torch.autocast(
                device_type="cuda",
                dtype=infer_runtime.amp_dtype,
                enabled=True,
            )
        else:
            amp_context = nullcontext()

        with amp_context:
            logits = model(
                donor_ids,
                donor_lengths,
                acceptor_ids,
                acceptor_lengths,
                concat_ids,
                concat_lengths,
            )
        probs_list.append(sigmoid_np(logits.float().cpu().numpy()))

    scores = np.concatenate(probs_list) if probs_list else np.array([])
    out_rows: list[dict[str, object]] = []
    for row, score in zip(pair_rows, scores):
        out_rows.append(
            {
                "transcript_id": str(row["transcript_id"]),
                "intron_index": int(row["intron_index"]),
                "site_type": "pair",
                "score": float(score),
            }
        )
    return out_rows


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register pair BiLSTM training arguments."""
    parser.add_argument("--epochs", type=str, default="20")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--train_target", choices=["pair"], default="pair")
    parser.add_argument(
        "--sequence_transform",
        choices=list(SEQUENCE_TRANSFORM_CHOICES),
        default="none",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--pair_arch",
        choices=list(PAIR_ARCH_CHOICES),
        default="separate",
        help="Pair encoder architecture: separate or concat.",
    )
    parser.add_argument(
        "--use_sep_token",
        type=int,
        choices=[0, 1],
        default=1,
        help="Insert SEP token between donor/acceptor in concat mode.",
    )
    parser.add_argument(
        "--input_mode",
        choices=list(INPUT_MODE_PARSE_CHOICES),
        default="dna",
        help="Input encoding mode: dna, kmer3, bpe, or onehot(alias for dna).",
    )
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument(
        "--bpe_pretrained_model_name",
        type=str,
        default=BPE_DEFAULT_MODEL_NAME,
        help="Pretrained tokenizer source for --input_mode=bpe.",
    )
    parser.add_argument(
        "--bpe_pretrained_revision",
        default=None,
        help="Optional tokenizer revision for --input_mode=bpe.",
    )
    parser.add_argument(
        "--bpe_trust_remote_code",
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to enable trust_remote_code for BPE tokenizer loading.",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_hidden", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eta_min_ratio", type=float, default=0.01)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument(
        "--loss",
        choices=list(LOSS_NAME_CHOICES),
        default="weighted_bce",
    )
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha_pos", type=float, default=None)
    parser.add_argument(
        "--f1_lambda",
        type=float,
        default=0.1,
        help="Mixing coefficient for --loss weighted_bce_f1 or focal_f1.",
    )
    parser.add_argument("--asym_gamma_pos", type=float, default=0.0)
    parser.add_argument("--asym_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asym_alpha_pos", type=float, default=None)
    parser.add_argument("--use_amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=["off", "on", "auto"],
        default="auto",
        help="Compilation mode for torch.compile.",
    )
    parser.add_argument("--allow_tf32", type=int, choices=[0, 1], default=1)
    parser.add_argument("--cudnn_benchmark", type=int, choices=[0, 1], default=1)
    parser.add_argument("--deterministic", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num_workers", default="auto")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", type=int, choices=[0, 1], default=1)
    parser.add_argument("--pin_memory", type=int, choices=[0, 1], default=1)
    parser.add_argument("--tag", default=None)


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register pair BiLSTM inference arguments."""
    parser.add_argument("--batch_size", type=int, default=512)
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
        "--sequence_transform",
        choices=list(SEQUENCE_TRANSFORM_CHOICES),
        default="none",
    )


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> dict[str, object]:
    """Train pair BiLSTM model with unified argument interface."""
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
    validate_window_args(donor_len=donor_len, acceptor_len=acceptor_len)

    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=False,
        tasks=("pair",),
    )
    pair_checkpoint_path = task_checkpoint_paths["pair"]

    train_target = resolve_train_target(model_args, allowed_targets=("pair",))
    tasks_to_train = resolve_tasks_to_train(train_target, both_tasks=("pair",))
    if tasks_to_train != ["pair"]:
        raise ValueError("bilstm_pair expects train_target=pair.")

    resolved_epochs, epochs_auto = resolve_training_epoch_budget(
        epochs_arg=model_args.epochs,
        max_epochs=int(model_args.max_epochs),
    )
    early_stop_patience, early_stop_min_delta = resolve_early_stopping_params(
        patience_arg=model_args.early_stop_patience,
        min_delta_arg=model_args.early_stop_min_delta,
    )
    effective_early_stop_patience = early_stop_patience if epochs_auto else 0

    train_params = _resolve_pair_train_params(model_args)
    pair_metrics = train_pair_model(
        pos_path=train_pos_path,
        neg_path=train_neg_path,
        checkpoint_path=pair_checkpoint_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        train_params=train_params,
        epochs=resolved_epochs,
        early_stop_patience=effective_early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        sequence_transform=model_args.sequence_transform,
        seed=common_args.seed,
        device=common_args.device,
        use_amp=model_args.use_amp,
        amp_dtype=model_args.amp_dtype,
        allow_tf32=model_args.allow_tf32,
        cudnn_benchmark=model_args.cudnn_benchmark,
        deterministic=model_args.deterministic,
        compile_model=model_args.compile,
        compile_mode=model_args.compile_mode,
        quick_phase=bool(getattr(common_args, "quick_phase", False)),
        num_workers=model_args.num_workers,
        prefetch_factor=model_args.prefetch_factor,
        persistent_workers=model_args.persistent_workers,
        pin_memory=model_args.pin_memory,
    )

    run_name = build_run_name(
        model_name="bilstm_pair",
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        lr=train_params.lr,
        batch_size=train_params.batch_size,
        epochs=resolved_epochs,
        tag=model_args.tag,
    )

    return {
        "model": "bilstm_pair",
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
        "sequence_transform": model_args.sequence_transform,
        "seed": common_args.seed,
        "device": common_args.device,
        "checkpoint_name": os.path.basename(pair_checkpoint_path),
        "pair_checkpoint_path": pair_checkpoint_path,
        "input_mode": train_params.input_mode,
        "bpe_pretrained_model_name": train_params.bpe_pretrained_model_name,
        "bpe_pretrained_revision": train_params.bpe_pretrained_revision,
        "bpe_trust_remote_code": train_params.bpe_trust_remote_code,
        "pair_arch": train_params.pair_arch,
        "use_sep_token": train_params.use_sep_token,
        "embedding_dim": train_params.embedding_dim,
        "hidden_size": train_params.hidden_size,
        "num_layers": train_params.num_layers,
        "dropout": train_params.dropout,
        "fc_hidden": train_params.fc_hidden,
        "weight_decay": train_params.weight_decay,
        "eta_min_ratio": train_params.eta_min_ratio,
        "val_frac": train_params.val_frac,
        "grad_clip": train_params.grad_clip,
        "use_amp": bool(model_args.use_amp),
        "amp_dtype": model_args.amp_dtype,
        "compile": bool(model_args.compile),
        "compile_mode": model_args.compile_mode,
        "allow_tf32": bool(model_args.allow_tf32),
        "cudnn_benchmark": bool(model_args.cudnn_benchmark),
        "deterministic": bool(model_args.deterministic),
        "num_workers": model_args.num_workers,
        "prefetch_factor": model_args.prefetch_factor,
        "persistent_workers": bool(model_args.persistent_workers),
        "pin_memory": bool(model_args.pin_memory),
        "loss": model_args.loss,
        "focal_gamma": model_args.focal_gamma,
        "focal_alpha_pos": model_args.focal_alpha_pos,
        "f1_lambda": model_args.f1_lambda,
        "asym_gamma_pos": model_args.asym_gamma_pos,
        "asym_gamma_neg": model_args.asym_gamma_neg,
        "asym_alpha_pos": model_args.asym_alpha_pos,
        "run_name": run_name,
        "inferred_train_len": inferred_train_len,
        "pair": pair_metrics,
        "task_hyperparameters": {
            "pair": {
                "batch_size": train_params.batch_size,
                "lr": train_params.lr,
                "loss": train_params.loss_name,
                "input_mode": train_params.input_mode,
                "pair_arch": train_params.pair_arch,
                "use_sep_token": train_params.use_sep_token,
                "embedding_dim": train_params.embedding_dim,
                "hidden_size": train_params.hidden_size,
                "num_layers": train_params.num_layers,
                "dropout": train_params.dropout,
                "fc_hidden": train_params.fc_hidden,
                "weight_decay": train_params.weight_decay,
                "eta_min_ratio": train_params.eta_min_ratio,
                "val_frac": train_params.val_frac,
                "grad_clip": train_params.grad_clip,
                "pos_weight_cap": train_params.pos_weight_cap,
                "focal_gamma": train_params.focal_gamma,
                "focal_alpha_pos": train_params.focal_alpha_pos,
                "f1_lambda": train_params.f1_lambda,
                "asym_gamma_pos": train_params.asym_gamma_pos,
                "asym_gamma_neg": train_params.asym_gamma_neg,
                "asym_alpha_pos": train_params.asym_alpha_pos,
                "bpe_pretrained_model_name": (
                    train_params.bpe_pretrained_model_name
                ),
                "bpe_pretrained_revision": train_params.bpe_pretrained_revision,
                "bpe_trust_remote_code": train_params.bpe_trust_remote_code,
            }
        },
    }


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Run pair-level site inference and return fixed-schema score rows."""
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
    validate_window_args(donor_len=donor_len, acceptor_len=acceptor_len)

    test_tsv = resolve_test_tsv(common_args.species, common_args.test_tsv)
    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=True,
        tasks=("pair",),
    )
    pair_model_path = task_checkpoint_paths["pair"]

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
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        device=common_args.device,
        batch_size=infer_batch_size,
        sequence_transform=model_args.sequence_transform,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
    )
