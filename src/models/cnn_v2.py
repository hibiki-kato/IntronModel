"""Pair CNN model implementation for intron-level splice scoring.

This module trains and infers one score per donor/acceptor pair. Donor and
acceptor branches run independent CNN encoders with configurable readout, then
branch features are mixed by one MLP head.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import product
import os
import random
import time
from typing import ContextManager, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.cnn_common import (
    CNN_HEAD_TYPE_CHOICES,
    CnnFeatureReadout,
    CnnGapEncoder,
    normalize_cnn_head_type,
    one_hot_encode_dna,
    parse_conv_channels,
    parse_kernel_sizes,
)
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
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

DEFAULT_MPS_MAX_BATCH_SIZE: int = 1024
INPUT_MODE_CHOICES: tuple[str, ...] = ("onehot", "kmer3", "bpe")
PAIR_MODE_CHOICES: tuple[str, ...] = ("pair", "independent")
PAIR_MODE_ALIASES: dict[str, str] = {
    "on": "pair",
    "off": "independent",
    "true": "pair",
    "false": "independent",
    "1": "pair",
    "0": "independent",
    "single": "independent",
    "site": "independent",
}
BPE_DEFAULT_MODEL_NAME: str = "zhihan1996/DNABERT-2-117M"
FUSION_MODE_CHOICES: tuple[str, ...] = ("late", "mid", "early")
FUSION_MODE_ALIASES: dict[str, str] = {"early_channel": "early"}
FUSION_MODE_PARSE_CHOICES: tuple[str, ...] = (
    *FUSION_MODE_CHOICES,
    *tuple(FUSION_MODE_ALIASES.keys()),
)
_TOKENIZER_CACHE: dict[tuple[str, Optional[str], bool], object] = {}


def _normalize_input_mode(raw_mode: object, *, arg_name: str) -> str:
    """Normalize input mode string."""
    mode = str(raw_mode).strip().lower()
    if mode not in INPUT_MODE_CHOICES:
        choices_text = ", ".join(INPUT_MODE_CHOICES)
        raise ValueError(f"{arg_name} must be one of: {choices_text}.")
    return mode


def _normalize_pair_mode(raw_mode: object, *, arg_name: str) -> str:
    """Normalize pair architecture mode string."""
    mode = str(raw_mode).strip().lower()
    if mode in PAIR_MODE_ALIASES:
        return PAIR_MODE_ALIASES[mode]
    if mode not in PAIR_MODE_CHOICES:
        choices_text = ", ".join(PAIR_MODE_CHOICES)
        raise ValueError(f"{arg_name} must be one of: {choices_text}.")
    return mode


def _build_kmer3_vocab() -> dict[str, int]:
    """Build fixed k=3 vocabulary over A/C/G/T."""
    return {
        "".join(chars): index
        for index, chars in enumerate(product(("A", "C", "G", "T"), repeat=3))
    }


def _load_bpe_tokenizer(
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


@dataclass(frozen=True)
class SequenceInputEncoder:
    """Encode DNA sequence into one-hot channels or token ids."""

    mode: str
    window_len: int
    kmer_vocab: Optional[Mapping[str, int]]
    bpe_tokenizer: Optional[object]

    @property
    def vocab_size(self) -> Optional[int]:
        """Return token vocabulary size for embedding-based modes."""
        if self.mode == "kmer3":
            assert self.kmer_vocab is not None
            return len(self.kmer_vocab) + 1
        if self.mode == "bpe":
            if self.bpe_tokenizer is None:
                return None
            return int(getattr(self.bpe_tokenizer, "vocab_size"))
        return None

    def encode(self, sequence: str) -> np.ndarray:
        """Encode one sequence according to configured mode."""
        normalized = sequence.upper()
        if self.mode == "onehot":
            return one_hot_encode_dna(normalized, self.window_len)
        if self.mode == "kmer3":
            assert self.kmer_vocab is not None
            clipped = normalized[: self.window_len]
            if len(clipped) < self.window_len:
                clipped = clipped + ("N" * (self.window_len - len(clipped)))
            token_count = max(1, self.window_len - 3 + 1)
            tokens = np.full(token_count, len(self.kmer_vocab), dtype=np.int64)
            for index in range(token_count):
                token = clipped[index : index + 3]
                if len(token) < 3:
                    break
                token_id = self.kmer_vocab.get(token)
                if token_id is not None:
                    tokens[index] = token_id
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
            padding="max_length",
            truncation=True,
            max_length=self.window_len,
            return_attention_mask=False,
        )
        if not isinstance(encoded_obj, Mapping):
            raise TypeError("Tokenizer output must be a mapping.")
        input_ids_obj = encoded_obj.get("input_ids")
        if isinstance(input_ids_obj, list):
            token_ids = np.asarray(input_ids_obj, dtype=np.int64)
        else:
            token_ids = np.asarray(input_ids_obj, dtype=np.int64)
        return token_ids


def _build_sequence_encoder(
    *,
    mode: str,
    window_len: int,
    bpe_pretrained_model_name: str,
    bpe_pretrained_revision: Optional[str],
    bpe_trust_remote_code: bool,
) -> SequenceInputEncoder:
    """Construct one sequence encoder with configured tokenization mode."""
    normalized_mode = _normalize_input_mode(mode, arg_name="input_mode")
    if normalized_mode == "onehot":
        return SequenceInputEncoder(
            mode=normalized_mode,
            window_len=window_len,
            kmer_vocab=None,
            bpe_tokenizer=None,
        )
    if normalized_mode == "kmer3":
        return SequenceInputEncoder(
            mode=normalized_mode,
            window_len=window_len,
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
        window_len=window_len,
        kmer_vocab=None,
        bpe_tokenizer=tokenizer,
    )


def _resolve_mps_max_batch_size() -> int:
    """Resolve MPS batch-size cap from environment with safe default."""
    return resolve_mps_max_batch_size(
        model_tag="cnn_pair",
        default_batch_size=DEFAULT_MPS_MAX_BATCH_SIZE,
    )


def _normalize_fusion_mode(raw_mode: object, *, arg_name: str) -> str:
    """Normalize fusion mode and resolve backward-compatible aliases."""
    mode = str(raw_mode).strip().lower()
    if mode in FUSION_MODE_ALIASES:
        return FUSION_MODE_ALIASES[mode]
    if mode in FUSION_MODE_CHOICES:
        return mode
    choices_text = ", ".join(FUSION_MODE_CHOICES)
    raise ValueError(f"{arg_name} must be one of: {choices_text}.")


def _extract_encoder_layout(encoder: CnnGapEncoder) -> tuple[list[int], list[int]]:
    """Extract per-layer channels and kernel sizes from one CNN encoder."""
    channels: list[int] = []
    kernels: list[int] = []
    for layer in encoder.conv_layers:
        if isinstance(layer, nn.Conv1d):
            channels.append(int(layer.out_channels))
            kernels.append(int(layer.kernel_size[0]))
    return channels, kernels


@dataclass(frozen=True)
class PairTrainParams:
    """Resolved pair-train hyperparameters."""

    batch_size: int
    lr: float
    loss_name: str
    input_mode: str
    pair_mode: str
    embedding_dim: int
    bpe_pretrained_model_name: str
    bpe_pretrained_revision: Optional[str]
    bpe_trust_remote_code: bool
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
    f1_lambda: float


@dataclass(frozen=True)
class InferRuntimeConfig:
    """Resolved runtime controls for pair-CNN inference."""

    batch_size: int
    use_amp: bool
    amp_dtype: Optional[torch.dtype]
    compile_enabled: bool


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
    ) = _compile_model_with_fallback(model)
    if (not compile_enabled_attempt) and compile_setup_error is not None:
        print(
            f"[{task_name}] infer torch.compile setup failed "
            f"({compile_setup_error.__class__.__name__}). Continue without compile."
        )
        return model
    return compiled_model


class PairDNADataset(Dataset):
    """Pair DNA dataset with optional pre-encoding cache."""

    def __init__(
        self,
        examples: Sequence[Tuple[str, str, int]],
        donor_encoder: SequenceInputEncoder,
        acceptor_encoder: SequenceInputEncoder,
        preencode: bool = False,
    ) -> None:
        self.examples: list[Tuple[str, str, int]] = list(examples)
        self.donor_encoder = donor_encoder
        self.acceptor_encoder = acceptor_encoder
        self.preencode = preencode
        self._cached_donor_x: Optional[torch.Tensor]
        self._cached_acceptor_x: Optional[torch.Tensor]
        self._cached_y: Optional[torch.Tensor]

        if preencode:
            donor_encoded = np.stack(
                [
                    self.donor_encoder.encode(donor_seq)
                    for donor_seq, _, _ in self.examples
                ]
            )
            acceptor_encoded = np.stack(
                [
                    self.acceptor_encoder.encode(acceptor_seq)
                    for _, acceptor_seq, _ in self.examples
                ]
            )
            labels = np.asarray(
                [label for _, _, label in self.examples],
                dtype=np.float32,
            )
            if self.donor_encoder.mode == "onehot":
                donor_encoded = donor_encoded.astype(np.float32, copy=False)
                acceptor_encoded = acceptor_encoded.astype(np.float32, copy=False)
                self._cached_donor_x = torch.from_numpy(donor_encoded)
                self._cached_acceptor_x = torch.from_numpy(acceptor_encoded)
            else:
                donor_encoded = donor_encoded.astype(np.int64, copy=False)
                acceptor_encoded = acceptor_encoded.astype(np.int64, copy=False)
                self._cached_donor_x = torch.from_numpy(donor_encoded)
                self._cached_acceptor_x = torch.from_numpy(acceptor_encoded)
            self._cached_y = torch.from_numpy(labels)
        else:
            self._cached_donor_x = None
            self._cached_acceptor_x = None
            self._cached_y = None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._cached_donor_x is not None
            and self._cached_acceptor_x is not None
            and self._cached_y is not None
        ):
            return (
                self._cached_donor_x[index],
                self._cached_acceptor_x[index],
                self._cached_y[index],
            )

        donor_seq, acceptor_seq, label = self.examples[index]
        donor_x = self.donor_encoder.encode(donor_seq)
        acceptor_x = self.acceptor_encoder.encode(acceptor_seq)
        if self.donor_encoder.mode == "onehot":
            donor_tensor = torch.from_numpy(donor_x)
            acceptor_tensor = torch.from_numpy(acceptor_x)
        else:
            donor_tensor = torch.from_numpy(donor_x.astype(np.int64, copy=False))
            acceptor_tensor = torch.from_numpy(acceptor_x.astype(np.int64, copy=False))
        return (
            donor_tensor,
            acceptor_tensor,
            torch.tensor(label, dtype=torch.float32),
        )


class MotifDilatedEncoder(nn.Module):
    """CNN encoder with multi-kernel motif, interaction, and dilated blocks."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.motif_layers = nn.ModuleList(
            [
                nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
                nn.Conv1d(in_channels, 64, kernel_size=11, padding=5),
                nn.Conv1d(in_channels, 64, kernel_size=17, padding=8),
            ]
        )
        self.motif_activation = nn.ReLU(inplace=True)
        self.interaction = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        dilations = (1, 2, 4, 8)
        self.dilated_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        256,
                        256,
                        kernel_size=3,
                        dilation=dilation,
                        padding=dilation,
                    ),
                    nn.ReLU(inplace=True),
                )
                for dilation in dilations
            ]
        )
        self.output_dim: int = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor of shape ``(batch, channels, length)``."""
        motif_features = [
            self.motif_activation(layer(x)) for layer in self.motif_layers
        ]
        x = torch.cat(motif_features, dim=1)
        x = self.interaction(x)
        for block in self.dilated_blocks:
            x = block(x)
        max_pool = torch.amax(x, dim=2)
        mean_pool = torch.mean(x, dim=2)
        return torch.cat([max_pool, mean_pool], dim=1)


class PairSpliceCNN(nn.Module):
    """Pair-scoring CNN with donor/acceptor branch encoders."""

    def __init__(
        self,
        input_mode: str,
        pair_mode: str,
        embedding_dim: int,
        vocab_size: Optional[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_mode = _normalize_input_mode(input_mode, arg_name="input_mode")
        self.pair_mode = _normalize_pair_mode(pair_mode, arg_name="pair_mode")
        if self.pair_mode != "pair":
            raise ValueError("PairSpliceCNN only supports pair_mode=pair.")
        if self.input_mode == "onehot":
            self.embedding = None
            in_channels = 4
        else:
            if embedding_dim <= 0:
                raise ValueError("embedding_dim must be positive.")
            if vocab_size is None or vocab_size <= 0:
                raise ValueError("vocab_size must be positive for token input modes.")
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            in_channels = embedding_dim

        self.donor_encoder = MotifDilatedEncoder(in_channels=in_channels)
        self.acceptor_encoder = MotifDilatedEncoder(in_channels=in_channels)
        classifier_input_dim = (
            self.donor_encoder.output_dim + self.acceptor_encoder.output_dim
        )

        self.fc = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Convert model input to channel-first float tensor."""
        if self.embedding is None:
            return x.float()
        return self.embedding(x.long()).transpose(1, 2).contiguous()

    def forward(self, donor_x: torch.Tensor, acceptor_x: torch.Tensor) -> torch.Tensor:
        """Return one pair logit per sample.

        Parameters
        ----------
        donor_x : torch.Tensor
            Donor tensor as one-hot or token ids.
        acceptor_x : torch.Tensor
            Acceptor tensor as one-hot or token ids.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch,)``.
        """
        donor_features = self._prepare_inputs(donor_x)
        acceptor_features = self._prepare_inputs(acceptor_x)
        mixed = torch.cat(
            [
                self.donor_encoder(donor_features),
                self.acceptor_encoder(acceptor_features),
            ],
            dim=1,
        )
        return self.fc(mixed).squeeze(-1)


def stratified_split_pair(
    examples: Sequence[Tuple[str, str, int]],
    *,
    val_frac: float,
    seed: int,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """Split pair examples into train/validation subsets preserving labels."""
    rng = random.Random(seed)
    pos = [example for example in examples if example[2] == 1]
    neg = [example for example in examples if example[2] == 0]

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
def evaluate_pair(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    """Evaluate pair model on validation data."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    use_non_blocking = device == "cuda"

    for donor_x, acceptor_x, y in loader:
        donor_x = donor_x.to(device, non_blocking=use_non_blocking)
        acceptor_x = acceptor_x.to(device, non_blocking=use_non_blocking)
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
            logits = model(donor_x, acceptor_x)
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


def _resolve_pair_train_params(model_args: argparse.Namespace) -> PairTrainParams:
    """Resolve pair train-time hyperparameters from CLI args."""
    input_mode = _normalize_input_mode(
        getattr(model_args, "input_mode", "onehot"),
        arg_name="--input_mode",
    )
    pair_mode = _normalize_pair_mode(
        getattr(model_args, "pair_mode", "pair"),
        arg_name="--pair_mode",
    )
    embedding_dim = int(getattr(model_args, "embedding_dim", 32))
    if embedding_dim <= 0:
        raise ValueError("--embedding_dim must be positive.")
    return PairTrainParams(
        batch_size=int(model_args.batch_size),
        lr=float(model_args.lr),
        loss_name=str(model_args.loss),
        input_mode=input_mode,
        pair_mode=pair_mode,
        embedding_dim=embedding_dim,
        bpe_pretrained_model_name=str(
            getattr(model_args, "bpe_pretrained_model_name", BPE_DEFAULT_MODEL_NAME)
        ),
        bpe_pretrained_revision=getattr(model_args, "bpe_pretrained_revision", None),
        bpe_trust_remote_code=_bool_from_flag(
            int(getattr(model_args, "bpe_trust_remote_code", 0))
        ),
        dropout=float(model_args.dropout),
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


def train_pair_model(
    *,
    pos_path: str,
    neg_path: str,
    checkpoint_path: str,
    donor_window_len: int,
    acceptor_window_len: int,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    train_params: PairTrainParams,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    sequence_transform: str,
    seed: int,
    lightweight: bool,
    compile_model: bool,
    compile_mode: str,
    device: str,
    use_amp: Union[bool, int],
    amp_dtype: str,
    allow_tf32: Union[bool, int],
    cudnn_benchmark: Union[bool, int],
    deterministic: Union[bool, int],
    num_workers: Union[str, int],
    prefetch_factor: int,
    persistent_workers: Union[bool, int],
    pin_memory: Union[bool, int],
    min_batch_size: int,
    max_oom_retries: int,
    quick_phase: bool,
    gpu_id: Optional[int],
) -> Dict[str, object]:
    """Train the pair CNN model."""
    if train_params.embedding_dim <= 0:
        raise ValueError("--embedding_dim must be positive.")
    if train_params.dropout < 0.0 or train_params.dropout >= 1.0:
        raise ValueError("--dropout must satisfy 0 <= dropout < 1.")
    if train_params.weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative.")
    if train_params.eta_min_ratio < 0.0:
        raise ValueError("--eta_min_ratio must be non-negative.")
    if train_params.val_frac <= 0.0 or train_params.val_frac >= 1.0:
        raise ValueError("--val_frac must satisfy 0 < val_frac < 1.")
    if train_params.grad_clip < 0.0:
        raise ValueError("--grad_clip must be non-negative.")
    if train_params.f1_lambda < 0.0:
        raise ValueError("--f1_lambda must be non-negative.")
    if prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive.")
    if min_batch_size <= 0:
        raise ValueError("--min_batch_size must be positive.")
    if max_oom_retries < 0:
        raise ValueError("--max_oom_retries must be >= 0.")
    if train_params.batch_size < min_batch_size:
        raise ValueError("--batch_size must be >= --min_batch_size.")
    if sequence_transform not in SEQUENCE_TRANSFORM_CHOICES:
        raise ValueError(
            "Unsupported --sequence_transform: "
            f"{sequence_transform}. Supported: {SEQUENCE_TRANSFORM_CHOICES}"
        )

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

    raw_examples = read_examples_pair_task_with_metadata(
        pos_path=pos_path,
        neg_path=neg_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        negative_pair_only=True,
    )
    resolved_sequence_transform = sequence_transform
    if sequence_transform != "none":
        missing_metadata_count = sum(
            item.intron_half_length is None for item in raw_examples
        )
        if missing_metadata_count > 0:
            print(
                "[cnn_v2] sequence_transform="
                f"{sequence_transform} requires intron_half_length metadata; "
                f"missing={missing_metadata_count}/{len(raw_examples)}. "
                "Falling back to sequence_transform=none."
            )
            resolved_sequence_transform = "none"
    examples: List[Tuple[str, str, int]] = []
    for item in raw_examples:
        transformed_pair = apply_pair_sequence_transform(
            PairSequenceRecord(
                donor_seq=item.donor_sequence,
                acceptor_seq=item.acceptor_sequence,
            ),
            transform_mode=resolved_sequence_transform,
            intron_half_length=item.intron_half_length,
        )
        examples.append(
            (
                transformed_pair.donor_seq,
                transformed_pair.acceptor_seq,
                item.label,
            )
        )

    n_pos = sum(label for _, _, label in examples)
    n_neg = len(examples) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Insufficient training examples for pair: pos={n_pos}, neg={n_neg}."
        )

    train_ex, val_ex = stratified_split_pair(
        examples,
        val_frac=train_params.val_frac,
        seed=seed,
    )
    print(
        f"[pair] device={device} total={len(examples)} "
        f"(pos={n_pos}, neg={n_neg}) train={len(train_ex)} val={len(val_ex)}"
    )
    preencode_dataset = device == "mps"
    if preencode_dataset:
        print("[pair] dataset pre-encoding enabled for mps.")

    donor_encoder = _build_sequence_encoder(
        mode=train_params.input_mode,
        window_len=donor_window_len,
        bpe_pretrained_model_name=train_params.bpe_pretrained_model_name,
        bpe_pretrained_revision=train_params.bpe_pretrained_revision,
        bpe_trust_remote_code=train_params.bpe_trust_remote_code,
    )
    acceptor_encoder = _build_sequence_encoder(
        mode=train_params.input_mode,
        window_len=acceptor_window_len,
        bpe_pretrained_model_name=train_params.bpe_pretrained_model_name,
        bpe_pretrained_revision=train_params.bpe_pretrained_revision,
        bpe_trust_remote_code=train_params.bpe_trust_remote_code,
    )

    train_ds = PairDNADataset(
        train_ex,
        donor_encoder=donor_encoder,
        acceptor_encoder=acceptor_encoder,
        preencode=preencode_dataset,
    )
    val_ds = PairDNADataset(
        val_ex,
        donor_encoder=donor_encoder,
        acceptor_encoder=acceptor_encoder,
        preencode=preencode_dataset,
    )

    train_pos = sum(label for _, _, label in train_ex)
    train_neg = len(train_ex) - train_pos
    criterion, loss_meta = build_binary_classification_loss(
        loss_name=train_params.loss_name,
        train_pos=train_pos,
        train_neg=train_neg,
        device=device,
        pos_weight_cap=train_params.pos_weight_cap,
        focal_gamma=train_params.focal_gamma,
        focal_alpha_pos=train_params.focal_alpha_pos,
        asym_gamma_pos=train_params.asym_gamma_pos,
        asym_gamma_neg=train_params.asym_gamma_neg,
        asym_alpha_pos=train_params.asym_alpha_pos,
        f1_lambda=train_params.f1_lambda,
    )

    effective_batch_size = train_params.batch_size
    if device == "mps":
        mps_max_batch_size = _resolve_mps_max_batch_size()
        if effective_batch_size > mps_max_batch_size:
            print(
                f"[pair] mps batch clamp: {effective_batch_size} -> "
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
        train_eval_loader_kwargs: dict[str, object] = {
            "dataset": train_ds,
            "batch_size": effective_batch_size,
            "shuffle": False,
            "num_workers": resolved_num_workers,
            "pin_memory": use_pin_memory,
        }
        if resolved_num_workers > 0:
            train_eval_loader_kwargs["prefetch_factor"] = prefetch_factor
            train_eval_loader_kwargs["persistent_workers"] = use_persistent_workers
        train_eval_loader = DataLoader(**train_eval_loader_kwargs)

        print(
            f"[pair] loader train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} batch_size={effective_batch_size} "
            f"workers={resolved_num_workers}"
        )

        try:
            model = PairSpliceCNN(
                input_mode=train_params.input_mode,
                pair_mode=train_params.pair_mode,
                embedding_dim=train_params.embedding_dim,
                vocab_size=donor_encoder.vocab_size,
                dropout=train_params.dropout,
            ).to(device)

            if compile_enabled_attempt:
                _configure_triton_tool_paths()
                _configure_torch_compile_runtime()
                ptxas_path = os.environ.get("TRITON_PTXAS_PATH")
                ptxas_blackwell_path = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH")
                print(
                    "[pair] torch.compile requested "
                    f"(ptxas={ptxas_path}, ptxas_blackwell={ptxas_blackwell_path})."
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
                        "[pair] torch.compile setup failed "
                        f"({compile_setup_error.__class__.__name__}). "
                        "Continue without compile."
                    )

            optimizer_impl = "adamw"
            adamw_kwargs: dict[str, object] = {
                "params": model.parameters(),
                "lr": train_params.lr,
                "weight_decay": train_params.weight_decay,
            }
            if device == "cuda":
                try:
                    optimizer = torch.optim.AdamW(**adamw_kwargs, fused=True)
                    optimizer_impl = "adamw_fused"
                except (TypeError, RuntimeError):
                    optimizer = torch.optim.AdamW(**adamw_kwargs)
            else:
                optimizer = torch.optim.AdamW(**adamw_kwargs)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=train_params.lr * train_params.eta_min_ratio,
            )
            scaler_enabled = (
                use_amp_bool
                and device == "cuda"
                and amp_dtype_resolved == torch.float16
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
            epochs_completed = 0
            epochs_since_improvement = 0
            stopped_early = False
            _ = apply_eta_process_title_placeholder()
            task_started_at = time.perf_counter()

            for epoch in range(1, epochs + 1):
                epochs_completed = epoch
                epoch_started_at = time.perf_counter()
                if device == "mps":
                    print(f"[pair] epoch {epoch}/{epochs} start")
                model.train()
                running_loss = torch.zeros((), dtype=torch.float64)

                for batch_index, (donor_x, acceptor_x, y) in enumerate(
                    train_loader,
                    start=1,
                ):
                    saw_training_batch = True
                    donor_x = donor_x.to(device, non_blocking=use_non_blocking)
                    acceptor_x = acceptor_x.to(device, non_blocking=use_non_blocking)
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
                        logits = model(donor_x, acceptor_x)
                        loss = criterion(logits, y)

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

                    if device == "mps" and batch_index == 1:
                        print(f"[pair] epoch {epoch}/{epochs} first batch done")

                    running_loss = running_loss + loss.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )

                scheduler.step()
                train_loss = float(running_loss / max(1, len(train_loader)))

                val_metrics = evaluate_pair(
                    model=model,
                    loader=val_loader,
                    device=device,
                    use_amp=use_amp_bool,
                    amp_dtype=amp_dtype_resolved,
                )
                train_metrics = evaluate_pair(
                    model=model,
                    loader=train_eval_loader,
                    device=device,
                    use_amp=use_amp_bool,
                    amp_dtype=amp_dtype_resolved,
                )
                pr_auc = val_metrics.get("pr_auc")
                roc_auc = val_metrics.get("roc_auc")
                max_f1 = val_metrics.get("max_f1")
                acc_at_0_5 = val_metrics.get("acc@0.5")
                train_pr_auc = train_metrics.get("pr_auc")
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
                            "task": "pair",
                            "donor_window_len": donor_window_len,
                            "acceptor_window_len": acceptor_window_len,
                            "model_config": {
                                "input_mode": train_params.input_mode,
                                "pair_mode": train_params.pair_mode,
                                "embedding_dim": train_params.embedding_dim,
                                "vocab_size": donor_encoder.vocab_size,
                                "bpe_pretrained_model_name": (
                                    train_params.bpe_pretrained_model_name
                                ),
                                "bpe_pretrained_revision": (
                                    train_params.bpe_pretrained_revision
                                ),
                                "bpe_trust_remote_code": (
                                    train_params.bpe_trust_remote_code
                                ),
                                "dropout": train_params.dropout,
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
                _ = apply_eta_process_title_from_epoch_progress(
                    task_started_at=task_started_at,
                    completed_epochs=epoch,
                    total_epochs=epochs,
                )

                mark = "*" if improved else "-"
                train_pr_auc_text = (
                    "nan" if train_pr_auc is None else f"{train_pr_auc:.4f}"
                )
                test_pr_auc_text = "nan" if pr_auc is None else f"{pr_auc:.4f}"
                objective_text = (
                    "" if score_name == "pr_auc" else f"{score_name}={score:.4f} "
                )
                print(
                    f"[pair] {mark} epoch {epoch}/{epochs} "
                    f"loss={train_loss:.4f} train_pr_auc={train_pr_auc_text} "
                    f"test_pr_auc={test_pr_auc_text} "
                    f"elapsed={epoch_elapsed_sec:.2f}s "
                    f"{objective_text}best={best_score:.4f} "
                    f"(ep {best_epoch})"
                )

                if (
                    early_stop_patience > 0
                    and epochs_since_improvement >= early_stop_patience
                ):
                    stopped_early = True
                    print(
                        f"[pair] early stop at epoch {epoch} "
                        f"(patience={early_stop_patience}, "
                        f"min_delta={early_stop_min_delta:g})"
                    )
                    break

            print(
                f"[pair] done best_{best_metric_name}={best_score:.4f} "
                f"at epoch {best_epoch}"
            )
            return {
                "task": "pair",
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
                "loss": train_params.loss_name,
                "pos_weight": loss_meta["pos_weight"],
                "focal_gamma": loss_meta["focal_gamma"],
                "focal_alpha_pos": loss_meta["focal_alpha_pos"],
                "asym_gamma_pos": loss_meta["asym_gamma_pos"],
                "asym_gamma_neg": loss_meta["asym_gamma_neg"],
                "asym_alpha_pos": loss_meta["asym_alpha_pos"],
                "f1_lambda": loss_meta["f1_lambda"],
                "input_mode": train_params.input_mode,
                "pair_mode": train_params.pair_mode,
                "embedding_dim": train_params.embedding_dim,
                "dropout": train_params.dropout,
                "weight_decay": train_params.weight_decay,
                "eta_min_ratio": train_params.eta_min_ratio,
                "val_frac": train_params.val_frac,
                "grad_clip": train_params.grad_clip,
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
                "sequence_transform": sequence_transform,
            }
        except RuntimeError as exc:
            is_compile_failure = compile_enabled_attempt and _is_compile_runtime_error(
                exc
            )
            if is_compile_failure:
                compile_enabled = False
                _record_compile_runtime_failure(compile_selected_mode)
                print(
                    "[pair] torch.compile runtime failed "
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
                f"[pair] {device.upper()} OOM detected. "
                "Retry with smaller batch size: "
                f"{effective_batch_size} -> {next_batch_size} "
                f"(retry {oom_retries}/{max_oom_retries})"
            )
            effective_batch_size = next_batch_size
            _empty_device_cache(device)


def load_pair_model(
    checkpoint_path: str,
    device: str,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Load pair model and checkpoint payload."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _normalize_checkpoint_state_dict(ckpt["model_state"])

    model_config = ckpt.get("model_config", {})
    input_mode = _normalize_input_mode(
        model_config.get("input_mode", "onehot"),
        arg_name="checkpoint input_mode",
    )
    pair_mode = _normalize_pair_mode(
        model_config.get("pair_mode", "pair"),
        arg_name="checkpoint pair_mode",
    )
    embedding_dim = int(model_config.get("embedding_dim", 32))
    vocab_size_raw = model_config.get("vocab_size")
    vocab_size = None if vocab_size_raw is None else int(vocab_size_raw)
    dropout = float(model_config.get("dropout", 0.3))
    if input_mode != "onehot" and vocab_size is None:
        raise ValueError("checkpoint is missing vocab_size for token input mode.")

    model = PairSpliceCNN(
        input_mode=input_mode,
        pair_mode=pair_mode,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_pair_sequences(
    model: nn.Module,
    pairs: Sequence[Tuple[str, str]],
    donor_window_len: int,
    acceptor_window_len: int,
    device: str,
    input_mode: str = "onehot",
    bpe_pretrained_model_name: str = BPE_DEFAULT_MODEL_NAME,
    bpe_pretrained_revision: Optional[str] = None,
    bpe_trust_remote_code: bool = False,
    batch_size: int = 512,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> np.ndarray:
    """Score donor/acceptor sequence pairs."""
    if not pairs:
        return np.array([])

    donor_encoder = _build_sequence_encoder(
        mode=input_mode,
        window_len=donor_window_len,
        bpe_pretrained_model_name=bpe_pretrained_model_name,
        bpe_pretrained_revision=bpe_pretrained_revision,
        bpe_trust_remote_code=bpe_trust_remote_code,
    )
    acceptor_encoder = _build_sequence_encoder(
        mode=input_mode,
        window_len=acceptor_window_len,
        bpe_pretrained_model_name=bpe_pretrained_model_name,
        bpe_pretrained_revision=bpe_pretrained_revision,
        bpe_trust_remote_code=bpe_trust_remote_code,
    )
    donor_encoded = np.stack(
        [donor_encoder.encode(donor_seq) for donor_seq, _ in pairs]
    )
    acceptor_encoded = np.stack(
        [acceptor_encoder.encode(acceptor_seq) for _, acceptor_seq in pairs]
    )

    if input_mode == "onehot":
        donor_x = torch.from_numpy(donor_encoded.astype(np.float32, copy=False)).to(
            device
        )
        acceptor_x = torch.from_numpy(
            acceptor_encoded.astype(np.float32, copy=False)
        ).to(device)
    else:
        donor_x = torch.from_numpy(donor_encoded.astype(np.int64, copy=False)).to(
            device
        )
        acceptor_x = torch.from_numpy(acceptor_encoded.astype(np.int64, copy=False)).to(
            device
        )

    outputs: list[np.ndarray] = []
    for index in range(0, len(donor_x), batch_size):
        batch_donor = donor_x[index : index + batch_size]
        batch_acceptor = acceptor_x[index : index + batch_size]
        if use_amp and device == "cuda" and amp_dtype is not None:
            amp_context: ContextManager[object] = torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            )
        else:
            amp_context = nullcontext()
        with amp_context:
            logits = model(batch_donor, batch_acceptor)
        probs = torch.sigmoid(logits).float().detach().cpu().numpy()
        outputs.append(probs)
    return np.concatenate(outputs)


def infer_pair_site_scores(
    *,
    pair_rows: List[Dict[str, object]],
    pair_model_path: str,
    device: str,
    batch_size: int,
    sequence_transform: str,
    infer_use_amp: Union[bool, int] = 0,
    infer_amp_dtype: str = "auto",
    infer_compile: Union[bool, int] = 0,
    infer_compile_mode: str = "off",
) -> List[Dict[str, object]]:
    """Infer site-level pair scores using one pair model."""
    if sequence_transform not in SEQUENCE_TRANSFORM_CHOICES:
        raise ValueError(
            "Unsupported --sequence_transform: "
            f"{sequence_transform}. Supported: {SEQUENCE_TRANSFORM_CHOICES}"
        )

    device_name = pick_device(device)
    infer_runtime = _resolve_infer_runtime_config(
        device=device_name,
        batch_size=batch_size,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
        infer_compile=infer_compile,
        infer_compile_mode=infer_compile_mode,
    )
    model, ckpt = load_pair_model(pair_model_path, device_name)
    model = _prepare_infer_model(
        model=model,
        task_name="pair",
        compile_enabled=infer_runtime.compile_enabled,
    )

    donor_window_len = int(ckpt.get("donor_window_len", 50))
    acceptor_window_len = int(ckpt.get("acceptor_window_len", 50))
    model_config = ckpt.get("model_config", {})
    input_mode = _normalize_input_mode(
        model_config.get("input_mode", "onehot"),
        arg_name="checkpoint input_mode",
    )
    bpe_pretrained_model_name = str(
        model_config.get("bpe_pretrained_model_name", BPE_DEFAULT_MODEL_NAME)
    )
    bpe_pretrained_revision = model_config.get("bpe_pretrained_revision")
    bpe_trust_remote_code = bool(model_config.get("bpe_trust_remote_code", False))

    transformed_pairs: list[Tuple[str, str]] = []
    resolved_sequence_transform = sequence_transform
    if sequence_transform != "none":
        missing_metadata_count = sum(
            row.get("intron_half_length") is None for row in pair_rows
        )
        if missing_metadata_count > 0:
            print(
                "[cnn_v2] infer sequence_transform="
                f"{sequence_transform} requires intron_half_length metadata; "
                f"missing={missing_metadata_count}/{len(pair_rows)}. "
                "Falling back to sequence_transform=none."
            )
            resolved_sequence_transform = "none"
    for row in pair_rows:
        transformed = apply_pair_sequence_transform(
            PairSequenceRecord(
                donor_seq=str(row["donor_seq"]),
                acceptor_seq=str(row["acceptor_seq"]),
            ),
            transform_mode=resolved_sequence_transform,
            intron_half_length=(
                int(row["intron_half_length"])
                if row.get("intron_half_length") is not None
                else None
            ),
        )
        transformed_pairs.append((transformed.donor_seq, transformed.acceptor_seq))

    scores = score_pair_sequences(
        model=model,
        pairs=transformed_pairs,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
        device=device_name,
        input_mode=input_mode,
        bpe_pretrained_model_name=bpe_pretrained_model_name,
        bpe_pretrained_revision=bpe_pretrained_revision,
        bpe_trust_remote_code=bpe_trust_remote_code,
        batch_size=infer_runtime.batch_size,
        use_amp=infer_runtime.use_amp,
        amp_dtype=infer_runtime.amp_dtype,
    )

    out_rows: List[Dict[str, object]] = []
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
    """Register pair CNN training arguments."""
    parser.add_argument("--epochs", type=str, default="20")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument(
        "--train_target",
        choices=["both", "donor", "acceptor", "pair"],
        default="pair",
    )
    parser.add_argument(
        "--sequence_transform",
        choices=list(SEQUENCE_TRANSFORM_CHOICES),
        default="none",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--input_mode",
        choices=list(INPUT_MODE_CHOICES),
        default="onehot",
        help="Input encoding mode: onehot, kmer3, or bpe.",
    )
    parser.add_argument(
        "--pair_mode",
        choices=list(PAIR_MODE_CHOICES),
        default="pair",
        help="Training mode: pair model or donor/acceptor independent model.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Embedding dimension for token modes kmer3/bpe.",
    )
    parser.add_argument(
        "--bpe_pretrained_model_name",
        type=str,
        default=BPE_DEFAULT_MODEL_NAME,
        help="Pretrained tokenizer source for --input_mode=bpe.",
    )
    parser.add_argument(
        "--bpe_pretrained_revision",
        type=str,
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
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--conv_channels", type=str, default=None)
    parser.add_argument(
        "--donor_conv_channels",
        type=str,
        default=None,
        help="Donor-branch override for --conv_channels.",
    )
    parser.add_argument(
        "--acceptor_conv_channels",
        type=str,
        default=None,
        help="Acceptor-branch override for --conv_channels.",
    )
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument(
        "--kernel_sizes",
        type=str,
        default=None,
        help="Shared per-layer kernel sizes, e.g. 11,7,5.",
    )
    parser.add_argument(
        "--donor_kernel_sizes",
        type=str,
        default=None,
        help="Donor-branch override for --kernel_sizes.",
    )
    parser.add_argument(
        "--acceptor_kernel_sizes",
        type=str,
        default=None,
        help="Acceptor-branch override for --kernel_sizes.",
    )
    parser.add_argument(
        "--max_pool_size",
        type=int,
        default=2,
        help="Max-pooling width after each conv block. Use 1 to disable pooling.",
    )
    parser.add_argument(
        "--conv_stride",
        type=int,
        default=1,
        help="Shared convolution stride applied to every conv block.",
    )
    parser.add_argument(
        "--head_type",
        choices=list(CNN_HEAD_TYPE_CHOICES),
        default="gap",
        help="CNN readout mode: gap or center.",
    )
    parser.add_argument(
        "--fusion_mode",
        choices=list(FUSION_MODE_PARSE_CHOICES),
        default="late",
        help=(
            "Pair feature fusion mode: late, mid, early. "
            "Backward-compatible alias early_channel is accepted."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_hidden", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eta_min_ratio", type=float, default=0.01)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--donor_batch_size", type=int, default=None)
    parser.add_argument("--acceptor_batch_size", type=int, default=None)
    parser.add_argument("--donor_lr", type=float, default=None)
    parser.add_argument("--acceptor_lr", type=float, default=None)
    parser.add_argument("--donor_kernel_size", type=int, default=None)
    parser.add_argument("--acceptor_kernel_size", type=int, default=None)
    parser.add_argument("--donor_dropout", type=float, default=None)
    parser.add_argument("--acceptor_dropout", type=float, default=None)
    parser.add_argument("--donor_max_pool_size", type=int, default=None)
    parser.add_argument("--acceptor_max_pool_size", type=int, default=None)
    parser.add_argument("--donor_conv_stride", type=int, default=None)
    parser.add_argument("--acceptor_conv_stride", type=int, default=None)
    parser.add_argument("--donor_head_type", type=str, default=None)
    parser.add_argument("--acceptor_head_type", type=str, default=None)
    parser.add_argument("--donor_fc_hidden", type=int, default=None)
    parser.add_argument("--acceptor_fc_hidden", type=int, default=None)
    parser.add_argument("--donor_weight_decay", type=float, default=None)
    parser.add_argument("--acceptor_weight_decay", type=float, default=None)
    parser.add_argument("--donor_eta_min_ratio", type=float, default=None)
    parser.add_argument("--acceptor_eta_min_ratio", type=float, default=None)
    parser.add_argument("--donor_val_frac", type=float, default=None)
    parser.add_argument("--acceptor_val_frac", type=float, default=None)
    parser.add_argument("--donor_grad_clip", type=float, default=None)
    parser.add_argument("--acceptor_grad_clip", type=float, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", choices=["off", "on", "auto"], default="auto")
    parser.add_argument("--use_amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--allow_tf32", type=int, choices=[0, 1], default=1)
    parser.add_argument("--cudnn_benchmark", type=int, choices=[0, 1], default=1)
    parser.add_argument("--deterministic", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num_workers", default="auto")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", type=int, choices=[0, 1], default=1)
    parser.add_argument("--pin_memory", type=int, choices=[0, 1], default=1)
    parser.add_argument("--min_batch_size", type=int, default=64)
    parser.add_argument("--max_oom_retries", type=int, default=8)
    parser.add_argument(
        "--loss",
        choices=list(LOSS_NAME_CHOICES),
        default="weighted_bce",
    )
    parser.add_argument("--donor_loss", type=str, default=None)
    parser.add_argument("--acceptor_loss", type=str, default=None)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--donor_pos_weight_cap", type=float, default=None)
    parser.add_argument("--acceptor_pos_weight_cap", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--donor_focal_gamma", type=float, default=None)
    parser.add_argument("--acceptor_focal_gamma", type=float, default=None)
    parser.add_argument("--focal_alpha_pos", type=float, default=None)
    parser.add_argument("--donor_focal_alpha_pos", type=float, default=None)
    parser.add_argument("--acceptor_focal_alpha_pos", type=float, default=None)
    parser.add_argument(
        "--f1_lambda",
        type=float,
        default=0.1,
        help=("Mixing coefficient for --loss weighted_bce_f1 or focal_f1."),
    )
    parser.add_argument("--donor_f1_lambda", type=float, default=None)
    parser.add_argument("--acceptor_f1_lambda", type=float, default=None)
    parser.add_argument("--asym_gamma_pos", type=float, default=0.0)
    parser.add_argument("--donor_asym_gamma_pos", type=float, default=None)
    parser.add_argument("--acceptor_asym_gamma_pos", type=float, default=None)
    parser.add_argument("--asym_gamma_neg", type=float, default=4.0)
    parser.add_argument("--donor_asym_gamma_neg", type=float, default=None)
    parser.add_argument("--acceptor_asym_gamma_neg", type=float, default=None)
    parser.add_argument("--asym_alpha_pos", type=float, default=None)
    parser.add_argument("--donor_asym_alpha_pos", type=float, default=None)
    parser.add_argument("--acceptor_asym_alpha_pos", type=float, default=None)
    parser.add_argument("--tag", default=None)


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register pair CNN inference arguments."""
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
    parser.add_argument(
        "--sequence_transform",
        choices=list(SEQUENCE_TRANSFORM_CHOICES),
        default="none",
    )


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train pair CNN model with unified argument interface."""
    reported_model_name = str(getattr(common_args, "model", "cnn_v2")).strip()
    if reported_model_name == "":
        reported_model_name = "cnn_v2"
    requested_pair_mode = _normalize_pair_mode(
        getattr(model_args, "pair_mode", "pair"),
        arg_name="--pair_mode",
    )
    requested_train_target = resolve_train_target(
        model_args,
        allowed_targets=("both", "donor", "acceptor", "pair"),
    )

    if requested_pair_mode == "independent":
        from models import cnn as cnn_site_module

        effective_train_target = requested_train_target
        if effective_train_target == "pair":
            print(
                "[cnn_v2] pair_mode=independent does not support "
                "--train_target=pair; using both."
            )
            effective_train_target = "both"

        site_model_args = argparse.Namespace(**vars(model_args))
        site_train_arg_parser = argparse.ArgumentParser(add_help=False)
        cnn_site_module.add_train_args(site_train_arg_parser)
        site_train_default_args = site_train_arg_parser.parse_args([])
        site_infer_arg_parser = argparse.ArgumentParser(add_help=False)
        cnn_site_module.add_infer_args(site_infer_arg_parser)
        site_infer_default_args = site_infer_arg_parser.parse_args([])
        site_default_values: dict[str, object] = vars(site_train_default_args)
        for key, value in vars(site_infer_default_args).items():
            site_default_values.setdefault(key, value)
        for key, value in site_default_values.items():
            if not hasattr(site_model_args, key):
                setattr(site_model_args, key, value)
        sequence_transform_value = str(
            getattr(site_model_args, "sequence_transform", "none")
        )
        if sequence_transform_value != "none":
            print(
                "[cnn_v2] pair_mode=independent uses site-level training; "
                f"overriding sequence_transform={sequence_transform_value} -> none."
            )
            site_model_args.sequence_transform = "none"
        site_model_args.train_target = effective_train_target
        summary = cnn_site_module.train(common_args, site_model_args)
        summary["model"] = reported_model_name
        summary["pair_mode"] = "independent"
        summary["train_target"] = effective_train_target
        summary["delegated_backend"] = "cnn"
        return summary

    requested_input_mode = _normalize_input_mode(
        getattr(model_args, "input_mode", "onehot"),
        arg_name="--input_mode",
    )
    if requested_input_mode == "onehot":
        from models import cnn_pair as legacy_pair_module

        summary = legacy_pair_module.train(common_args, model_args)
        summary["model"] = reported_model_name
        summary["pair_mode"] = "pair"
        summary["train_target"] = "pair"
        summary["input_mode"] = "onehot"
        summary["delegated_backend"] = "cnn_pair"
        return summary

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

    donor_window_len = donor_len if donor_len is not None else 50
    acceptor_window_len = acceptor_len if acceptor_len is not None else 50

    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=False,
        tasks=("pair",),
    )
    pair_checkpoint_path = task_checkpoint_paths["pair"]

    if requested_train_target != "pair":
        print("[cnn_v2] pair_mode=pair forces --train_target=pair.")
    train_target = "pair"

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
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        train_params=train_params,
        epochs=resolved_epochs,
        early_stop_patience=effective_early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        sequence_transform=model_args.sequence_transform,
        seed=common_args.seed,
        lightweight=model_args.lightweight,
        compile_model=model_args.compile,
        compile_mode=model_args.compile_mode,
        device=common_args.device,
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

    run_name = build_run_name(
        model_name=reported_model_name,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        lr=train_params.lr,
        batch_size=train_params.batch_size,
        epochs=resolved_epochs,
        tag=model_args.tag,
    )

    summary: Dict[str, object] = {
        "model": reported_model_name,
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
        "lightweight": model_args.lightweight,
        "input_mode": train_params.input_mode,
        "pair_mode": train_params.pair_mode,
        "embedding_dim": train_params.embedding_dim,
        "bpe_pretrained_model_name": train_params.bpe_pretrained_model_name,
        "bpe_pretrained_revision": train_params.bpe_pretrained_revision,
        "bpe_trust_remote_code": train_params.bpe_trust_remote_code,
        "dropout": train_params.dropout,
        "weight_decay": train_params.weight_decay,
        "eta_min_ratio": train_params.eta_min_ratio,
        "val_frac": train_params.val_frac,
        "grad_clip": train_params.grad_clip,
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
                "pair_mode": train_params.pair_mode,
                "embedding_dim": train_params.embedding_dim,
                "bpe_pretrained_model_name": (train_params.bpe_pretrained_model_name),
                "bpe_pretrained_revision": train_params.bpe_pretrained_revision,
                "bpe_trust_remote_code": train_params.bpe_trust_remote_code,
                "dropout": train_params.dropout,
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
            }
        },
    }
    return summary


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Run pair-level site inference and return rows with fixed schema."""
    requested_pair_mode = _normalize_pair_mode(
        getattr(model_args, "pair_mode", "pair"),
        arg_name="--pair_mode",
    )
    if requested_pair_mode == "independent":
        from models import cnn as cnn_site_module

        site_rows = cnn_site_module.infer_site(common_args, model_args)
        donor_scores: dict[tuple[str, int], float] = {}
        acceptor_scores: dict[tuple[str, int], float] = {}
        for row in site_rows:
            row_site_type = str(row.get("site_type", "")).strip().lower()
            key = (str(row["transcript_id"]), int(row["intron_index"]))
            score = float(row["score"])
            if row_site_type == "donor":
                donor_scores[key] = score
            elif row_site_type == "acceptor":
                acceptor_scores[key] = score

        pair_rows: list[dict[str, object]] = []
        for key in sorted(set(donor_scores) & set(acceptor_scores)):
            donor_score = donor_scores[key]
            acceptor_score = acceptor_scores[key]
            pair_rows.append(
                {
                    "transcript_id": key[0],
                    "intron_index": key[1],
                    "site_type": "pair",
                    "score": donor_score * acceptor_score,
                }
            )
        return pair_rows

    requested_input_mode = _normalize_input_mode(
        getattr(model_args, "input_mode", "onehot"),
        arg_name="--input_mode",
    )
    if requested_input_mode == "onehot":
        from models import cnn_pair as legacy_pair_module

        try:
            return legacy_pair_module.infer_site(common_args, model_args)
        except Exception as exc:
            print(
                "[cnn_v2] legacy cnn_pair inference backend failed "
                f"({exc.__class__.__name__}). Falling back to cnn_v2 backend."
            )

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
        sequence_transform=model_args.sequence_transform,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
        infer_compile=infer_compile,
        infer_compile_mode=infer_compile_mode,
    )
