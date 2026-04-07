"""Organic residual-dilated pair CNN v3 for intron-level splice scoring.

This module redefines ``cnn_v3`` as a pair-only CNN with donor and acceptor
branches built from residual dilated blocks. During tuning, the search layer
may materialize high-level mutation controls into explicit per-branch block
layouts before training starts.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import os
import time
from typing import ContextManager, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import cnn_v2
from models.cnn_common import (
    CNN_HEAD_TYPE_CHOICES,
    _pad_batch_to_fixed_size,
    _readout_sequence_features,
    normalize_cnn_head_type,
    parse_conv_channels,
)
from util.data_proc import (
    build_run_name,
    infer_default_train_paths,
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
    is_compile_runtime_error as _is_compile_runtime_error,
    is_cuda_oom_error as _is_cuda_oom_error,
    is_mps_oom_error as _is_mps_oom_error,
    log10_sigmoid_np,
    normalize_checkpoint_state_dict as _normalize_checkpoint_state_dict,
    pick_device,
    resolve_amp_dtype as _resolve_amp_dtype,
    resolve_compile_enabled as _resolve_compile_enabled,
    resolve_num_workers as _resolve_num_workers,
    record_compile_runtime_failure as _record_compile_runtime_failure,
    seed_worker as _seed_worker,
    set_seed,
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
    get_metric_value,
    resolve_early_stopping_params,
    resolve_training_epoch_budget,
    resolve_validation_metric,
    select_validation_score,
)
from util.transcript_eval import SCORE_SPACE_FIELD, SCORE_SPACE_LOG10

BPE_DEFAULT_MODEL_NAME: str = cnn_v2.BPE_DEFAULT_MODEL_NAME


def _coerce_positive_int(raw_value: object, *, arg_name: str) -> int:
    """Normalize one positive integer argument."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arg_name} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"{arg_name} must be positive.")
    return value


def _coerce_optional_int_list(
    raw_value: object,
    *,
    arg_name: str,
) -> Optional[list[int]]:
    """Normalize one optional integer-list argument."""
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        parsed = parse_conv_channels(raw_value, arg_name=arg_name)
        return None if parsed is None else list(parsed)
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        parsed: list[int] = []
        for item in raw_value:
            value = _coerce_positive_int(item, arg_name=arg_name)
            parsed.append(value)
        return parsed if parsed else None
    raise ValueError(f"{arg_name} must be a comma-separated string or a list.")


def _align_positive_int_list(
    raw_value: object,
    *,
    depth: int,
    arg_name: str,
    require_odd: bool = False,
    default_values: Optional[Sequence[int]] = None,
) -> list[int]:
    """Resolve one positive integer list to the requested depth."""
    if depth <= 0:
        raise ValueError("depth must be positive.")

    parsed = _coerce_optional_int_list(raw_value, arg_name=arg_name)
    if parsed is None:
        if default_values is None or not default_values:
            raise ValueError(f"{arg_name} requires at least one value.")
        parsed = [int(value) for value in default_values]

    if len(parsed) == 1:
        resolved = parsed * depth
    elif len(parsed) < depth:
        resolved = parsed + ([parsed[-1]] * (depth - len(parsed)))
    else:
        resolved = parsed[:depth]

    if require_odd and any(value % 2 == 0 for value in resolved):
        raise ValueError(f"{arg_name} values must be odd.")
    return resolved


def _default_dilation_schedule(depth: int) -> list[int]:
    """Return one default cyclic dilation schedule."""
    base_cycle = [1, 2, 4, 8]
    return [base_cycle[index % len(base_cycle)] for index in range(depth)]


def _default_residual_channels(channels: Sequence[int]) -> list[int]:
    """Return one bottleneck width per residual block."""
    resolved: list[int] = []
    for channel in channels:
        resolved.append(max(32, int(channel) // 2))
    return resolved


def _shared_if_equal(
    left: Sequence[int],
    right: Sequence[int],
) -> Optional[list[int]]:
    """Return one shared list when both branches match."""
    left_list = [int(value) for value in left]
    right_list = [int(value) for value in right]
    if left_list == right_list:
        return left_list
    return None


@dataclass(frozen=True)
class OrganicBranchLayout:
    """Per-branch residual-dilated block layout.

    Attributes
    ----------
    channels : list[int]
        Output channel count per residual block.
    kernel_sizes : list[int]
        Odd convolution kernel size per block.
    dilations : list[int]
        Dilation factor per block.
    residual_channels : list[int]
        Bottleneck channel count inside each residual block.
    """

    channels: list[int]
    kernel_sizes: list[int]
    dilations: list[int]
    residual_channels: list[int]


@dataclass(frozen=True)
class PairOrganicArchParams:
    """Resolved pair-model architecture parameters."""

    donor: OrganicBranchLayout
    acceptor: OrganicBranchLayout
    max_pool_size: int
    pool_every: int
    head_type: str
    fc_hidden: int


def _resolve_branch_layout(
    *,
    branch_name: str,
    model_args: argparse.Namespace,
    shared_channels: Optional[list[int]],
    shared_kernel_sizes: Optional[list[int]],
    shared_dilations: Optional[list[int]],
    shared_residual_channels: Optional[list[int]],
    lightweight: bool,
) -> OrganicBranchLayout:
    """Resolve one donor or acceptor branch layout."""
    branch_channels = _coerce_optional_int_list(
        getattr(model_args, f"{branch_name}_conv_channels", None),
        arg_name=f"--{branch_name}_conv_channels",
    )
    if branch_channels is None:
        branch_channels = shared_channels
    if branch_channels is None:
        branch_channels = [48, 96, 192] if lightweight else [64, 128, 256, 384]
    depth = len(branch_channels)

    branch_kernel_sizes = _coerce_optional_int_list(
        getattr(model_args, f"{branch_name}_kernel_sizes", None),
        arg_name=f"--{branch_name}_kernel_sizes",
    )
    if branch_kernel_sizes is None:
        branch_kernel_sizes = shared_kernel_sizes
    kernel_sizes = _align_positive_int_list(
        branch_kernel_sizes,
        depth=depth,
        arg_name=f"--{branch_name}_kernel_sizes",
        require_odd=True,
        default_values=[9, 7, 5, 5],
    )

    branch_dilations = _coerce_optional_int_list(
        getattr(model_args, f"{branch_name}_block_dilations", None),
        arg_name=f"--{branch_name}_block_dilations",
    )
    if branch_dilations is None:
        branch_dilations = shared_dilations
    dilations = _align_positive_int_list(
        branch_dilations,
        depth=depth,
        arg_name=f"--{branch_name}_block_dilations",
        default_values=_default_dilation_schedule(depth),
    )

    branch_residual_channels = _coerce_optional_int_list(
        getattr(model_args, f"{branch_name}_residual_channels", None),
        arg_name=f"--{branch_name}_residual_channels",
    )
    if branch_residual_channels is None:
        branch_residual_channels = shared_residual_channels
    residual_channels = _align_positive_int_list(
        branch_residual_channels,
        depth=depth,
        arg_name=f"--{branch_name}_residual_channels",
        default_values=_default_residual_channels(branch_channels),
    )

    return OrganicBranchLayout(
        channels=list(branch_channels),
        kernel_sizes=kernel_sizes,
        dilations=dilations,
        residual_channels=residual_channels,
    )


def _resolve_pair_arch_params(
    model_args: argparse.Namespace,
    *,
    lightweight: bool = False,
) -> PairOrganicArchParams:
    """Resolve pair-model architecture parameters from CLI args."""
    shared_channels = _coerce_optional_int_list(
        getattr(model_args, "conv_channels", None),
        arg_name="--conv_channels",
    )
    shared_kernel_sizes = _coerce_optional_int_list(
        getattr(model_args, "kernel_sizes", None),
        arg_name="--kernel_sizes",
    )
    shared_dilations = _coerce_optional_int_list(
        getattr(model_args, "block_dilations", None),
        arg_name="--block_dilations",
    )
    shared_residual_channels = _coerce_optional_int_list(
        getattr(model_args, "residual_channels", None),
        arg_name="--residual_channels",
    )

    max_pool_size = _coerce_positive_int(
        getattr(model_args, "max_pool_size", 2),
        arg_name="--max_pool_size",
    )
    pool_every = _coerce_positive_int(
        getattr(model_args, "pool_every", 2),
        arg_name="--pool_every",
    )
    head_type = normalize_cnn_head_type(
        getattr(model_args, "head_type", "gap"),
        arg_name="--head_type",
    )
    fc_hidden = _coerce_positive_int(
        getattr(model_args, "fc_hidden", 192),
        arg_name="--fc_hidden",
    )

    donor = _resolve_branch_layout(
        branch_name="donor",
        model_args=model_args,
        shared_channels=shared_channels,
        shared_kernel_sizes=shared_kernel_sizes,
        shared_dilations=shared_dilations,
        shared_residual_channels=shared_residual_channels,
        lightweight=lightweight,
    )
    acceptor = _resolve_branch_layout(
        branch_name="acceptor",
        model_args=model_args,
        shared_channels=shared_channels,
        shared_kernel_sizes=shared_kernel_sizes,
        shared_dilations=shared_dilations,
        shared_residual_channels=shared_residual_channels,
        lightweight=lightweight,
    )
    return PairOrganicArchParams(
        donor=donor,
        acceptor=acceptor,
        max_pool_size=max_pool_size,
        pool_every=pool_every,
        head_type=head_type,
        fc_hidden=fc_hidden,
    )


class ResidualDilatedBlock(nn.Module):
    """Residual 1D CNN block with dilated convolutions.

    Parameters
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    kernel_size : int
        Odd convolution kernel size.
    dilation : int
        Dilation factor for both convolutions.
    residual_channels : int
        Hidden bottleneck width inside the block.
    dropout : float
        Dropout probability applied between the two convolutions.
    max_pool_size : int
        Max-pooling width applied when ``apply_pool`` is ``True``.
    apply_pool : bool
        Whether to apply pooling after the residual merge.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        residual_channels: int,
        dropout: float,
        max_pool_size: int,
        apply_pool: bool,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd.")
        if dilation <= 0:
            raise ValueError("dilation must be positive.")
        if residual_channels <= 0:
            raise ValueError("residual_channels must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        if max_pool_size <= 0:
            raise ValueError("max_pool_size must be positive.")

        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            residual_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(residual_channels)
        self.conv2 = nn.Conv1d(
            residual_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.projection: nn.Module
        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool: Optional[nn.MaxPool1d]
        if apply_pool and max_pool_size > 1:
            self.pool = nn.MaxPool1d(max_pool_size)
        else:
            self.pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to one feature map.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Output tensor with shape ``(batch, out_channels, length')``.
        """
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out + residual)
        if self.pool is not None:
            out = self.pool(out)
        return out


class ResidualDilatedBranchEncoder(nn.Module):
    """Residual-dilated encoder for one donor or acceptor sequence."""

    def __init__(
        self,
        *,
        in_channels: int,
        layout: OrganicBranchLayout,
        max_pool_size: int,
        pool_every: int,
        head_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if pool_every <= 0:
            raise ValueError("pool_every must be positive.")
        if not layout.channels:
            raise ValueError("layout.channels must contain at least one block.")
        self.head_type = normalize_cnn_head_type(head_type, arg_name="head_type")
        self.blocks = nn.ModuleList()

        current_in_channels = in_channels
        for index, (
            channel,
            kernel_size,
            dilation,
            residual_channels,
        ) in enumerate(
            zip(
                layout.channels,
                layout.kernel_sizes,
                layout.dilations,
                layout.residual_channels,
                strict=True,
            )
        ):
            apply_pool = (index + 1) % pool_every == 0
            self.blocks.append(
                ResidualDilatedBlock(
                    in_channels=current_in_channels,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    residual_channels=residual_channels,
                    dropout=dropout,
                    max_pool_size=max_pool_size,
                    apply_pool=apply_pool,
                )
            )
            current_in_channels = channel
        self.output_dim = current_in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one feature map to a fixed-size feature vector."""
        for block in self.blocks:
            x = block(x)
        return _readout_sequence_features(x, self.head_type)


class PairOrganicResDilCNN(nn.Module):
    """Pair-scoring CNN with residual-dilated donor and acceptor encoders."""

    def __init__(
        self,
        *,
        input_mode: str,
        pair_mode: str,
        embedding_dim: int,
        vocab_size: Optional[int],
        arch_params: PairOrganicArchParams,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_mode = cnn_v2._normalize_input_mode(
            input_mode,
            arg_name="input_mode",
        )
        self.pair_mode = cnn_v2._normalize_pair_mode(
            pair_mode,
            arg_name="pair_mode",
        )
        if self.pair_mode != "pair":
            raise ValueError("cnn_pair_v3 only supports pair_mode=pair.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        if self.input_mode == "onehot":
            self.embedding = None
            branch_in_channels = 4
        else:
            if embedding_dim <= 0:
                raise ValueError("embedding_dim must be positive.")
            if vocab_size is None or vocab_size <= 0:
                raise ValueError("vocab_size must be positive for token input.")
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            branch_in_channels = embedding_dim

        self.donor_encoder = ResidualDilatedBranchEncoder(
            in_channels=branch_in_channels,
            layout=arch_params.donor,
            max_pool_size=arch_params.max_pool_size,
            pool_every=arch_params.pool_every,
            head_type=arch_params.head_type,
            dropout=dropout,
        )
        self.acceptor_encoder = ResidualDilatedBranchEncoder(
            in_channels=branch_in_channels,
            layout=arch_params.acceptor,
            max_pool_size=arch_params.max_pool_size,
            pool_every=arch_params.pool_every,
            head_type=arch_params.head_type,
            dropout=dropout,
        )
        classifier_input_dim = (
            self.donor_encoder.output_dim + self.acceptor_encoder.output_dim
        )
        self.fc = nn.Sequential(
            nn.Linear(classifier_input_dim, arch_params.fc_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(arch_params.fc_hidden, 1),
        )

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Convert model input to one channel-first float tensor."""
        if self.embedding is None:
            if x.ndim != 3:
                raise ValueError(
                    "One-hot inputs must have shape (batch, channels, length)."
                )
            return x.float()
        if x.ndim != 2:
            raise ValueError("Token inputs must have shape (batch, length).")
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
        donor_features = self.donor_encoder(self._prepare_inputs(donor_x))
        acceptor_features = self.acceptor_encoder(self._prepare_inputs(acceptor_x))
        logits = self.fc(torch.cat([donor_features, acceptor_features], dim=1))
        return logits[:, 0]


def train_pair_model(
    *,
    pos_path: str,
    neg_path: str,
    checkpoint_path: str,
    donor_window_len: int,
    acceptor_window_len: int,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    model_args: argparse.Namespace,
    train_params: cnn_v2.PairTrainParams,
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
    validation_metric: str = "pr_auc",
    report_train_metrics: Union[bool, int] = 1,
    gpu_id: Optional[int] = None,
) -> Dict[str, object]:
    """Train the residual-dilated pair CNN."""
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

    arch_params = _resolve_pair_arch_params(
        model_args,
        lightweight=lightweight,
    )
    resolved_validation_metric = resolve_validation_metric(validation_metric)

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

    raw_examples = cnn_v2.read_examples_pair_task_with_metadata(
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
                "[cnn_v3] sequence_transform="
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

    train_ex, val_ex = cnn_v2.stratified_split_pair(
        examples,
        val_frac=train_params.val_frac,
        seed=seed,
    )
    print(
        f"[cnn_v3] device={device} total={len(examples)} "
        f"(pos={n_pos}, neg={n_neg}) train={len(train_ex)} val={len(val_ex)}"
    )
    preencode_dataset = device == "mps"
    if preencode_dataset:
        print("[cnn_v3] dataset pre-encoding enabled for mps.")

    donor_encoder = cnn_v2._build_sequence_encoder(
        mode=train_params.input_mode,
        window_len=donor_window_len,
        bpe_pretrained_model_name=train_params.bpe_pretrained_model_name,
        bpe_pretrained_revision=train_params.bpe_pretrained_revision,
        bpe_trust_remote_code=train_params.bpe_trust_remote_code,
    )
    acceptor_encoder = cnn_v2._build_sequence_encoder(
        mode=train_params.input_mode,
        window_len=acceptor_window_len,
        bpe_pretrained_model_name=train_params.bpe_pretrained_model_name,
        bpe_pretrained_revision=train_params.bpe_pretrained_revision,
        bpe_trust_remote_code=train_params.bpe_trust_remote_code,
    )
    train_ds = cnn_v2.PairDNADataset(
        train_ex,
        donor_encoder=donor_encoder,
        acceptor_encoder=acceptor_encoder,
        preencode=preencode_dataset,
    )
    val_ds = cnn_v2.PairDNADataset(
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
        mps_max_batch_size = cnn_v2._resolve_mps_max_batch_size()
        if effective_batch_size > mps_max_batch_size:
            print(
                f"[cnn_v3] mps batch clamp: {effective_batch_size} -> "
                f"{mps_max_batch_size} "
                "(set INTRONMODEL_MPS_MAX_BATCH_SIZE to change)."
            )
            effective_batch_size = mps_max_batch_size

    oom_retries = 0
    use_non_blocking = device == "cuda"
    report_train_metrics_bool = _bool_from_flag(report_train_metrics)
    while True:
        saw_training_batch = False
        compile_enabled_attempt = compile_enabled and hasattr(torch, "compile")
        compile_selected_mode: str | None = None
        fixed_shape_loader = compile_enabled

        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
        train_loader_batch_size, train_loader_drop_last = (
            cnn_v2._resolve_loader_batch_size_and_drop_last(
                requested_batch_size=effective_batch_size,
                dataset_size=len(train_ds),
                fixed_shape=fixed_shape_loader,
            )
        )
        train_loader_kwargs: dict[str, object] = {
            "dataset": train_ds,
            "batch_size": train_loader_batch_size,
            "shuffle": True,
            "num_workers": resolved_num_workers,
            "pin_memory": use_pin_memory,
            "worker_init_fn": _seed_worker if resolved_num_workers > 0 else None,
            "generator": loader_generator,
            "drop_last": train_loader_drop_last,
        }
        if resolved_num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = prefetch_factor
            train_loader_kwargs["persistent_workers"] = use_persistent_workers
        train_loader = DataLoader(**train_loader_kwargs)

        val_loader_batch_size, val_loader_drop_last = (
            cnn_v2._resolve_loader_batch_size_and_drop_last(
                requested_batch_size=effective_batch_size,
                dataset_size=len(val_ds),
                fixed_shape=fixed_shape_loader,
            )
        )
        val_loader_kwargs: dict[str, object] = {
            "dataset": val_ds,
            "batch_size": val_loader_batch_size,
            "shuffle": False,
            "num_workers": resolved_num_workers,
            "pin_memory": use_pin_memory,
            "drop_last": val_loader_drop_last,
        }
        if resolved_num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = prefetch_factor
            val_loader_kwargs["persistent_workers"] = use_persistent_workers
        val_loader = DataLoader(**val_loader_kwargs)

        train_eval_loader: Optional[DataLoader] = None
        if report_train_metrics_bool:
            train_eval_loader_batch_size, train_eval_loader_drop_last = (
                cnn_v2._resolve_loader_batch_size_and_drop_last(
                    requested_batch_size=effective_batch_size,
                    dataset_size=len(train_ds),
                    fixed_shape=fixed_shape_loader,
                )
            )
            train_eval_loader_kwargs: dict[str, object] = {
                "dataset": train_ds,
                "batch_size": train_eval_loader_batch_size,
                "shuffle": False,
                "num_workers": resolved_num_workers,
                "pin_memory": use_pin_memory,
                "drop_last": train_eval_loader_drop_last,
            }
            if resolved_num_workers > 0:
                train_eval_loader_kwargs["prefetch_factor"] = prefetch_factor
                train_eval_loader_kwargs["persistent_workers"] = use_persistent_workers
            train_eval_loader = DataLoader(**train_eval_loader_kwargs)

        print(
            f"[cnn_v3] loader train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} batch_size={effective_batch_size} "
            f"workers={resolved_num_workers} "
            f"train_eval={'on' if report_train_metrics_bool else 'off'} "
            f"fixed_shape={'on' if fixed_shape_loader else 'off'}"
        )

        try:
            model = PairOrganicResDilCNN(
                input_mode=train_params.input_mode,
                pair_mode=train_params.pair_mode,
                embedding_dim=train_params.embedding_dim,
                vocab_size=donor_encoder.vocab_size,
                arch_params=arch_params,
                dropout=train_params.dropout,
            ).to(device)

            if compile_enabled_attempt:
                _configure_triton_tool_paths()
                _configure_torch_compile_runtime()
                ptxas_path = os.environ.get("TRITON_PTXAS_PATH")
                ptxas_blackwell_path = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH")
                print(
                    "[cnn_v3] torch.compile requested "
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
                        "[cnn_v3] torch.compile setup failed "
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
                model.train()
                running_loss = torch.zeros((), dtype=torch.float64)

                for donor_x, acceptor_x, y in train_loader:
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

                    running_loss = running_loss + loss.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )

                scheduler.step()
                train_loss = float(running_loss / max(1, len(train_loader)))
                val_metrics = cnn_v2.evaluate_pair(
                    model=model,
                    loader=val_loader,
                    device=device,
                    use_amp=use_amp_bool,
                    amp_dtype=amp_dtype_resolved,
                )
                pr_auc = val_metrics.get("pr_auc")
                roc_auc = val_metrics.get("roc_auc")
                max_f1 = val_metrics.get("max_f1")
                acc_at_0_5 = val_metrics.get("acc@0.5")
                train_metrics: dict[str, float] = {}
                train_pr_auc: Optional[float] = None
                if report_train_metrics_bool and train_eval_loader is not None:
                    train_metrics = cnn_v2.evaluate_pair(
                        model=model,
                        loader=train_eval_loader,
                        device=device,
                        use_amp=use_amp_bool,
                        amp_dtype=amp_dtype_resolved,
                    )
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

                score, score_name = select_validation_score(
                    metrics=val_metrics,
                    validation_metric=resolved_validation_metric,
                )
                train_score = get_metric_value(train_metrics, score_name)
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
                                "pair_arch": "organic_resdil",
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
                                "conv_channels": _shared_if_equal(
                                    arch_params.donor.channels,
                                    arch_params.acceptor.channels,
                                ),
                                "kernel_sizes": _shared_if_equal(
                                    arch_params.donor.kernel_sizes,
                                    arch_params.acceptor.kernel_sizes,
                                ),
                                "block_dilations": _shared_if_equal(
                                    arch_params.donor.dilations,
                                    arch_params.acceptor.dilations,
                                ),
                                "residual_channels": _shared_if_equal(
                                    arch_params.donor.residual_channels,
                                    arch_params.acceptor.residual_channels,
                                ),
                                "donor_conv_channels": list(arch_params.donor.channels),
                                "acceptor_conv_channels": list(
                                    arch_params.acceptor.channels
                                ),
                                "donor_kernel_sizes": list(
                                    arch_params.donor.kernel_sizes
                                ),
                                "acceptor_kernel_sizes": list(
                                    arch_params.acceptor.kernel_sizes
                                ),
                                "donor_block_dilations": list(
                                    arch_params.donor.dilations
                                ),
                                "acceptor_block_dilations": list(
                                    arch_params.acceptor.dilations
                                ),
                                "donor_residual_channels": list(
                                    arch_params.donor.residual_channels
                                ),
                                "acceptor_residual_channels": list(
                                    arch_params.acceptor.residual_channels
                                ),
                                "max_pool_size": arch_params.max_pool_size,
                                "pool_every": arch_params.pool_every,
                                "head_type": arch_params.head_type,
                                "dropout": train_params.dropout,
                                "fc_hidden": arch_params.fc_hidden,
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
                        "train_score": train_score,
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

                if report_train_metrics_bool:
                    mark = "*" if improved else "-"
                    train_score_text = (
                        "nan" if train_score is None else f"{train_score:.4f}"
                    )
                    val_score_text = f"{score:.4f}"
                    print(
                        f"[cnn_v3] {mark} epoch {epoch}/{epochs} "
                        f"loss={train_loss:.4f} score_metric={score_name} "
                        f"train_score={train_score_text} "
                        f"val_score={val_score_text} "
                        f"elapsed={epoch_elapsed_sec:.2f}s "
                        f"best={best_score:.4f} "
                        f"(ep {best_epoch})"
                    )

                if (
                    early_stop_patience > 0
                    and epochs_since_improvement >= early_stop_patience
                ):
                    stopped_early = True
                    print(
                        f"[cnn_v3] early stop at epoch {epoch} "
                        f"(patience={early_stop_patience}, "
                        f"min_delta={early_stop_min_delta:g})"
                    )
                    break

            print(
                f"[cnn_v3] done best_{best_metric_name}={best_score:.4f} "
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
                "validation_metric": resolved_validation_metric,
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
                "conv_channels": _shared_if_equal(
                    arch_params.donor.channels,
                    arch_params.acceptor.channels,
                ),
                "kernel_sizes": _shared_if_equal(
                    arch_params.donor.kernel_sizes,
                    arch_params.acceptor.kernel_sizes,
                ),
                "block_dilations": _shared_if_equal(
                    arch_params.donor.dilations,
                    arch_params.acceptor.dilations,
                ),
                "residual_channels": _shared_if_equal(
                    arch_params.donor.residual_channels,
                    arch_params.acceptor.residual_channels,
                ),
                "donor_conv_channels": list(arch_params.donor.channels),
                "acceptor_conv_channels": list(arch_params.acceptor.channels),
                "donor_kernel_sizes": list(arch_params.donor.kernel_sizes),
                "acceptor_kernel_sizes": list(arch_params.acceptor.kernel_sizes),
                "donor_block_dilations": list(arch_params.donor.dilations),
                "acceptor_block_dilations": list(arch_params.acceptor.dilations),
                "donor_residual_channels": list(arch_params.donor.residual_channels),
                "acceptor_residual_channels": list(
                    arch_params.acceptor.residual_channels
                ),
                "max_pool_size": arch_params.max_pool_size,
                "pool_every": arch_params.pool_every,
                "head_type": arch_params.head_type,
                "fc_hidden": arch_params.fc_hidden,
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
                    "[cnn_v3] torch.compile runtime failed "
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
                f"[cnn_v3] {device.upper()} OOM detected. "
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
    """Load one trained ``cnn_pair_v3`` pair model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _normalize_checkpoint_state_dict(ckpt["model_state"])
    model_config = ckpt.get("model_config", {})
    input_mode = cnn_v2._normalize_input_mode(
        model_config.get("input_mode", "onehot"),
        arg_name="checkpoint input_mode",
    )
    pair_mode = cnn_v2._normalize_pair_mode(
        model_config.get("pair_mode", "pair"),
        arg_name="checkpoint pair_mode",
    )
    embedding_dim = int(model_config.get("embedding_dim", 32))
    vocab_size_raw = model_config.get("vocab_size")
    vocab_size = None if vocab_size_raw is None else int(vocab_size_raw)
    dropout = float(model_config.get("dropout", 0.3))
    if input_mode != "onehot" and vocab_size is None:
        raise ValueError("checkpoint is missing vocab_size for token input mode.")
    arch_params = _resolve_pair_arch_params(argparse.Namespace(**model_config))
    model = PairOrganicResDilCNN(
        input_mode=input_mode,
        pair_mode=pair_mode,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        arch_params=arch_params,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_pair_sequences(
    *,
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
    """Score donor/acceptor sequence pairs with a trained model."""
    if not pairs:
        return np.array([])

    donor_encoder = cnn_v2._build_sequence_encoder(
        mode=input_mode,
        window_len=donor_window_len,
        bpe_pretrained_model_name=bpe_pretrained_model_name,
        bpe_pretrained_revision=bpe_pretrained_revision,
        bpe_trust_remote_code=bpe_trust_remote_code,
    )
    acceptor_encoder = cnn_v2._build_sequence_encoder(
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
        batch_donor, valid_batch_size = _pad_batch_to_fixed_size(
            donor_x[index : index + batch_size],
            batch_size,
        )
        batch_acceptor, _ = _pad_batch_to_fixed_size(
            acceptor_x[index : index + batch_size],
            batch_size,
        )
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
        log_scores = log10_sigmoid_np(logits.float().detach().cpu().numpy())[
            :valid_batch_size
        ]
        outputs.append(log_scores)
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
    """Infer pair-site scores using a trained ``cnn_pair_v3`` checkpoint."""
    if sequence_transform not in SEQUENCE_TRANSFORM_CHOICES:
        raise ValueError(
            "Unsupported --sequence_transform: "
            f"{sequence_transform}. Supported: {SEQUENCE_TRANSFORM_CHOICES}"
        )
    device_name = pick_device(device)
    infer_runtime = cnn_v2._resolve_infer_runtime_config(
        device=device_name,
        batch_size=batch_size,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
        infer_compile=infer_compile,
        infer_compile_mode=infer_compile_mode,
    )
    model, ckpt = load_pair_model(pair_model_path, device_name)
    model = cnn_v2._prepare_infer_model(
        model=model,
        task_name="pair",
        compile_enabled=infer_runtime.compile_enabled,
        compile_mode=infer_compile_mode,
    )

    donor_window_len = int(ckpt.get("donor_window_len", 50))
    acceptor_window_len = int(ckpt.get("acceptor_window_len", 50))
    model_config = ckpt.get("model_config", {})
    input_mode = cnn_v2._normalize_input_mode(
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
                "[cnn_v3] infer sequence_transform="
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
    if len(scores) != len(pair_rows):
        raise ValueError(
            "Pair score count does not match pair row count: "
            f"{len(scores)} != {len(pair_rows)}"
        )

    out_rows: List[Dict[str, object]] = []
    for row, score in zip(pair_rows, scores, strict=True):
        out_rows.append(
            {
                "transcript_id": str(row["transcript_id"]),
                "intron_index": int(row["intron_index"]),
                "site_type": "pair",
                "score": float(score),
                SCORE_SPACE_FIELD: SCORE_SPACE_LOG10,
            }
        )
    return out_rows


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register ``cnn_pair_v3`` training arguments."""
    cnn_v2.add_train_args(parser)
    parser.set_defaults(
        pair_mode="pair", train_target="pair", validation_metric="pr_auc"
    )
    parser.add_argument(
        "--block_dilations",
        type=str,
        default=None,
        help="Shared per-block dilations, e.g. 1,2,4,8.",
    )
    parser.add_argument(
        "--donor_block_dilations",
        type=str,
        default=None,
        help="Donor-branch override for --block_dilations.",
    )
    parser.add_argument(
        "--acceptor_block_dilations",
        type=str,
        default=None,
        help="Acceptor-branch override for --block_dilations.",
    )
    parser.add_argument(
        "--residual_channels",
        type=str,
        default=None,
        help="Shared bottleneck channels inside residual blocks.",
    )
    parser.add_argument(
        "--donor_residual_channels",
        type=str,
        default=None,
        help="Donor-branch override for --residual_channels.",
    )
    parser.add_argument(
        "--acceptor_residual_channels",
        type=str,
        default=None,
        help="Acceptor-branch override for --residual_channels.",
    )
    parser.add_argument(
        "--pool_every",
        type=int,
        default=2,
        help="Apply max-pooling after every N residual blocks.",
    )


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register ``cnn_pair_v3`` inference arguments."""
    cnn_v2.add_infer_args(parser)


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train ``cnn_v3`` with the unified runtime interface."""
    reported_model_name = str(getattr(common_args, "model", "cnn_pair_v3")).strip()
    if reported_model_name == "":
        reported_model_name = "cnn_pair_v3"
    requested_pair_mode = cnn_v2._normalize_pair_mode(
        getattr(model_args, "pair_mode", "pair"),
        arg_name="--pair_mode",
    )
    if requested_pair_mode != "pair":
        raise ValueError("cnn_pair_v3 supports only --pair_mode=pair.")

    requested_train_target = resolve_train_target(
        model_args,
        allowed_targets=("pair",),
    )
    if requested_train_target != "pair":
        raise ValueError("cnn_pair_v3 supports only --train_target=pair.")

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

    resolved_epochs, epochs_auto = resolve_training_epoch_budget(
        epochs_arg=model_args.epochs,
        max_epochs=int(model_args.max_epochs),
    )
    early_stop_patience, early_stop_min_delta = resolve_early_stopping_params(
        patience_arg=model_args.early_stop_patience,
        min_delta_arg=model_args.early_stop_min_delta,
    )
    # Enable early stopping only for auto epoch budget mode.
    effective_early_stop_patience = early_stop_patience if epochs_auto else 0

    train_params = cnn_v2._resolve_pair_train_params(model_args)
    pair_metrics = train_pair_model(
        pos_path=train_pos_path,
        neg_path=train_neg_path,
        checkpoint_path=pair_checkpoint_path,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        model_args=model_args,
        train_params=train_params,
        epochs=resolved_epochs,
        early_stop_patience=effective_early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        validation_metric=model_args.validation_metric,
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
        report_train_metrics=getattr(model_args, "report_train_metrics", 1),
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
    return {
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
        "validation_metric": resolve_validation_metric(model_args.validation_metric),
        "batch_size": model_args.batch_size,
        "lr": model_args.lr,
        "train_target": "pair",
        "sequence_transform": model_args.sequence_transform,
        "seed": common_args.seed,
        "device": common_args.device,
        "checkpoint_name": os.path.basename(pair_checkpoint_path),
        "pair_checkpoint_path": pair_checkpoint_path,
        "lightweight": model_args.lightweight,
        "input_mode": train_params.input_mode,
        "pair_mode": train_params.pair_mode,
        "embedding_dim": train_params.embedding_dim,
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
    }


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Run pair inference and return site-score rows."""
    requested_pair_mode = cnn_v2._normalize_pair_mode(
        getattr(model_args, "pair_mode", "pair"),
        arg_name="--pair_mode",
    )
    if requested_pair_mode != "pair":
        raise ValueError("cnn_pair_v3 supports only --pair_mode=pair.")

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
