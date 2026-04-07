"""Unified pipeline CLI for training, inference, transcript scoring, and eval.

This script is the single public executable for model workflows.
It dispatches to the selected model module and runs the default pipeline:
train -> infer -> transcript -> eval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

from models.registry import available_models, load_model_module
from util.checkpoint_prune import prune_species_model_checkpoints
from util.checkpoint_io import read_json_object
from util.data_proc import (
    NAME_FIELD_CHOICES,
    NAME_FIELD_LABELS,
    default_intron_output_path,
    default_site_output_path,
    default_transcript_output_path,
    infer_default_train_paths,
    model_root,
    normalize_tag_name_value,
    parse_name_fields,
    project_root,
    resolve_test_tsv,
    resolve_effective_window_lengths,
    species_data_dirs,
)
from util.model_task_paths import checkpoint_tasks_for_model
from util.model_runtime import (
    HIGH_LEVEL_COMPILE_MODE_CHOICES,
    is_compile_runtime_error as _is_compile_runtime_error,
    normalize_high_level_compile_mode as _normalize_high_level_compile_mode,
    record_compile_runtime_failure as _record_compile_runtime_failure,
)
from util.process_title import (
    apply_eta_process_title_placeholder,
    apply_process_title_from_env,
)
from util.sequence_transform import SEQUENCE_TRANSFORM_CHOICES
from util.validation_protocol import (
    build_validation_protocol,
    compute_validation_signature,
)
from util.path_format import relativize_path_fields
from util.transcript_eval import (
    INTRON_SCORE_OP_CHOICES,
    TRANSCRIPT_SCORE_AGG_CHOICES,
    aggregate_pair_transcript_scores,
    aggregate_transcript_scores,
    build_intron_scores,
    read_site_scores,
    write_intron_scores,
    write_site_scores,
    write_transcript_scores,
)
from util.unique_intron import (
    UNIQUE_MAP_TSV_NAME,
    UNIQUE_TRANSCRIPTS_TSV_NAME,
    UniqueMapMember,
    invert_unique_map,
    load_unique_map,
)
from util.versioned_artifacts import (
    refresh_published_version_if_improved,
    is_active_public_model,
    normalize_published_run_checkpoints,
    resolve_latest_published_name,
    resolve_published_run_assets,
    resolve_latest_published_run_assets,
)

try:
    from util.losses import LOSS_NAME_CHOICES
except ModuleNotFoundError:  # pragma: no cover
    LOSS_NAME_CHOICES = (
        "bce",
        "weighted_bce",
        "focal",
        "asymmetric_focal",
        "f1",
        "weighted_bce_f1",
        "focal_f1",
    )

_ = apply_process_title_from_env()


CHECKPOINT_NAME_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "species",
        "device",
        "perf_mode",
        "name_fields",
        "train_pos_path",
        "train_neg_path",
        "test_tsv",
        "site_score_tsv",
        "site_collapse_score_tolerance",
        "site_output_tsv",
        "intron_output_tsv",
        "transcript_output_tsv",
        "metrics_json",
        "class_file",
        "eval_output_txt",
        "skip_train",
        "continue_train",
        "train_only",
        "use_amp",
        "amp_dtype",
        "allow_tf32",
        "cudnn_benchmark",
        "deterministic",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "report_train_metrics",
        "compile",
        "compile_mode",
        "infer_batch_size",
        "infer_use_amp",
        "infer_amp_dtype",
        "infer_compile",
        "infer_compile_mode",
        "min_batch_size",
        "max_oom_retries",
        "gpu_id",
        "quick_phase",
        "intron_score_op",
        "transcript_score_agg",
        "softmin_tau",
        "ref_gff",
        "visualize",
        "output_png",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "donor_checkpoint_path",
        "acceptor_checkpoint_path",
        "pair_checkpoint_path",
        "pretrained_model_name",
    }
)

MAX_CHECKPOINT_STEM_LENGTH: int = 200
CHECKPOINT_STEM_HASH_CHARS: int = 12
DEFAULT_SITE_COLLAPSE_SCORE_TOLERANCE: float = 1e-6
TUNED_IDENTITY_IGNORED_FIXED_RUN_ARGS: frozenset[str] = frozenset(
    {"seed", "script_name", "train_target"}
)
TUNED_IDENTITY_IGNORED_SAMPLED_PARAMS: frozenset[str] = frozenset(
    {
        "model",
        "species",
        "seed",
        "train_target",
        "donor_len",
        "acceptor_len",
        "input_mode",
        "pair_mode",
        "sequence_transform",
        "embedding_dim",
        "bpe_pretrained_model_name",
        "bpe_pretrained_revision",
        "bpe_trust_remote_code",
    }
)


def _set_csv_field_limit_max() -> None:
    """Set CSV field-size limit to the largest supported value."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _add_shared_common_args(parser: argparse.ArgumentParser) -> None:
    """Register model-agnostic arguments."""
    parser.add_argument("--model", choices=available_models(), default="cnn")
    parser.add_argument("--species", default="Dmel")
    parser.add_argument("--donor_len", type=int, default=None)
    parser.add_argument("--acceptor_len", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Global random seed for reproducible training and inference.",
    )
    parser.add_argument(
        "--name_fields",
        default="bp_avg",
        help=(
            "Comma-separated fields used for output filename naming. "
            f"Supported: {', '.join(NAME_FIELD_CHOICES)}, none."
        ),
    )


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Register pipeline arguments shared across all stages."""
    parser.add_argument("--train_pos_path", default=None)
    parser.add_argument("--train_neg_path", default=None)
    parser.add_argument("--test_tsv", default=None)
    parser.add_argument("--site_score_tsv", default=None)
    parser.add_argument(
        "--site_collapse_score_tolerance",
        type=float,
        default=DEFAULT_SITE_COLLAPSE_SCORE_TOLERANCE,
        help=(
            "Absolute score tolerance when collapsing duplicate site-score rows "
            "to one unique intron/site key."
        ),
    )
    parser.add_argument("--site_output_tsv", default=None)
    parser.add_argument("--intron_output_tsv", default=None)
    parser.add_argument("--transcript_output_tsv", default=None)
    parser.add_argument("--metrics_json", default=None)
    parser.add_argument("--donor_tuned_config_path", default=None)
    parser.add_argument("--acceptor_tuned_config_path", default=None)
    parser.add_argument("--pair_tuned_config_path", default=None)
    parser.add_argument("--class_file", default=None)
    parser.add_argument(
        "--ref_gff",
        default=None,
        help=(
            "Reference GFF for evaluation. If omitted, it is auto-resolved "
            "from data/<species>/raw."
        ),
    )
    parser.add_argument("--eval_output_txt", default=None)
    parser.add_argument(
        "--skip_train",
        "--skip-training",
        action="store_true",
        help="Skip training and use existing checkpoints.",
    )
    parser.add_argument(
        "--continue_train",
        "--continue-training",
        action="store_true",
        help="Continue training from existing checkpoints as initial weights.",
    )
    parser.add_argument(
        "--train_only",
        "--train-only",
        action="store_true",
        help=(
            "Run only the training stage and write train summary JSON. "
            "Inference, transcript aggregation, and eval are skipped."
        ),
    )
    parser.add_argument(
        "--intron_score_op",
        choices=list(INTRON_SCORE_OP_CHOICES),
        default="+",
        help="How to combine donor and acceptor scores into intron score.",
    )
    parser.add_argument(
        "--transcript_score_agg",
        choices=list(TRANSCRIPT_SCORE_AGG_CHOICES),
        default="min",
        help="How to aggregate intron scores into transcript score.",
    )
    parser.add_argument(
        "--softmin_tau",
        type=float,
        default=1.0,
        help=(
            "Temperature used when --transcript_score_agg is softmin or softmin_wavg."
        ),
    )
    parser.add_argument(
        "--visualize",
        choices=["none", "true", "interactive"],
        default="none",
        help="Evaluation plot mode: none, true (save), interactive (save + show).",
    )
    parser.add_argument("--output_png", default=None)
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    parser.add_argument(
        "--checkpoint_top_k",
        type=int,
        default=3,
        help=(
            "Keep top-k checkpoints per species/model/task/validation_signature bucket."
        ),
    )
    parser.add_argument(
        "--checkpoint_prune_dry_run",
        type=int,
        choices=[0, 1],
        default=0,
        help="When set to 1, print pruning candidates without deleting files.",
    )


def _add_cnn_fallback_train_args(parser: argparse.ArgumentParser) -> None:
    """Add CNN train args without importing torch-dependent modules."""
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
        "--validation_metric",
        type=str,
        default="pr_auc",
        help="Validation metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--train_target",
        choices=["both", "donor", "acceptor"],
        default="both",
    )
    parser.add_argument(
        "--sequence_transform",
        choices=list(SEQUENCE_TRANSFORM_CHOICES),
        default="none",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--conv_channels", type=str, default=None)
    parser.add_argument("--kernel_sizes", default=None)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--max_pool_size", type=int, default=2)
    parser.add_argument("--conv_stride", type=int, default=1)
    parser.add_argument("--head_type", type=str, default="gap")
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
    parser.add_argument("--donor_conv_channels", type=str, default=None)
    parser.add_argument("--acceptor_conv_channels", type=str, default=None)
    parser.add_argument("--donor_kernel_sizes", default=None)
    parser.add_argument("--acceptor_kernel_sizes", default=None)
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
    parser.add_argument(
        "--compile_mode",
        choices=list(HIGH_LEVEL_COMPILE_MODE_CHOICES),
        default="auto",
        help=(
            "Compilation mode for torch.compile. "
            "Use quick for reduce-overhead only, or full for "
            "max-autotune-first compilation."
        ),
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
        "--report_train_metrics",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Compute train-split PR-AUC every epoch when set to 1. "
            "Set to 0 to skip extra train-eval pass."
        ),
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
    )
    parser.add_argument(
        "--acceptor_loss",
        choices=list(LOSS_NAME_CHOICES),
        default=None,
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
        "--f1_lambda",
        type=float,
        default=0.1,
        help=("Mixing coefficient for --loss weighted_bce_f1 or focal_f1."),
    )
    parser.add_argument("--donor_f1_lambda", type=float, default=None)
    parser.add_argument("--acceptor_f1_lambda", type=float, default=None)
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
    parser.add_argument("--tag", default=None)


def _add_cnn_fallback_infer_args(parser: argparse.ArgumentParser) -> None:
    """Add CNN infer args without importing torch-dependent modules."""
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--sequence_transform",
        choices=list(SEQUENCE_TRANSFORM_CHOICES),
        default="none",
    )


def _add_cnn_pair_fallback_train_args(parser: argparse.ArgumentParser) -> None:
    """Add cnn_pair train args without importing torch-dependent modules."""
    _add_cnn_fallback_train_args(parser)
    parser.add_argument(
        "--fusion_mode",
        choices=["late", "mid", "early", "early_channel"],
        default="late",
    )
    parser.add_argument(
        "--train_target",
        choices=["pair"],
        default="pair",
    )


def _add_cnn_v2_fallback_train_args(parser: argparse.ArgumentParser) -> None:
    """Add cnn_v2 train args without importing torch-dependent modules."""
    _add_cnn_pair_fallback_train_args(parser)
    parser.add_argument(
        "--pair_mode",
        choices=["pair", "independent"],
        default="independent",
    )
    parser.add_argument(
        "--train_target",
        choices=["donor", "acceptor"],
        default="donor",
    )


def _add_cnn_pair_v2_fallback_train_args(parser: argparse.ArgumentParser) -> None:
    """Add cnn_pair_v2 train args without importing torch-dependent modules."""
    _add_cnn_pair_fallback_train_args(parser)
    parser.add_argument(
        "--pair_mode",
        choices=["pair", "independent"],
        default="pair",
    )
    parser.add_argument(
        "--train_target",
        choices=["pair"],
        default="pair",
    )


def _add_cnn_v3_fallback_train_args(parser: argparse.ArgumentParser) -> None:
    """Add cnn_v3 train args without importing torch-dependent modules."""
    _add_cnn_fallback_train_args(parser)
    parser.add_argument(
        "--pair_mode",
        choices=["pair", "independent"],
        default="independent",
    )
    parser.add_argument(
        "--block_dilations",
        type=str,
        default=None,
        help="Shared residual-block dilations.",
    )
    parser.add_argument("--donor_block_dilations", type=str, default=None)
    parser.add_argument("--acceptor_block_dilations", type=str, default=None)
    parser.add_argument(
        "--residual_channels",
        type=str,
        default=None,
        help="Shared residual bottleneck channels.",
    )
    parser.add_argument("--donor_residual_channels", type=str, default=None)
    parser.add_argument("--acceptor_residual_channels", type=str, default=None)
    parser.add_argument(
        "--pool_every",
        type=int,
        default=2,
        help="Apply pooling after every N residual blocks.",
    )
    parser.add_argument("--donor_pool_every", type=int, default=None)
    parser.add_argument("--acceptor_pool_every", type=int, default=None)


def _add_cnn_pair_v3_fallback_train_args(parser: argparse.ArgumentParser) -> None:
    """Add cnn_pair_v3 train args without importing torch-dependent modules."""
    _add_cnn_pair_v2_fallback_train_args(parser)
    parser.add_argument(
        "--block_dilations",
        type=str,
        default=None,
        help="Shared residual-block dilations.",
    )
    parser.add_argument("--donor_block_dilations", type=str, default=None)
    parser.add_argument("--acceptor_block_dilations", type=str, default=None)
    parser.add_argument(
        "--residual_channels",
        type=str,
        default=None,
        help="Shared residual bottleneck channels.",
    )
    parser.add_argument("--donor_residual_channels", type=str, default=None)
    parser.add_argument("--acceptor_residual_channels", type=str, default=None)
    parser.add_argument(
        "--pool_every",
        type=int,
        default=2,
        help="Apply pooling after every N residual blocks.",
    )
    parser.add_argument("--donor_pool_every", type=int, default=None)
    parser.add_argument("--acceptor_pool_every", type=int, default=None)


def _add_cnn_v3_meta_fallback_train_args(parser: argparse.ArgumentParser) -> None:
    """Add cnn_v3_meta train args without importing torch-dependent modules."""
    _add_cnn_pair_v2_fallback_train_args(parser)
    parser.add_argument(
        "--base_pair_checkpoints",
        type=str,
        default=None,
        help=(
            "Comma-separated pretrained pair checkpoint paths used as "
            "meta-model inputs."
        ),
    )
    parser.add_argument(
        "--meta_hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension for meta MLP.",
    )
    parser.add_argument(
        "--meta_dropout",
        type=float,
        default=0.2,
        help="Dropout for meta MLP.",
    )


def _add_cnn_pair_fallback_infer_args(parser: argparse.ArgumentParser) -> None:
    """Add cnn_pair infer args without importing torch-dependent modules."""
    _add_cnn_fallback_infer_args(parser)


def _build_parser(
    selected_model: str,
    skip_model_import_error: bool = False,
) -> argparse.ArgumentParser:
    """Build parser and inject model-specific arguments."""
    parser = argparse.ArgumentParser(
        description="Unified model pipeline runner for splice-site workflows.",
        conflict_handler="resolve",
    )
    _add_shared_common_args(parser)
    _add_pipeline_args(parser)

    model_module = None
    try:
        model_module = load_model_module(selected_model)
    except (RuntimeError, ModuleNotFoundError, ImportError):
        if not skip_model_import_error:
            raise

    if model_module is not None:
        model_module.add_train_args(parser)
        model_module.add_infer_args(parser)
    elif selected_model in {"cnn", "cnn_resdil", "tcn"}:
        _add_cnn_fallback_train_args(parser)
        _add_cnn_fallback_infer_args(parser)
    elif selected_model == "cnn_pair":
        _add_cnn_pair_fallback_train_args(parser)
        _add_cnn_pair_fallback_infer_args(parser)
    elif selected_model == "cnn_v2":
        _add_cnn_v2_fallback_train_args(parser)
        _add_cnn_pair_fallback_infer_args(parser)
    elif selected_model == "cnn_pair_v2":
        _add_cnn_pair_v2_fallback_train_args(parser)
        _add_cnn_pair_fallback_infer_args(parser)
    elif selected_model == "cnn_v3":
        _add_cnn_v3_fallback_train_args(parser)
        _add_cnn_pair_fallback_infer_args(parser)
    elif selected_model == "cnn_pair_v3":
        _add_cnn_pair_v3_fallback_train_args(parser)
        _add_cnn_pair_fallback_infer_args(parser)
    elif selected_model == "cnn_v3_meta":
        _add_cnn_v3_meta_fallback_train_args(parser)
        _add_cnn_pair_fallback_infer_args(parser)

    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI args with model-aware two-phase parsing."""
    is_help_mode = any(token in {"-h", "--help"} for token in argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--model", default="cnn", choices=available_models())
    probed, _ = probe.parse_known_args(argv)
    parser = _build_parser(
        selected_model=probed.model,
        skip_model_import_error=is_help_mode,
    )
    return parser.parse_args(argv)


def _infer_window_defaults(
    species: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Infer effective donor/acceptor lengths for naming and preprocessing."""
    inferred_train_len: Optional[int] = None
    if donor_len is None or acceptor_len is None:
        dirs = species_data_dirs(species)
        try:
            _, _, inferred_train_len = infer_default_train_paths(
                train_dir=dirs["raw"],
                donor_len=donor_len,
                acceptor_len=acceptor_len,
            )
        except ValueError:
            inferred_train_len = None

    resolved_donor_len, resolved_acceptor_len = resolve_effective_window_lengths(
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        inferred_train_len=inferred_train_len,
    )
    return resolved_donor_len, resolved_acceptor_len, inferred_train_len


def _format_name_value(value: object) -> str:
    """Format and sanitize one filename token value."""
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    text = text.replace("+", "plus")
    text = text.replace("*", "x")
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]", "", text)


def _build_checkpoint_stem_from_params(
    model_name: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    inferred_train_len: Optional[int],
    raw_params: Mapping[str, object],
) -> str:
    """Build checkpoint stem from model-relevant runtime parameters."""
    donor_len_eff = donor_len
    acceptor_len_eff = acceptor_len
    if donor_len_eff is None and inferred_train_len is not None:
        donor_len_eff = inferred_train_len
    if acceptor_len_eff is None and inferred_train_len is not None:
        acceptor_len_eff = inferred_train_len

    pieces: list[str] = []
    if donor_len_eff is not None:
        pieces.append(f"dlen{_format_name_value(donor_len_eff)}")
    if acceptor_len_eff is not None:
        pieces.append(f"alen{_format_name_value(acceptor_len_eff)}")

    params = dict(raw_params)
    train_target_raw = params.get("train_target")
    train_target = (
        str(train_target_raw).strip().lower()
        if train_target_raw is not None
        else ("donor" if model_name == "cnn_v2" else "both")
    )
    for key in sorted(params):
        if key in CHECKPOINT_NAME_EXCLUDED_FIELDS:
            continue
        if key in {"donor_len", "acceptor_len"}:
            continue
        if train_target == "donor" and key.startswith("acceptor_"):
            continue
        if train_target == "acceptor" and key.startswith("donor_"):
            continue
        value = params[key]
        if value is None:
            continue
        if key == "train_target" and str(value) == "both":
            continue
        if key == "tag":
            normalized_tag = normalize_tag_name_value(model_name, value)
            if normalized_tag != "":
                pieces.append(normalized_tag)
            continue
        label = NAME_FIELD_LABELS.get(key, key)
        pieces.append(f"{label}{_format_name_value(value)}")

    if not pieces:
        return model_name

    stem = f"{model_name}_{'_'.join(pieces)}"
    if len(stem) <= MAX_CHECKPOINT_STEM_LENGTH:
        return stem

    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:CHECKPOINT_STEM_HASH_CHARS]
    suffix = f"_h{digest}"
    max_prefix_len = MAX_CHECKPOINT_STEM_LENGTH - len(suffix)
    if max_prefix_len <= 0:
        return f"h{digest}"
    trimmed_prefix = stem[:max_prefix_len].rstrip("_")
    return f"{trimmed_prefix}{suffix}"


def _build_checkpoint_paths(
    species: str,
    stem: str,
    tasks: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Build strict checkpoint paths for the provided tasks."""
    root = model_root()
    task_names = tuple(tasks) if tasks is not None else ("donor", "acceptor")
    paths: dict[str, str] = {}
    for task in task_names:
        paths[task] = os.path.join(root, species, task, f"{stem}.pt")
    return paths


def _assert_checkpoint_paths_exist(
    paths: dict[str, str],
    required_tasks: Optional[Sequence[str]] = None,
) -> None:
    """Assert checkpoint files exist for required tasks."""
    task_names = (
        tuple(required_tasks) if required_tasks is not None else tuple(paths.keys())
    )
    for task in task_names:
        if task not in paths:
            raise ValueError(f"Unknown checkpoint task requested: {task}")
        path = paths[task]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{task.capitalize()} checkpoint not found: {path}")


def _latest_checkpoint_for_task(
    species: str,
    task: str,
    *,
    model_name: str | None = None,
) -> Optional[str]:
    """Return the newest checkpoint path for one species/task.

    Parameters
    ----------
    species : str
        Species identifier.
    task : str
        Task name (for example, ``donor``).
    model_name : str | None, default=None
        Optional model-name prefix filter. When set, only checkpoints whose
        filename starts with ``"{model_name}_"`` are considered.

    Returns
    -------
    str | None
        Newest checkpoint path, or ``None`` if no candidate exists.
    """
    task_dir = os.path.join(model_root(), species, task)
    if not os.path.isdir(task_dir):
        return None

    candidates: list[str] = []
    normalized_model = None
    allowed_prefixes: tuple[str, ...] | None = None
    if model_name is not None:
        stripped = model_name.strip()
        if stripped != "":
            normalized_model = stripped
            allowed_prefixes = (f"{normalized_model}_",)
            if is_active_public_model(normalized_model):
                allowed_prefixes = (*allowed_prefixes, f"{normalized_model}.")
    for file_name in os.listdir(task_dir):
        if not file_name.endswith(".pt"):
            continue
        if (
            normalized_model is not None
            and allowed_prefixes is not None
            and not file_name.startswith(allowed_prefixes)
        ):
            continue
        path = os.path.join(task_dir, file_name)
        if os.path.isfile(path):
            candidates.append(path)
    if not candidates:
        return None

    candidates.sort(
        key=lambda path: (os.path.getmtime(path), os.path.basename(path)),
        reverse=True,
    )
    return candidates[0]


def _resolve_missing_checkpoints_for_skip_train(
    *,
    species: str,
    model_name: str,
    paths: dict[str, str],
    required_tasks: Sequence[str],
) -> dict[str, str]:
    """Fill missing checkpoint paths with latest available task checkpoints.

    Notes
    -----
    This is a skip-training fallback only. It prioritizes run continuity when
    strict filename matching is unavailable (for example after checkpoint
    pruning). The selected checkpoint is the latest file by mtime under each
    task directory.
    """
    resolved = dict(paths)
    for task in required_tasks:
        current = resolved.get(task)
        if current is None:
            raise ValueError(f"Unknown checkpoint task requested: {task}")
        if os.path.exists(current):
            continue
        fallback = _latest_checkpoint_for_task(
            species=species,
            task=task,
            model_name=model_name,
        )
        if fallback is None:
            continue
        resolved[task] = fallback
        print(
            "[pipeline] checkpoint fallback: "
            f"{task} strict path not found; using latest checkpoint: {fallback}"
        )
    return resolved


def _resolve_pipeline_paths(
    args: argparse.Namespace,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    inferred_train_len: Optional[int],
) -> tuple[str, str, str, str, str, str]:
    """Resolve default paths for pipeline artifacts."""
    dirs = species_data_dirs(args.species)
    name_fields = parse_name_fields(args.name_fields)
    name_params = dict(vars(args))

    if bool(getattr(args, "train_only", False)):
        test_tsv = str(args.test_tsv) if args.test_tsv is not None else ""
    else:
        test_tsv = resolve_test_tsv(args.species, args.test_tsv)
    if args.class_file is not None:
        class_file = str(args.class_file)
    else:
        processed_class_file = os.path.join(
            dirs["processed"],
            "transcript_class.txt",
        )
        raw_class_file = os.path.join(dirs["raw"], "transcript_class.txt")
        class_file = (
            processed_class_file
            if os.path.isfile(processed_class_file)
            else raw_class_file
        )
    site_output_tsv = args.site_output_tsv or default_site_output_path(
        species=args.species,
        model_name=args.model,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        fallback_train_len=inferred_train_len,
        name_fields=name_fields,
        name_params=name_params,
    )
    transcript_output_tsv = (
        args.transcript_output_tsv
        or default_transcript_output_path(
            species=args.species,
            model_name=args.model,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
            fallback_train_len=inferred_train_len,
            name_fields=name_fields,
            name_params=name_params,
        )
    )
    intron_output_tsv = args.intron_output_tsv or default_intron_output_path(
        species=args.species,
        model_name=args.model,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        fallback_train_len=inferred_train_len,
        name_fields=name_fields,
        name_params=name_params,
    )

    if args.eval_output_txt:
        eval_output_txt = args.eval_output_txt
    else:
        base = os.path.splitext(os.path.basename(transcript_output_tsv))[0]
        eval_output_txt = os.path.join(dirs["eval_score"], f"{base}.txt")

    return (
        test_tsv,
        class_file,
        site_output_tsv,
        intron_output_tsv,
        transcript_output_tsv,
        eval_output_txt,
    )


def _apply_skip_train_published_version(
    *,
    args: argparse.Namespace,
    checkpoint_paths: dict[str, str],
    model_tasks: Sequence[str],
) -> dict[str, str]:
    """Attach latest published paths when skip-train runs load canonical best."""
    if not bool(getattr(args, "skip_train", False)):
        return checkpoint_paths
    if not is_active_public_model(str(getattr(args, "model", ""))):
        return checkpoint_paths

    published_assets = resolve_latest_published_run_assets(
        project_root=Path(project_root()),
        species=str(args.species),
        model_name=str(args.model),
    )
    if published_assets is None:
        return checkpoint_paths

    if not args.site_output_tsv:
        args.site_output_tsv = published_assets["site_output_tsv"]
    if not args.intron_output_tsv:
        args.intron_output_tsv = published_assets["intron_output_tsv"]
    if not args.transcript_output_tsv:
        args.transcript_output_tsv = published_assets["transcript_output_tsv"]
    if not args.eval_output_txt:
        args.eval_output_txt = published_assets["eval_output_txt"]
    if getattr(args, "metrics_json", None) is None:
        args.metrics_json = published_assets["metrics_json"]

    resolved_paths = dict(checkpoint_paths)
    for task in model_tasks:
        asset_key = f"{task}_checkpoint_path"
        raw_path = published_assets.get(asset_key)
        if not isinstance(raw_path, str) or raw_path.strip() == "":
            continue
        resolved_paths[task] = raw_path
        setattr(args, asset_key, raw_path)

    published_name = str(published_assets["published_name"])
    print(f"[pipeline] Skip training uses published version: {published_name}")
    return resolved_paths


def _load_optional_intron_labels(
    species: str,
    labeled_name: str = "intron_eval_flank10.unique.tsv",
) -> dict[tuple[str, int], int]:
    """Load optional intron labels from ``data/<species>/processed``."""
    _set_csv_field_limit_max()
    species_dirs = species_data_dirs(species)
    labeled_path = os.path.join(species_dirs["base"], "processed", labeled_name)
    if not os.path.exists(labeled_path):
        return {}

    labels: dict[tuple[str, int], int] = {}
    with open(labeled_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"transcript_id", "intron_index", "label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            return {}
        for raw in reader:
            transcript_id = str(raw["transcript_id"]).strip()
            if transcript_id == "":
                continue
            try:
                intron_index = int(str(raw["intron_index"]))
                label = int(str(raw["label"]))
            except ValueError:
                continue
            if label not in {0, 1}:
                continue
            labels[(transcript_id, intron_index)] = label
    return labels


def _load_required_unique_intron_map(
    *,
    species: str,
    map_name: str = UNIQUE_MAP_TSV_NAME,
) -> dict[tuple[str, int], list[UniqueMapMember]]:
    """Load required unique->original transcript intron mapping.

    Parameters
    ----------
    species : str
        Species folder name under ``data``.
    map_name : str, default=UNIQUE_MAP_TSV_NAME
        Unique map TSV filename under ``data/<species>/processed``.

    Returns
    -------
    dict[tuple[str, int], list[object]]
        Unique map keyed by ``(unique_transcript_id, unique_intron_index)``.

    Raises
    ------
    FileNotFoundError
        If the unique map TSV does not exist.
    ValueError
        If the map TSV is malformed.
    """
    species_dirs = species_data_dirs(species)
    map_path = Path(species_dirs["base"]) / "processed" / map_name
    return load_unique_map(map_path)


def _uses_default_unique_test_tsv(*, species: str, test_tsv: str) -> bool:
    """Return True when ``test_tsv`` is the canonical unique transcript TSV."""
    if test_tsv.strip() == "":
        return False

    species_dirs = species_data_dirs(species)
    default_unique_path = (
        Path(species_dirs["base"]) / "processed" / UNIQUE_TRANSCRIPTS_TSV_NAME
    )
    try:
        return Path(test_tsv).resolve(strict=False) == default_unique_path.resolve(
            strict=False
        )
    except OSError:
        return False


def _collapse_site_rows_to_unique(
    *,
    site_score_rows: list[dict[str, object]],
    unique_map: dict[tuple[str, int], list[UniqueMapMember]],
    score_tolerance: float = DEFAULT_SITE_COLLAPSE_SCORE_TOLERANCE,
) -> list[dict[str, object]]:
    """Collapse site-score rows to unique intron keys.

    Parameters
    ----------
    site_score_rows : list[dict[str, object]]
        Site-score rows keyed by either original or unique intron IDs.
    unique_map : dict[tuple[str, int], list[UniqueMapMember]]
        Unique intron map loaded from ``transcripts.unique.map.tsv``.
    score_tolerance : float, default=1e-6
        Allowed absolute score difference when duplicate rows collapse to one
        unique key and site type.

    Returns
    -------
    list[dict[str, object]]
        Site-score rows keyed by unique intron IDs only.

    Raises
    ------
    ValueError
        If rows contain unsupported site types, unmapped intron keys, or
        conflicting scores for one collapsed unique key.
    """
    if score_tolerance < 0.0:
        raise ValueError("score_tolerance must be >= 0.")

    reverse_unique_map = invert_unique_map(unique_map)
    collapsed_rows: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in site_score_rows:
        transcript_id = str(row["transcript_id"]).strip()
        if transcript_id == "":
            raise ValueError("Site-score row has empty transcript_id.")
        intron_index = int(row["intron_index"])
        site_type = str(row["site_type"]).strip().lower()
        if site_type not in {"donor", "acceptor", "pair"}:
            raise ValueError(f"Unsupported site_type in site-score row: {site_type}")
        score = float(row["score"])

        source_key = (transcript_id, intron_index)
        unique_key = reverse_unique_map.get(source_key, source_key)
        if unique_key not in unique_map:
            raise ValueError(
                "Site-score row key is not present in unique intron map. "
                "Ensure inference/evaluation uses processed unique assets. "
                f"key={source_key[0]}:{source_key[1]}"
            )

        collapse_key = (unique_key[0], unique_key[1], site_type)
        previous = collapsed_rows.get(collapse_key)
        if previous is not None:
            previous_score = float(previous["score"])
            if abs(previous_score - score) > score_tolerance:
                raise ValueError(
                    "Conflicting scores among rows collapsed to one unique "
                    f"intron/site key={collapse_key} "
                    f"score_a={previous_score:.8g} score_b={score:.8g}"
                )
            continue

        copied = dict(row)
        copied["transcript_id"] = unique_key[0]
        copied["intron_index"] = unique_key[1]
        copied["site_type"] = site_type
        copied["score"] = score
        collapsed_rows[collapse_key] = copied

    sorted_keys = sorted(collapsed_rows.keys(), key=lambda item: item)
    return [collapsed_rows[key] for key in sorted_keys]


def _expand_unique_site_rows(
    *,
    site_score_rows: list[dict[str, object]],
    unique_map: dict[tuple[str, int], list[UniqueMapMember]],
) -> list[dict[str, object]]:
    """Expand unique-intron site rows back to original transcript introns.

    Parameters
    ----------
    site_score_rows : list[dict[str, object]]
        Site-level rows keyed by unique transcript id.
    unique_map : dict[tuple[str, int], list[object]]
        Unique map loaded from ``transcripts.unique.map.tsv``.

    Returns
    -------
    list[dict[str, object]]
        Expanded rows keyed by original ``(transcript_id, intron_index)``.

    Raises
    ------
    ValueError
        If some site rows cannot be mapped back to original transcript introns.

    Notes
    -----
    Some inference paths already emit rows keyed by original intron IDs
    instead of unique IDs. Those rows are passed through unchanged.
    """
    expanded_rows: list[dict[str, object]] = []
    missing_keys: set[tuple[str, int]] = set()
    reverse_unique_map = invert_unique_map(unique_map)
    for row in site_score_rows:
        unique_transcript_id = str(row["transcript_id"]).strip()
        unique_intron_index = int(row["intron_index"])
        unique_key = (unique_transcript_id, unique_intron_index)
        mapped_members = unique_map.get(unique_key)
        if mapped_members is None:
            if unique_key in reverse_unique_map:
                expanded_rows.append(dict(row))
                continue
            missing_keys.add(unique_key)
            continue
        for member in mapped_members:
            copied = dict(row)
            copied["transcript_id"] = member.transcript_id
            copied["intron_index"] = member.intron_index
            expanded_rows.append(copied)
    if missing_keys:
        examples = ", ".join(
            f"{transcript_id}:{intron_index}"
            for transcript_id, intron_index in sorted(missing_keys)[:5]
        )
        raise ValueError(
            "Unique site-score rows contain unmapped introns. "
            "Ensure processed unique assets are generated and aligned with "
            "--site_score_tsv. "
            f"examples={examples}"
        )
    return expanded_rows


def _resolve_ref_gff_file(
    species: str,
    configured_path: Optional[str],
) -> str:
    """Resolve reference GFF path for evaluation.

    Parameters
    ----------
    species : str
        Species folder name under ``data``.
    configured_path : str | None
        Optional explicit path provided by ``--ref_gff``.

    Returns
    -------
    str
        Existing path to the reference GFF file.

    Raises
    ------
    FileNotFoundError
        If a valid reference GFF cannot be resolved.
    """

    if configured_path not in (None, "", "None"):
        if os.path.exists(configured_path):
            return configured_path
        raise FileNotFoundError(f"Reference GFF not found: {configured_path}")

    raw_dir = species_data_dirs(species)["raw"]
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    candidates: list[str] = []
    for name in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".gff") or name.endswith(".gff3") or ".gff." in name:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Reference GFF not found under raw directory. Set --ref_gff explicitly."
        )

    preferred = [
        path
        for path in candidates
        if path.endswith(".fix.gff") or path.endswith(".gff.fix")
    ]
    if preferred:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    return candidates[0]


def _safe_float(value: object, default: float = float("-inf")) -> float:
    """Convert one scalar-like value to float with fallback."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _normalize_hparam_value(value: object) -> str:
    """Normalize one CLI or JSON hyperparameter value for equality checks."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, (list, tuple)):
        return ",".join(_normalize_hparam_value(item) for item in value)
    if value is None:
        return ""
    return str(value).strip()


def _resolve_tuned_config_paths(
    *,
    args: argparse.Namespace,
    model_tasks: Sequence[str],
) -> dict[str, Path]:
    """Collect explicitly provided tuned-config paths for active tasks."""
    resolved: dict[str, Path] = {}
    for task in model_tasks:
        raw_path = getattr(args, f"{task}_tuned_config_path", None)
        if not isinstance(raw_path, str) or raw_path.strip() == "":
            continue
        resolved[task] = Path(raw_path).resolve()
    return resolved


def _load_tuned_payloads(
    *,
    args: argparse.Namespace,
    model_tasks: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Load tuned best-config payloads keyed by task."""
    payloads: dict[str, dict[str, object]] = {}
    for task, path in _resolve_tuned_config_paths(
        args=args, model_tasks=model_tasks
    ).items():
        payload = read_json_object(path)
        if payload is None or payload.get("status") != "ok":
            continue
        payloads[task] = dict(payload)
    return payloads


def _resolve_tuned_runtime_value(
    *,
    args: argparse.Namespace,
    task: str,
    tuned_key: str,
) -> object:
    """Resolve one runtime value corresponding to a tuned key."""
    prefixed_name = f"{task}_{tuned_key}"
    if hasattr(args, prefixed_name):
        prefixed_value = getattr(args, prefixed_name)
        if prefixed_value is not None:
            return prefixed_value
    return getattr(args, tuned_key, None)


def _tuned_identity_matches_args(
    *,
    args: argparse.Namespace,
    model_tasks: Sequence[str],
    tuned_payloads: Mapping[str, Mapping[str, object]],
) -> bool:
    """Return whether current runtime args match tuned hparam identity."""
    if not tuned_payloads:
        return False

    for task in model_tasks:
        payload = tuned_payloads.get(task)
        if payload is None:
            continue
        sampled_params = payload.get("sampled_params")
        if isinstance(sampled_params, Mapping):
            for tuned_key, tuned_value in sampled_params.items():
                key_name = str(tuned_key).strip()
                if (
                    key_name == ""
                    or key_name == "train_target"
                    or key_name in TUNED_IDENTITY_IGNORED_SAMPLED_PARAMS
                ):
                    continue
                runtime_value = _resolve_tuned_runtime_value(
                    args=args,
                    task=task,
                    tuned_key=key_name,
                )
                if _normalize_hparam_value(runtime_value) != _normalize_hparam_value(
                    tuned_value
                ):
                    return False

        hparam_context = payload.get("hparam_context")
        if not isinstance(hparam_context, Mapping):
            continue
        fixed_run_args = hparam_context.get("fixed_run_args")
        if not isinstance(fixed_run_args, Mapping):
            continue
        for key, tuned_value in fixed_run_args.items():
            key_name = str(key).strip()
            if (
                key_name == ""
                or key_name in TUNED_IDENTITY_IGNORED_FIXED_RUN_ARGS
                or not hasattr(args, key_name)
            ):
                continue
            runtime_value = getattr(args, key_name)
            if _normalize_hparam_value(runtime_value) != _normalize_hparam_value(
                tuned_value
            ):
                return False
    return True


def _resolve_published_name_from_tuned_payloads(
    tuned_payloads: Mapping[str, Mapping[str, object]],
) -> str | None:
    """Resolve one shared published version name from tuned payloads."""
    published_names = {
        str(payload.get("published_name", "")).strip()
        for payload in tuned_payloads.values()
        if str(payload.get("published_name", "")).strip() != ""
    }
    if not published_names:
        return None
    if len(published_names) != 1:
        raise ValueError(
            "Tuned configs disagree on published_name; cannot warm-start safely."
        )
    return next(iter(published_names))


def _resolve_effective_published_name_for_tuned_run(
    *,
    args: argparse.Namespace,
    tuned_payloads: Mapping[str, Mapping[str, object]],
    model_tasks: Sequence[str],
) -> str | None:
    """Resolve one published name for tuned runs.

    Prefer the explicit ``published_name`` stored in tuned payloads. When that
    annotation is missing, fall back to the latest live published version from
    version history, but only for runs whose effective tuned identity matches
    the current CLI arguments.
    """
    published_name = _resolve_published_name_from_tuned_payloads(tuned_payloads)
    if published_name is not None:
        return published_name
    if not is_active_public_model(str(args.model)):
        return None
    if not _tuned_identity_matches_args(
        args=args,
        model_tasks=model_tasks,
        tuned_payloads=tuned_payloads,
    ):
        return None
    data_root = Path(species_data_dirs(str(args.species))["base"]).parent
    return resolve_latest_published_name(
        data_root=data_root,
        species=str(args.species),
        model_name=str(args.model),
    )


def _resolve_required_published_name_for_tuned_continue(
    *,
    args: argparse.Namespace,
    tuned_payloads: Mapping[str, Mapping[str, object]],
) -> str:
    """Resolve the published version name required for tuned continue runs.

    For ``--continue_train`` with tuned configs, warm-start must use the
    versioned checkpoint that corresponds to the tuned configuration identity.
    Therefore this resolver requires an explicit ``published_name`` annotation
    in tuned payloads and does not fall back to "latest" publication.

    Raises
    ------
    ValueError
        If published_name is missing or inconsistent across tuned payloads.
    """
    published_name = _resolve_published_name_from_tuned_payloads(tuned_payloads)
    if published_name is not None:
        return published_name
    data_root = Path(species_data_dirs(str(args.species))["base"]).parent
    latest = resolve_latest_published_name(
        data_root=data_root,
        species=str(args.species),
        model_name=str(args.model),
    )
    if latest is None:
        raise ValueError(
            "Tuned continue-train requires published_name in best_config.json, "
            "but none was found and no published version history exists for "
            f"species={args.species}, model={args.model}. "
            "Seed one published version first (versioning rollout), or add "
            "published_name to tuned best_config files."
        )
    raise ValueError(
        "Tuned continue-train requires published_name in best_config.json "
        f"for each active task, but no published_name was found "
        f"(species={args.species}, model={args.model}, latest={latest})."
    )


def _resolve_tuned_published_assets(
    *,
    args: argparse.Namespace,
    tuned_payloads: Mapping[str, Mapping[str, object]],
    model_tasks: Sequence[str],
    allow_missing_checkpoints: bool,
) -> tuple[str, dict[str, str]] | None:
    """Resolve published assets for one tuned run when identity matches."""
    published_name = _resolve_effective_published_name_for_tuned_run(
        args=args,
        tuned_payloads=tuned_payloads,
        model_tasks=model_tasks,
    )
    if published_name is None:
        return None

    published_assets = resolve_published_run_assets(
        project_root=Path(project_root()),
        species=str(args.species),
        model_name=str(args.model),
        published_name=published_name,
        allow_missing_checkpoints=allow_missing_checkpoints,
    )
    if published_assets is None:
        return None
    return published_name, published_assets


def _apply_published_output_targets(
    *,
    args: argparse.Namespace,
    published_assets: Mapping[str, str],
) -> None:
    """Fill unset output paths from one published-asset payload."""
    if not args.site_output_tsv:
        args.site_output_tsv = published_assets["site_output_tsv"]
    if not args.intron_output_tsv:
        args.intron_output_tsv = published_assets["intron_output_tsv"]
    if not args.transcript_output_tsv:
        args.transcript_output_tsv = published_assets["transcript_output_tsv"]
    if not args.eval_output_txt:
        args.eval_output_txt = published_assets["eval_output_txt"]
    if getattr(args, "metrics_json", None) in {None, ""}:
        args.metrics_json = published_assets["metrics_json"]


def _apply_tuned_continue_warm_start(
    *,
    args: argparse.Namespace,
    checkpoint_paths: dict[str, str],
    model_tasks: Sequence[str],
    tasks_to_train: Sequence[str],
    tuned_payloads: Mapping[str, Mapping[str, object]],
) -> bool:
    """Inject published checkpoints as init weights for tuned continue runs."""
    if not tuned_payloads:
        return False

    published_name = _resolve_required_published_name_for_tuned_continue(
        args=args,
        tuned_payloads=tuned_payloads,
    )
    normalized = normalize_published_run_checkpoints(
        project_root=Path(project_root()),
        species=str(args.species),
        model_name=str(args.model),
        published_name=published_name,
    )
    if normalized:
        for task_name, source_path in normalized.items():
            print(
                "[pipeline] Normalized published checkpoint: "
                f"task={task_name} source={source_path} "
                f"target={published_name}.pt"
            )
    published_assets = resolve_published_run_assets(
        project_root=Path(project_root()),
        species=str(args.species),
        model_name=str(args.model),
        published_name=published_name,
        allow_missing_checkpoints=False,
    )
    if published_assets is None:
        raise ValueError(
            "Published assets were not found for tuned continue-train "
            f"(species={args.species}, model={args.model}, "
            f"published_name={published_name})."
        )

    for task in tasks_to_train:
        asset_key = f"{task}_checkpoint_path"
        checkpoint_path = published_assets.get(asset_key)
        if not isinstance(checkpoint_path, str) or checkpoint_path.strip() == "":
            return False

    for task in tasks_to_train:
        checkpoint_path = str(published_assets[f"{task}_checkpoint_path"])
        setattr(args, f"{task}_init_checkpoint_path", checkpoint_path)
        setattr(args, f"{task}_checkpoint_path", checkpoint_paths[task])
        print(
            "[pipeline] Continue warm-start source: "
            f"task={task} checkpoint={checkpoint_path}"
        )

    print(
        "[pipeline] Continue training (--continue_train): "
        f"use published checkpoints from {published_name} as initialization."
    )
    return True


def _resolve_tuned_published_checkpoint_targets(
    *,
    args: argparse.Namespace,
    model_tasks: Sequence[str],
    tuned_payloads: Mapping[str, Mapping[str, object]],
) -> dict[str, str] | None:
    """Resolve published checkpoint paths as canonical save targets.

    When one run exactly matches the tuned hparam identity, the checkpoint
    destination should be the published version path itself rather than one
    transient stem-specific raw filename.
    """
    resolved = _resolve_tuned_published_assets(
        args=args,
        tuned_payloads=tuned_payloads,
        model_tasks=model_tasks,
        allow_missing_checkpoints=True,
    )
    if resolved is None:
        return None
    _published_name, published_assets = resolved

    resolved_paths: dict[str, str] = {}
    for task in model_tasks:
        asset_key = f"{task}_checkpoint_path"
        checkpoint_path = published_assets.get(asset_key)
        if not isinstance(checkpoint_path, str) or checkpoint_path.strip() == "":
            return None
        resolved_paths[task] = checkpoint_path
    return resolved_paths


def _apply_tuned_published_output_targets(
    *,
    args: argparse.Namespace,
    tuned_payloads: Mapping[str, Mapping[str, object]],
    model_tasks: Sequence[str],
) -> str | None:
    """Point pipeline outputs at the published version when identity matches."""
    resolved = _resolve_tuned_published_assets(
        args=args,
        tuned_payloads=tuned_payloads,
        model_tasks=model_tasks,
        allow_missing_checkpoints=True,
    )
    if resolved is None:
        return None
    published_name, published_assets = resolved
    _apply_published_output_targets(args=args, published_assets=published_assets)
    return published_name


def _build_refresh_task_payloads(
    *,
    summary: Mapping[str, object],
    tasks_to_train: Sequence[str],
    metrics_json: str,
) -> dict[str, dict[str, object]]:
    """Build version-refresh payloads from one completed training summary."""
    payloads: dict[str, dict[str, object]] = {}
    for task in tasks_to_train:
        task_summary = summary.get(task)
        if not isinstance(task_summary, Mapping):
            continue
        checkpoint = str(task_summary.get("checkpoint", "")).strip()
        metric = str(task_summary.get("best_metric", "")).strip()
        score = task_summary.get("best_score")
        if checkpoint == "" or metric == "" or score is None:
            continue
        payloads[task] = {
            f"{task}_checkpoint_path": checkpoint,
            "objective_metric": metric,
            "objective_score": score,
            "metrics_json": metrics_json,
        }
    return payloads


def _attach_validation_metadata(
    *,
    summary: dict[str, object],
    args: argparse.Namespace,
) -> None:
    """Attach validation protocol/signature and per-task selection score."""
    metric_primary = "pr_auc"
    task_names = checkpoint_tasks_for_model(str(getattr(args, "model", "")))
    primary_task = task_names[0] if task_names else "donor"
    primary_task_summary = summary.get(primary_task)
    if isinstance(primary_task_summary, dict):
        primary_metric = primary_task_summary.get("best_metric")
        if isinstance(primary_metric, str) and primary_metric.strip():
            metric_primary = primary_metric.strip()

    model_name = str(getattr(args, "model", "")).strip().lower()
    train_target = str(getattr(args, "train_target", "")).strip().lower()
    pair_mode = str(getattr(args, "pair_mode", "")).strip().lower()
    include_pair_mixed_negatives = False
    if train_target == "pair":
        include_pair_mixed_negatives = True
    elif model_name == "cnn_pair_v2":
        include_pair_mixed_negatives = True
    elif model_name in {"cnn_pair", "bilstm_pair", "cnn_pair_v3"}:
        include_pair_mixed_negatives = True

    protocol = build_validation_protocol(
        val_frac=getattr(args, "val_frac", None),
        seed=getattr(args, "seed", None),
        train_pos_path=(
            str(summary.get("train_pos_path"))
            if isinstance(summary.get("train_pos_path"), str)
            else getattr(args, "train_pos_path", None)
        ),
        train_neg_path=(
            str(summary.get("train_neg_path"))
            if isinstance(summary.get("train_neg_path"), str)
            else getattr(args, "train_neg_path", None)
        ),
        metric_primary=metric_primary,
        split_type="stratified_site",
        include_pair_mixed_negatives=include_pair_mixed_negatives,
    )
    signature = compute_validation_signature(protocol)
    summary["validation_protocol"] = protocol
    summary["validation_signature"] = signature

    selection_by_task: dict[str, float] = {}
    for task in task_names:
        task_payload = summary.get(task)
        if not isinstance(task_payload, dict):
            continue
        best_score = _safe_float(task_payload.get("best_score"))
        if best_score == float("-inf"):
            continue
        selection_by_task[task] = best_score
    if selection_by_task:
        summary["selection_score_by_task"] = selection_by_task
        for task, value in selection_by_task.items():
            summary[f"selection_score_{task}"] = value


def _bool_from_cli_flag(value: object) -> bool:
    """Parse common CLI flag values into one strict boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "on", "yes"}:
            return True
        if normalized in {"0", "false", "off", "no", ""}:
            return False
    return False


def _infer_compile_requested(args: argparse.Namespace) -> bool:
    """Return whether inference may attempt ``torch.compile`` for this run."""
    compile_flag = _bool_from_cli_flag(getattr(args, "compile", False))
    compile_mode = _normalize_high_level_compile_mode(
        str(getattr(args, "compile_mode", "off"))
    )

    infer_compile_raw = getattr(args, "infer_compile", None)
    infer_mode_raw = getattr(args, "infer_compile_mode", None)
    infer_compile_flag = (
        compile_flag
        if infer_compile_raw is None
        else _bool_from_cli_flag(infer_compile_raw)
    )
    infer_mode = (
        compile_mode
        if infer_mode_raw is None
        else _normalize_high_level_compile_mode(str(infer_mode_raw))
    )
    return infer_compile_flag and infer_mode == "on"


def _disable_infer_compile_flags(args: argparse.Namespace) -> None:
    """Disable inference compile flags in-place for one retry."""
    if hasattr(args, "infer_compile"):
        setattr(args, "infer_compile", 0)
    if hasattr(args, "infer_compile_mode"):
        setattr(args, "infer_compile_mode", "off")


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the model pipeline with optional stage skipping."""
    from evaluate_scores import evaluate_score_file, plot_eval_scores

    model_module = load_model_module(args.model)
    model_tasks = checkpoint_tasks_for_model(args.model)
    if args.model == "cnn_v2":
        args.pair_mode = "independent"
        model_tasks = ("donor", "acceptor")
    elif args.model == "cnn_pair_v2":
        args.pair_mode = "pair"
        model_tasks = ("pair",)
    elif args.model == "cnn_pair_v3":
        args.pair_mode = "pair"
        model_tasks = ("pair",)
    default_train_target = (
        model_tasks[0]
        if args.model == "cnn_v2"
        else ("both" if len(model_tasks) > 1 else model_tasks[0])
    )
    train_target = (
        str(getattr(args, "train_target", default_train_target)).strip().lower()
    )
    if args.model in {"cnn_pair_v2", "cnn_pair_v3"}:
        if train_target != "pair":
            train_target = "pair"
            args.train_target = "pair"
    allowed_targets = (
        ("both", *model_tasks) if len(model_tasks) > 1 else tuple(model_tasks)
    )
    if train_target not in allowed_targets:
        allowed_text = ", ".join(allowed_targets)
        raise ValueError(f"--train_target must be one of: {allowed_text}.")
    if (
        args.model != "cnn_v2"
        and len(model_tasks) > 1
        and (not args.train_only)
        and train_target != "both"
    ):
        task_text = "/".join(model_tasks)
        raise ValueError(
            f"--train_target {task_text} requires --train_only. "
            "Inference and transcript scoring require all task checkpoints."
        )
    if (
        len(model_tasks) == 1
        and (not args.train_only)
        and train_target != model_tasks[0]
    ):
        raise ValueError(
            f"--train_target must be '{model_tasks[0]}' for model {args.model}."
        )
    tasks_to_train = model_tasks if train_target == "both" else (train_target,)
    tuned_payloads = _load_tuned_payloads(args=args, model_tasks=model_tasks)
    published_output_name = _apply_tuned_published_output_targets(
        args=args,
        tuned_payloads=tuned_payloads,
        model_tasks=model_tasks,
    )
    donor_len, acceptor_len, inferred_train_len = _infer_window_defaults(
        species=args.species,
        donor_len=args.donor_len,
        acceptor_len=args.acceptor_len,
    )

    checkpoint_stem = _build_checkpoint_stem_from_params(
        model_name=args.model,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        inferred_train_len=inferred_train_len,
        raw_params=dict(vars(args)),
    )
    checkpoint_paths = _build_checkpoint_paths(
        args.species,
        checkpoint_stem,
        tasks=model_tasks,
    )
    published_checkpoint_targets = _resolve_tuned_published_checkpoint_targets(
        args=args,
        model_tasks=model_tasks,
        tuned_payloads=tuned_payloads,
    )
    if published_checkpoint_targets is not None:
        checkpoint_paths = published_checkpoint_targets
    for task in model_tasks:
        setattr(args, f"{task}_checkpoint_path", checkpoint_paths[task])
        setattr(args, f"{task}_init_checkpoint_path", "")
    checkpoint_paths = _apply_skip_train_published_version(
        args=args,
        checkpoint_paths=checkpoint_paths,
        model_tasks=model_tasks,
    )

    (
        args.test_tsv,
        class_file,
        site_output_tsv,
        intron_output_tsv,
        transcript_output_tsv,
        eval_output_txt,
    ) = _resolve_pipeline_paths(
        args=args,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        inferred_train_len=inferred_train_len,
    )
    if published_output_name is not None:
        print(f"[pipeline] Versioned output targets: {published_output_name}")

    if args.skip_train and args.continue_train:
        raise ValueError("--continue_train cannot be combined with --skip_train.")

    if args.skip_train:
        print("[pipeline] Skip training (--skip_train).")
    else:
        _ = apply_eta_process_title_placeholder()
        if args.continue_train:
            used_published_warm_start = _apply_tuned_continue_warm_start(
                args=args,
                checkpoint_paths=checkpoint_paths,
                model_tasks=model_tasks,
                tasks_to_train=tasks_to_train,
                tuned_payloads=tuned_payloads,
            )
            if not used_published_warm_start:
                _assert_checkpoint_paths_exist(
                    checkpoint_paths,
                    required_tasks=tasks_to_train,
                )
                for task in tasks_to_train:
                    setattr(
                        args, f"{task}_init_checkpoint_path", checkpoint_paths[task]
                    )
                print(
                    "[pipeline] Continue training (--continue_train): "
                    "use existing checkpoints as initialization."
                )
        summary = model_module.train(common_args=args, model_args=args)
        _attach_validation_metadata(summary=summary, args=args)
        metrics_json = args.metrics_json
        if metrics_json is None:
            dirs = species_data_dirs(args.species)
            os.makedirs(dirs["learning_metric"], exist_ok=True)
            metrics_json = os.path.join(
                dirs["learning_metric"],
                f"{checkpoint_stem}.train.json",
            )
        else:
            metrics_json_parent = Path(metrics_json).parent
            metrics_json_parent.mkdir(parents=True, exist_ok=True)

        serializable_summary = relativize_path_fields(summary)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(serializable_summary, f, indent=2)
        print(f"Saved training summary: {metrics_json}")
        published_name = _resolve_effective_published_name_for_tuned_run(
            args=args,
            tuned_payloads=tuned_payloads,
            model_tasks=model_tasks,
        )
        if published_name is not None:
            refresh_payloads = _build_refresh_task_payloads(
                summary=summary,
                tasks_to_train=tasks_to_train,
                metrics_json=str(metrics_json),
            )
            refreshed_entry = refresh_published_version_if_improved(
                project_root=Path(project_root()),
                species=str(args.species),
                model_name=str(args.model),
                published_name=published_name,
                task_payloads=refresh_payloads,
                metrics_json=str(metrics_json),
            )
            if refreshed_entry is not None:
                print(
                    "[pipeline] Refreshed published version in place: "
                    f"{refreshed_entry.published_name} "
                    f"updated_side={refreshed_entry.updated_side}"
                )
        for task in model_tasks:
            task_summary = summary.get(task)
            if isinstance(task_summary, dict) and "checkpoint" in task_summary:
                print(f"{task.capitalize()} checkpoint: {task_summary['checkpoint']}")
        top_k = int(getattr(args, "checkpoint_top_k", 3))
        if top_k <= 0:
            raise ValueError("--checkpoint_top_k must be > 0.")
        prune_report = prune_species_model_checkpoints(
            data_root=Path(species_data_dirs(args.species)["base"]).parent,
            species=args.species,
            model_name=args.model,
            top_k=top_k,
            dry_run=bool(int(getattr(args, "checkpoint_prune_dry_run", 0))),
        )
        print(
            "[pipeline] checkpoint prune: "
            f"total={prune_report.total_candidates} "
            f"kept={prune_report.kept_count} "
            f"deleted={prune_report.deleted_count} "
            f"dry_run={prune_report.dry_run}"
        )

    if args.train_only:
        if args.skip_train:
            checkpoint_paths = _resolve_missing_checkpoints_for_skip_train(
                species=args.species,
                model_name=args.model,
                paths=checkpoint_paths,
                required_tasks=tasks_to_train,
            )
            for task in model_tasks:
                setattr(args, f"{task}_checkpoint_path", checkpoint_paths[task])
            _assert_checkpoint_paths_exist(
                checkpoint_paths,
                required_tasks=tasks_to_train,
            )
            print("[pipeline] --train_only with --skip_train: checkpoints verified.")
        print("[pipeline] --train_only requested. Stop after training stage.")
        return

    unique_map = _load_required_unique_intron_map(species=args.species)

    infer_stage_started_at = time.perf_counter()
    if args.site_score_tsv:
        site_score_tsv = args.site_score_tsv
        site_rows = read_site_scores(site_score_tsv)
        print(f"[pipeline] Skip infer (use --site_score_tsv): {site_score_tsv}")
    else:
        checkpoint_paths = _resolve_missing_checkpoints_for_skip_train(
            species=args.species,
            model_name=args.model,
            paths=checkpoint_paths,
            required_tasks=model_tasks,
        )
        for task in model_tasks:
            setattr(args, f"{task}_checkpoint_path", checkpoint_paths[task])
        _assert_checkpoint_paths_exist(checkpoint_paths, required_tasks=model_tasks)
        try:
            site_rows = model_module.infer_site(common_args=args, model_args=args)
        except Exception as exc:
            should_retry_without_compile = _infer_compile_requested(args) and (
                isinstance(exc, NotImplementedError) or _is_compile_runtime_error(exc)
            )
            if not should_retry_without_compile:
                raise
            print(
                "[pipeline] inference torch.compile runtime failed "
                f"({exc.__class__.__name__}). "
                "Retry once with infer_compile=0 and infer_compile_mode=off."
            )
            _record_compile_runtime_failure(selected_mode=None)
            _disable_infer_compile_flags(args)
            site_rows = model_module.infer_site(common_args=args, model_args=args)

    unique_site_rows = _collapse_site_rows_to_unique(
        site_score_rows=site_rows,
        unique_map=unique_map,
        score_tolerance=args.site_collapse_score_tolerance,
    )
    if len(unique_site_rows) != len(site_rows):
        print(
            "[pipeline] site-score unique collapse: "
            f"input_rows={len(site_rows)} unique_rows={len(unique_site_rows)}"
        )
    if args.site_score_tsv:
        print(
            "[pipeline] intron evaluation uses unique-collapsed site rows from "
            f"--site_score_tsv: {site_score_tsv}"
        )
    else:
        intron_labels = _load_optional_intron_labels(args.species)
        write_site_scores(
            site_output_tsv,
            unique_site_rows,
            labels=intron_labels,
        )
        site_score_tsv = site_output_tsv
        print(f"Saved site scores: {site_output_tsv}")
    if args.site_score_tsv:
        intron_labels = _load_optional_intron_labels(args.species)
    infer_stage_elapsed_sec = time.perf_counter() - infer_stage_started_at
    print(f"[pipeline] inference stage elapsed: {infer_stage_elapsed_sec:.3f}s")

    intron_rows = build_intron_scores(
        site_score_rows=unique_site_rows,
        intron_score_op=args.intron_score_op,
    )
    write_intron_scores(
        intron_output_tsv,
        intron_rows,
        labels=intron_labels,
    )
    print(f"Saved intron scores: {intron_output_tsv}")
    print(f"Total introns: {len(intron_rows)}")

    mapped_site_rows = _expand_unique_site_rows(
        site_score_rows=unique_site_rows,
        unique_map=unique_map,
    )

    if model_tasks == ("pair",):
        transcript_rows = aggregate_pair_transcript_scores(
            site_score_rows=mapped_site_rows,
            transcript_score_agg=args.transcript_score_agg,
            softmin_tau=args.softmin_tau,
        )
    else:
        transcript_rows = aggregate_transcript_scores(
            site_score_rows=mapped_site_rows,
            intron_score_op=args.intron_score_op,
            transcript_score_agg=args.transcript_score_agg,
            softmin_tau=args.softmin_tau,
        )
    write_transcript_scores(transcript_output_tsv, transcript_rows)
    print(f"Saved transcript scores: {transcript_output_tsv}")
    print(f"Total transcripts: {len(transcript_rows)}")

    ref_gff = _resolve_ref_gff_file(
        species=args.species,
        configured_path=args.ref_gff,
    )
    print(f"[pipeline] Evaluation reference GFF: {ref_gff}")

    output_lines = evaluate_score_file(
        class_file=class_file,
        score_file=transcript_output_tsv,
        ref_gff=ref_gff,
    )
    eval_out_dir = os.path.dirname(eval_output_txt)
    if eval_out_dir:
        os.makedirs(eval_out_dir, exist_ok=True)
    with open(eval_output_txt, "w", encoding="utf-8") as f:
        if output_lines:
            f.write("\n".join(output_lines))
            f.write("\n")
    print(f"Evaluation scores saved to {eval_output_txt}")

    if args.visualize != "none":
        print(
            "[pipeline] Plot request: "
            f"species={args.species} "
            f"x=({args.x_min}, {args.x_max}) "
            f"y=({args.y_min}, {args.y_max})"
        )
        plot_eval_scores(
            species=args.species,
            output_png=args.output_png,
            interactive=(args.visualize == "interactive"),
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI main entrypoint."""
    actual_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(actual_argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
