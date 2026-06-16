"""Residual-dilated CNN v3 for site-level splice scoring.

This module implements the independent ``cnn_v3`` variant. Donor and acceptor
tasks are trained as separate site classifiers that share one residual-dilated
CNN design but keep their own checkpoints.
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

from models import cnn
from models.cnn_common import score_sequences as _score_sequences
from models.cnn_pair_v3 import (
    OrganicBranchLayout,
    ResidualDilatedBranchEncoder,
    _align_positive_int_list,
    _coerce_optional_int_list,
    _coerce_positive_int,
    _default_dilation_schedule,
    _default_residual_channels,
)
from util.data_proc import (
    build_run_name,
    infer_default_train_paths,
    read_test_site_rows,
    resolve_effective_window_lengths,
    resolve_test_tsv,
    resolve_train_paths,
    species_data_dirs,
    validate_window_args,
)
from util.losses import build_binary_classification_loss
from util.model_task_paths import (
    attach_init_checkpoint_summary,
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
    is_compile_runtime_error as _is_compile_runtime_error,
    is_cuda_oom_error as _is_cuda_oom_error,
    is_mps_oom_error as _is_mps_oom_error,
    normalize_checkpoint_state_dict as _normalize_checkpoint_state_dict,
    pick_device,
    resolve_amp_dtype as _resolve_amp_dtype,
    resolve_compile_enabled as _resolve_compile_enabled,
    resolve_num_workers as _resolve_num_workers,
    record_compile_runtime_failure as _record_compile_runtime_failure,
    seed_worker as _seed_worker,
    set_seed,
    warm_start_model as _warm_start_model,
)
from util.process_title import (
    apply_eta_process_title_from_epoch_progress,
    apply_eta_process_title_placeholder,
)
from util.training_control import (
    get_metric_value,
    resolve_training_schedule,
    resolve_validation_metric,
    select_validation_score,
)
from util.transcript_eval import SCORE_SPACE_FIELD, SCORE_SPACE_LOG10


@dataclass(frozen=True)
class TaskOrganicArchParams:
    """Resolved residual-dilated architecture for one site classifier."""

    layout: OrganicBranchLayout
    head_type: str
    fc_hidden: int


def _resolve_task_arch_params(
    task: str,
    model_args: argparse.Namespace,
    *,
    lightweight: bool = False,
) -> TaskOrganicArchParams:
    """Resolve one donor or acceptor residual-dilated architecture."""
    if task not in {"donor", "acceptor"}:
        raise ValueError(f"Unsupported task: {task}")

    prefix = f"{task}_"

    def _override_or_default(name: str, default: object) -> object:
        override = getattr(model_args, f"{prefix}{name}", None)
        return default if override is None else override

    shared_channels = _coerce_optional_int_list(
        getattr(model_args, "conv_channels", None),
        arg_name="--conv_channels",
    )
    shared_kernel_sizes = _coerce_optional_int_list(
        getattr(model_args, "kernel_sizes", None),
        arg_name="--kernel_sizes",
    )
    if shared_kernel_sizes is None and getattr(model_args, "kernel_size", None):
        shared_kernel_sizes = [
            _coerce_positive_int(
                getattr(model_args, "kernel_size"),
                arg_name="--kernel_size",
            )
        ]
    shared_dilations = _coerce_optional_int_list(
        getattr(model_args, "block_dilations", None),
        arg_name="--block_dilations",
    )
    shared_residual_channels = _coerce_optional_int_list(
        getattr(model_args, "residual_channels", None),
        arg_name="--residual_channels",
    )

    branch_channels = _coerce_optional_int_list(
        getattr(model_args, f"{task}_conv_channels", None),
        arg_name=f"--{task}_conv_channels",
    )
    if branch_channels is None:
        branch_channels = shared_channels
    if branch_channels is None:
        branch_channels = [48, 96, 192] if lightweight else [64, 128, 256, 384]
    depth = len(branch_channels)

    branch_kernel_sizes = _coerce_optional_int_list(
        getattr(model_args, f"{task}_kernel_sizes", None),
        arg_name=f"--{task}_kernel_sizes",
    )
    if branch_kernel_sizes is None and getattr(model_args, f"{task}_kernel_size", None):
        branch_kernel_sizes = [
            _coerce_positive_int(
                getattr(model_args, f"{task}_kernel_size"),
                arg_name=f"--{task}_kernel_size",
            )
        ]
    if branch_kernel_sizes is None:
        branch_kernel_sizes = shared_kernel_sizes
    kernel_sizes = _align_positive_int_list(
        branch_kernel_sizes,
        depth=depth,
        arg_name=f"--{task}_kernel_sizes",
        require_odd=True,
        default_values=[9, 7, 5, 5],
    )

    branch_dilations = _coerce_optional_int_list(
        getattr(model_args, f"{task}_block_dilations", None),
        arg_name=f"--{task}_block_dilations",
    )
    if branch_dilations is None:
        branch_dilations = shared_dilations
    dilations = _align_positive_int_list(
        branch_dilations,
        depth=depth,
        arg_name=f"--{task}_block_dilations",
        default_values=_default_dilation_schedule(depth),
    )

    branch_residual_channels = _coerce_optional_int_list(
        getattr(model_args, f"{task}_residual_channels", None),
        arg_name=f"--{task}_residual_channels",
    )
    if branch_residual_channels is None:
        branch_residual_channels = shared_residual_channels
    residual_channels = _align_positive_int_list(
        branch_residual_channels,
        depth=depth,
        arg_name=f"--{task}_residual_channels",
        default_values=_default_residual_channels(branch_channels),
    )

    head_type = cnn._normalize_cnn_head_type(
        _override_or_default("head_type", getattr(model_args, "head_type", "gap")),
        arg_name=f"--{prefix}head_type",
    )
    fc_hidden = _coerce_positive_int(
        _override_or_default("fc_hidden", getattr(model_args, "fc_hidden", 192)),
        arg_name=f"--{prefix}fc_hidden",
    )
    return TaskOrganicArchParams(
        layout=OrganicBranchLayout(
            channels=list(branch_channels),
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            residual_channels=residual_channels,
        ),
        head_type=head_type,
        fc_hidden=fc_hidden,
    )


class OrganicSiteCNN(nn.Module):
    """Residual-dilated site classifier for one splice-site task."""

    def __init__(
        self,
        *,
        arch_params: TaskOrganicArchParams,
        dropout: float,
    ) -> None:
        super().__init__()
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        self.encoder = ResidualDilatedBranchEncoder(
            in_channels=4,
            layout=arch_params.layout,
            head_type=arch_params.head_type,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.output_dim, arch_params.fc_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(arch_params.fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, 4, length)``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch,)``.
        """
        if x.ndim != 3:
            raise ValueError("Inputs must have shape (batch, channels, length).")
        features = self.encoder(x.float())
        return self.fc(features)[:, 0]


def train_task_model(
    *,
    task: str,
    pos_path: str,
    neg_path: str,
    checkpoint_path: str,
    init_checkpoint_path: Optional[str],
    window_len: int,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    model_args: argparse.Namespace,
    task_params: cnn.TaskTrainParams,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
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
    sequence_transform: str = "none",
    report_train_metrics: Union[bool, int] = 1,
    gpu_id: Optional[int] = None,
) -> Dict[str, object]:
    """Train one residual-dilated donor or acceptor model."""
    arch_params = _resolve_task_arch_params(
        task=task,
        model_args=model_args,
        lightweight=lightweight,
    )
    if task_params.dropout < 0.0 or task_params.dropout >= 1.0:
        raise ValueError("--dropout must satisfy 0 <= dropout < 1.")
    if task_params.weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative.")
    if task_params.eta_min_ratio < 0.0:
        raise ValueError("--eta_min_ratio must be non-negative.")
    if task_params.val_frac <= 0.0 or task_params.val_frac >= 1.0:
        raise ValueError("--val_frac must satisfy 0 < val_frac < 1.")
    if task_params.grad_clip < 0.0:
        raise ValueError("--grad_clip must be non-negative.")
    if task_params.f1_lambda < 0.0:
        raise ValueError("--f1_lambda must be non-negative.")
    if prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive.")
    if min_batch_size <= 0:
        raise ValueError("--min_batch_size must be positive.")
    if max_oom_retries < 0:
        raise ValueError("--max_oom_retries must be >= 0.")
    if task_params.batch_size < min_batch_size:
        raise ValueError("--batch_size must be >= --min_batch_size.")
    if init_checkpoint_path is not None and init_checkpoint_path.strip() == "":
        init_checkpoint_path = None
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
    report_train_metrics_bool = _bool_from_flag(report_train_metrics)

    set_seed(
        seed=seed,
        deterministic=deterministic_bool,
        cudnn_benchmark=cudnn_benchmark_bool,
        allow_tf32=allow_tf32_bool,
    )
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    examples = cnn._load_task_examples_with_transform(
        pos_path=pos_path,
        neg_path=neg_path,
        task=task,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        sequence_transform=sequence_transform,
    )
    n_pos = sum(label for _, label in examples)
    n_neg = len(examples) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Insufficient training examples for {task}: pos={n_pos}, neg={n_neg}."
        )

    train_ex, val_ex = cnn.stratified_split(
        examples,
        val_frac=task_params.val_frac,
        seed=seed,
    )
    print(
        f"[cnn_v3:{task}] device={device} total={len(examples)} "
        f"(pos={n_pos}, neg={n_neg}) train={len(train_ex)} val={len(val_ex)}"
    )
    preencode_dataset = device == "mps"
    if preencode_dataset:
        print(f"[cnn_v3:{task}] dataset pre-encoding enabled for mps.")
    train_ds = cnn.DNADataset(
        train_ex,
        window_len=window_len,
        preencode=preencode_dataset,
    )
    val_ds = cnn.DNADataset(
        val_ex,
        window_len=window_len,
        preencode=preencode_dataset,
    )

    train_pos = sum(label for _, label in train_ex)
    train_neg = len(train_ex) - train_pos
    criterion, loss_meta = build_binary_classification_loss(
        loss_name=task_params.loss_name,
        train_pos=train_pos,
        train_neg=train_neg,
        device=device,
        pos_weight_cap=task_params.pos_weight_cap,
        focal_gamma=task_params.focal_gamma,
        focal_alpha_pos=task_params.focal_alpha_pos,
        asym_gamma_pos=task_params.asym_gamma_pos,
        asym_gamma_neg=task_params.asym_gamma_neg,
        asym_alpha_pos=task_params.asym_alpha_pos,
        f1_lambda=task_params.f1_lambda,
    )

    effective_batch_size = task_params.batch_size
    if device == "mps":
        mps_max_batch_size = cnn._resolve_mps_max_batch_size()
        if effective_batch_size > mps_max_batch_size:
            print(
                f"[cnn_v3:{task}] mps batch clamp: {effective_batch_size} -> "
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
        fixed_shape_loader = compile_enabled
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)

        train_loader_batch_size, train_loader_drop_last = (
            cnn._resolve_loader_batch_size_and_drop_last(
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
            cnn._resolve_loader_batch_size_and_drop_last(
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
                cnn._resolve_loader_batch_size_and_drop_last(
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
            f"[cnn_v3:{task}] loader train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} batch_size={effective_batch_size} "
            f"workers={resolved_num_workers} "
            f"train_eval={'on' if report_train_metrics_bool else 'off'} "
            f"fixed_shape={'on' if fixed_shape_loader else 'off'}"
        )

        try:
            model = OrganicSiteCNN(
                arch_params=arch_params,
                dropout=task_params.dropout,
            ).to(device)
            warm_start_result = _warm_start_model(
                model,
                init_checkpoint_path=init_checkpoint_path,
                device=device,
                log_prefix=f"cnn_v3:{task}",
            )
            initialized_from_checkpoint = (
                warm_start_result.initialized_from_checkpoint
            )
            init_checkpoint_path = warm_start_result.init_checkpoint_path

            if compile_enabled_attempt:
                _configure_triton_tool_paths()
                _configure_torch_compile_runtime()
                ptxas_path = os.environ.get("TRITON_PTXAS_PATH")
                ptxas_blackwell_path = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH")
                print(
                    f"[cnn_v3:{task}] torch.compile requested "
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
                        f"[cnn_v3:{task}] torch.compile setup failed "
                        f"({compile_setup_error.__class__.__name__}). "
                        "Continue without compile."
                    )

            optimizer_impl = "adamw"
            adamw_kwargs: dict[str, object] = {
                "params": model.parameters(),
                "lr": task_params.lr,
                "weight_decay": task_params.weight_decay,
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
                eta_min=task_params.lr * task_params.eta_min_ratio,
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
                running_loss = 0.0

                for x, y in train_loader:
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
                        if task_params.grad_clip > 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                task_params.grad_clip,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if task_params.grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                task_params.grad_clip,
                            )
                        optimizer.step()
                    running_loss += float(loss.detach().item())

                scheduler.step()
                train_loss = float(running_loss / max(1, len(train_loader)))
                val_metrics = cnn.evaluate(
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
                    train_metrics = cnn.evaluate(
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
                            "task": task,
                            "window_len": window_len,
                            "model_config": {
                                "site_arch": "organic_resdil",
                                "conv_channels": list(arch_params.layout.channels),
                                "kernel_sizes": list(arch_params.layout.kernel_sizes),
                                "block_dilations": list(arch_params.layout.dilations),
                                "residual_channels": list(
                                    arch_params.layout.residual_channels
                                ),
                                "head_type": arch_params.head_type,
                                "dropout": task_params.dropout,
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
                        f"[cnn_v3:{task}] {mark} epoch {epoch}/{epochs} "
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
                        f"[cnn_v3:{task}] early stop at epoch {epoch} "
                        f"(patience={early_stop_patience}, "
                        f"min_delta={early_stop_min_delta:g})"
                    )
                    break

            print(
                f"[cnn_v3:{task}] done best_{best_metric_name}={best_score:.4f} "
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
                "validation_metric": resolved_validation_metric,
                "checkpoint": checkpoint_path,
                "loss": task_params.loss_name,
                "pos_weight": loss_meta["pos_weight"],
                "focal_gamma": loss_meta["focal_gamma"],
                "focal_alpha_pos": loss_meta["focal_alpha_pos"],
                "asym_gamma_pos": loss_meta["asym_gamma_pos"],
                "asym_gamma_neg": loss_meta["asym_gamma_neg"],
                "asym_alpha_pos": loss_meta["asym_alpha_pos"],
                "f1_lambda": loss_meta["f1_lambda"],
                "conv_channels": list(arch_params.layout.channels),
                "kernel_sizes": list(arch_params.layout.kernel_sizes),
                "block_dilations": list(arch_params.layout.dilations),
                "residual_channels": list(arch_params.layout.residual_channels),
                "head_type": arch_params.head_type,
                "fc_hidden": arch_params.fc_hidden,
                "dropout": task_params.dropout,
                "weight_decay": task_params.weight_decay,
                "eta_min_ratio": task_params.eta_min_ratio,
                "val_frac": task_params.val_frac,
                "grad_clip": task_params.grad_clip,
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
                "report_train_metrics": report_train_metrics_bool,
                "optimizer_impl": optimizer_impl,
                "sequence_transform": sequence_transform,
                "initialized_from_checkpoint": initialized_from_checkpoint,
                "init_checkpoint_path": init_checkpoint_path,
            }
        except RuntimeError as exc:
            is_compile_failure = compile_enabled_attempt and _is_compile_runtime_error(
                exc
            )
            if is_compile_failure:
                compile_enabled = False
                _record_compile_runtime_failure(compile_selected_mode)
                print(
                    f"[cnn_v3:{task}] torch.compile runtime failed "
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
                f"[cnn_v3:{task}] {device.upper()} OOM detected. "
                "Retry with smaller batch size: "
                f"{effective_batch_size} -> {next_batch_size} "
                f"(retry {oom_retries}/{max_oom_retries})"
            )
            effective_batch_size = next_batch_size
            _empty_device_cache(device)


def load_task_model(checkpoint_path: str, device: str) -> Tuple[nn.Module, Dict]:
    """Load one trained ``cnn_v3`` site model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _normalize_checkpoint_state_dict(ckpt["model_state"])
    model_config = ckpt.get("model_config", {})
    task = str(ckpt.get("task", "donor")).strip().lower() or "donor"
    dropout = float(model_config.get("dropout", 0.3))
    arch_params = _resolve_task_arch_params(task, argparse.Namespace(**model_config))
    model = OrganicSiteCNN(
        arch_params=arch_params,
        dropout=dropout,
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
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> np.ndarray:
    """Score sequences with one trained ``cnn_v3`` site model."""
    return _score_sequences(
        model=model,
        sequences=sequences,
        window_len=window_len,
        device=device,
        batch_size=batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )


def infer_site_scores(
    site_rows: List[Dict[str, object]],
    donor_model_path: str,
    acceptor_model_path: str,
    device: str = "auto",
    batch_size: int = 512,
    sequence_transform: str = "none",
    infer_use_amp: Union[bool, int] = 0,
    infer_amp_dtype: str = "auto",
    infer_compile: Union[bool, int] = 0,
    infer_compile_mode: str = "off",
) -> List[Dict[str, object]]:
    """Run donor/acceptor site scoring with ``cnn_v3`` checkpoints."""
    if sequence_transform not in cnn.SEQUENCE_TRANSFORM_CHOICES:
        raise ValueError(
            "Unsupported --sequence_transform: "
            f"{sequence_transform}. Supported: {cnn.SEQUENCE_TRANSFORM_CHOICES}"
        )
    device_name = pick_device(device)
    infer_runtime = cnn._resolve_infer_runtime_config(
        device=device_name,
        batch_size=batch_size,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
        infer_compile=infer_compile,
        infer_compile_mode=infer_compile_mode,
    )

    donor_model, donor_ckpt = load_task_model(donor_model_path, device_name)
    acceptor_model, acceptor_ckpt = load_task_model(acceptor_model_path, device_name)
    donor_model = cnn._prepare_infer_model(
        model=donor_model,
        task_name="donor",
        compile_enabled=infer_runtime.compile_enabled,
        compile_mode=infer_compile_mode,
    )
    acceptor_model = cnn._prepare_infer_model(
        model=acceptor_model,
        task_name="acceptor",
        compile_enabled=infer_runtime.compile_enabled,
        compile_mode=infer_compile_mode,
    )

    donor_window_len = int(donor_ckpt.get("window_len", 50))
    acceptor_window_len = int(acceptor_ckpt.get("window_len", 50))

    transformed_rows: List[Dict[str, object]] = []
    for row in site_rows:
        site_type = str(row["site_type"])
        transformed_seq = cnn.apply_site_sequence_transform(
            str(row["seq"]),
            site_type=site_type,
            transform_mode=sequence_transform,
            intron_half_length=(
                int(row["intron_half_length"])
                if row.get("intron_half_length") is not None
                else None
            ),
        )
        next_row = dict(row)
        next_row["seq"] = transformed_seq
        transformed_rows.append(next_row)

    donor_seqs = [
        str(row["seq"]) for row in transformed_rows if row["site_type"] == "donor"
    ]
    acceptor_seqs = [
        str(row["seq"]) for row in transformed_rows if row["site_type"] == "acceptor"
    ]

    donor_scores = score_sequences(
        donor_model,
        donor_seqs,
        donor_window_len,
        device_name,
        batch_size=infer_runtime.batch_size,
        use_amp=infer_runtime.use_amp,
        amp_dtype=infer_runtime.amp_dtype,
    )
    acceptor_scores = score_sequences(
        acceptor_model,
        acceptor_seqs,
        acceptor_window_len,
        device_name,
        batch_size=infer_runtime.batch_size,
        use_amp=infer_runtime.use_amp,
        amp_dtype=infer_runtime.amp_dtype,
    )
    if len(donor_scores) != len(donor_seqs):
        raise ValueError("Donor score count does not match donor sequence count.")
    if len(acceptor_scores) != len(acceptor_seqs):
        raise ValueError("Acceptor score count does not match acceptor sequence count.")

    out_rows: List[Dict[str, object]] = []
    donor_index = 0
    acceptor_index = 0
    for row in transformed_rows:
        site_type = str(row["site_type"])
        if site_type == "donor":
            score = float(donor_scores[donor_index])
            donor_index += 1
        else:
            score = float(acceptor_scores[acceptor_index])
            acceptor_index += 1
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


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register ``cnn_v3`` training arguments."""
    cnn.add_train_args(parser)
    parser.set_defaults(validation_metric="pr_auc")
    parser.add_argument(
        "--pair_mode",
        choices=["pair", "independent"],
        default="independent",
        help=(
            "Compatibility flag for shared tuning infrastructure. "
            "cnn_v3 always runs in independent donor/acceptor mode."
        ),
    )
    parser.add_argument(
        "--block_dilations",
        type=str,
        default=None,
        help="Shared per-block dilations, e.g. 1,2,4,8.",
    )
    parser.add_argument("--donor_block_dilations", type=str, default=None)
    parser.add_argument("--acceptor_block_dilations", type=str, default=None)
    parser.add_argument(
        "--residual_channels",
        type=str,
        default=None,
        help="Shared bottleneck channels inside residual blocks.",
    )
    parser.add_argument("--donor_residual_channels", type=str, default=None)
    parser.add_argument("--acceptor_residual_channels", type=str, default=None)


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register ``cnn_v3`` inference arguments."""
    cnn.add_infer_args(parser)


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train donor/acceptor ``cnn_v3`` models with the unified runtime."""
    shared_conv_channels = cnn.parse_conv_channels(model_args.conv_channels)
    donor_conv_channels = cnn.parse_conv_channels(
        getattr(model_args, "donor_conv_channels", None),
        arg_name="--donor_conv_channels",
    )
    acceptor_conv_channels = cnn.parse_conv_channels(
        getattr(model_args, "acceptor_conv_channels", None),
        arg_name="--acceptor_conv_channels",
    )
    shared_kernel_sizes = cnn.parse_kernel_sizes(
        getattr(model_args, "kernel_sizes", None),
        arg_name="--kernel_sizes",
    )
    donor_kernel_sizes = cnn.parse_kernel_sizes(
        getattr(model_args, "donor_kernel_sizes", None),
        arg_name="--donor_kernel_sizes",
    )
    acceptor_kernel_sizes = cnn.parse_kernel_sizes(
        getattr(model_args, "acceptor_kernel_sizes", None),
        arg_name="--acceptor_kernel_sizes",
    )
    if donor_kernel_sizes is None and getattr(model_args, "donor_kernel_size", None):
        donor_kernel_sizes = [int(getattr(model_args, "donor_kernel_size"))]
    if acceptor_kernel_sizes is None and getattr(
        model_args, "acceptor_kernel_size", None
    ):
        acceptor_kernel_sizes = [int(getattr(model_args, "acceptor_kernel_size"))]
    if shared_kernel_sizes is None and getattr(model_args, "kernel_size", None):
        shared_kernel_sizes = [int(model_args.kernel_size)]

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
    )
    donor_checkpoint_path = task_checkpoint_paths["donor"]
    acceptor_checkpoint_path = task_checkpoint_paths["acceptor"]
    train_target = resolve_train_target(model_args)

    schedule = resolve_training_schedule(
        epochs_arg=model_args.epochs,
        max_epochs=int(model_args.max_epochs),
        patience_arg=model_args.early_stop_patience,
        min_delta_arg=model_args.early_stop_min_delta,
    )

    tasks_to_train = resolve_tasks_to_train(train_target)
    task_window_len = {"donor": donor_window_len, "acceptor": acceptor_window_len}

    task_hparams: dict[str, cnn.TaskTrainParams] = {}
    task_metrics: dict[str, Dict[str, object]] = {}
    task_init_checkpoint_paths = resolve_task_init_checkpoint_paths(common_args)
    for task in tasks_to_train:
        resolved = cnn._resolve_task_train_params(
            task=task,
            model_args=model_args,
            shared_conv_channels=shared_conv_channels,
            donor_conv_channels=donor_conv_channels,
            acceptor_conv_channels=acceptor_conv_channels,
            shared_kernel_sizes=shared_kernel_sizes,
            donor_kernel_sizes=donor_kernel_sizes,
            acceptor_kernel_sizes=acceptor_kernel_sizes,
        )
        task_hparams[task] = resolved
        task_metrics[task] = train_task_model(
            task=task,
            pos_path=train_pos_path,
            neg_path=train_neg_path,
            checkpoint_path=task_checkpoint_paths[task],
            init_checkpoint_path=task_init_checkpoint_paths[task],
            window_len=task_window_len[task],
            donor_len=donor_len,
            acceptor_len=acceptor_len,
            model_args=model_args,
            task_params=resolved,
            epochs=schedule.resolved_epochs,
            early_stop_patience=schedule.effective_early_stop_patience,
            early_stop_min_delta=schedule.early_stop_min_delta,
            validation_metric=model_args.validation_metric,
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
            sequence_transform=model_args.sequence_transform,
            quick_phase=bool(getattr(common_args, "quick_phase", False)),
            report_train_metrics=model_args.report_train_metrics,
            gpu_id=getattr(common_args, "gpu_id", None),
        )

    run_name_lr = model_args.lr
    run_name_batch_size = model_args.batch_size
    if train_target != "both":
        selected_params = task_hparams[tasks_to_train[0]]
        run_name_lr = selected_params.lr
        run_name_batch_size = selected_params.batch_size
    run_name = build_run_name(
        model_name="cnn_v3",
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        lr=run_name_lr,
        batch_size=run_name_batch_size,
        epochs=schedule.resolved_epochs,
        tag=model_args.tag,
    )

    summary: Dict[str, object] = {
        "model": "cnn_v3",
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
        "validation_metric": resolve_validation_metric(model_args.validation_metric),
        "batch_size": model_args.batch_size,
        "lr": model_args.lr,
        "train_target": train_target,
        "sequence_transform": model_args.sequence_transform,
        "seed": common_args.seed,
        "device": common_args.device,
        "checkpoint_name": os.path.basename(donor_checkpoint_path),
        "donor_checkpoint_path": donor_checkpoint_path,
        "acceptor_checkpoint_path": acceptor_checkpoint_path,
        "lightweight": model_args.lightweight,
        "conv_channels": (
            None if shared_conv_channels is None else list(shared_conv_channels)
        ),
        "kernel_sizes": (
            None if shared_kernel_sizes is None else list(shared_kernel_sizes)
        ),
        "block_dilations": _coerce_optional_int_list(
            getattr(model_args, "block_dilations", None),
            arg_name="--block_dilations",
        ),
        "residual_channels": _coerce_optional_int_list(
            getattr(model_args, "residual_channels", None),
            arg_name="--residual_channels",
        ),
        "head_type": cnn._normalize_cnn_head_type(
            model_args.head_type,
            arg_name="--head_type",
        ),
        "donor_kernel_sizes": (
            None if donor_kernel_sizes is None else list(donor_kernel_sizes)
        ),
        "acceptor_kernel_sizes": (
            None if acceptor_kernel_sizes is None else list(acceptor_kernel_sizes)
        ),
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
        "report_train_metrics": bool(model_args.report_train_metrics),
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
    }
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
    )
    donor_model_path = task_checkpoint_paths["donor"]
    acceptor_model_path = task_checkpoint_paths["acceptor"]
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
        sequence_transform=model_args.sequence_transform,
        infer_use_amp=infer_use_amp,
        infer_amp_dtype=infer_amp_dtype,
        infer_compile=infer_compile,
        infer_compile_mode=infer_compile_mode,
    )
