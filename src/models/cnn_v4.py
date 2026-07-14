"""Grouped deformable CNN v4 for independent splice-site scoring.

The model retains the independent donor/acceptor training contract of
:mod:`models.cnn_v3`.  Checkpoints are species-local, while its architectural
parameters can be supplied from one task-specific shared tuning configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dev.IntronModel.src.models import cnn, cnn_v3
from dev.IntronModel.src.util.model_runtime import (
    normalize_checkpoint_state_dict as _normalize_checkpoint_state_dict,
)


@dataclass(frozen=True)
class GroupedDeformableParams:
    """Validated grouped-deformable stem parameters for one site task."""

    groups: int
    kernel_size: int


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _resolve_task_deformable_params(
    task: str,
    model_args: argparse.Namespace,
) -> GroupedDeformableParams:
    """Resolve task-specific grouped/deformable stem settings."""
    if task not in {"donor", "acceptor"}:
        raise ValueError(f"Unsupported task: {task}")
    prefix = f"{task}_"
    groups = getattr(model_args, f"{prefix}deformable_groups", None)
    if groups is None:
        groups = getattr(model_args, "deformable_groups", 2)
    kernel_size = getattr(model_args, f"{prefix}deformable_kernel_size", None)
    if kernel_size is None:
        kernel_size = getattr(model_args, "deformable_kernel_size", 3)
    groups = _positive_int(groups, name=f"--{prefix}deformable_groups")
    kernel_size = _positive_int(
        kernel_size,
        name=f"--{prefix}deformable_kernel_size",
    )
    if kernel_size % 2 == 0:
        raise ValueError("--deformable_kernel_size must be odd.")
    return GroupedDeformableParams(groups=groups, kernel_size=kernel_size)


class DeformableConv1d(nn.Module):
    """A differentiable grouped deformable 1D convolution.

    This is a torch-native compatibility implementation.  Unlike silently
    replacing the operation with ``Conv1d`` when torchvision custom ops are
    unavailable, it samples offset positions with ``grid_sample`` on every
    supported PyTorch device.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if groups <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError(
                "groups must divide both in_channels and out_channels."
            )
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = kernel_size // 2
        self.offset = nn.Conv1d(in_channels, groups * kernel_size, kernel_size,
                                padding=self.padding, bias=True)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = (self.in_channels // self.groups) * self.kernel_size
            bound = 1.0 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Inputs must have shape (batch, channels, length).")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )
        batch_size, _, length = x.shape
        offsets = self.offset(x).view(
            batch_size, self.groups, self.kernel_size, length
        )
        base = torch.arange(length, device=x.device, dtype=x.dtype)
        kernel = torch.arange(self.kernel_size, device=x.device, dtype=x.dtype)
        positions = (
            base.view(1, 1, 1, length)
            + kernel.view(1, 1, self.kernel_size, 1)
            - self.padding
            + offsets
        )
        if length == 1:
            x_coords = torch.zeros_like(positions)
        else:
            x_coords = positions.mul(2.0 / (length - 1)).sub(1.0)
        grid = torch.stack((x_coords, torch.zeros_like(x_coords)), dim=-1)
        grid = grid.reshape(batch_size * self.groups, 1, self.kernel_size * length, 2)
        grouped_x = x.reshape(batch_size * self.groups, self.in_channels // self.groups, 1, length)
        sampled = F.grid_sample(
            grouped_x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(
            batch_size,
            self.groups,
            self.in_channels // self.groups,
            self.kernel_size,
            length,
        )
        grouped_weight = self.weight.reshape(
            self.groups,
            self.out_channels // self.groups,
            self.in_channels // self.groups,
            self.kernel_size,
        )
        output = torch.einsum("bgckl,gock->bgol", sampled, grouped_weight)
        output = output.reshape(batch_size, self.out_channels, length)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output


class GroupedDeformableStem(nn.Module):
    """Grouped pointwise convolution followed by grouped deformable sampling."""

    def __init__(self, *, groups: int, kernel_size: int) -> None:
        super().__init__()
        if groups <= 0 or 4 % groups:
            raise ValueError("--deformable_groups must divide the 4 one-hot channels.")
        self.groups = groups
        self.grouped_conv = nn.Conv1d(4, 4, kernel_size=1, groups=groups, bias=False)
        self.deformable_conv = DeformableConv1d(
            4,
            4,
            kernel_size,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(4)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.norm(self.deformable_conv(self.grouped_conv(x))))


class OrganicSiteCNN(nn.Module):
    """cnn_v3 encoder preceded by a grouped deformable one-hot stem."""

    def __init__(
        self,
        *,
        arch_params: cnn_v3.TaskOrganicArchParams,
        dropout: float,
        deformable_params: GroupedDeformableParams,
    ) -> None:
        super().__init__()
        self.stem = GroupedDeformableStem(
            groups=deformable_params.groups,
            kernel_size=deformable_params.kernel_size,
        )
        self.encoder = cnn_v3.ResidualDilatedBranchEncoder(
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
        if x.ndim != 3:
            raise ValueError("Inputs must have shape (batch, channels, length).")
        features = self.encoder(self.stem(x.float()))
        return self.fc(features)[:, 0]


def _checkpoint_extra(task: str, model_args: argparse.Namespace) -> dict[str, object]:
    params = _resolve_task_deformable_params(task, model_args)
    return {
        "site_arch": "organic_resdil_grouped_deformable",
        "deformable_groups": params.groups,
        "deformable_kernel_size": params.kernel_size,
    }


def _decorate_checkpoint(path: str, task: str, model_args: argparse.Namespace) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cnn_v4 checkpoint payload: {path}")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError(f"cnn_v4 checkpoint lacks model_config: {path}")
    config.update(_checkpoint_extra(task, model_args))
    payload["model_config"] = config
    torch.save(payload, path)


def train(common_args: argparse.Namespace, model_args: argparse.Namespace) -> Dict[str, object]:
    """Train species-local cnn_v4 donor/acceptor checkpoints."""
    def factory(*, arch_params: cnn_v3.TaskOrganicArchParams, dropout: float) -> OrganicSiteCNN:
        task = getattr(factory, "task")
        return OrganicSiteCNN(
            arch_params=arch_params,
            dropout=dropout,
            deformable_params=_resolve_task_deformable_params(task, model_args),
        )

    original = cnn_v3.OrganicSiteCNN
    cnn_v3.OrganicSiteCNN = factory  # type: ignore[assignment]
    try:
        # cnn_v3 invokes the constructor once per task, so attach the selected
        # task through its architecture resolver rather than sharing weights.
        original_resolver = cnn_v3._resolve_task_arch_params
        def resolver(task: str, *args: object, **kwargs: object) -> cnn_v3.TaskOrganicArchParams:
            factory.task = task  # type: ignore[attr-defined]
            return original_resolver(task, *args, **kwargs)
        cnn_v3._resolve_task_arch_params = resolver  # type: ignore[assignment]
        try:
            summary = cnn_v3.train(common_args, model_args)
        finally:
            cnn_v3._resolve_task_arch_params = original_resolver  # type: ignore[assignment]
    finally:
        cnn_v3.OrganicSiteCNN = original  # type: ignore[assignment]

    summary["model"] = "cnn_v4"
    for task in ("donor", "acceptor"):
        task_summary = summary.get(task)
        if isinstance(task_summary, dict):
            checkpoint = task_summary.get("checkpoint")
            if isinstance(checkpoint, str):
                _decorate_checkpoint(checkpoint, task, model_args)
            task_summary.update(_checkpoint_extra(task, model_args))
    return summary


def load_task_model(checkpoint_path: str, device: str) -> Tuple[nn.Module, Dict]:
    """Load one cnn_v4 checkpoint without accepting a v3 architecture silently."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid cnn_v4 checkpoint payload.")
    model_config = ckpt.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("cnn_v4 checkpoint lacks model_config.")
    if model_config.get("site_arch") != "organic_resdil_grouped_deformable":
        raise ValueError("Checkpoint is not a cnn_v4 grouped-deformable model.")
    task = str(ckpt.get("task", "donor")).strip().lower() or "donor"
    arch_params = cnn_v3._resolve_task_arch_params(task, argparse.Namespace(**model_config))
    params = GroupedDeformableParams(
        groups=_positive_int(model_config.get("deformable_groups"), name="deformable_groups"),
        kernel_size=_positive_int(model_config.get("deformable_kernel_size"), name="deformable_kernel_size"),
    )
    model = OrganicSiteCNN(
        arch_params=arch_params,
        dropout=float(model_config.get("dropout", 0.3)),
        deformable_params=params,
    ).to(device)
    model.load_state_dict(_normalize_checkpoint_state_dict(ckpt["model_state"]))
    model.eval()
    return model, ckpt


def score_sequences(*args: object, **kwargs: object):
    """Delegate sequence encoding and scoring to the shared CNN helper."""
    return cnn_v3.score_sequences(*args, **kwargs)


def infer_site_scores(*args: object, **kwargs: object) -> List[Dict[str, object]]:
    """Score cnn_v4 donor and acceptor checkpoints."""
    original = cnn_v3.load_task_model
    cnn_v3.load_task_model = load_task_model  # type: ignore[assignment]
    try:
        return cnn_v3.infer_site_scores(*args, **kwargs)
    finally:
        cnn_v3.load_task_model = original  # type: ignore[assignment]


def infer_site(common_args: argparse.Namespace, model_args: argparse.Namespace) -> List[Dict[str, object]]:
    """Run unified site inference for cnn_v4 checkpoints."""
    original = cnn_v3.infer_site_scores
    cnn_v3.infer_site_scores = infer_site_scores  # type: ignore[assignment]
    try:
        return cnn_v3.infer_site(common_args, model_args)
    finally:
        cnn_v3.infer_site_scores = original  # type: ignore[assignment]


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register cnn_v4 training arguments."""
    cnn_v3.add_train_args(parser)
    parser.add_argument("--deformable_groups", type=int, default=2)
    parser.add_argument("--donor_deformable_groups", type=int, default=None)
    parser.add_argument("--acceptor_deformable_groups", type=int, default=None)
    parser.add_argument("--deformable_kernel_size", type=int, default=3)
    parser.add_argument("--donor_deformable_kernel_size", type=int, default=None)
    parser.add_argument("--acceptor_deformable_kernel_size", type=int, default=None)


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register cnn_v4 inference arguments."""
    cnn_v3.add_infer_args(parser)
