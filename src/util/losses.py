"""Shared loss functions for binary classification site models.

This module centralizes reusable loss implementations so model files can focus
on architecture and training flow. It currently provides:

- BCE
- weighted BCE
- focal loss
- asymmetric focal loss (recommended for heavily imbalanced negatives)
- soft F1 loss
- weighted BCE + soft F1 mixed loss
- focal + soft F1 mixed loss
"""

from __future__ import annotations

import math
from typing import Optional, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import sigmoid_focal_loss as torchvision_sigmoid_focal_loss
except ImportError:  # pragma: no cover
    torchvision_sigmoid_focal_loss = None

LOSS_NAME_CHOICES: tuple[str, ...] = (
    "bce",
    "weighted_bce",
    "focal",
    "asymmetric_focal",
    "f1",
    "weighted_bce_f1",
    "focal_f1",
)


class LossMeta(TypedDict):
    """Metadata for configured loss behavior."""

    pos_weight: float
    focal_gamma: float
    focal_alpha_pos: float
    asym_gamma_pos: float
    asym_gamma_neg: float
    asym_alpha_pos: float
    f1_lambda: float


class BinaryFocalLoss(nn.Module):
    """Binary focal loss for imbalanced classification.

    If available, this uses ``torchvision.ops.sigmoid_focal_loss``.
    Otherwise, it falls back to an equivalent local implementation.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter. Higher values down-weight easy samples more.
    alpha_pos : float, default=0.75
        Class weight for positive targets. Negative class weight is
        ``1 - alpha_pos``.
    reduction : str, default="mean"
        One of ``mean``, ``sum``, or ``none``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_pos: float = 0.75,
        reduction: str = "mean",
    ) -> None:
        """Initialize focal-loss module."""
        super().__init__()
        if gamma < 0.0:
            raise ValueError("gamma must be >= 0.0")
        if not (0.0 < alpha_pos < 1.0):
            raise ValueError("alpha_pos must satisfy 0 < alpha_pos < 1")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits of shape ``(N,)``.
        targets : torch.Tensor
            Binary labels of shape ``(N,)`` with values in ``{0, 1}``.

        Returns
        -------
        torch.Tensor
            Reduced focal loss scalar (or per-sample tensor for ``none``).
        """
        targets = targets.float()
        if torchvision_sigmoid_focal_loss is not None:
            return torchvision_sigmoid_focal_loss(
                inputs=logits,
                targets=targets,
                alpha=self.alpha_pos,
                gamma=self.gamma,
                reduction=self.reduction,
            )

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha_pos * targets + (1.0 - self.alpha_pos) * (
            1.0 - targets
        )
        focal_factor = torch.pow(1.0 - pt, self.gamma)
        loss = alpha_t * focal_factor * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricBinaryFocalLoss(nn.Module):
    """Asymmetric focal loss with separate focusing for positive and negative.

    This variant is useful for severe class imbalance where easy negatives
    dominate gradients. Setting ``gamma_neg`` higher than ``gamma_pos``
    suppresses easy negatives more aggressively.

    Parameters
    ----------
    gamma_pos : float, default=0.0
        Focusing parameter for positive samples.
    gamma_neg : float, default=4.0
        Focusing parameter for negative samples.
    alpha_pos : float, default=0.75
        Positive class weight. Negative class uses ``1 - alpha_pos``.
    reduction : str, default="mean"
        One of ``mean``, ``sum``, or ``none``.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        alpha_pos: float = 0.75,
        reduction: str = "mean",
    ) -> None:
        """Initialize asymmetric focal-loss module."""
        super().__init__()
        if gamma_pos < 0.0 or gamma_neg < 0.0:
            raise ValueError("gamma_pos and gamma_neg must be >= 0.0")
        if not (0.0 < alpha_pos < 1.0):
            raise ValueError("alpha_pos must satisfy 0 < alpha_pos < 1")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha_pos = alpha_pos
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric focal loss.

        Complexity
        ----------
        O(N), where N is the number of samples in the batch.
        """
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        gamma_t = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        alpha_t = self.alpha_pos * targets + (1.0 - self.alpha_pos) * (
            1.0 - targets
        )
        loss = alpha_t * torch.pow(1.0 - pt, gamma_t) * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SoftBinaryF1Loss(nn.Module):
    """Differentiable soft F1 loss from logits.

    This computes a smooth approximation of the positive-class F1 score:

    ``f1 = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)``

    where soft counts are computed from probabilities
    ``p = sigmoid(logits)``.

    Parameters
    ----------
    smooth : float, default=1e-7
        Numerical stabilizer added to numerator and denominator.
    reduction : str, default="mean"
        Only ``mean`` is supported because F1 is computed batch-wise.

    Notes
    -----
    Complexity is O(N), where N is batch size.
    """

    def __init__(self, smooth: float = 1e-7, reduction: str = "mean") -> None:
        """Initialize soft F1 loss."""
        super().__init__()
        if smooth <= 0.0:
            raise ValueError("smooth must be > 0.0")
        if reduction != "mean":
            raise ValueError("SoftBinaryF1Loss only supports reduction='mean'")
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute batch soft F1 loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits of shape ``(N,)``.
        targets : torch.Tensor
            Binary labels of shape ``(N,)`` with values in ``{0, 1}``.

        Returns
        -------
        torch.Tensor
            Scalar loss ``1 - soft_f1``.
        """
        probs = torch.sigmoid(logits).float().reshape(-1)
        labels = targets.float().reshape(-1)
        tp = torch.sum(labels * probs)
        fp = torch.sum((1.0 - labels) * probs)
        fn = torch.sum(labels * (1.0 - probs))
        f1 = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        return 1.0 - f1


class WeightedBceSoftF1Loss(nn.Module):
    """Weighted BCE + soft F1 mixed loss.

    This loss is defined as:

    ``loss = weighted_bce + f1_lambda * soft_f1_loss``

    Parameters
    ----------
    pos_weight : torch.Tensor
        Positive class weight tensor of shape ``(1,)``.
    f1_lambda : float, default=0.1
        Mixing coefficient for the soft F1 term.
    f1_smooth : float, default=1e-7
        Numerical stabilizer used by the soft F1 term.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor,
        f1_lambda: float = 0.1,
        f1_smooth: float = 1e-7,
    ) -> None:
        """Initialize weighted BCE + soft F1 mixed loss."""
        super().__init__()
        if pos_weight.numel() != 1:
            raise ValueError("pos_weight must contain exactly one scalar value.")
        if not math.isfinite(f1_lambda) or f1_lambda < 0.0:
            raise ValueError("f1_lambda must be a finite non-negative float.")
        self.register_buffer(
            "pos_weight",
            pos_weight.detach().clone().reshape(1).to(dtype=torch.float32),
        )
        self.f1_lambda = float(f1_lambda)
        self.soft_f1_loss = SoftBinaryF1Loss(smooth=f1_smooth, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE + soft F1 mixed loss.

        Complexity
        ----------
        O(N), where N is the number of samples in the batch.
        """
        labels = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight,
            reduction="mean",
        )
        soft_f1 = self.soft_f1_loss(logits, labels)
        return bce + self.f1_lambda * soft_f1


class FocalSoftF1Loss(nn.Module):
    """Focal + soft F1 mixed loss.

    This loss is defined as:

    ``loss = focal_loss + f1_lambda * soft_f1_loss``

    Parameters
    ----------
    gamma : float, default=2.0
        Focal-loss focusing parameter.
    alpha_pos : float, default=0.75
        Positive-class weight used by focal loss.
    f1_lambda : float, default=0.1
        Mixing coefficient for the soft F1 term.
    f1_smooth : float, default=1e-7
        Numerical stabilizer used by the soft F1 term.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_pos: float = 0.75,
        f1_lambda: float = 0.1,
        f1_smooth: float = 1e-7,
    ) -> None:
        """Initialize focal + soft F1 mixed loss."""
        super().__init__()
        if not math.isfinite(f1_lambda) or f1_lambda < 0.0:
            raise ValueError("f1_lambda must be a finite non-negative float.")
        self.focal_loss = BinaryFocalLoss(
            gamma=gamma,
            alpha_pos=alpha_pos,
            reduction="mean",
        )
        self.soft_f1_loss = SoftBinaryF1Loss(smooth=f1_smooth, reduction="mean")
        self.f1_lambda = float(f1_lambda)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal + soft F1 mixed loss.

        Complexity
        ----------
        O(N), where N is the number of samples in the batch.
        """
        labels = targets.float()
        focal = self.focal_loss(logits, labels)
        soft_f1 = self.soft_f1_loss(logits, labels)
        return focal + self.f1_lambda * soft_f1


def _resolve_alpha_pos(
    alpha_pos: Optional[float],
    train_pos: int,
    train_neg: int,
) -> float:
    """Resolve positive-class alpha from argument or class imbalance."""
    if alpha_pos is not None:
        resolved = float(alpha_pos)
    else:
        total = max(1, train_pos + train_neg)
        resolved = float(train_neg / total)
    if not (0.0 < resolved < 1.0):
        raise ValueError("Resolved alpha_pos must satisfy 0 < alpha_pos < 1")
    return resolved


def build_binary_classification_loss(
    loss_name: str,
    train_pos: int,
    train_neg: int,
    device: str,
    pos_weight_cap: float,
    focal_gamma: float,
    focal_alpha_pos: Optional[float],
    asym_gamma_pos: float,
    asym_gamma_neg: float,
    asym_alpha_pos: Optional[float],
    f1_lambda: float = 0.1,
) -> tuple[nn.Module, LossMeta]:
    """Create configured binary loss module and resolved metadata.

    Parameters
    ----------
    loss_name : str
        One of values in ``LOSS_NAME_CHOICES``.
    train_pos : int
        Number of positive samples in training split.
    train_neg : int
        Number of negative samples in training split.
    device : str
        Torch device string.
    pos_weight_cap : float
        Upper bound for positive class weight in ``weighted_bce``.
    focal_gamma : float
        Gamma parameter for ``focal``.
    focal_alpha_pos : float | None
        Positive class alpha for ``focal``. If ``None``, infer from imbalance.
    asym_gamma_pos : float
        Positive-class gamma for ``asymmetric_focal``.
    asym_gamma_neg : float
        Negative-class gamma for ``asymmetric_focal``.
    asym_alpha_pos : float | None
        Positive class alpha for ``asymmetric_focal``. If ``None``, infer from
        imbalance.
    f1_lambda : float, default=0.1
        Mixing coefficient used when ``loss_name`` is ``weighted_bce_f1`` or
        ``focal_f1``.

    Returns
    -------
    tuple[nn.Module, LossMeta]
        Loss object and resolved metadata.

    Raises
    ------
    ValueError
        If ``loss_name`` is unsupported or parameters are invalid.
    """
    if loss_name == "bce":
        return (
            nn.BCEWithLogitsLoss(reduction="mean"),
            LossMeta(
                pos_weight=1.0,
                focal_gamma=0.0,
                focal_alpha_pos=0.5,
                asym_gamma_pos=0.0,
                asym_gamma_neg=0.0,
                asym_alpha_pos=0.5,
                f1_lambda=0.0,
            ),
        )

    if loss_name == "weighted_bce":
        pos_weight_raw = (train_neg / max(1, train_pos)) if train_pos > 0 else 1.0
        pos_weight = min(pos_weight_raw, pos_weight_cap)
        pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        return (
            nn.BCEWithLogitsLoss(pos_weight=pos_weight_t, reduction="mean"),
            LossMeta(
                pos_weight=float(pos_weight),
                focal_gamma=0.0,
                focal_alpha_pos=0.5,
                asym_gamma_pos=0.0,
                asym_gamma_neg=0.0,
                asym_alpha_pos=0.5,
                f1_lambda=0.0,
            ),
        )

    if loss_name == "focal":
        alpha_pos = _resolve_alpha_pos(
            alpha_pos=focal_alpha_pos,
            train_pos=train_pos,
            train_neg=train_neg,
        )
        criterion = BinaryFocalLoss(
            gamma=float(focal_gamma),
            alpha_pos=alpha_pos,
            reduction="mean",
        )
        return (
            criterion,
            LossMeta(
                pos_weight=1.0,
                focal_gamma=float(focal_gamma),
                focal_alpha_pos=alpha_pos,
                asym_gamma_pos=0.0,
                asym_gamma_neg=0.0,
                asym_alpha_pos=0.5,
                f1_lambda=0.0,
            ),
        )

    if loss_name == "asymmetric_focal":
        alpha_pos = _resolve_alpha_pos(
            alpha_pos=asym_alpha_pos,
            train_pos=train_pos,
            train_neg=train_neg,
        )
        criterion = AsymmetricBinaryFocalLoss(
            gamma_pos=float(asym_gamma_pos),
            gamma_neg=float(asym_gamma_neg),
            alpha_pos=alpha_pos,
            reduction="mean",
        )
        return (
            criterion,
            LossMeta(
                pos_weight=1.0,
                focal_gamma=0.0,
                focal_alpha_pos=0.5,
                asym_gamma_pos=float(asym_gamma_pos),
                asym_gamma_neg=float(asym_gamma_neg),
                asym_alpha_pos=alpha_pos,
                f1_lambda=0.0,
            ),
        )

    if loss_name == "f1":
        return (
            SoftBinaryF1Loss(reduction="mean"),
            LossMeta(
                pos_weight=1.0,
                focal_gamma=0.0,
                focal_alpha_pos=0.5,
                asym_gamma_pos=0.0,
                asym_gamma_neg=0.0,
                asym_alpha_pos=0.5,
                f1_lambda=0.0,
            ),
        )

    if loss_name == "weighted_bce_f1":
        pos_weight_raw = (train_neg / max(1, train_pos)) if train_pos > 0 else 1.0
        pos_weight = min(pos_weight_raw, pos_weight_cap)
        pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        criterion = WeightedBceSoftF1Loss(
            pos_weight=pos_weight_t,
            f1_lambda=float(f1_lambda),
            f1_smooth=1e-7,
        )
        return (
            criterion,
            LossMeta(
                pos_weight=float(pos_weight),
                focal_gamma=0.0,
                focal_alpha_pos=0.5,
                asym_gamma_pos=0.0,
                asym_gamma_neg=0.0,
                asym_alpha_pos=0.5,
                f1_lambda=float(f1_lambda),
            ),
        )

    if loss_name == "focal_f1":
        alpha_pos = _resolve_alpha_pos(
            alpha_pos=focal_alpha_pos,
            train_pos=train_pos,
            train_neg=train_neg,
        )
        criterion = FocalSoftF1Loss(
            gamma=float(focal_gamma),
            alpha_pos=alpha_pos,
            f1_lambda=float(f1_lambda),
            f1_smooth=1e-7,
        )
        return (
            criterion,
            LossMeta(
                pos_weight=1.0,
                focal_gamma=float(focal_gamma),
                focal_alpha_pos=alpha_pos,
                asym_gamma_pos=0.0,
                asym_gamma_neg=0.0,
                asym_alpha_pos=0.5,
                f1_lambda=float(f1_lambda),
            ),
        )

    known = ", ".join(LOSS_NAME_CHOICES)
    raise ValueError(f"Unsupported --loss value '{loss_name}'. Supported: {known}")
