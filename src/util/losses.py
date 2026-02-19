"""Shared loss functions for binary classification site models.

This module centralizes reusable loss implementations so model files can focus
on architecture and training flow. It currently provides:

- BCE
- weighted BCE
- focal loss
- asymmetric focal loss (recommended for heavily imbalanced negatives)
"""

from __future__ import annotations

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
)


class LossMeta(TypedDict):
    """Metadata for configured loss behavior."""

    pos_weight: float
    focal_gamma: float
    focal_alpha_pos: float
    asym_gamma_pos: float
    asym_gamma_neg: float
    asym_alpha_pos: float


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
            ),
        )

    known = ", ".join(LOSS_NAME_CHOICES)
    raise ValueError(f"Unsupported --loss value '{loss_name}'. Supported: {known}")
