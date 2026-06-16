from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from util.losses import (  # noqa: E402
    FocalSoftF1Loss,
    LOSS_NAME_CHOICES,
    SoftBinaryF1Loss,
    WeightedBceSoftF1Loss,
    build_binary_classification_loss,
)


def test_loss_name_choices_includes_f1() -> None:
    assert "f1" in LOSS_NAME_CHOICES
    assert "weighted_bce_f1" in LOSS_NAME_CHOICES
    assert "focal_f1" in LOSS_NAME_CHOICES


def test_soft_binary_f1_loss_is_small_for_perfect_predictions() -> None:
    criterion = SoftBinaryF1Loss()
    logits = torch.tensor([8.0, -8.0, 9.0, -9.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    loss = criterion(logits, targets)
    assert float(loss.item()) < 1e-3


def test_soft_binary_f1_loss_is_higher_for_bad_predictions() -> None:
    criterion = SoftBinaryF1Loss()
    logits_good = torch.tensor([8.0, -8.0, 9.0, -9.0], dtype=torch.float32)
    logits_bad = -logits_good
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    loss_good = criterion(logits_good, targets)
    loss_bad = criterion(logits_bad, targets)
    assert float(loss_bad.item()) > float(loss_good.item())


def test_soft_binary_f1_loss_supports_backward() -> None:
    criterion = SoftBinaryF1Loss()
    logits = torch.tensor([0.5, -0.2, 1.1, -1.7], dtype=torch.float32)
    logits.requires_grad_(True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    loss = criterion(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_soft_binary_f1_loss_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="smooth must be > 0.0"):
        _ = SoftBinaryF1Loss(smooth=0.0)
    with pytest.raises(ValueError, match="reduction='mean'"):
        _ = SoftBinaryF1Loss(reduction="sum")


def test_build_binary_classification_loss_supports_f1() -> None:
    criterion, meta = build_binary_classification_loss(
        loss_name="f1",
        train_pos=10,
        train_neg=90,
        device="cpu",
        pos_weight_cap=20.0,
        focal_gamma=2.0,
        focal_alpha_pos=None,
        asym_gamma_pos=0.0,
        asym_gamma_neg=4.0,
        asym_alpha_pos=None,
    )
    assert isinstance(criterion, SoftBinaryF1Loss)
    assert meta["pos_weight"] == pytest.approx(1.0)
    assert meta["focal_gamma"] == pytest.approx(0.0)
    assert meta["asym_gamma_neg"] == pytest.approx(0.0)
    assert meta["f1_lambda"] == pytest.approx(0.0)


def test_build_binary_classification_loss_supports_weighted_bce_f1() -> None:
    criterion, meta = build_binary_classification_loss(
        loss_name="weighted_bce_f1",
        train_pos=10,
        train_neg=90,
        device="cpu",
        pos_weight_cap=20.0,
        focal_gamma=2.0,
        focal_alpha_pos=None,
        asym_gamma_pos=0.0,
        asym_gamma_neg=4.0,
        asym_alpha_pos=None,
        f1_lambda=0.25,
    )
    assert isinstance(criterion, WeightedBceSoftF1Loss)
    logits = torch.tensor([0.2, -0.3, 1.0, -1.2], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    loss = criterion(logits, targets)
    assert float(loss.item()) > 0.0
    assert meta["pos_weight"] == pytest.approx(9.0)
    assert meta["f1_lambda"] == pytest.approx(0.25)


def test_build_binary_classification_loss_supports_focal_f1() -> None:
    criterion, meta = build_binary_classification_loss(
        loss_name="focal_f1",
        train_pos=10,
        train_neg=90,
        device="cpu",
        pos_weight_cap=20.0,
        focal_gamma=2.5,
        focal_alpha_pos=None,
        asym_gamma_pos=0.0,
        asym_gamma_neg=4.0,
        asym_alpha_pos=None,
        f1_lambda=0.15,
    )
    assert isinstance(criterion, FocalSoftF1Loss)
    logits = torch.tensor([0.2, -0.3, 1.0, -1.2], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    loss = criterion(logits, targets)
    assert float(loss.item()) > 0.0
    assert meta["pos_weight"] == pytest.approx(1.0)
    assert meta["focal_gamma"] == pytest.approx(2.5)
    assert meta["f1_lambda"] == pytest.approx(0.15)
