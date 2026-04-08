from __future__ import annotations

import pytest

from util.training_control import (
    get_metric_value,
    resolve_training_schedule,
    resolve_validation_metric,
    select_validation_score,
)


def test_resolve_validation_metric_normalizes_legacy_alias() -> None:
    assert resolve_validation_metric("f1_max") == "max_f1"


def test_resolve_validation_metric_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="validation_metric"):
        resolve_validation_metric("unknown_metric")


def test_select_validation_score_prefers_requested_metric() -> None:
    score, metric_name = select_validation_score(
        metrics={
            "pr_auc": 0.81,
            "roc_auc": 0.82,
            "max_f1": 0.77,
            "acc@0.5": 0.79,
        },
        validation_metric="max_f1",
    )

    assert score == pytest.approx(0.77)
    assert metric_name == "max_f1"


def test_select_validation_score_falls_back_for_pr_auc() -> None:
    score, metric_name = select_validation_score(
        metrics={
            "roc_auc": 0.74,
            "acc@0.5": 0.71,
        },
        validation_metric="pr_auc",
    )

    assert score == pytest.approx(0.74)
    assert metric_name == "roc_auc"


def test_get_metric_value_returns_float_for_existing_metric() -> None:
    value = get_metric_value(
        metrics={
            "max_f1": 0.81,
            "pr_auc": 0.92,
        },
        metric_name="max_f1",
    )

    assert value == pytest.approx(0.81)


def test_get_metric_value_returns_none_for_missing_metric() -> None:
    assert get_metric_value(metrics={"pr_auc": 0.92}, metric_name="max_f1") is None


def test_resolve_training_schedule_disables_early_stop_for_fixed_epochs() -> None:
    schedule = resolve_training_schedule(
        epochs_arg="6",
        max_epochs=20,
        patience_arg=7,
        min_delta_arg=0.01,
    )

    assert schedule.resolved_epochs == 6
    assert schedule.epochs_auto is False
    assert schedule.early_stop_patience == 7
    assert schedule.early_stop_min_delta == pytest.approx(0.01)
    assert schedule.effective_early_stop_patience == 0


def test_resolve_training_schedule_preserves_patience_for_auto_epochs() -> None:
    schedule = resolve_training_schedule(
        epochs_arg="auto",
        max_epochs=14,
        patience_arg=5,
        min_delta_arg=0.0,
    )

    assert schedule.resolved_epochs == 14
    assert schedule.epochs_auto is True
    assert schedule.effective_early_stop_patience == 5
