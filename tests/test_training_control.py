from __future__ import annotations

import pytest

from util.training_control import (
    get_metric_value,
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
