from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.plot_learning_curve import plot_curves


def test_plot_curves_supports_pair_only_metrics(tmp_path: Path) -> None:
    """Render one learning-curve PNG from pair-only metrics history."""
    metrics_json = tmp_path / "pair.train.json"
    output_png = tmp_path / "pair_learning_curve.png"
    metrics_json.write_text(
        json.dumps(
            {
                "pair": {
                    "epoch_history": [
                        {
                            "epoch": 1,
                            "train_loss": 0.50,
                            "pr_auc": 0.81,
                            "roc_auc": 0.91,
                            "max_f1": 0.71,
                            "acc@0.5": 0.75,
                            "objective_metric": "pr_auc",
                            "objective_score": 0.81,
                            "improved": True,
                            "best_metric": "pr_auc",
                            "best_score": 0.81,
                            "best_epoch": 1,
                        },
                        {
                            "epoch": 2,
                            "train_loss": 0.35,
                            "pr_auc": 0.84,
                            "roc_auc": 0.93,
                            "max_f1": 0.74,
                            "acc@0.5": 0.79,
                            "objective_metric": "pr_auc",
                            "objective_score": 0.84,
                            "improved": True,
                            "best_metric": "pr_auc",
                            "best_score": 0.84,
                            "best_epoch": 2,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    plot_curves(metrics_json=metrics_json, output_png=output_png, dpi=100)

    assert output_png.exists()
    assert output_png.stat().st_size > 0


def test_plot_curves_rejects_missing_supported_task_history(tmp_path: Path) -> None:
    """Fail clearly when no supported task exposes epoch history."""
    metrics_json = tmp_path / "missing.train.json"
    output_png = tmp_path / "missing_learning_curve.png"
    metrics_json.write_text(json.dumps({"meta": {"epoch_history": []}}), encoding="utf-8")

    with pytest.raises(ValueError, match="supported task metrics"):
        plot_curves(metrics_json=metrics_json, output_png=output_png, dpi=100)
