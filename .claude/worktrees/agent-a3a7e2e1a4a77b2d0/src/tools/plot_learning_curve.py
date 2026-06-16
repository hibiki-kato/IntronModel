"""Plot training/validation learning curves from a run metrics JSON file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

SUPPORTED_TASK_NAMES: tuple[str, ...] = ("donor", "acceptor", "pair")


class EpochRecord(TypedDict):
    """One epoch-level metrics record for a single task."""

    epoch: int
    train_loss: float | None
    pr_auc: float | None
    roc_auc: float | None
    max_f1: float | None
    acc_at_0_5: float | None
    objective_metric: str
    objective_score: float | None
    improved: bool
    best_metric: str
    best_score: float | None
    best_epoch: int


@dataclass(frozen=True)
class TaskCurve:
    """Structured curve data for one task."""

    name: str
    records: list[EpochRecord]


def _as_float(value: object) -> float | None:
    """Return finite float from JSON scalar-like value."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if parsed == parsed:
            return parsed
    return None


def _normalize_epoch_record(raw: object) -> EpochRecord | None:
    """Validate and normalize a raw epoch record object."""

    if not isinstance(raw, dict):
        return None
    epoch_raw = raw.get("epoch")
    if not isinstance(epoch_raw, int) or epoch_raw <= 0:
        return None

    objective_metric_raw = raw.get("objective_metric")
    best_metric_raw = raw.get("best_metric")
    if not isinstance(objective_metric_raw, str) or not objective_metric_raw:
        return None
    if not isinstance(best_metric_raw, str) or not best_metric_raw:
        return None

    improved_raw = raw.get("improved")
    if not isinstance(improved_raw, bool):
        improved_raw = False

    best_epoch_raw = raw.get("best_epoch")
    if not isinstance(best_epoch_raw, int) or best_epoch_raw <= 0:
        best_epoch_raw = epoch_raw

    return EpochRecord(
        epoch=epoch_raw,
        train_loss=_as_float(raw.get("train_loss")),
        pr_auc=_as_float(raw.get("pr_auc")),
        roc_auc=_as_float(raw.get("roc_auc")),
        max_f1=_as_float(raw.get("max_f1")),
        acc_at_0_5=_as_float(raw.get("acc@0.5")),
        objective_metric=objective_metric_raw,
        objective_score=_as_float(raw.get("objective_score")),
        improved=improved_raw,
        best_metric=best_metric_raw,
        best_score=_as_float(raw.get("best_score")),
        best_epoch=best_epoch_raw,
    )


def _load_task_curve(payload: dict[str, object], task_name: str) -> TaskCurve | None:
    """Load one task curve from top-level metrics payload."""

    task_payload = payload.get(task_name)
    if not isinstance(task_payload, dict):
        return None
    history_raw = task_payload.get("epoch_history")
    if not isinstance(history_raw, list):
        return None

    records: list[EpochRecord] = []
    for raw_row in history_raw:
        normalized = _normalize_epoch_record(raw_row)
        if normalized is not None:
            records.append(normalized)

    if not records:
        return None

    records.sort(key=lambda row: row["epoch"])
    return TaskCurve(name=task_name, records=records)


def _load_task_curves(payload: dict[str, object]) -> list[TaskCurve]:
    """Load all supported task curves present in one metrics payload."""

    task_curves: list[TaskCurve] = []
    for task_name in SUPPORTED_TASK_NAMES:
        curve = _load_task_curve(payload, task_name)
        if curve is not None:
            task_curves.append(curve)
    return task_curves


def _default_output_path(metrics_json: Path) -> Path:
    """Build default output PNG path for a metrics JSON path."""

    stem = metrics_json.name
    if stem.endswith(".train.json"):
        stem = stem[: -len(".train.json")]
    else:
        stem = metrics_json.stem
    return metrics_json.with_name(f"{stem}_learning_curve.png")


def plot_curves(*, metrics_json: Path, output_png: Path, dpi: int) -> None:
    """Render and save learning curves for all supported training tasks."""

    payload_obj = json.loads(metrics_json.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError("Metrics JSON top-level value must be an object.")

    task_curves = _load_task_curves(payload_obj)
    if not task_curves:
        supported = ", ".join(SUPPORTED_TASK_NAMES)
        raise ValueError(
            f"No epoch_history found in supported task metrics: {supported}."
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is not available.") from exc

    fig, axes = plt.subplots(
        nrows=len(task_curves),
        ncols=2,
        figsize=(12, 4.5 * len(task_curves)),
        dpi=dpi,
        squeeze=False,
    )

    for row_index, task_curve in enumerate(task_curves):
        records = task_curve.records
        epochs = [row["epoch"] for row in records]

        ax_loss = axes[row_index][0]
        ax_metric = axes[row_index][1]

        train_loss = [row["train_loss"] for row in records]
        ax_loss.plot(epochs, train_loss, color="#0077b6", marker="o", ms=3)
        ax_loss.set_title(f"{task_curve.name}: train loss")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(alpha=0.25, linestyle="--")

        pr_auc = [row["pr_auc"] for row in records]
        roc_auc = [row["roc_auc"] for row in records]
        max_f1 = [row["max_f1"] for row in records]
        acc_at_0_5 = [row["acc_at_0_5"] for row in records]
        objective = [row["objective_score"] for row in records]

        if any(value is not None for value in pr_auc):
            ax_metric.plot(epochs, pr_auc, label="pr_auc", linewidth=1.5)
        if any(value is not None for value in roc_auc):
            ax_metric.plot(epochs, roc_auc, label="roc_auc", linewidth=1.5)
        if any(value is not None for value in max_f1):
            ax_metric.plot(epochs, max_f1, label="max_f1", linewidth=1.5)
        if any(value is not None for value in acc_at_0_5):
            ax_metric.plot(epochs, acc_at_0_5, label="acc@0.5", linewidth=1.5)
        if any(value is not None for value in objective):
            ax_metric.plot(
                epochs,
                objective,
                label="objective",
                linewidth=2.0,
                color="black",
                alpha=0.8,
            )

        improved_epochs = [
            row["epoch"]
            for row in records
            if row["improved"] and row["objective_score"] is not None
        ]
        improved_scores = [
            row["objective_score"]
            for row in records
            if row["improved"] and row["objective_score"] is not None
        ]
        if improved_epochs:
            ax_metric.scatter(
                improved_epochs,
                improved_scores,
                s=24,
                color="#d00000",
                marker="*",
                label="improved",
                zorder=5,
            )

        ax_metric.set_title(f"{task_curve.name}: validation metrics")
        ax_metric.set_xlabel("epoch")
        ax_metric.set_ylabel("score")
        ax_metric.grid(alpha=0.25, linestyle="--")
        labels = ax_metric.get_legend_handles_labels()[1]
        if labels:
            ax_metric.legend(loc="best")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Plot learning curves from metrics JSON generated by run_model.",
    )
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = _parse_args()
    metrics_json = Path(str(args.metrics_json)).resolve()
    if not metrics_json.exists():
        raise FileNotFoundError(f"metrics JSON not found: {metrics_json}")
    output_raw = str(args.output).strip()
    output_png = (
        Path(output_raw).resolve()
        if output_raw
        else _default_output_path(metrics_json)
    )
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    plot_curves(
        metrics_json=metrics_json,
        output_png=output_png,
        dpi=int(args.dpi),
    )
    print(f"[plot_learning_curve] wrote {output_png}")


if __name__ == "__main__":
    main()
