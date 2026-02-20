"""Plot aggregated tuning history for double-descent inspection.

This utility reads all successful tuning trials under:
``data/<species>/tuning/cnn/<target>/<run_id>/{quick_trials,full_trials}.tsv``
and generates one pooled complexity-vs-score figure.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from tools.hparam_search import ArgValue, Scalar, estimate_cnn_param_complexity
except ModuleNotFoundError:
    from hparam_search import ArgValue, Scalar, estimate_cnn_param_complexity


@dataclass(frozen=True)
class TrialPoint:
    """One plotted trial point."""

    complexity: int
    score: float
    phase: str
    run_id: str
    objective_metric: str


def _safe_float(value: object) -> Optional[float]:
    """Convert to float when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_sampled_params(row: dict[str, str]) -> dict[str, Scalar]:
    """Extract sampled hyperparameters from one TSV row."""
    fixed_cols = {
        "phase",
        "trial_id",
        "status",
        "gpu_id",
        "effective_batch_size",
        "oom_retries",
        "donor_pr_auc",
        "acceptor_pr_auc",
        "mean_pr_auc",
        "objective_metric",
        "objective_score",
        "return_code",
        "duration_sec",
        "metrics_json",
        "log_file",
        "error_message",
    }
    sampled: dict[str, Scalar] = {}
    for key, raw in row.items():
        if key in fixed_cols or raw is None:
            continue
        text = raw.strip()
        if text == "":
            continue
        float_value = _safe_float(text)
        if float_value is None:
            sampled[key] = text
            continue
        if float_value.is_integer():
            sampled[key] = int(float_value)
        else:
            sampled[key] = float_value
    return sampled


def load_points(
    *,
    project_root: Path,
    species: str,
    target: str,
) -> list[TrialPoint]:
    """Load all successful historical points for one species/target."""
    root = project_root / "data" / species / "tuning" / "cnn" / target
    if not root.exists():
        return []

    points: list[TrialPoint] = []
    run_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    for run_dir in run_dirs:
        for file_name in ("quick_trials.tsv", "full_trials.tsv"):
            tsv_path = run_dir / file_name
            if not tsv_path.exists():
                continue
            phase = "quick" if "quick" in file_name else "full"
            with tsv_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    if row.get("status") != "success":
                        continue
                    score = _safe_float(row.get("objective_score", ""))
                    if score is None:
                        continue
                    sampled = _extract_sampled_params(row)
                    complexity = estimate_cnn_param_complexity(
                        sampled_params=sampled,
                        base_args={},
                    )
                    if complexity is None:
                        continue
                    objective_metric = str(
                        row.get("objective_metric", f"{target}_pr_auc")
                    )
                    points.append(
                        TrialPoint(
                            complexity=complexity,
                            score=score,
                            phase=phase,
                            run_id=run_dir.name,
                            objective_metric=objective_metric,
                        )
                    )
    return points


def _rolling_curve(
    points: list[tuple[int, float]],
    window_size: int,
) -> tuple[list[float], list[float], list[float]]:
    """Return rolling mean and percentile envelope."""
    if not points:
        return [], [], []
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1

    xs: list[float] = []
    means: list[float] = []
    spreads: list[float] = []
    half = window_size // 2
    for idx, (x_value, _) in enumerate(points):
        start = max(0, idx - half)
        end = min(len(points), idx + half + 1)
        segment = points[start:end]
        scores = sorted([score for _, score in segment])
        xs.append(float(x_value))
        means.append(sum(scores) / len(scores))
        lower_idx = int(0.2 * (len(scores) - 1))
        upper_idx = int(0.8 * (len(scores) - 1))
        spreads.append((scores[upper_idx] - scores[lower_idx]) / 2.0)
    return xs, means, spreads


def render_plot(
    *,
    points: list[TrialPoint],
    species: str,
    target: str,
    output_png: Path,
) -> str:
    """Render pooled double-descent style plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib is not available."

    quick_points = [point for point in points if point.phase == "quick"]
    full_points = [point for point in points if point.phase == "full"]
    objective_metric = points[0].objective_metric if points else f"{target}_pr_auc"

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    if quick_points:
        ax.scatter(
            [point.complexity for point in quick_points],
            [point.score for point in quick_points],
            s=18,
            alpha=0.35,
            label="quick (all runs)",
            marker="o",
        )
    if full_points:
        ax.scatter(
            [point.complexity for point in full_points],
            [point.score for point in full_points],
            s=40,
            alpha=0.75,
            label="full (all runs)",
            marker="^",
        )

    sorted_points = sorted(
        [(point.complexity, point.score) for point in points],
        key=lambda item: item[0],
    )
    if len(sorted_points) >= 9:
        window_size = max(9, len(sorted_points) // 10)
        xs, means, spreads = _rolling_curve(sorted_points, window_size)
        if xs:
            ax.plot(
                xs,
                means,
                linewidth=2.0,
                color="black",
                label=f"trend (rolling mean, w={window_size})",
            )
            lower = [mean - spread for mean, spread in zip(means, spreads)]
            upper = [mean + spread for mean, spread in zip(means, spreads)]
            ax.fill_between(
                xs,
                lower,
                upper,
                color="gray",
                alpha=0.15,
                label="local spread (20-80%)",
            )

    best = max(points, key=lambda point: point.score)
    ax.scatter(
        [best.complexity],
        [best.score],
        s=150,
        marker="*",
        color="gold",
        edgecolors="black",
        linewidths=0.8,
        label=f"best ({best.score:.4f})",
        zorder=6,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Estimated model complexity (trainable parameters, log scale)")
    ax.set_ylabel(objective_metric)
    ax.set_title(f"{species} {target}: pooled tuning history for double-descent check")
    ax.grid(alpha=0.3, linestyle="--")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="best")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    return ""


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot pooled tuning history for complexity-vs-score analysis."
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--target", choices=["donor", "acceptor"], required=True)
    parser.add_argument("--output_png", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""
    import sys

    args = parse_args(sys.argv[1:] if argv is None else argv)
    project_root = Path(args.project_root).resolve()
    species = str(args.species)
    target = str(args.target)
    points = load_points(
        project_root=project_root,
        species=species,
        target=target,
    )
    if not points:
        print(
            f"[plot_tuning_double_descent] no points found for {species}/{target}.",
            flush=True,
        )
        return 0

    if args.output_png:
        output_png = Path(args.output_png).resolve()
    else:
        output_png = (
            project_root
            / "data"
            / species
            / "tuning"
            / "cnn"
            / target
            / f"{species}_{target}_double_descent.png"
        )
    error = render_plot(
        points=points,
        species=species,
        target=target,
        output_png=output_png,
    )
    if error:
        print(f"[plot_tuning_double_descent] skipped: {error}", flush=True)
        return 0
    print(
        "[plot_tuning_double_descent] wrote "
        f"{output_png} (points={len(points)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
