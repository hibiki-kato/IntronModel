from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from evaluate_scores import resolve_plot_bounds, resolve_plot_output
from evaluate_scores import plot_eval_scores
from evaluate_scores import run_plot_command


def test_resolve_plot_bounds_for_known_species() -> None:
    bounds = resolve_plot_bounds(
        species="Mmus",
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )
    assert bounds == (10.0, 18.0, 40.0, 46.0)


def test_resolve_plot_bounds_for_hsap_defaults() -> None:
    bounds = resolve_plot_bounds(
        species="Hsap",
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )
    assert bounds == (10.0, 19.0, 27.0, 38.0)


def test_resolve_plot_bounds_unknown_species_requires_explicit_bounds() -> None:
    with pytest.raises(ValueError, match="Unknown species"):
        resolve_plot_bounds(
            species="Mmel",
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
        )


def test_resolve_plot_bounds_unknown_species_with_explicit_bounds() -> None:
    bounds = resolve_plot_bounds(
        species="Mmel",
        x_min=1.0,
        x_max=2.0,
        y_min=3.0,
        y_max=4.0,
    )
    assert bounds == (1.0, 2.0, 3.0, 4.0)


def test_resolve_plot_output_defaults_to_species_snpr_name() -> None:
    output_path = resolve_plot_output(species="Mmus", output_png=None)
    assert output_path.endswith("/data/Mmus/Mmus_snpr.png")


def test_plot_eval_scores_places_legend_outside_right(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summary SN-PR legend should sit outside the plotting area."""

    eval_dir = tmp_path / "eval_score"
    eval_dir.mkdir()
    (eval_dir / "model_a.txt").write_text(
        "\n".join(
            [
                "tx1 0.10 = 10.0 20.0 13.3",
                "tx2 0.20 = 20.0 30.0 24.0",
                "tx3 0.30 = 30.0 40.0 34.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "evaluate_scores.resolve_eval_dir",
        lambda species: str(eval_dir),
    )

    output_png = tmp_path / "plot.png"
    plot_eval_scores(
        species="SpX",
        output_png=str(output_png),
        interactive=False,
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
    )

    legend = plt.gcf().axes[0].get_legend()
    assert legend is not None
    assert legend._loc == 2
    bbox = legend.get_bbox_to_anchor()._bbox
    assert bbox.x0 == pytest.approx(1.02)
    assert bbox.y0 == pytest.approx(1.0)

    plt.close("all")


def test_run_plot_command_archives_stale_eval_scores_before_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_dir = tmp_path / "data" / "SpX" / "eval_score"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "cnn_v2.01.txt").write_text("old\n", encoding="utf-8")
    (eval_dir / "cnn_v2.02.txt").write_text("new\n", encoding="utf-8")

    calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr("evaluate_scores.resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "evaluate_scores.finalize_ready_published_outputs_for_species",
        lambda *, project_root, species: [],
    )
    monkeypatch.setattr(
        "evaluate_scores.plot_eval_scores",
        lambda *,
        species,
        output_png,
        interactive,
        x_min,
        x_max,
        y_min,
        y_max: calls.append((species, output_png)),
    )

    run_plot_command(
        SimpleNamespace(
            species="SpX",
            output_png=None,
            interactive=False,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
        )
    )

    assert calls == [("SpX", None)]
    assert not (eval_dir / "cnn_v2.01.txt").exists()
    assert (eval_dir / "cnn_v2.02.txt").exists()
    assert (
        tmp_path
        / "archive"
        / "eval_score_latest_only"
        / "SpX"
        / "cnn_v2"
        / "cnn_v2.01.txt"
    ).is_file()
