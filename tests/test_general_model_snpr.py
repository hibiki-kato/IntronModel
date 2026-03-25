from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.general_model_snpr import (  # noqa: E402
    _compute_species_curves,
    _build_arg_parser,
    _resolve_eval_species,
    _write_interactive_html,
    go,
)


def _create_species_like_directory(base_dir: Path, species: str) -> None:
    species_dir = base_dir / species
    (species_dir / "raw").mkdir(parents=True)
    (species_dir / "processed").mkdir(parents=True)
    (species_dir / "intron_score").mkdir(parents=True)
    (species_dir / "raw" / "transcript_class.txt").write_text(
        "tx1 =\n",
        encoding="utf-8",
    )
    (species_dir / "processed" / "transcripts.tsv").write_text(
        "transcript_id\tgene_id\tsite_type\tintron_index\n",
        encoding="utf-8",
    )
    (species_dir / "processed" / "transcripts.unique.map.tsv").write_text(
        "unique_transcript_id\tunique_intron_index\ttranscript_id\tintron_index"
        "\tchrom\tstrand\tintron_start\tintron_end\n",
        encoding="utf-8",
    )


def test_resolve_eval_species_all_filters_non_species_directories(
    tmp_path: Path,
) -> None:
    _create_species_like_directory(tmp_path, "Hsap")
    _create_species_like_directory(tmp_path, "Mmus")
    (tmp_path / "tuning").mkdir()

    resolved = _resolve_eval_species(tmp_path, "all")

    assert resolved == ["Hsap", "Mmus"]


def test_compute_species_curves_returns_species_aware_summary() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    model_to_scores = {
        "baseline_min": np.asarray([0.1, 0.7, 0.8, 0.2], dtype=np.float64)
    }

    summaries, curves = _compute_species_curves(
        eval_species="Hsap",
        model_to_scores=model_to_scores,
        labels=labels,
        sensitivity_denominator=10,
    )

    assert len(summaries) == 1
    assert len(curves) == 1
    assert summaries[0].eval_species == "Hsap"
    assert summaries[0].point_count == 3
    assert summaries[0].sensitivity_denominator == 10
    assert curves[0].eval_species == "Hsap"
    assert curves[0].model_name == "baseline_min"


def test_write_interactive_html_requires_plotly_or_writes_file(
    tmp_path: Path,
) -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    model_to_scores = {
        "baseline_min": np.asarray([0.1, 0.7, 0.8, 0.2], dtype=np.float64)
    }
    summaries, curves = _compute_species_curves(
        eval_species="Hsap",
        model_to_scores=model_to_scores,
        labels=labels,
        sensitivity_denominator=10,
    )
    output_html = tmp_path / "snpr.html"

    if go is None:
        with pytest.raises(ImportError):
            _write_interactive_html(
                curves=curves,
                summaries=summaries,
                output_html=output_html,
                title="test",
            )
    else:
        _write_interactive_html(
            curves=curves,
            summaries=summaries,
            output_html=output_html,
            title="test",
        )
        assert output_html.is_file()
        content = output_html.read_text(encoding="utf-8")
        assert "plotly" in content.lower()


def test_write_interactive_html_places_legend_outside_right(
    tmp_path: Path,
) -> None:
    """The Plotly SN-PR legend should be rendered outside the plot area."""

    if go is None:
        pytest.skip("plotly is not available")

    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    model_to_scores = {
        "baseline_min": np.asarray([0.1, 0.7, 0.8, 0.2], dtype=np.float64)
    }
    summaries, curves = _compute_species_curves(
        eval_species="Hsap",
        model_to_scores=model_to_scores,
        labels=labels,
        sensitivity_denominator=10,
    )
    output_html = tmp_path / "snpr.html"

    _write_interactive_html(
        curves=curves,
        summaries=summaries,
        output_html=output_html,
        title="test",
    )

    content = output_html.read_text(encoding="utf-8").replace(" ", "")
    assert '"legend":{"x":1.02' in content
    assert '"xanchor":"left"' in content
    assert '"yanchor":"top"' in content
    assert '"orientation":"v"' in content
    assert '"r":220' in content


def test_build_arg_parser_includes_logreg_c() -> None:
    """The SN-PR CLI should forward the L1 logistic regularization strength."""

    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--train-species",
            "Mmus",
            "--eval-species",
            "Hsap",
            "--logreg-c",
            "0.25",
        ]
    )

    assert args.logreg_c == 0.25
