from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.transcript_intron_yield_curves import (  # noqa: E402
    build_transcript_intron_yield_curve_figure,
    load_transcript_intron_yield_curves,
    save_transcript_intron_yield_curve_figure,
    yield_curve_output_path,
)


def test_load_transcript_intron_yield_curves_splits_positive_and_negative(
    tmp_path: Path,
) -> None:
    """Load one selected intron-count bin and split it by label."""

    data_dir = tmp_path / "data" / "SpX"
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"
    intron_dir = data_dir / "intron_score"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    intron_dir.mkdir(parents=True, exist_ok=True)

    (processed_dir / "transcripts.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index",
                "tx_pos\t1",
                "tx_pos\t2",
                "tx_neg\t1",
                "tx_neg\t2",
                "tx_other\t1",
                "tx_other\t2",
                "tx_other\t3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "transcript_class.txt").write_text(
        "\n".join(
            [
                "tx_pos =",
                "tx_neg -",
                "tx_other =",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    score_file = intron_dir / "model.tsv"
    score_file.write_text(
        "\n".join(
            [
                "source_transcript_id\tscore\tlabel",
                "tx_pos\t0.30\t1",
                "tx_pos\t0.10\t0",
                "tx_neg\t0.90\t1",
                "tx_neg\t0.20\t0",
                "tx_other\t0.50\t1",
                "tx_other\t0.40\t1",
                "tx_other\t0.10\t0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_transcript_intron_yield_curves(
        repo_root=tmp_path,
        species="SpX",
        score_file=score_file,
        selected_intron_count=2,
    )

    assert result.total_rows == 7
    assert result.selected_transcript_count == 2
    assert len(result.positive_curves) == 1
    assert len(result.negative_curves) == 1
    assert result.selected_false_intron_count == 2
    assert result.negative_curves[0].sorted_scores == (0.2, 0.9)
    assert result.negative_curves[0].sorted_labels == (0, 1)
    assert result.positive_curves[0].sorted_scores == (0.1, 0.3)
    assert result.positive_curves[0].sorted_labels == (0, 1)

    figure = build_transcript_intron_yield_curve_figure(
        result,
        highlight_false_introns=True,
        false_intron_highlight_limit=10,
    )
    output_path = tmp_path / "figure.png"
    save_transcript_intron_yield_curve_figure(figure, output_path)

    assert output_path.is_file()
    assert len(figure.axes) == 2
    figure.clf()


def test_yield_curve_output_path_builds_expected_name(tmp_path: Path) -> None:
    """Build the PNG path used by the notebook renderer."""

    output_path = yield_curve_output_path(
        output_dir=tmp_path,
        species="SpX",
        model_name="model",
        selected_intron_count=3,
    )

    assert output_path == (
        tmp_path / "SpX_model_intron_score_yield_curves_3intron.png"
    )
