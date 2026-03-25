from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.profile_score_distributions import (  # noqa: E402
    ScoreDistributionResult,
    build_score_distribution_figure,
    load_intron_score_distribution,
    load_transcript_score_distribution,
    save_score_distribution_figure,
)


def test_load_intron_score_distribution_reads_labels(tmp_path: Path) -> None:
    """Split intron scores into positive and negative groups."""

    intron_dir = tmp_path / "data" / "SpX" / "intron_score"
    intron_dir.mkdir(parents=True, exist_ok=True)
    score_file = intron_dir / "model.tsv"
    score_file.write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tscore\tlabel",
                "tx1\t1\t0.90\t1",
                "tx2\t1\t0.10\t0",
                "tx3\t1\t0.80\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_intron_score_distribution(species="SpX", score_file=score_file)

    assert result.score_kind == "intron_score"
    assert result.model_name == "model"
    assert result.positive_scores == (0.9, 0.8)
    assert result.negative_scores == (0.1,)
    assert result.total_rows == 3
    assert result.used_rows == 3
    assert result.skipped_rows == 0


def test_load_transcript_score_distribution_uses_class_file_and_column(
    tmp_path: Path,
) -> None:
    """Resolve transcript labels and the default score column."""

    raw_dir = tmp_path / "data" / "SpX" / "raw"
    score_dir = tmp_path / "data" / "SpX" / "trans_score"
    raw_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "transcript_class.txt").write_text(
        "\n".join(
            [
                "tx1 =",
                "tx2 j",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    score_file = score_dir / "model.tsv"
    score_file.write_text(
        "\n".join(
            [
                "transcript_id\tmin_intron_index\tScore_donor\tScore_acceptor"
                "\ttrans_score",
                "tx1\t1\t0.1\t0.1\t0.90",
                "tx2\t1\t0.2\t0.2\t0.10",
                "tx3\t1\t0.3\t0.3\t0.50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_transcript_score_distribution(
        repo_root=tmp_path,
        species="SpX",
        score_file=score_file,
    )

    assert result.score_kind == "transcript_score"
    assert result.score_column_used == "trans_score"
    assert result.positive_scores == (0.9,)
    assert result.negative_scores == (0.1,)
    assert result.total_rows == 3
    assert result.used_rows == 2
    assert result.skipped_rows == 1


def test_build_score_distribution_figure_writes_png(tmp_path: Path) -> None:
    """Render a combined intron/transcript distribution figure."""

    intron_result = ScoreDistributionResult(
        species="SpX",
        model_name="model",
        score_kind="intron_score",
        source_file=tmp_path / "intron.tsv",
        score_column_used=None,
        positive_scores=(0.9, 0.8),
        negative_scores=(0.1, 0.2),
        total_rows=4,
        used_rows=4,
        skipped_rows=0,
    )
    transcript_result = ScoreDistributionResult(
        species="SpX",
        model_name="model",
        score_kind="transcript_score",
        source_file=tmp_path / "trans.tsv",
        score_column_used="trans_score",
        positive_scores=(0.7, 0.6),
        negative_scores=(0.3, 0.4),
        total_rows=4,
        used_rows=4,
        skipped_rows=0,
    )

    figure = build_score_distribution_figure(
        species="SpX",
        model_name="model",
        results=[intron_result, transcript_result],
        bins=10,
    )
    output_path = tmp_path / "figure.png"
    save_score_distribution_figure(figure, output_path)

    assert output_path.is_file()
    assert len(figure.axes) == 2
    assert "intron score" in figure.axes[0].get_title()
    assert "transcript score" in figure.axes[1].get_title()
    assert figure.axes[0].get_ylabel() == "Count"
    figure.clf()


def test_build_score_distribution_figure_supports_density_toggle(
    tmp_path: Path,
) -> None:
    """Switch histogram normalization on and off with one option."""

    result = ScoreDistributionResult(
        species="SpX",
        model_name="model",
        score_kind="intron_score",
        source_file=tmp_path / "intron.tsv",
        score_column_used=None,
        positive_scores=(0.9, 0.8),
        negative_scores=(0.1, 0.2),
        total_rows=4,
        used_rows=4,
        skipped_rows=0,
    )

    count_figure = build_score_distribution_figure(
        species="SpX",
        model_name="model",
        results=[result],
        bins=10,
        density=False,
    )
    density_figure = build_score_distribution_figure(
        species="SpX",
        model_name="model",
        results=[result],
        bins=10,
        density=True,
    )

    assert count_figure.axes[0].get_ylabel() == "Count"
    assert density_figure.axes[0].get_ylabel() == "Density"
    count_figure.clf()
    density_figure.clf()


def test_load_intron_score_distribution_rejects_single_label_group(
    tmp_path: Path,
) -> None:
    """Fail clearly when the input only contains positive rows."""

    intron_dir = tmp_path / "data" / "SpX" / "intron_score"
    intron_dir.mkdir(parents=True, exist_ok=True)
    score_file = intron_dir / "model.tsv"
    score_file.write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tscore\tlabel",
                "tx1\t1\t0.90\t1",
                "tx2\t1\t0.80\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_intron_score_distribution(species="SpX", score_file=score_file)
    except ValueError as error:
        assert "No negative scores" in str(error)
    else:
        raise AssertionError("Expected ValueError for single-label input")
