from __future__ import annotations

from pathlib import Path
import sys

import pytest

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from score.profile_score_vs_intron_count import (  # noqa: E402
    ScoreFileLoadResult,
    TranscriptScoreRecord,
    build_correlation_row,
    build_intron_bin_rows,
    collect_records_from_trans_score,
    load_transcript_intron_counts,
    spearman_correlation,
)


def test_load_transcript_intron_counts_counts_unique_intron_indices(
    tmp_path: Path,
) -> None:
    raw_tsv = tmp_path / "transcripts.tsv"
    raw_tsv.write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tsite_type",
                "tx1\t1\tdonor",
                "tx1\t1\tacceptor",
                "tx1\t2\tdonor",
                "tx2\t3\tdonor",
                "tx2\t3\tacceptor",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    counts = load_transcript_intron_counts(raw_tsv)

    assert counts == {"tx1": 2, "tx2": 1}


def test_spearman_correlation_handles_monotonic_series() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys_pos = [10.0, 20.0, 30.0, 40.0]
    ys_neg = [40.0, 30.0, 20.0, 10.0]

    assert spearman_correlation(xs, ys_pos) == pytest.approx(1.0)
    assert spearman_correlation(xs, ys_neg) == pytest.approx(-1.0)


def test_collect_records_from_trans_score_counts_missing_transcripts(
    tmp_path: Path,
) -> None:
    score_tsv = tmp_path / "model.tsv"
    score_tsv.write_text(
        "\n".join(
            [
                "transcript_id\tmin_donor_plus_acceptor",
                "tx1\t0.9",
                "tx_missing\t0.1",
                "tx2\tnan_text",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    intron_counts = {"tx1": 2, "tx2": 3}

    result = collect_records_from_trans_score(
        species="SpX",
        trans_score_tsv=score_tsv,
        intron_count_by_transcript=intron_counts,
        score_column="min_donor_plus_acceptor",
    )

    assert result.total_rows == 3
    assert result.used_rows == 1
    assert result.invalid_score_count == 1
    assert result.missing_transcript_count == 1
    assert result.records[0].transcript_id == "tx1"


def test_collect_records_from_trans_score_supports_legacy_score_column(
    tmp_path: Path,
) -> None:
    score_tsv = tmp_path / "legacy.tsv"
    score_tsv.write_text(
        "\n".join(
            [
                "transcript_id\tmin_donor_times_acceptor",
                "tx1\t0.25",
                "tx2\t0.50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    intron_counts = {"tx1": 1, "tx2": 3}

    result = collect_records_from_trans_score(
        species="SpX",
        trans_score_tsv=score_tsv,
        intron_count_by_transcript=intron_counts,
        score_column="min_donor_plus_acceptor",
    )

    assert result.used_rows == 2
    assert result.score_column_used == "min_donor_times_acceptor"
    assert result.records[0].final_score == pytest.approx(0.25)


def test_build_correlation_row_and_bins_capture_negative_association() -> None:
    result = ScoreFileLoadResult(
        species="SpX",
        model_name="cnn",
        file_path=Path("data/SpX/trans_score/cnn.tsv"),
        score_column_used="min_donor_plus_acceptor",
        total_rows=4,
        used_rows=4,
        invalid_score_count=0,
        missing_transcript_count=0,
        records=(
            TranscriptScoreRecord(
                species="SpX",
                model_name="cnn",
                file_path=Path("x.tsv"),
                transcript_id="tx1",
                intron_count=1,
                final_score=0.9,
            ),
            TranscriptScoreRecord(
                species="SpX",
                model_name="cnn",
                file_path=Path("x.tsv"),
                transcript_id="tx2",
                intron_count=2,
                final_score=0.6,
            ),
            TranscriptScoreRecord(
                species="SpX",
                model_name="cnn",
                file_path=Path("x.tsv"),
                transcript_id="tx3",
                intron_count=5,
                final_score=0.2,
            ),
            TranscriptScoreRecord(
                species="SpX",
                model_name="cnn",
                file_path=Path("x.tsv"),
                transcript_id="tx4",
                intron_count=7,
                final_score=0.1,
            ),
        ),
    )

    row = build_correlation_row(result=result, score_floor=1e-12)
    bins = build_intron_bin_rows(result)

    pearson_corr = row["pearson_corr"]
    assert isinstance(pearson_corr, float)
    assert pearson_corr < 0.0
    assert row["score_median_intron_le_2"] == pytest.approx(0.75)
    assert row["score_median_intron_ge_5"] == pytest.approx(0.15)
    assert len(bins) == 4
    bin_counts = {
        int(record["intron_count"]): int(record["transcript_count"])
        for record in bins
    }
    assert bin_counts == {1: 1, 2: 1, 5: 1, 7: 1}
