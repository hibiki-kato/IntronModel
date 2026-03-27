from __future__ import annotations

import math
from pathlib import Path

import pytest

from util.transcript_eval import (
    aggregate_pair_transcript_scores,
    aggregate_transcript_scores,
    build_intron_scores,
    read_site_scores,
    write_intron_scores,
    write_transcript_scores,
    write_site_scores,
)


def test_aggregate_transcript_scores_softmin_exp_sum() -> None:
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "donor",
            "score": 0.5,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": 0.5,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 2,
            "site_type": "donor",
            "score": 0.3,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 2,
            "site_type": "acceptor",
            "score": 0.3,
        },
    ]
    rows = aggregate_transcript_scores(
        site_score_rows=site_rows,
        intron_score_op="+",
        transcript_score_agg="softmin",
        softmin_tau=1.0,
    )

    assert len(rows) == 1
    result = rows[0]
    intron_scores = [math.log10(1.0), math.log10(0.6)]
    min_score = min(intron_scores)
    shifted_sum = sum(math.exp(-(score - min_score)) for score in intron_scores)
    expected = min_score - math.log(shifted_sum)

    assert result["min_intron_index"] == 2
    assert float(result["trans_score"]) == pytest.approx(expected)


def test_aggregate_transcript_scores_softmin_wavg() -> None:
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "donor",
            "score": 0.25,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": 0.25,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 2,
            "site_type": "donor",
            "score": 0.75,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 2,
            "site_type": "acceptor",
            "score": 0.75,
        },
    ]
    rows = aggregate_transcript_scores(
        site_score_rows=site_rows,
        intron_score_op="+",
        transcript_score_agg="softmin_wavg",
        softmin_tau=1.0,
    )

    assert len(rows) == 1
    result = rows[0]
    intron_scores = [math.log10(0.5), math.log10(1.5)]
    min_score = min(intron_scores)
    weights = [math.exp(-(score - min_score)) for score in intron_scores]
    expected = (
        weights[0] * intron_scores[0] + weights[1] * intron_scores[1]
    ) / sum(weights)

    assert result["min_intron_index"] == 1
    assert float(result["trans_score"]) == pytest.approx(expected)


@pytest.mark.parametrize("tau", [0.0, -0.5])
def test_aggregate_transcript_scores_softmin_tau_validation(tau: float) -> None:
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "donor",
            "score": 0.5,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": 0.5,
        },
    ]

    with pytest.raises(ValueError, match="softmin_tau must be positive"):
        _ = aggregate_transcript_scores(
            site_score_rows=site_rows,
            intron_score_op="+",
            transcript_score_agg="softmin_wavg",
            softmin_tau=tau,
        )


def test_aggregate_pair_transcript_scores_keeps_5col_compatibility() -> None:
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "pair",
            "score": 0.3,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 2,
            "site_type": "pair",
            "score": 0.7,
        },
    ]

    rows = aggregate_pair_transcript_scores(
        site_score_rows=site_rows,
        transcript_score_agg="min",
        softmin_tau=1.0,
    )

    assert rows == [
        {
            "transcript_id": "tx1",
            "min_intron_index": 1,
            "Score_donor": pytest.approx(math.log10(0.3)),
            "Score_acceptor": pytest.approx(math.log10(0.3)),
            "trans_score": pytest.approx(math.log10(0.3)),
            "_score_space": "log10",
        }
    ]


def test_aggregate_pair_transcript_scores_softmin_wavg() -> None:
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "pair",
            "score": 0.2,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 2,
            "site_type": "pair",
            "score": 1.2,
        },
    ]

    rows = aggregate_pair_transcript_scores(
        site_score_rows=site_rows,
        transcript_score_agg="softmin_wavg",
        softmin_tau=1.0,
    )
    scores = [0.2, 1.2]
    min_score = min(scores)
    weights = [math.exp(-(score - min_score)) for score in scores]
    expected = (weights[0] * scores[0] + weights[1] * scores[1]) / sum(weights)

    assert rows[0]["min_intron_index"] == 1
    assert float(rows[0]["trans_score"]) == pytest.approx(expected)


def test_aggregate_transcript_scores_uses_pair_rows_when_available() -> None:
    """Aggregate pair-only site rows without collapsing them to zero."""
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "pair",
            "score": 0.25,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 2,
            "site_type": "pair",
            "score": 0.75,
        },
    ]

    rows = aggregate_transcript_scores(
        site_score_rows=site_rows,
        intron_score_op="*",
        transcript_score_agg="min",
    )

    assert rows == [
        {
            "transcript_id": "tx1",
            "min_intron_index": 1,
            "Score_donor": pytest.approx(math.log10(0.25)),
            "Score_acceptor": pytest.approx(math.log10(0.25)),
            "trans_score": pytest.approx(math.log10(0.25)),
            "_score_space": "log10",
        }
    ]


def test_write_transcript_scores_outputs_trans_score_header(
    tmp_path: Path,
) -> None:
    """Write transcript-score TSV with the renamed score column."""

    output_tsv = tmp_path / "transcript.tsv"
    rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "min_intron_index": 1,
            "Score_donor": 0.3,
            "Score_acceptor": 0.3,
            "trans_score": 0.3,
        }
    ]
    write_transcript_scores(str(output_tsv), rows)
    lines = output_tsv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == (
        "transcript_id\tmin_intron_index\tScore_donor\tScore_acceptor\t"
        "trans_score"
    )
    assert lines[1] == "tx1\t1\t-0.522879\t-0.522879\t-0.522879"


def test_write_intron_scores_outputs_eight_decimal_places(
    tmp_path: Path,
) -> None:
    """Write intron-score TSV with fixed eight-decimal precision."""
    output_tsv = tmp_path / "intron.tsv"
    rows: list[dict[str, object]] = [
        {
            "intron_id": "uintron_00000001",
            "score": 0.12345678,
        }
    ]
    write_intron_scores(str(output_tsv), rows)
    lines = output_tsv.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "intron_id\tscore\tlabel"
    assert lines[1] == "uintron_00000001\t-0.908485\t"


def test_write_intron_scores_preserves_zero_log10_scores(
    tmp_path: Path,
) -> None:
    """Keep zero-valued log10 intron scores as ``0.000000``."""
    output_tsv = tmp_path / "intron_zero.tsv"
    rows = build_intron_scores(
        site_score_rows=[
            {
                "transcript_id": "tx1",
                "intron_index": 1,
                "site_type": "donor",
                "score": 0.5,
            },
            {
                "transcript_id": "tx1",
                "intron_index": 1,
                "site_type": "acceptor",
                "score": 0.5,
            },
        ],
        intron_score_op="+",
    )

    assert rows[0]["score"] == pytest.approx(0.0)
    assert rows[0]["_score_space"] == "log10"

    write_intron_scores(
        str(output_tsv),
        rows,
        labels={("tx1", 1): 1},
    )
    lines = output_tsv.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "tx1\t0.000000\t1"


def test_write_site_scores_outputs_wide_format(tmp_path: Path) -> None:
    """Write wide-format site_score TSV with one row per intron."""
    output_tsv = tmp_path / "site.tsv"
    rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "donor",
            "score": 0.91,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": 0.82,
        },
    ]
    write_site_scores(str(output_tsv), rows)
    lines = output_tsv.read_text(encoding="utf-8").strip().splitlines()
    assert (
        lines[0]
        == "transcript_id\tintron_index\tdonor_score\tacceptor_score\tlabel"
    )
    assert lines[1] == "tx1\t1\t-0.040959\t-0.086186"


def test_write_site_scores_fills_label_from_mapping(tmp_path: Path) -> None:
    """Fill wide-format label column from optional intron label mapping."""
    output_tsv = tmp_path / "site_with_labels.tsv"
    rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "donor",
            "score": 0.91,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": 0.82,
        },
    ]
    write_site_scores(
        str(output_tsv),
        rows,
        labels={("tx1", 1): 1},
    )
    lines = output_tsv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[1] == "tx1\t1\t-0.040959\t-0.086186\t1"


def test_read_site_scores_supports_wide_format(tmp_path: Path) -> None:
    """Read wide-format TSV into donor/acceptor row dictionaries."""
    site_score_tsv = tmp_path / "site.tsv"
    site_score_tsv.write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tdonor_score\tacceptor_score\tlabel",
                "tx1\t1\t0.900000\t0.800000\t1",
                "tx2\t2\t\t\t0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = read_site_scores(str(site_score_tsv))
    assert rows == [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "donor",
            "score": pytest.approx(math.log10(0.9)),
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": pytest.approx(math.log10(0.8)),
        },
    ]


def test_write_site_scores_keeps_pair_rows_as_blank_scores(tmp_path: Path) -> None:
    """Keep pair-only rows with blank donor/acceptor in wide output."""
    output_tsv = tmp_path / "pair.tsv"
    rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx_pair",
            "intron_index": 3,
            "site_type": "pair",
            "score": 0.55,
        }
    ]
    write_site_scores(str(output_tsv), rows)
    lines = output_tsv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[1] == "tx_pair\t3\t\t\t"


def test_write_transcript_scores_preserves_zero_log10_scores(
    tmp_path: Path,
) -> None:
    """Keep zero-valued log10 transcript scores as ``0.000000``."""
    output_tsv = tmp_path / "transcript_zero.tsv"
    rows = aggregate_pair_transcript_scores(
        site_score_rows=[
            {
                "transcript_id": "tx1",
                "intron_index": 1,
                "site_type": "pair",
                "score": 1.0,
            }
        ],
        transcript_score_agg="min",
    )

    assert rows[0]["Score_donor"] == pytest.approx(0.0)
    assert rows[0]["Score_acceptor"] == pytest.approx(0.0)
    assert rows[0]["trans_score"] == pytest.approx(0.0)
    assert rows[0]["_score_space"] == "log10"

    write_transcript_scores(str(output_tsv), rows)
    lines = output_tsv.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "tx1\t1\t0.000000\t0.000000\t0.000000"


def test_build_intron_scores_uses_pair_or_donor_acceptor() -> None:
    """Build intron scores from mixed donor/acceptor and pair rows."""
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "donor",
            "score": 0.5,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": 0.2,
        },
        {
            "transcript_id": "tx2",
            "intron_index": 2,
            "site_type": "pair",
            "score": 0.7,
        },
    ]
    rows = build_intron_scores(site_score_rows=site_rows, intron_score_op="+")
    assert rows == [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "score": pytest.approx(math.log10(0.7)),
            "_score_space": "log10",
        },
        {
            "transcript_id": "tx2",
            "intron_index": 2,
            "score": pytest.approx(math.log10(0.7)),
            "_score_space": "log10",
        },
    ]
