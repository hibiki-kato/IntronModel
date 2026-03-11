from __future__ import annotations

import math
from pathlib import Path

import pytest

from util.transcript_eval import (
    aggregate_pair_transcript_scores,
    aggregate_transcript_scores,
    read_site_scores,
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
            "score": 1.5,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 2,
            "site_type": "acceptor",
            "score": 1.5,
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
    intron_scores = [1.0, 3.0]
    expected = math.exp(-1.0) + math.exp(-3.0)

    assert result["min_intron_index"] == 1
    assert float(result["min_donor_plus_acceptor"]) == pytest.approx(expected)


def test_aggregate_transcript_scores_softmin_wavg() -> None:
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
            "score": 1.5,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 2,
            "site_type": "acceptor",
            "score": 1.5,
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
    intron_scores = [1.0, 3.0]
    min_score = min(intron_scores)
    weights = [math.exp(-(score - min_score)) for score in intron_scores]
    expected = (
        weights[0] * intron_scores[0] + weights[1] * intron_scores[1]
    ) / sum(weights)

    assert result["min_intron_index"] == 1
    assert float(result["min_donor_plus_acceptor"]) == pytest.approx(expected)


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
            "Score_donor": 0.3,
            "Score_acceptor": 0.3,
            "min_donor_plus_acceptor": 0.3,
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
    assert float(rows[0]["min_donor_plus_acceptor"]) == pytest.approx(expected)


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
    assert lines[0] == "Transcript number\tdonor score\tacceptor score\tlabel"
    assert lines[1] == "tx1:1:1.730000\t0.910000\t0.820000"


def test_read_site_scores_supports_wide_format(tmp_path: Path) -> None:
    """Read wide-format TSV into donor/acceptor row dictionaries."""
    site_score_tsv = tmp_path / "site.tsv"
    site_score_tsv.write_text(
        "\n".join(
            [
                "Transcript number\tdonor score\tacceptor score\tlabel",
                "tx1:1:1.700000\t0.900000\t0.800000\t1",
                "tx2:2:0.250000\t\t\t0",
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
            "score": 0.9,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": 0.8,
        },
        {
            "transcript_id": "tx2",
            "intron_index": 2,
            "site_type": "pair",
            "score": 0.25,
        },
    ]
