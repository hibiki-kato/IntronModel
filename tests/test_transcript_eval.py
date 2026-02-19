from __future__ import annotations

import math

import pytest

from util.transcript_eval import aggregate_transcript_scores


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
