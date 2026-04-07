from __future__ import annotations

from util.score_test_suite_pair_filter import (
    PairCandidate,
    apply_pair_score_filter,
    build_pair_candidates,
    extract_acceptor_window,
    extract_donor_window,
)


def test_extract_donor_window_uses_pair_training_alignment() -> None:
    sequence = "ABCDEFGHIJKLMN"

    extracted = extract_donor_window(sequence, coordinate=4, window_len=6)

    assert extracted == "CDEFGH"


def test_extract_acceptor_window_uses_pair_training_alignment() -> None:
    sequence = "ABCDEFGHIJKLMN"

    extracted = extract_acceptor_window(sequence, coordinate=8, window_len=7)

    assert extracted == "GHIJKLM"


def test_build_pair_candidates_filters_by_order_distance_and_activity() -> None:
    sequence = "A" * 80
    donor_scores = {5: 0.5, 12: -1000.0}
    acceptor_scores = {20: 0.2, 33: 0.8, 40: -1000.0}

    candidates = build_pair_candidates(
        sequence=sequence,
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        donor_window_len=6,
        acceptor_window_len=7,
        inactive_score=-1000.0,
        min_intron_length=20,
    )

    coordinates = [
        (item.donor_coordinate, item.acceptor_coordinate) for item in candidates
    ]

    assert coordinates == [(5, 33)]


def test_apply_pair_score_filter_prunes_sites_without_good_pairs() -> None:
    donor_scores = {5: 10.0, 8: 20.0}
    acceptor_scores = {20: 30.0, 25: 40.0}
    pair_candidates = [
        PairCandidate(5, 20, "AAAAAA", "CCCCCCC"),
        PairCandidate(5, 25, "AAAAAA", "GGGGGGG"),
        PairCandidate(8, 25, "TTTTTT", "GGGGGGG"),
    ]

    donor_output, acceptor_output, summary = apply_pair_score_filter(
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        pair_candidates=pair_candidates,
        pair_scores=[-2.4, -1.0, -3.0],
        inactive_score=-1000.0,
        pair_keep_threshold=-1.5,
    )

    assert donor_output == {5: 10.0, 8: -1000.0}
    assert acceptor_output == {20: -1000.0, 25: 40.0}
    assert summary.donor_input_active_count == 2
    assert summary.acceptor_input_active_count == 2
    assert summary.pair_candidate_count == 3
    assert summary.donor_pruned_count == 1
    assert summary.acceptor_pruned_count == 1


def test_apply_pair_score_filter_prunes_all_active_sites_without_pairs() -> None:
    donor_output, acceptor_output, summary = apply_pair_score_filter(
        donor_scores={5: 1.0, 10: -1000.0},
        acceptor_scores={20: 2.0},
        pair_candidates=[],
        pair_scores=[],
        inactive_score=-1000.0,
        pair_keep_threshold=-2.0,
    )

    assert donor_output == {5: -1000.0, 10: -1000.0}
    assert acceptor_output == {20: -1000.0}
    assert summary.pair_candidate_count == 0
    assert summary.donor_pruned_count == 1
    assert summary.acceptor_pruned_count == 1
