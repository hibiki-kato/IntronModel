from __future__ import annotations

from tools.filter_score_test_suite_pairs import (
    backend_names_for_model_name,
    format_adjustment_summary,
    format_pair_score_summary,
)
from util.score_test_suite_pair_filter import (
    PairCandidate,
    apply_pair_score_adjustments,
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


def test_apply_pair_score_adjustments_applies_bonus_and_penalty() -> None:
    donor_scores = {5: 100.0, 8: 100.0}
    acceptor_scores = {20: 100.0, 25: 100.0}
    pair_candidates = [
        PairCandidate(5, 20, "AAAAAA", "CCCCCCC"),
        PairCandidate(5, 25, "AAAAAA", "GGGGGGG"),
        PairCandidate(8, 25, "TTTTTT", "GGGGGGG"),
    ]

    donor_output, acceptor_output, summary = apply_pair_score_adjustments(
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        pair_candidates=pair_candidates,
        pair_scores=[-2.4, -1.0, -3.0],
        inactive_score=-1000.0,
        pair_score_center=-2.0,
        pair_score_scale=50.0,
        pair_delta_min=-150.0,
        pair_delta_max=100.0,
        no_pair_penalty=-150.0,
    )

    assert donor_output == {5: 150.0, 8: 50.0}
    assert acceptor_output == {20: 80.0, 25: 150.0}
    assert summary.donor_bonus_count == 1
    assert summary.donor_penalty_count == 1
    assert summary.donor_no_pair_count == 0
    assert summary.acceptor_bonus_count == 1
    assert summary.acceptor_penalty_count == 1
    assert summary.acceptor_no_pair_count == 0


def test_apply_pair_score_adjustments_uses_no_pair_penalty_without_reject() -> None:
    donor_output, acceptor_output, summary = apply_pair_score_adjustments(
        donor_scores={5: -950.0},
        acceptor_scores={20: -980.0},
        pair_candidates=[],
        pair_scores=[],
        inactive_score=-1000.0,
        pair_score_center=-2.0,
        pair_score_scale=50.0,
        pair_delta_min=-150.0,
        pair_delta_max=100.0,
        no_pair_penalty=-200.0,
    )

    assert donor_output == {5: -999.0}
    assert acceptor_output == {20: -999.0}
    assert summary.donor_no_pair_count == 1
    assert summary.acceptor_no_pair_count == 1


def test_backend_names_for_model_name_prefers_real_pair_backends() -> None:
    assert backend_names_for_model_name("cnn_pair_v2") == ("cnn_v2",)
    assert backend_names_for_model_name("cnn_pair_v2.01") == ("cnn_v2",)
    assert backend_names_for_model_name("cnn_pair_v3") == ("cnn_pair_v3",)


def test_format_pair_score_summary_reports_threshold_hits() -> None:
    summary = format_pair_score_summary(
        [-3.0, -2.5, -0.2, -0.1],
        threshold=-2.0,
    )

    assert "2/4<=thresh(0.500)" in summary
    assert "median=-0.200" in summary


def test_format_adjustment_summary_reports_bonus_penalty_counts() -> None:
    donor_scores = {5: 10.0}
    acceptor_scores = {20: 10.0}

    _, _, summary = apply_pair_score_adjustments(
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        pair_candidates=[],
        pair_scores=[],
        inactive_score=-1000.0,
        pair_score_center=-2.0,
        pair_score_scale=50.0,
        pair_delta_min=-150.0,
        pair_delta_max=100.0,
        no_pair_penalty=-150.0,
    )

    formatted = format_adjustment_summary(summary)

    assert "donor(+0/-1/nopair1)" in formatted
    assert "acceptor(+0/-1/nopair1)" in formatted
