"""Helpers for pair-based pruning in the score test suite workflow.

The score test suite currently feeds sparse donor/acceptor score tables into a
Viterbi-style HMM. This module adds one lightweight pre-filtering stage:

1. Keep only donor/acceptor candidates whose site score is still active.
2. Enumerate valid donor/acceptor combinations.
3. Score those combinations with a pair model.
4. Drop site candidates that never participate in a sufficiently good pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from util.score_format import format_score_text


@dataclass(frozen=True)
class PairCandidate:
    """One score-ready donor/acceptor pair candidate.

    Attributes
    ----------
    donor_coordinate : int
        0-based coordinate of the donor-side ``G`` in ``GT``.
    acceptor_coordinate : int
        0-based coordinate of the acceptor-side ``A`` in ``AG``.
    donor_window : str
        Fixed-length donor window used by the pair model.
    acceptor_window : str
        Fixed-length acceptor window used by the pair model.
    """

    donor_coordinate: int
    acceptor_coordinate: int
    donor_window: str
    acceptor_window: str


@dataclass(frozen=True)
class PairFilterSummary:
    """Summary of one pair-pruning pass.

    Attributes
    ----------
    donor_input_active_count : int
        Number of donor candidates active before pair pruning.
    acceptor_input_active_count : int
        Number of acceptor candidates active before pair pruning.
    pair_candidate_count : int
        Number of donor/acceptor combinations scored by the pair model.
    donor_pruned_count : int
        Number of donor candidates deactivated by pair pruning.
    acceptor_pruned_count : int
        Number of acceptor candidates deactivated by pair pruning.
    """

    donor_input_active_count: int
    acceptor_input_active_count: int
    pair_candidate_count: int
    donor_pruned_count: int
    acceptor_pruned_count: int


@dataclass(frozen=True)
class PairScoreAdjustmentSummary:
    """Summary of one pair-score additive update pass.

    Attributes
    ----------
    donor_input_active_count : int
        Number of donor candidates active before pair-based reweighting.
    acceptor_input_active_count : int
        Number of acceptor candidates active before pair-based reweighting.
    pair_candidate_count : int
        Number of donor/acceptor combinations scored by the pair model.
    donor_bonus_count : int
        Number of donor candidates whose score increased.
    donor_penalty_count : int
        Number of donor candidates whose score decreased.
    donor_no_pair_count : int
        Number of donor candidates that had no valid pair support.
    acceptor_bonus_count : int
        Number of acceptor candidates whose score increased.
    acceptor_penalty_count : int
        Number of acceptor candidates whose score decreased.
    acceptor_no_pair_count : int
        Number of acceptor candidates that had no valid pair support.
    """

    donor_input_active_count: int
    acceptor_input_active_count: int
    pair_candidate_count: int
    donor_bonus_count: int
    donor_penalty_count: int
    donor_no_pair_count: int
    acceptor_bonus_count: int
    acceptor_penalty_count: int
    acceptor_no_pair_count: int


def read_fasta_sequence(path: Path) -> str:
    """Read one FASTA file into an upper-cased contiguous sequence.

    Parameters
    ----------
    path : Path
        FASTA file path.

    Returns
    -------
    str
        Upper-cased sequence with headers and whitespace removed.

    Raises
    ------
    ValueError
        If the file contains no sequence content.
    """
    parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "" or line.startswith(">"):
                continue
            parts.append(line)

    sequence = "".join(parts).upper()
    if sequence == "":
        raise ValueError(f"FASTA sequence is empty: {path}")
    return sequence


def read_sparse_scores(path: Path) -> dict[int, float]:
    """Read one sparse coordinate-score table.

    Parameters
    ----------
    path : Path
        Input text file with ``coordinate<TAB>score`` rows.

    Returns
    -------
    dict[int, float]
        Mapping from 0-based coordinate to floating-point score.

    Raises
    ------
    ValueError
        If one row is malformed, duplicated, or contains an invalid value.
    """
    scores: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Expected coordinate and score at {path}:{line_no}.")
            coordinate = int(fields[0])
            if coordinate < 0:
                raise ValueError(
                    f"Coordinate must be non-negative at {path}:{line_no}."
                )
            score = float(fields[1])
            if coordinate in scores:
                raise ValueError(f"Duplicate coordinate {coordinate} found in {path}.")
            scores[coordinate] = score
    return scores


def write_sparse_scores(scores: Mapping[int, float], path: Path) -> None:
    """Write one sparse coordinate-score table.

    Parameters
    ----------
    scores : Mapping[int, float]
        Coordinate-score mapping to serialize.
    path : Path
        Destination file path.

    Returns
    -------
    None
        This function writes the file as a side effect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for coordinate in sorted(scores):
            handle.write(
                f"{coordinate}\t{format_score_text(float(scores[coordinate]))}\n"
            )


def extract_donor_window(sequence: str, coordinate: int, window_len: int) -> str | None:
    """Extract one donor window in pair-model training layout.

    The pair-data builder keeps three exonic bases on the left of the donor and
    places the donor-side ``G`` at index ``2`` of the extracted window.

    Parameters
    ----------
    sequence : str
        Full transcript sequence.
    coordinate : int
        0-based donor ``G`` coordinate.
    window_len : int
        Fixed donor window length.

    Returns
    -------
    str | None
        Fixed donor window, or ``None`` when the full context is unavailable.

    Raises
    ------
    ValueError
        If ``window_len`` is smaller than the canonical donor motif context.
    """
    if window_len < 5:
        raise ValueError("window_len must be at least 5 for donor extraction.")
    start = coordinate - 2
    end = start + window_len
    if start < 0 or end > len(sequence):
        return None
    return sequence[start:end]


def extract_acceptor_window(
    sequence: str,
    coordinate: int,
    window_len: int,
) -> str | None:
    """Extract one acceptor window in pair-model training layout.

    The pair-data builder keeps the terminal ``AG`` near the right edge and
    stores three exonic bases after the splice boundary. For an acceptor
    coordinate pointing at the ``A`` in ``AG``, the required left offset is
    ``window_len - 5``.

    Parameters
    ----------
    sequence : str
        Full transcript sequence.
    coordinate : int
        0-based acceptor ``A`` coordinate.
    window_len : int
        Fixed acceptor window length.

    Returns
    -------
    str | None
        Fixed acceptor window, or ``None`` when the full context is unavailable.

    Raises
    ------
    ValueError
        If ``window_len`` is smaller than the canonical acceptor motif context.
    """
    if window_len < 5:
        raise ValueError("window_len must be at least 5 for acceptor extraction.")
    start = coordinate - (window_len - 5)
    end = start + window_len
    if start < 0 or end > len(sequence):
        return None
    return sequence[start:end]


def active_coordinates(
    scores: Mapping[int, float],
    *,
    inactive_score: float,
) -> list[int]:
    """Return active coordinates sorted in ascending genomic order.

    Parameters
    ----------
    scores : Mapping[int, float]
        Sparse coordinate-score table.
    inactive_score : float
        Sentinel score used to mark impossible sites.

    Returns
    -------
    list[int]
        Coordinates whose score remains strictly above ``inactive_score``.
    """
    return sorted(
        coordinate
        for coordinate, score in scores.items()
        if float(score) > inactive_score
    )


def build_pair_candidates(
    *,
    sequence: str,
    donor_scores: Mapping[int, float],
    acceptor_scores: Mapping[int, float],
    donor_window_len: int,
    acceptor_window_len: int,
    inactive_score: float,
    min_intron_length: int,
) -> list[PairCandidate]:
    """Build valid donor/acceptor combinations for pair-model scoring.

    The algorithm precomputes extractable windows for active donor and acceptor
    candidates, then enumerates valid ordered combinations. The time complexity
    is ``O(D + A + D*A)`` where ``D`` and ``A`` are the numbers of active donor
    and acceptor candidates after the initial site-level pruning.

    Parameters
    ----------
    sequence : str
        Full transcript sequence.
    donor_scores : Mapping[int, float]
        Sparse donor score table.
    acceptor_scores : Mapping[int, float]
        Sparse acceptor score table.
    donor_window_len : int
        Fixed donor window length required by the pair model.
    acceptor_window_len : int
        Fixed acceptor window length required by the pair model.
    inactive_score : float
        Sentinel score used to mark impossible sites.
    min_intron_length : int
        Minimum donor-to-acceptor distance to consider pair-feasible.

    Returns
    -------
    list[PairCandidate]
        Score-ready donor/acceptor pair candidates.

    Raises
    ------
    ValueError
        If ``min_intron_length`` is negative.
    """
    if min_intron_length < 0:
        raise ValueError("min_intron_length must be non-negative.")

    donor_windows: dict[int, str] = {}
    for coordinate in active_coordinates(
        donor_scores,
        inactive_score=inactive_score,
    ):
        window = extract_donor_window(sequence, coordinate, donor_window_len)
        if window is not None:
            donor_windows[coordinate] = window

    acceptor_windows: dict[int, str] = {}
    for coordinate in active_coordinates(
        acceptor_scores,
        inactive_score=inactive_score,
    ):
        window = extract_acceptor_window(sequence, coordinate, acceptor_window_len)
        if window is not None:
            acceptor_windows[coordinate] = window

    candidates: list[PairCandidate] = []
    for donor_coordinate in sorted(donor_windows):
        donor_window = donor_windows[donor_coordinate]
        for acceptor_coordinate in sorted(acceptor_windows):
            if acceptor_coordinate <= donor_coordinate:
                continue
            if acceptor_coordinate - donor_coordinate < min_intron_length:
                continue
            candidates.append(
                PairCandidate(
                    donor_coordinate=donor_coordinate,
                    acceptor_coordinate=acceptor_coordinate,
                    donor_window=donor_window,
                    acceptor_window=acceptor_windows[acceptor_coordinate],
                )
            )
    return candidates


def compute_best_pair_scores(
    *,
    donor_scores: Mapping[int, float],
    acceptor_scores: Mapping[int, float],
    pair_candidates: Sequence[PairCandidate],
    pair_scores: Sequence[float],
    inactive_score: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return best pair score touching each active donor and acceptor site.

    Parameters
    ----------
    donor_scores : Mapping[int, float]
        Original sparse donor score table.
    acceptor_scores : Mapping[int, float]
        Original sparse acceptor score table.
    pair_candidates : Sequence[PairCandidate]
        Pair candidates in the same order used for pair-model inference.
    pair_scores : Sequence[float]
        Pair-model scores aligned with ``pair_candidates``.
    inactive_score : float
        Sentinel score used to mark impossible sites.

    Returns
    -------
    tuple[dict[int, float], dict[int, float]]
        Best pair score for each active donor coordinate and each active
        acceptor coordinate. Coordinates without any valid pair keep
        ``-inf``.

    Raises
    ------
    ValueError
        If ``pair_candidates`` and ``pair_scores`` do not have the same length.
    """
    if len(pair_candidates) != len(pair_scores):
        raise ValueError("pair_candidates and pair_scores must have the same length.")

    donor_best: dict[int, float] = {
        coordinate: float("-inf")
        for coordinate in active_coordinates(
            donor_scores,
            inactive_score=inactive_score,
        )
    }
    acceptor_best: dict[int, float] = {
        coordinate: float("-inf")
        for coordinate in active_coordinates(
            acceptor_scores,
            inactive_score=inactive_score,
        )
    }

    for pair_candidate, pair_score in zip(pair_candidates, pair_scores, strict=True):
        donor_best[pair_candidate.donor_coordinate] = max(
            donor_best.get(pair_candidate.donor_coordinate, float("-inf")),
            float(pair_score),
        )
        acceptor_best[pair_candidate.acceptor_coordinate] = max(
            acceptor_best.get(pair_candidate.acceptor_coordinate, float("-inf")),
            float(pair_score),
        )

    return donor_best, acceptor_best


def apply_pair_score_filter(
    *,
    donor_scores: Mapping[int, float],
    acceptor_scores: Mapping[int, float],
    pair_candidates: Sequence[PairCandidate],
    pair_scores: Sequence[float],
    inactive_score: float,
    pair_keep_threshold: float,
) -> tuple[dict[int, float], dict[int, float], PairFilterSummary]:
    """Deactivate sites that never appear in a sufficiently good pair.

    The filter computes the best pair score touching each active donor and each
    active acceptor. Candidates whose best achievable pair score remains below
    ``pair_keep_threshold`` are reset to ``inactive_score``. This scan is
    linear in the number of pair candidates, namely ``O(P)``.

    Parameters
    ----------
    donor_scores : Mapping[int, float]
        Original sparse donor score table.
    acceptor_scores : Mapping[int, float]
        Original sparse acceptor score table.
    pair_candidates : Sequence[PairCandidate]
        Pair candidates in the same order used for pair-model inference.
    pair_scores : Sequence[float]
        Pair-model scores aligned with ``pair_candidates``.
    inactive_score : float
        Sentinel score written for pruned sites.
    pair_keep_threshold : float
        Minimum acceptable pair score. Sites with no pair at or above this
        value are deactivated.

    Returns
    -------
    tuple[dict[int, float], dict[int, float], PairFilterSummary]
        Pruned donor scores, pruned acceptor scores, and a summary object.

    Raises
    ------
    ValueError
        If ``pair_candidates`` and ``pair_scores`` do not have the same length.
    """
    donor_output = {int(key): float(value) for key, value in donor_scores.items()}
    acceptor_output = {int(key): float(value) for key, value in acceptor_scores.items()}

    donor_best, acceptor_best = compute_best_pair_scores(
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        pair_candidates=pair_candidates,
        pair_scores=pair_scores,
        inactive_score=inactive_score,
    )

    donor_pruned_count = 0
    for coordinate, best_score in donor_best.items():
        if best_score < pair_keep_threshold:
            donor_output[coordinate] = inactive_score
            donor_pruned_count += 1

    acceptor_pruned_count = 0
    for coordinate, best_score in acceptor_best.items():
        if best_score < pair_keep_threshold:
            acceptor_output[coordinate] = inactive_score
            acceptor_pruned_count += 1

    summary = PairFilterSummary(
        donor_input_active_count=len(donor_best),
        acceptor_input_active_count=len(acceptor_best),
        pair_candidate_count=len(pair_candidates),
        donor_pruned_count=donor_pruned_count,
        acceptor_pruned_count=acceptor_pruned_count,
    )
    return donor_output, acceptor_output, summary


def compute_pair_score_delta(
    *,
    best_pair_score: float,
    pair_score_center: float,
    pair_score_scale: float,
    pair_delta_min: float,
    pair_delta_max: float,
    no_pair_penalty: float,
) -> tuple[float, bool]:
    """Map one best pair score to one additive site-score delta.

    Parameters
    ----------
    best_pair_score : float
        Best pair score touching one site. ``-inf`` means no valid pair.
    pair_score_center : float
        Neutral pair score. Values above this score receive positive deltas,
        and values below it receive negative deltas.
    pair_score_scale : float
        Multiplier applied to the centered pair score difference.
    pair_delta_min : float
        Minimum allowed additive delta.
    pair_delta_max : float
        Maximum allowed additive delta.
    no_pair_penalty : float
        Additive penalty applied when no pair score is available.

    Returns
    -------
    tuple[float, bool]
        ``(delta, had_no_pair)`` where ``delta`` is the additive score update.

    Raises
    ------
    ValueError
        If the delta bounds are invalid or the scale is negative.
    """
    if pair_score_scale < 0.0:
        raise ValueError("pair_score_scale must be non-negative.")
    if pair_delta_min > pair_delta_max:
        raise ValueError("pair_delta_min must be <= pair_delta_max.")

    if best_pair_score == float("-inf"):
        return float(no_pair_penalty), True

    raw_delta = (float(best_pair_score) - pair_score_center) * pair_score_scale
    bounded_delta = min(max(raw_delta, pair_delta_min), pair_delta_max)
    return float(bounded_delta), False


def apply_pair_score_adjustments(
    *,
    donor_scores: Mapping[int, float],
    acceptor_scores: Mapping[int, float],
    pair_candidates: Sequence[PairCandidate],
    pair_scores: Sequence[float],
    inactive_score: float,
    pair_score_center: float,
    pair_score_scale: float,
    pair_delta_min: float,
    pair_delta_max: float,
    no_pair_penalty: float,
) -> tuple[dict[int, float], dict[int, float], PairScoreAdjustmentSummary]:
    """Adjust donor and acceptor site scores using pair support.

    The algorithm first computes the best pair score touching each active site,
    then converts that best score into an additive delta. Scores better than
    ``pair_score_center`` get a positive bonus, while worse scores receive a
    penalty. Sites without any valid pair receive ``no_pair_penalty``. The
    update is clipped to stay above the inactive sentinel, so this mode never
    performs a hard reject.

    Parameters
    ----------
    donor_scores : Mapping[int, float]
        Original sparse donor score table.
    acceptor_scores : Mapping[int, float]
        Original sparse acceptor score table.
    pair_candidates : Sequence[PairCandidate]
        Pair candidates in the same order used for pair-model inference.
    pair_scores : Sequence[float]
        Pair-model scores aligned with ``pair_candidates``.
    inactive_score : float
        Sentinel score used to mark impossible sites.
    pair_score_center : float
        Neutral pair score for additive updates.
    pair_score_scale : float
        Scale factor applied to centered pair scores.
    pair_delta_min : float
        Minimum allowed additive delta.
    pair_delta_max : float
        Maximum allowed additive delta.
    no_pair_penalty : float
        Additive penalty applied when a site has no valid pair support.

    Returns
    -------
    tuple[dict[int, float], dict[int, float], PairScoreAdjustmentSummary]
        Updated donor scores, updated acceptor scores, and one summary object.
    """
    donor_output = {int(key): float(value) for key, value in donor_scores.items()}
    acceptor_output = {int(key): float(value) for key, value in acceptor_scores.items()}
    donor_best, acceptor_best = compute_best_pair_scores(
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        pair_candidates=pair_candidates,
        pair_scores=pair_scores,
        inactive_score=inactive_score,
    )

    minimum_active_score = float(inactive_score) + 1.0
    donor_bonus_count = 0
    donor_penalty_count = 0
    donor_no_pair_count = 0
    for coordinate, best_score in donor_best.items():
        delta, had_no_pair = compute_pair_score_delta(
            best_pair_score=best_score,
            pair_score_center=pair_score_center,
            pair_score_scale=pair_score_scale,
            pair_delta_min=pair_delta_min,
            pair_delta_max=pair_delta_max,
            no_pair_penalty=no_pair_penalty,
        )
        if had_no_pair:
            donor_no_pair_count += 1
        if delta > 0.0:
            donor_bonus_count += 1
        elif delta < 0.0:
            donor_penalty_count += 1
        donor_output[coordinate] = max(
            minimum_active_score,
            donor_output[coordinate] + delta,
        )

    acceptor_bonus_count = 0
    acceptor_penalty_count = 0
    acceptor_no_pair_count = 0
    for coordinate, best_score in acceptor_best.items():
        delta, had_no_pair = compute_pair_score_delta(
            best_pair_score=best_score,
            pair_score_center=pair_score_center,
            pair_score_scale=pair_score_scale,
            pair_delta_min=pair_delta_min,
            pair_delta_max=pair_delta_max,
            no_pair_penalty=no_pair_penalty,
        )
        if had_no_pair:
            acceptor_no_pair_count += 1
        if delta > 0.0:
            acceptor_bonus_count += 1
        elif delta < 0.0:
            acceptor_penalty_count += 1
        acceptor_output[coordinate] = max(
            minimum_active_score,
            acceptor_output[coordinate] + delta,
        )

    summary = PairScoreAdjustmentSummary(
        donor_input_active_count=len(donor_best),
        acceptor_input_active_count=len(acceptor_best),
        pair_candidate_count=len(pair_candidates),
        donor_bonus_count=donor_bonus_count,
        donor_penalty_count=donor_penalty_count,
        donor_no_pair_count=donor_no_pair_count,
        acceptor_bonus_count=acceptor_bonus_count,
        acceptor_penalty_count=acceptor_penalty_count,
        acceptor_no_pair_count=acceptor_no_pair_count,
    )
    return donor_output, acceptor_output, summary
