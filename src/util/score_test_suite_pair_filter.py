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
                raise ValueError(
                    f"Expected coordinate and score at {path}:{line_no}."
                )
            coordinate = int(fields[0])
            if coordinate < 0:
                raise ValueError(
                    f"Coordinate must be non-negative at {path}:{line_no}."
                )
            score = float(fields[1])
            if coordinate in scores:
                raise ValueError(
                    f"Duplicate coordinate {coordinate} found in {path}."
                )
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
            handle.write(f"{coordinate}\t{scores[coordinate]:.6f}\n")


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
    if len(pair_candidates) != len(pair_scores):
        raise ValueError("pair_candidates and pair_scores must have the same length.")

    donor_output = {int(key): float(value) for key, value in donor_scores.items()}
    acceptor_output = {
        int(key): float(value) for key, value in acceptor_scores.items()
    }

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
