"""Shared transcript-level aggregation utilities.

This module converts site-level donor/acceptor scores into transcript-level
summary scores and provides read/write helpers for score TSV files.
"""

from __future__ import annotations

import math
import os
import csv
from collections import defaultdict
from statistics import median
from typing import Dict, Iterable, List, Mapping, Sequence

INTRON_SCORE_OP_CHOICES: tuple[str, ...] = ("+", "*", "harmonic", "min")
TRANSCRIPT_SCORE_COLUMN: str = "trans_score"
SCORE_OUTPUT_PRECISION: int = 6
SCORE_SPACE_LOG10: str = "log10"
SCORE_SPACE_FIELD: str = "_score_space"
LEGACY_TRANSCRIPT_SCORE_COLUMNS: tuple[str, ...] = (
    "min_donor_plus_acceptor",
    "min_donor_times_acceptor",
)
TRANSCRIPT_SCORE_AGG_CHOICES: tuple[str, ...] = (
    "min",
    "softmin",
    "softmin_wavg",
    "+",
    "*",
    "mean",
    "avg",
    "median",
    "max",
)


def _combine_intron_score(donor_score: float, acceptor_score: float, op: str) -> float:
    """Combine donor/acceptor log10 scores into one intron score."""
    if op == "+":
        return _log10_sum([donor_score, acceptor_score])
    if op == "*":
        return donor_score + acceptor_score
    if op == "harmonic":
        return math.log10(2.0) + donor_score + acceptor_score - _log10_sum(
            [donor_score, acceptor_score]
        )
    if op == "min":
        return float(min(donor_score, acceptor_score))
    raise ValueError(f"Unsupported intron score operation: {op}")


def probability_to_log10_score(value: float) -> float:
    """Convert a probability score to log10 space.

    Parameters
    ----------
    value : float
        Probability score in ``[0, 1]``.

    Returns
    -------
    float
        Base-10 logarithm of the score. Zero maps to ``-inf``.

    Raises
    ------
    ValueError
        If ``value`` is negative.
    """
    if value < 0.0:
        raise ValueError(f"Probability score must be non-negative, got: {value}")
    if value == 0.0:
        return -math.inf
    return math.log10(value)


def log10_to_probability_score(value: float) -> float:
    """Convert a log10 score back to probability space.

    Parameters
    ----------
    value : float
        Base-10 logarithm of a probability score.

    Returns
    -------
    float
        Probability score in ``[0, 1]``. ``-inf`` maps to ``0.0``.
    """
    if math.isinf(value) and value < 0.0:
        return 0.0
    return float(10.0 ** value)


def _format_log10_score(value: float) -> str:
    """Format a log10 score for TSV output."""
    if math.isinf(value) and value < 0.0:
        return "-inf"
    return f"{value:.{SCORE_OUTPUT_PRECISION}f}"


def _row_scores_look_like_log10(scores: Sequence[float]) -> bool:
    """Return whether raw scores appear to already be in log10 space."""
    if not scores:
        return False
    return any(score < 0.0 or score > 1.0 for score in scores)


def _row_has_log10_score_space(row: Mapping[str, object]) -> bool:
    """Return whether one row is explicitly marked as log10-valued."""
    return str(row.get(SCORE_SPACE_FIELD, "")).strip().lower() == SCORE_SPACE_LOG10


def _normalize_scores_to_log10(scores: Sequence[float]) -> list[float]:
    """Normalize a score sequence to log10 space."""
    if _row_scores_look_like_log10(scores):
        return [float(score) for score in scores]
    return [probability_to_log10_score(float(score)) for score in scores]


def _log10_sum(scores: Sequence[float]) -> float:
    """Return ``log10(sum(10**score for score in scores))`` stably."""
    if not scores:
        raise ValueError("scores must not be empty")
    max_score = max(scores)
    if math.isinf(max_score) and max_score < 0.0:
        return -math.inf
    return float(
        max_score
        + math.log10(math.fsum(10.0 ** (score - max_score) for score in scores))
    )


def _softmin_exponential_sum(scores: Sequence[float], tau: float) -> float:
    """Return a soft minimum in the same scale as the input scores.

    Parameters
    ----------
    scores : Sequence[float]
        Intron score sequence.
    tau : float
        Positive temperature. Smaller values approach hard minimum.

    Returns
    -------
    float
        Soft minimum score.

    Raises
    ------
    ValueError
        If ``tau`` is not positive.
    """
    if tau <= 0.0:
        raise ValueError(f"softmin_tau must be positive, got: {tau}")

    min_score = min(scores)
    shifted_sum = math.fsum(
        math.exp(-(score - min_score) / tau) for score in scores
    )
    return float(min_score - tau * math.log(shifted_sum))


def _softmin_weighted_average(scores: Sequence[float], tau: float) -> float:
    """Return softmin-weighted average: ``sum_i w_i * scores_i``."""
    if tau <= 0.0:
        raise ValueError(f"softmin_tau must be positive, got: {tau}")

    min_score = min(scores)
    weights = [math.exp(-(score - min_score) / tau) for score in scores]
    weight_sum = math.fsum(weights)
    if weight_sum == 0.0:
        raise ValueError("Numerical underflow while computing softmin weights")
    weighted_sum = math.fsum(weight * score for weight, score in zip(weights, scores))
    return float(weighted_sum / weight_sum)


def _aggregate_transcript_score(
    scores: Sequence[float],
    agg: str,
    softmin_tau: float = 1.0,
) -> float:
    """Aggregate intron scores into a transcript score."""
    if not scores:
        raise ValueError("scores must not be empty")
    if softmin_tau <= 0.0:
        raise ValueError(f"softmin_tau must be positive, got: {softmin_tau}")

    if agg == "min":
        return float(min(scores))
    if agg == "softmin":
        return _softmin_exponential_sum(scores=scores, tau=softmin_tau)
    if agg == "softmin_wavg":
        return _softmin_weighted_average(scores=scores, tau=softmin_tau)
    if agg == "+":
        max_score = max(scores)
        return float(
            max_score
            + math.log10(
                math.fsum(10.0 ** (score - max_score) for score in scores)
            )
        )
    if agg == "*":
        return float(math.fsum(scores))
    if agg in {"mean", "avg"}:
        return float(sum(scores) / len(scores))
    if agg == "median":
        return float(median(scores))
    if agg == "max":
        return float(max(scores))
    raise ValueError(f"Unsupported transcript score aggregation: {agg}")


def aggregate_transcript_scores(
    site_score_rows: Iterable[Dict[str, object]],
    intron_score_op: str = "+",
    transcript_score_agg: str = "min",
    softmin_tau: float = 1.0,
) -> List[Dict[str, object]]:
    """Aggregate site-level scores into transcript-level rows.

    Parameters
    ----------
    site_score_rows : Iterable[dict[str, object]]
        Input row format:
        ``transcript_id``, ``intron_index``, ``site_type``
        (``donor``, ``acceptor``, or ``pair``), ``score``.
    intron_score_op : str, default="+"
        Intron score operation. Supported: ``+``, ``*``, ``harmonic``, ``min``.
    transcript_score_agg : str, default="min"
        Transcript aggregation over intron scores.
        Supported: ``min``, ``softmin``, ``softmin_wavg``, ``+``, ``*``,
        ``mean``, ``avg``, ``median``, ``max``.
    softmin_tau : float, default=1.0
        Temperature for ``softmin`` and ``softmin_wavg``. Must be positive.

    Returns
    -------
    list[dict[str, object]]
        Transcript-level rows. For compatibility, output schema stays:
        ``transcript_id``, ``min_intron_index``, ``Score_donor``,
        ``Score_acceptor``, ``trans_score``.
    """
    if intron_score_op not in INTRON_SCORE_OP_CHOICES:
        raise ValueError(
            "Unsupported intron score operation: "
            f"{intron_score_op}. Supported: {INTRON_SCORE_OP_CHOICES}"
        )
    if transcript_score_agg not in TRANSCRIPT_SCORE_AGG_CHOICES:
        raise ValueError(
            "Unsupported transcript score aggregation: "
            f"{transcript_score_agg}. Supported: {TRANSCRIPT_SCORE_AGG_CHOICES}"
        )
    if softmin_tau <= 0.0:
        raise ValueError(f"softmin_tau must be positive, got: {softmin_tau}")

    transcript_introns: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for row in site_score_rows:
        tid = str(row["transcript_id"])
        iidx = int(row["intron_index"])
        stype = str(row["site_type"])
        score = float(row["score"])
        transcript_introns[tid][iidx][stype] = score

    results: List[Dict[str, object]] = []
    for tid, introns in transcript_introns.items():
        intron_scores: dict[int, tuple[float, float, float]] = {}
        for iidx, per_site in introns.items():
            donor_raw = per_site.get("donor")
            acceptor_raw = per_site.get("acceptor")
            pair_raw = per_site.get("pair")

            if donor_raw is not None and acceptor_raw is not None:
                donor_score, acceptor_score = _normalize_scores_to_log10(
                    [float(donor_raw), float(acceptor_raw)]
                )
                intron_score = _combine_intron_score(
                    donor_score=donor_score,
                    acceptor_score=acceptor_score,
                    op=intron_score_op,
                )
            elif pair_raw is not None:
                intron_score = _normalize_scores_to_log10([float(pair_raw)])[0]
                donor_score = intron_score
                acceptor_score = intron_score
            else:
                continue

            intron_scores[iidx] = (donor_score, acceptor_score, intron_score)

        if not intron_scores:
            continue

        min_iidx = min(intron_scores.keys(), key=lambda idx: intron_scores[idx][2])
        donor_score, acceptor_score, _ = intron_scores[min_iidx]
        transcript_score = _aggregate_transcript_score(
            scores=[v[2] for v in intron_scores.values()],
            agg=transcript_score_agg,
            softmin_tau=softmin_tau,
        )
        results.append(
            {
                "transcript_id": tid,
                "min_intron_index": min_iidx,
                "Score_donor": donor_score,
                "Score_acceptor": acceptor_score,
                TRANSCRIPT_SCORE_COLUMN: transcript_score,
                SCORE_SPACE_FIELD: SCORE_SPACE_LOG10,
            }
        )

    results.sort(key=lambda x: str(x["transcript_id"]))
    return results


def aggregate_min_intron_scores(
    site_score_rows: Iterable[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Backward-compatible alias of transcript aggregation defaults."""
    return aggregate_transcript_scores(
        site_score_rows=site_score_rows,
        intron_score_op="+",
        transcript_score_agg="min",
    )


def aggregate_pair_transcript_scores(
    site_score_rows: Iterable[Dict[str, object]],
    transcript_score_agg: str = "min",
    softmin_tau: float = 1.0,
) -> List[Dict[str, object]]:
    """Aggregate pair-model site scores into transcript-level rows.

    Parameters
    ----------
    site_score_rows : Iterable[dict[str, object]]
        Input row format:
        ``transcript_id``, ``intron_index``, ``score``.
    transcript_score_agg : str, default="min"
        Transcript aggregation over per-intron pair scores.
    softmin_tau : float, default=1.0
        Temperature for ``softmin`` and ``softmin_wavg``. Must be positive.

    Returns
    -------
    list[dict[str, object]]
        Compatibility schema:
        ``transcript_id``, ``min_intron_index``, ``Score_donor``,
        ``Score_acceptor``, ``trans_score``.
        ``Score_donor`` and ``Score_acceptor`` are identical for pair mode.
    """
    if transcript_score_agg not in TRANSCRIPT_SCORE_AGG_CHOICES:
        raise ValueError(
            "Unsupported transcript score aggregation: "
            f"{transcript_score_agg}. Supported: {TRANSCRIPT_SCORE_AGG_CHOICES}"
        )
    if softmin_tau <= 0.0:
        raise ValueError(f"softmin_tau must be positive, got: {softmin_tau}")

    transcript_introns: dict[str, dict[int, float]] = defaultdict(dict)
    for row in site_score_rows:
        tid = str(row["transcript_id"])
        iidx = int(row["intron_index"])
        score = float(row["score"])
        transcript_introns[tid][iidx] = score

    results: List[Dict[str, object]] = []
    for tid, introns in transcript_introns.items():
        if not introns:
            continue
        normalized_scores = _normalize_scores_to_log10(list(introns.values()))
        normalized_items = list(zip(introns.keys(), normalized_scores, strict=True))
        min_iidx = min(normalized_items, key=lambda item: item[1])[0]
        min_score = dict(normalized_items)[min_iidx]
        transcript_score = _aggregate_transcript_score(
            scores=normalized_scores,
            agg=transcript_score_agg,
            softmin_tau=softmin_tau,
        )
        results.append(
            {
                "transcript_id": tid,
                "min_intron_index": min_iidx,
                "Score_donor": min_score,
                "Score_acceptor": min_score,
                TRANSCRIPT_SCORE_COLUMN: transcript_score,
                SCORE_SPACE_FIELD: SCORE_SPACE_LOG10,
            }
        )

    results.sort(key=lambda x: str(x["transcript_id"]))
    return results


def build_intron_scores(
    site_score_rows: Iterable[Dict[str, object]],
    intron_score_op: str = "+",
) -> List[Dict[str, object]]:
    """Build one intron-level score row per ``(transcript_id, intron_index)``.

    Parameters
    ----------
    site_score_rows : Iterable[dict[str, object]]
        Input row format:
        ``transcript_id``, ``intron_index``, ``site_type``, ``score``.
    intron_score_op : str, default="+"
        Donor/acceptor combination operator when pair score is unavailable.

    Returns
    -------
    list[dict[str, object]]
        Row schema:
        ``transcript_id``, ``intron_index``, ``score``.
        Rows without derivable score are skipped.
    """
    if intron_score_op not in INTRON_SCORE_OP_CHOICES:
        raise ValueError(
            "Unsupported intron score operation: "
            f"{intron_score_op}. Supported: {INTRON_SCORE_OP_CHOICES}"
        )

    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in site_score_rows:
        transcript_id = str(row["transcript_id"])
        intron_index = int(row["intron_index"])
        site_type = str(row["site_type"]).strip().lower()
        score = float(row["score"])
        grouped[(transcript_id, intron_index)][site_type] = score

    results: List[Dict[str, object]] = []
    for key in sorted(grouped.keys()):
        transcript_id, intron_index = key
        per_site = grouped[key]
        if "pair" in per_site:
            intron_score = _normalize_scores_to_log10([float(per_site["pair"])])[0]
        elif "donor" in per_site and "acceptor" in per_site:
            donor_score, acceptor_score = _normalize_scores_to_log10(
                [float(per_site["donor"]), float(per_site["acceptor"])]
            )
            intron_score = _combine_intron_score(
                donor_score=donor_score,
                acceptor_score=acceptor_score,
                op=intron_score_op,
            )
        else:
            continue
        results.append(
            {
                "transcript_id": transcript_id,
                "intron_index": intron_index,
                "score": intron_score,
                SCORE_SPACE_FIELD: SCORE_SPACE_LOG10,
            }
        )
    return results


def write_intron_scores(
    output_tsv: str,
    rows: List[Dict[str, object]],
    labels: Mapping[str | tuple[str, int], int] | None = None,
) -> None:
    """Write intron-level scores with optional labels.

    Parameters
    ----------
    output_tsv : str
        Output TSV path.
    rows : list[dict[str, object]]
        Row schema:
        ``intron_id`` and ``score``. For backward compatibility, legacy
        rows containing ``transcript_id`` and ``intron_index`` are also
        accepted.
    labels : Mapping[str | tuple[str, int], int] | None, default=None
        Optional intron labels keyed by ``intron_id`` or by the legacy
        ``(transcript_id, intron_index)`` pair.

    Returns
    -------
    None
        The TSV is written to ``output_tsv``.

    Raises
    ------
    KeyError
        If one row is missing both ``intron_id`` and ``transcript_id``.
    ValueError
        If ``score`` cannot be parsed as float.
    """
    outdir = os.path.dirname(output_tsv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    label_map = labels or {}

    with open(output_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["intron_id", "score", "label"])
        for row in rows:
            log10_score_space = _row_has_log10_score_space(row)
            intron_id = str(row.get("intron_id", "")).strip()
            transcript_id = str(row.get("transcript_id", "")).strip()
            intron_index_text = str(row.get("intron_index", "")).strip()
            if intron_id == "":
                if transcript_id == "":
                    raise KeyError(
                        "Intron score row must contain intron_id or "
                        "transcript_id."
                    )
                intron_id = transcript_id
            label = label_map.get(intron_id)
            if label is None and transcript_id != "" and intron_index_text != "":
                label = label_map.get((transcript_id, int(intron_index_text)))
            score = float(row["score"])
            label_text = "" if label is None else str(int(label))
            if log10_score_space or _row_scores_look_like_log10([score]):
                score_text = _format_log10_score(score)
            else:
                score_text = _format_log10_score(probability_to_log10_score(score))
            writer.writerow([intron_id, score_text, label_text])


def _get_transcript_score(row: Dict[str, object]) -> float:
    """Return the transcript score value from new or legacy row keys."""

    value = row.get(TRANSCRIPT_SCORE_COLUMN)
    if value is None:
        for legacy_key in LEGACY_TRANSCRIPT_SCORE_COLUMNS:
            value = row.get(legacy_key)
            if value is not None:
                break
    if value is None:
        raise KeyError(
            "Transcript score row is missing the trans_score column."
        )
    return float(value)


def write_transcript_scores(output_tsv: str, rows: List[Dict[str, object]]) -> None:
    """Write transcript-level score rows to a 5-column TSV file.

    Parameters
    ----------
    output_tsv : str
        Output TSV path.
    rows : list[dict[str, object]]
        Transcript-score rows containing ``transcript_id``,
        ``min_intron_index``, ``Score_donor``, ``Score_acceptor``, and
        ``trans_score`` or a legacy transcript-score key.

    Returns
    -------
    None
        This function writes a TSV file in place.

    Raises
    ------
    KeyError
        If a row does not include any transcript-score field.
    """
    outdir = os.path.dirname(output_tsv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(output_tsv, "w", encoding="utf-8", newline="") as f:
        f.write(
            "transcript_id\tmin_intron_index\tScore_donor\t"
            f"Score_acceptor\t{TRANSCRIPT_SCORE_COLUMN}\n"
        )
        for r in rows:
            log10_score_space = _row_has_log10_score_space(r)
            raw_scores = [
                float(r["Score_donor"]),
                float(r["Score_acceptor"]),
                _get_transcript_score(r),
            ]
            if log10_score_space or _row_scores_look_like_log10(raw_scores):
                donor_text = _format_log10_score(float(r["Score_donor"]))
                acceptor_text = _format_log10_score(float(r["Score_acceptor"]))
                transcript_text = _format_log10_score(_get_transcript_score(r))
            else:
                donor_text = _format_log10_score(
                    probability_to_log10_score(float(r["Score_donor"]))
                )
                acceptor_text = _format_log10_score(
                    probability_to_log10_score(float(r["Score_acceptor"]))
                )
                transcript_text = _format_log10_score(
                    probability_to_log10_score(_get_transcript_score(r))
                )
            f.write(
                f"{r['transcript_id']}\t"
                f"{r['min_intron_index']}\t"
                f"{donor_text}\t"
                f"{acceptor_text}\t"
                f"{transcript_text}\n"
            )


def write_site_scores(
    output_tsv: str,
    rows: List[Dict[str, object]],
    labels: Dict[tuple[str, int], int] | None = None,
) -> None:
    """Write site-level scores to the wide 5-column TSV format.

    Output schema:
    ``transcript_id``, ``intron_index``, ``donor_score``,
    ``acceptor_score``, ``label``.
    Donor/acceptor columns are written in log10 space. Probability inputs are
    converted to log10 values, while log10 inputs are preserved. When
    `labels` is provided, `label` is filled from
    ``(transcript_id, intron_index) -> {0,1}`` mapping.
    """
    outdir = os.path.dirname(output_tsv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    label_map = labels or {}

    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        transcript_id = str(row["transcript_id"])
        intron_index = int(row["intron_index"])
        site_type = str(row["site_type"]).strip().lower()
        score = float(row["score"])
        grouped[(transcript_id, intron_index)][site_type] = score

    with open(output_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "transcript_id",
                "intron_index",
                "donor_score",
                "acceptor_score",
                "label",
            ]
        )
        for key in sorted(grouped.keys()):
            transcript_id, intron_index = key
            per_site = grouped[key]
            donor_score = per_site.get("donor")
            acceptor_score = per_site.get("acceptor")
            pair_score = per_site.get("pair")
            raw_scores = [
                score for score in (donor_score, acceptor_score, pair_score)
                if score is not None
            ]
            if (
                donor_score is None
                and acceptor_score is None
                and pair_score is None
            ):
                continue

            if _row_scores_look_like_log10(raw_scores):
                donor_text = (
                    "" if donor_score is None else _format_log10_score(donor_score)
                )
                acceptor_text = (
                    ""
                    if acceptor_score is None
                    else _format_log10_score(acceptor_score)
                )
            else:
                donor_text = (
                    ""
                    if donor_score is None
                    else _format_log10_score(probability_to_log10_score(donor_score))
                )
                acceptor_text = (
                    ""
                    if acceptor_score is None
                    else _format_log10_score(
                        probability_to_log10_score(acceptor_score)
                    )
                )
            label = label_map.get((transcript_id, intron_index))
            label_text = "" if label is None else str(int(label))
            writer.writerow(
                [
                    transcript_id,
                    str(intron_index),
                    donor_text,
                    acceptor_text,
                    label_text,
                ]
            )


def _parse_transcript_number(value: str) -> tuple[str, int, float | None]:
    """Parse ``<transcript_id>:<intron_index>:<combined_score>`` token."""
    token = value.strip()
    parts = token.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "Transcript number must be '<transcript_id>:<intron_index>:"
            f"<combined_score>', got: {value}"
        )
    transcript_id = parts[0].strip()
    if transcript_id == "":
        raise ValueError("Transcript number contains empty transcript_id.")
    intron_index = int(parts[1])
    combined_score = float(parts[2]) if parts[2].strip() != "" else None
    return transcript_id, intron_index, combined_score


def read_site_scores(site_score_tsv: str) -> List[Dict[str, object]]:
    """Read site-score TSV in legacy long or new wide format."""
    rows: List[Dict[str, object]] = []
    raw_scores: list[float] = []

    def _append_row(row: Dict[str, object]) -> None:
        rows.append(row)
        raw_scores.append(float(row["score"]))

    def _finalize_rows() -> None:
        if _row_scores_look_like_log10(raw_scores):
            return
        for row in rows:
            row["score"] = probability_to_log10_score(float(row["score"]))

    with open(site_score_tsv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return rows
        fieldnames = set(reader.fieldnames)
        legacy_required = {"transcript_id", "intron_index", "site_type", "score"}
        wide_required = {
            "transcript_id",
            "intron_index",
            "donor_score",
            "acceptor_score",
        }
        prior_wide_required = {"Transcript number", "donor score", "acceptor score"}

        if legacy_required.issubset(fieldnames):
            for raw in reader:
                _append_row(
                    {
                        "transcript_id": str(raw["transcript_id"]).strip(),
                        "intron_index": int(str(raw["intron_index"])),
                        "site_type": str(raw["site_type"]).strip().lower(),
                        "score": float(str(raw["score"])),
                    }
                )
            _finalize_rows()
            return rows

        if wide_required.issubset(fieldnames):
            for raw in reader:
                transcript_id = str(raw["transcript_id"]).strip()
                intron_index = int(str(raw["intron_index"]))
                donor_raw = str(raw["donor_score"]).strip()
                acceptor_raw = str(raw["acceptor_score"]).strip()
                donor_score = float(donor_raw) if donor_raw != "" else None
                acceptor_score = float(acceptor_raw) if acceptor_raw != "" else None
                if donor_score is not None:
                    _append_row(
                        {
                            "transcript_id": transcript_id,
                            "intron_index": intron_index,
                            "site_type": "donor",
                            "score": donor_score,
                        }
                    )
                if acceptor_score is not None:
                    _append_row(
                        {
                            "transcript_id": transcript_id,
                            "intron_index": intron_index,
                            "site_type": "acceptor",
                            "score": acceptor_score,
                        }
                    )
            _finalize_rows()
            return rows

        if prior_wide_required.issubset(fieldnames):
            for raw in reader:
                transcript_id, intron_index, combined_score = _parse_transcript_number(
                    str(raw["Transcript number"])
                )
                donor_raw = str(raw["donor score"]).strip()
                acceptor_raw = str(raw["acceptor score"]).strip()
                donor_score = float(donor_raw) if donor_raw != "" else None
                acceptor_score = float(acceptor_raw) if acceptor_raw != "" else None
                if donor_score is not None:
                    _append_row(
                        {
                            "transcript_id": transcript_id,
                            "intron_index": intron_index,
                            "site_type": "donor",
                            "score": donor_score,
                        }
                    )
                if acceptor_score is not None:
                    _append_row(
                        {
                            "transcript_id": transcript_id,
                            "intron_index": intron_index,
                            "site_type": "acceptor",
                            "score": acceptor_score,
                        }
                    )
                if (
                    donor_score is None
                    and acceptor_score is None
                    and combined_score is not None
                ):
                    _append_row(
                        {
                            "transcript_id": transcript_id,
                            "intron_index": intron_index,
                            "site_type": "pair",
                            "score": float(combined_score),
                        }
                    )
            _finalize_rows()
            return rows

    raise ValueError(
        "Unsupported site_score TSV schema. Expected either "
        "legacy columns (transcript_id, intron_index, site_type, score) or "
        "wide columns "
        "(transcript_id, intron_index, donor_score, acceptor_score)."
    )
