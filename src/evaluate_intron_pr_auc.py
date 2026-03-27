"""Evaluate intron-level PR-AUC from labeled intron and site-score TSV files.

This utility joins:
- labeled introns built by ``make_labeled_intron_eval_data.py``
- model site scores written by ``run_model.py``

and computes binary ranking metrics on intron candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
from util.transcript_eval import SCORE_OUTPUT_PRECISION
from util.unique_intron import UNIQUE_MAP_TSV_NAME, invert_unique_map, load_unique_map

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

INTRON_SCORE_OP_CHOICES: tuple[str, ...] = ("+", "*", "harmonic", "min")
SCORE_SOURCE_CHOICES: tuple[str, ...] = (
    "auto",
    "donor_acceptor",
    "pair",
    "donor",
    "acceptor",
)
SCORE_COLLAPSE_TOLERANCE: float = 2e-4


def _format_log10_score(value: float) -> str:
    """Format a log10 score for TSV output."""
    if math.isinf(value) and value < 0.0:
        return "-inf"
    return f"{value:.{SCORE_OUTPUT_PRECISION}f}"


def _log10_sum(values: list[float]) -> float:
    """Return log10(sum(10**value for value in values)) in a stable way."""
    if not values:
        raise ValueError("values must not be empty")
    finite_values = [value for value in values if not math.isinf(value)]
    if not finite_values:
        return -math.inf
    max_value = max(finite_values)
    total = math.fsum(10.0 ** (value - max_value) for value in finite_values)
    return max_value + math.log10(total)


def _set_csv_field_limit_max() -> None:
    """Set CSV parser field-size limit to the largest supported value."""
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


@dataclass(frozen=True)
class IntronEvalRow:
    """One evaluated intron row.

    Attributes
    ----------
    transcript_id : str
        Transcript identifier.
    intron_index : int
        1-based intron index in transcript order.
    label : int
        Binary truth label, ``1`` for true intron and ``0`` otherwise.
    intron_score : float
        Intron-level model score used for ranking.
    donor_score : float | None
        Donor site score when available.
    acceptor_score : float | None
        Acceptor site score when available.
    pair_score : float | None
        Pair-model score when available.
    seen_train_pos_coord : int
        1 when the coordinate appears in train-positive introns.
    seen_train_neg_seq : int
        1 when donor/acceptor sequence pair appears in train negatives.
    train_leak : int
        OR of ``seen_train_pos_coord`` and ``seen_train_neg_seq``.
    """

    transcript_id: str
    intron_index: int
    label: int
    intron_score: float
    donor_score: float | None
    acceptor_score: float | None
    pair_score: float | None
    seen_train_pos_coord: int
    seen_train_neg_seq: int
    train_leak: int


@dataclass(frozen=True)
class LabeledIntronRecord:
    """One labeled intron record keyed by ``(transcript_id, intron_index)``."""

    label: int
    seen_train_pos_coord: int
    seen_train_neg_seq: int
    train_leak: int


@dataclass(frozen=True)
class IntronEvalSummary:
    """Aggregated intron-level metric summary."""

    labeled_introns: int
    site_score_introns: int
    used_introns: int
    skipped_missing_score_introns: int
    unlabeled_site_score_introns: int
    train_leak_introns: int
    non_train_leak_introns: int
    seen_train_pos_coord_introns: int
    seen_train_neg_seq_introns: int
    positive_count: int
    negative_count: int
    positive_fraction: float
    pr_auc: float
    roc_auc: float
    intron_score_op: str
    score_source: str
    labeled_tsv: str
    site_score_tsv: str


def _binary_clf_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative false/true positives over score thresholds.

    Parameters
    ----------
    labels : np.ndarray
        1-D integer array of 0/1 labels with shape ``(n,)``.
    scores : np.ndarray
        1-D float score array with shape ``(n,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        False positives and true positives at each distinct threshold.

    Raises
    ------
    ValueError
        If inputs are not valid aligned 1-D arrays.

    Notes
    -----
    Complexity is ``O(n log n)`` time and ``O(n)`` memory.
    """
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and scores must be 1-D arrays.")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length.")
    if labels.size == 0:
        raise ValueError("labels and scores must be non-empty.")

    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(np.int64, copy=False)
    scores_sorted = scores[order]

    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_indices = np.r_[distinct_indices, labels_sorted.size - 1]

    true_positives = np.cumsum(labels_sorted)[threshold_indices]
    false_positives = (threshold_indices + 1) - true_positives
    return (
        false_positives.astype(np.float64, copy=False),
        true_positives.astype(np.float64, copy=False),
    )


def _fallback_average_precision(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute average precision without scikit-learn."""
    positives = float(np.sum(labels == 1))
    if positives <= 0.0:
        raise ValueError("At least one positive label is required.")

    false_positives, true_positives = _binary_clf_curve(labels, probs)
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / positives

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _fallback_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute ROC-AUC without scikit-learn."""
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    if positives <= 0.0 or negatives <= 0.0:
        raise ValueError("Both positive and negative labels are required.")

    false_positives, true_positives = _binary_clf_curve(labels, probs)
    fpr = np.r_[0.0, false_positives / negatives, 1.0]
    tpr = np.r_[0.0, true_positives / positives, 1.0]
    return float(np.trapezoid(tpr, fpr))


def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute PR-AUC (average precision) with sklearn fallback.

    Parameters
    ----------
    labels : np.ndarray
        1-D binary labels with shape ``(n,)``.
    scores : np.ndarray
        1-D ranking scores with shape ``(n,)``.

    Returns
    -------
    float
        Average precision score in ``[0, 1]``.
    """
    if average_precision_score is not None:
        try:
            return float(average_precision_score(labels, scores))
        except Exception:
            pass
    return _fallback_average_precision(labels, scores)


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC with sklearn fallback.

    Parameters
    ----------
    labels : np.ndarray
        1-D binary labels with shape ``(n,)``.
    scores : np.ndarray
        1-D ranking scores with shape ``(n,)``.

    Returns
    -------
    float
        ROC-AUC score in ``[0, 1]``.
    """
    if roc_auc_score is not None:
        try:
            return float(roc_auc_score(labels, scores))
        except Exception:
            pass
    return _fallback_roc_auc(labels, scores)


def _parse_optional_seen_flag(
    *,
    raw_value: str,
    column_name: str,
    path: Path,
    line_no: int,
) -> int:
    """Parse one optional seen flag to ``0`` or ``1``."""
    value = raw_value.strip().lower()
    if value in {"", "0", "false", "f", "no", "n"}:
        return 0
    if value in {"1", "true", "t", "yes", "y"}:
        return 1
    raise ValueError(
        f"Invalid {column_name} at {path}:{line_no}: expected 0/1, got {raw_value}"
    )


def _read_labeled_introns(path: Path) -> dict[tuple[str, int], LabeledIntronRecord]:
    """Read labeled intron TSV keyed by ``(transcript_id, intron_index)``."""
    if not path.exists():
        raise FileNotFoundError(f"Labeled intron TSV not found: {path}")
    _set_csv_field_limit_max()

    required = {"transcript_id", "intron_index", "label"}
    labels: dict[tuple[str, int], LabeledIntronRecord] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Labeled intron TSV must include columns: "
                "transcript_id, intron_index, label"
            )

        for line_no, raw in enumerate(reader, start=2):
            transcript_id = str(raw["transcript_id"]).strip()
            if transcript_id == "":
                raise ValueError(f"Empty transcript_id at {path}:{line_no}")
            try:
                intron_index = int(str(raw["intron_index"]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid intron_index at {path}:{line_no}"
                ) from exc
            try:
                label = int(str(raw["label"]))
            except ValueError as exc:
                raise ValueError(f"Invalid label at {path}:{line_no}") from exc
            if label not in {0, 1}:
                raise ValueError(
                    f"Label must be 0/1 at {path}:{line_no}; got {label}"
                )

            seen_train_pos_coord = _parse_optional_seen_flag(
                raw_value=str(raw.get("seen_train_pos_coord", "")),
                column_name="seen_train_pos_coord",
                path=path,
                line_no=line_no,
            )
            seen_train_neg_seq = _parse_optional_seen_flag(
                raw_value=str(raw.get("seen_train_neg_seq", "")),
                column_name="seen_train_neg_seq",
                path=path,
                line_no=line_no,
            )
            train_leak = _parse_optional_seen_flag(
                raw_value=str(raw.get("train_leak", "")),
                column_name="train_leak",
                path=path,
                line_no=line_no,
            )
            expected_seen_any = int(
                seen_train_pos_coord == 1 or seen_train_neg_seq == 1
            )
            if train_leak != expected_seen_any:
                train_leak = expected_seen_any

            key = (transcript_id, intron_index)
            record = LabeledIntronRecord(
                label=label,
                seen_train_pos_coord=seen_train_pos_coord,
                seen_train_neg_seq=seen_train_neg_seq,
                train_leak=train_leak,
            )
            previous = labels.get(key)
            if previous is not None and previous != record:
                raise ValueError(
                    "Conflicting labels for key "
                    f"{transcript_id}:{intron_index} in {path}"
                )
            labels[key] = record

    if not labels:
        raise ValueError(f"No valid labeled introns found: {path}")
    return labels


def _read_site_scores(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    """Read site-score TSV keyed by ``(transcript_id, intron_index)``."""
    if not path.exists():
        raise FileNotFoundError(f"Site score TSV not found: {path}")
    _set_csv_field_limit_max()

    legacy_required = {"transcript_id", "intron_index", "site_type", "score"}
    wide_required = {
        "transcript_id",
        "intron_index",
        "donor_score",
        "acceptor_score",
    }
    prior_wide_required = {"Transcript number", "donor score", "acceptor score"}
    site_scores: dict[tuple[str, int], dict[str, float]] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Site score TSV is missing header: {path}")
        fieldnames = set(reader.fieldnames)

        if legacy_required.issubset(fieldnames):
            for line_no, raw in enumerate(reader, start=2):
                transcript_id = str(raw["transcript_id"]).strip()
                if transcript_id == "":
                    raise ValueError(f"Empty transcript_id at {path}:{line_no}")
                try:
                    intron_index = int(str(raw["intron_index"]))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid intron_index at {path}:{line_no}"
                    ) from exc
                site_type = str(raw["site_type"]).strip().lower()
                if site_type not in {"donor", "acceptor", "pair"}:
                    raise ValueError(
                        f"Unsupported site_type '{site_type}' at {path}:{line_no}"
                    )
                try:
                    score = float(str(raw["score"]))
                except ValueError as exc:
                    raise ValueError(f"Invalid score at {path}:{line_no}") from exc

                key = (transcript_id, intron_index)
                per_site = site_scores.setdefault(key, {})
                if site_type in per_site:
                    raise ValueError(
                        "Duplicate site score for key "
                        f"{transcript_id}:{intron_index}:{site_type} in {path}"
                    )
                per_site[site_type] = score
        elif wide_required.issubset(fieldnames):
            for line_no, raw in enumerate(reader, start=2):
                transcript_id = str(raw["transcript_id"]).strip()
                if transcript_id == "":
                    raise ValueError(f"Empty transcript_id at {path}:{line_no}")
                try:
                    intron_index = int(str(raw["intron_index"]))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid intron_index at {path}:{line_no}"
                    ) from exc

                donor_raw = str(raw["donor_score"]).strip()
                acceptor_raw = str(raw["acceptor_score"]).strip()
                donor_score = float(donor_raw) if donor_raw != "" else None
                acceptor_score = float(acceptor_raw) if acceptor_raw != "" else None

                key = (transcript_id, intron_index)
                per_site = site_scores.setdefault(key, {})
                if donor_score is not None:
                    per_site["donor"] = donor_score
                if acceptor_score is not None:
                    per_site["acceptor"] = acceptor_score
        elif prior_wide_required.issubset(fieldnames):
            for line_no, raw in enumerate(reader, start=2):
                transcript_number = str(raw["Transcript number"]).strip()
                parts = transcript_number.rsplit(":", 2)
                if len(parts) != 3:
                    raise ValueError(
                        "Transcript number must be "
                        "'<transcript_id>:<intron_index>:<combined_score>' "
                        f"at {path}:{line_no}"
                    )
                transcript_id = parts[0].strip()
                if transcript_id == "":
                    raise ValueError(f"Empty transcript_id at {path}:{line_no}")
                try:
                    intron_index = int(parts[1].strip())
                except ValueError as exc:
                    raise ValueError(
                        "Invalid intron_index in Transcript number at "
                        f"{path}:{line_no}"
                    ) from exc
                combined_raw = parts[2].strip()
                try:
                    combined_score = (
                        float(combined_raw) if combined_raw != "" else None
                    )
                except ValueError as exc:
                    raise ValueError(
                        "Invalid combined score in Transcript number at "
                        f"{path}:{line_no}"
                    ) from exc

                donor_raw = str(raw["donor score"]).strip()
                acceptor_raw = str(raw["acceptor score"]).strip()
                donor_score = float(donor_raw) if donor_raw != "" else None
                acceptor_score = float(acceptor_raw) if acceptor_raw != "" else None

                key = (transcript_id, intron_index)
                per_site = site_scores.setdefault(key, {})
                if donor_score is not None:
                    per_site["donor"] = donor_score
                if acceptor_score is not None:
                    per_site["acceptor"] = acceptor_score
                if (
                    donor_score is None
                    and acceptor_score is None
                    and combined_score is not None
                ):
                    per_site["pair"] = combined_score
        else:
            raise ValueError(
                "Site score TSV must include columns: "
                "transcript_id, intron_index, site_type, score "
                "or transcript_id, intron_index, donor_score, acceptor_score"
            )

    if not site_scores:
        raise ValueError(f"No valid site score rows found: {path}")
    return site_scores


def _uses_unique_intron_ids(keys: set[tuple[str, int]]) -> bool:
    """Return whether all keys use canonical unique intron transcript IDs."""
    if not keys:
        return False
    return all(transcript_id.startswith("uintron_") for transcript_id, _ in keys)


def _resolve_unique_map_path(
    *,
    labeled_tsv: Path,
    site_score_tsv: Path,
    unique_map_tsv: Path | None,
) -> Path | None:
    """Resolve unique-map path from explicit option or input-relative defaults."""
    if unique_map_tsv is not None:
        if not unique_map_tsv.is_file():
            raise FileNotFoundError(f"Unique map TSV not found: {unique_map_tsv}")
        return unique_map_tsv

    candidates = (
        labeled_tsv.parent / UNIQUE_MAP_TSV_NAME,
        site_score_tsv.parent.parent / "processed" / UNIQUE_MAP_TSV_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _collapse_site_scores_to_unique(
    *,
    site_scores_by_key: dict[tuple[str, int], dict[str, float]],
    original_to_unique: dict[tuple[str, int], tuple[str, int]],
    tolerance: float = SCORE_COLLAPSE_TOLERANCE,
) -> dict[tuple[str, int], dict[str, float]]:
    """Collapse site scores to unique intron keys via unique-map reverse lookup.

    Rows keyed by original transcript intron IDs are mapped to one unique key.
    If multiple rows map to one unique key, donor/acceptor/pair scores must be
    consistent within tolerance.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0.")

    collapsed: dict[tuple[str, int], dict[str, float]] = {}
    for key, per_site in site_scores_by_key.items():
        unique_key = original_to_unique.get(key, key)
        target = collapsed.setdefault(unique_key, {})
        for site_type, score in per_site.items():
            previous = target.get(site_type)
            if previous is not None and abs(previous - score) > tolerance:
                raise ValueError(
                    "Conflicting scores after unique collapse. "
                    f"key={unique_key} site_type={site_type} "
                    f"score_a={previous:.8g} score_b={score:.8g}"
                )
            target[site_type] = score
    return collapsed


def _combine_intron_score(donor_score: float, acceptor_score: float, op: str) -> float:
    """Combine donor/acceptor log10 scores into one intron score."""
    if op == "+":
        return _log10_sum([donor_score, acceptor_score])
    if op == "*":
        return donor_score + acceptor_score
    if op == "harmonic":
        # log10(2ab / (a + b)) = log10(2) + log10(a) + log10(b) - log10(a + b)
        return (
            math.log10(2.0)
            + donor_score
            + acceptor_score
            - _log10_sum([donor_score, acceptor_score])
        )
    if op == "min":
        return float(min(donor_score, acceptor_score))
    raise ValueError(
        f"Unsupported intron score operation: {op}. "
        f"Supported: {INTRON_SCORE_OP_CHOICES}"
    )


def _resolve_intron_score(
    *,
    site_scores: dict[str, float],
    intron_score_op: str,
    score_source: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Resolve one intron score from donor/acceptor/pair site scores."""
    donor_score = site_scores.get("donor")
    acceptor_score = site_scores.get("acceptor")
    pair_score = site_scores.get("pair")

    if score_source == "pair":
        return pair_score, donor_score, acceptor_score, pair_score
    if score_source == "donor":
        return donor_score, donor_score, acceptor_score, pair_score
    if score_source == "acceptor":
        return acceptor_score, donor_score, acceptor_score, pair_score
    if score_source == "donor_acceptor":
        if donor_score is None or acceptor_score is None:
            return None, donor_score, acceptor_score, pair_score
        intron_score = _combine_intron_score(
            donor_score=donor_score,
            acceptor_score=acceptor_score,
            op=intron_score_op,
        )
        return intron_score, donor_score, acceptor_score, pair_score
    if score_source != "auto":
        raise ValueError(
            f"Unsupported score source: {score_source}. "
            f"Supported: {SCORE_SOURCE_CHOICES}"
        )

    if donor_score is not None and acceptor_score is not None:
        intron_score = _combine_intron_score(
            donor_score=donor_score,
            acceptor_score=acceptor_score,
            op=intron_score_op,
        )
        return intron_score, donor_score, acceptor_score, pair_score
    if pair_score is not None:
        return pair_score, donor_score, acceptor_score, pair_score
    return None, donor_score, acceptor_score, pair_score


def evaluate_labeled_introns(
    *,
    labeled_tsv: Path,
    site_score_tsv: Path,
    intron_score_op: str = "*",
    score_source: str = "auto",
    strict_missing: bool = False,
    unique_map_tsv: Path | None = None,
) -> tuple[IntronEvalSummary, list[IntronEvalRow]]:
    """Evaluate intron-level PR-AUC from label and score TSV files.

    Parameters
    ----------
    labeled_tsv : Path
        Path to intron label TSV produced by
        ``make_labeled_intron_eval_data.py``.
    site_score_tsv : Path
        Path to model site-score TSV produced by ``run_model.py``.
    intron_score_op : str, default="*"
        Donor/acceptor score merge operator.
    score_source : str, default="auto"
        Intron score source mode:
        ``auto`` uses donor+acceptor when both exist, otherwise pair;
        ``donor_acceptor`` requires donor+acceptor;
        ``pair`` requires pair score;
        ``donor`` uses donor score only;
        ``acceptor`` uses acceptor score only.
    strict_missing : bool, default=False
        If ``True``, raise when some labeled introns cannot be scored.
    unique_map_tsv : Path | None, default=None
        Optional path to ``transcripts.unique.map.tsv`` used to collapse
        original-keyed site scores into unique intron keys.

    Returns
    -------
    tuple[IntronEvalSummary, list[IntronEvalRow]]
        Summary metrics and per-intron scored rows.

    Raises
    ------
    ValueError
        If arguments are invalid, required labels/scores are missing,
        or labels do not contain both classes.
    FileNotFoundError
        If required inputs do not exist.
    """
    if intron_score_op not in INTRON_SCORE_OP_CHOICES:
        raise ValueError(
            "Unsupported intron score operation: "
            f"{intron_score_op}. Supported: {INTRON_SCORE_OP_CHOICES}"
        )
    if score_source not in SCORE_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported score source: {score_source}. "
            f"Supported: {SCORE_SOURCE_CHOICES}"
        )

    labels_by_key = _read_labeled_introns(labeled_tsv)
    site_scores_by_key = _read_site_scores(site_score_tsv)
    label_keys = set(labels_by_key.keys())
    score_keys = set(site_scores_by_key.keys())
    labels_are_unique = _uses_unique_intron_ids(label_keys)
    scores_are_unique = _uses_unique_intron_ids(score_keys)

    if labels_are_unique and not scores_are_unique:
        resolved_unique_map_path = _resolve_unique_map_path(
            labeled_tsv=labeled_tsv,
            site_score_tsv=site_score_tsv,
            unique_map_tsv=unique_map_tsv,
        )
        if resolved_unique_map_path is None:
            raise ValueError(
                "Labeled TSV uses unique intron IDs but site-score TSV does not. "
                "Could not resolve transcripts.unique.map.tsv automatically. "
                "Pass --unique-map-tsv explicitly or rewrite score files to "
                "unique keys first."
            )
        unique_map = load_unique_map(resolved_unique_map_path)
        original_to_unique = invert_unique_map(unique_map)
        site_scores_by_key = _collapse_site_scores_to_unique(
            site_scores_by_key=site_scores_by_key,
            original_to_unique=original_to_unique,
        )

    rows: list[IntronEvalRow] = []
    skipped_missing_score_introns = 0

    for key in sorted(labels_by_key.keys()):
        label_record = labels_by_key[key]
        site_scores = site_scores_by_key.get(key)
        if site_scores is None:
            skipped_missing_score_introns += 1
            continue

        (
            intron_score,
            donor_score,
            acceptor_score,
            pair_score,
        ) = _resolve_intron_score(
            site_scores=site_scores,
            intron_score_op=intron_score_op,
            score_source=score_source,
        )
        if intron_score is None:
            skipped_missing_score_introns += 1
            continue

        rows.append(
            IntronEvalRow(
                transcript_id=key[0],
                intron_index=key[1],
                label=label_record.label,
                intron_score=float(intron_score),
                donor_score=donor_score,
                acceptor_score=acceptor_score,
                pair_score=pair_score,
                seen_train_pos_coord=label_record.seen_train_pos_coord,
                seen_train_neg_seq=label_record.seen_train_neg_seq,
                train_leak=label_record.train_leak,
            )
        )

    if strict_missing and skipped_missing_score_introns > 0:
        raise ValueError(
            "Missing intron scores for labeled rows: "
            f"{skipped_missing_score_introns}"
        )
    if not rows:
        raise ValueError("No introns are scoreable after joining labels and scores.")

    labels = np.array([row.label for row in rows], dtype=np.int32)
    scores = np.array([row.intron_score for row in rows], dtype=np.float64)

    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError(
            "Both positive and negative labels are required for PR-AUC/ROC-AUC."
        )

    train_leak_introns = int(sum(row.train_leak for row in rows))
    seen_train_pos_coord_introns = int(
        sum(row.seen_train_pos_coord for row in rows)
    )
    seen_train_neg_seq_introns = int(sum(row.seen_train_neg_seq for row in rows))
    non_train_leak_introns = int(len(rows) - train_leak_introns)

    pr_auc = compute_average_precision(labels, scores)
    roc_auc = compute_roc_auc(labels, scores)

    unlabeled_site_score_introns = len(
        set(site_scores_by_key.keys()) - set(labels_by_key.keys())
    )
    summary = IntronEvalSummary(
        labeled_introns=len(labels_by_key),
        site_score_introns=len(site_scores_by_key),
        used_introns=len(rows),
        skipped_missing_score_introns=skipped_missing_score_introns,
        unlabeled_site_score_introns=unlabeled_site_score_introns,
        train_leak_introns=train_leak_introns,
        non_train_leak_introns=non_train_leak_introns,
        seen_train_pos_coord_introns=seen_train_pos_coord_introns,
        seen_train_neg_seq_introns=seen_train_neg_seq_introns,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_fraction=float(positive_count / len(rows)),
        pr_auc=float(pr_auc),
        roc_auc=float(roc_auc),
        intron_score_op=intron_score_op,
        score_source=score_source,
        labeled_tsv=str(labeled_tsv),
        site_score_tsv=str(site_score_tsv),
    )
    return summary, rows


def _write_eval_rows_tsv(path: Path, rows: list[IntronEvalRow]) -> None:
    """Write scored per-intron rows to TSV."""
    fieldnames = [
        "transcript_id",
        "intron_index",
        "label",
        "intron_score",
        "donor_score",
        "acceptor_score",
        "pair_score",
        "seen_train_pos_coord",
        "seen_train_neg_seq",
        "train_leak",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "transcript_id": row.transcript_id,
                    "intron_index": row.intron_index,
                    "label": row.label,
                    "intron_score": _format_log10_score(row.intron_score),
                    "donor_score": (
                        ""
                        if row.donor_score is None
                        else _format_log10_score(row.donor_score)
                    ),
                    "acceptor_score": (
                        ""
                        if row.acceptor_score is None
                        else _format_log10_score(row.acceptor_score)
                    ),
                    "pair_score": (
                        ""
                        if row.pair_score is None
                        else _format_log10_score(row.pair_score)
                    ),
                    "seen_train_pos_coord": row.seen_train_pos_coord,
                    "seen_train_neg_seq": row.seen_train_neg_seq,
                    "train_leak": row.train_leak,
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate intron-level PR-AUC by joining labeled introns and "
            "site-score TSV."
        )
    )
    parser.add_argument(
        "--labeled-intron-tsv",
        required=True,
        help="Path to labeled intron TSV (from make_labeled_intron_eval_data).",
    )
    parser.add_argument(
        "--site-score-tsv",
        required=True,
        help="Path to site score TSV (from run_model).",
    )
    parser.add_argument(
        "--intron-score-op",
        choices=list(INTRON_SCORE_OP_CHOICES),
        default="*",
        help="How to combine donor/acceptor into intron score.",
    )
    parser.add_argument(
        "--score-source",
        choices=list(SCORE_SOURCE_CHOICES),
        default="auto",
        help="Which score source to use for intron ranking.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if some labeled introns do not have usable scores.",
    )
    parser.add_argument(
        "--unique-map-tsv",
        default="",
        help=(
            "Optional transcripts.unique.map.tsv path for collapsing "
            "original-keyed site scores to unique intron IDs."
        ),
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON output path for metric summary.",
    )
    parser.add_argument(
        "--output-tsv",
        default="",
        help="Optional TSV output path for per-intron scored rows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    unique_map_tsv_text = str(args.unique_map_tsv).strip()
    unique_map_tsv = Path(unique_map_tsv_text) if unique_map_tsv_text != "" else None
    summary, rows = evaluate_labeled_introns(
        labeled_tsv=Path(str(args.labeled_intron_tsv)),
        site_score_tsv=Path(str(args.site_score_tsv)),
        intron_score_op=str(args.intron_score_op),
        score_source=str(args.score_source),
        strict_missing=bool(args.strict_missing),
        unique_map_tsv=unique_map_tsv,
    )

    print(
        "[intron_pr_auc] "
        f"used={summary.used_introns} "
        f"pos={summary.positive_count} "
        f"neg={summary.negative_count} "
        f"train_leak={summary.train_leak_introns} "
        f"non_train_leak={summary.non_train_leak_introns} "
        f"missing={summary.skipped_missing_score_introns} "
        f"unlabeled_site_only={summary.unlabeled_site_score_introns} "
        f"pr_auc={summary.pr_auc:.6f} "
        f"roc_auc={summary.roc_auc:.6f}"
    )

    output_json = str(args.output_json).strip()
    if output_json:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(asdict(summary), indent=2),
            encoding="utf-8",
        )
        print(f"[intron_pr_auc] wrote summary: {output_json_path}")

    output_tsv = str(args.output_tsv).strip()
    if output_tsv:
        output_tsv_path = Path(output_tsv)
        _write_eval_rows_tsv(output_tsv_path, rows)
        print(f"[intron_pr_auc] wrote rows: {output_tsv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
