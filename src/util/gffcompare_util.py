"""Utilities for parsing gffcompare outputs.

This module extracts transcript-level classification and summary counts from
gffcompare output files.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GffCompareCounts:
    """Counts used by the evaluation stage.

    Attributes
    ----------
    good : int
        Number of query transcripts considered "good" (class code '=').
    total : int
        Number of query transcripts considered in precision denominator.
    ref : int
        Number of reference transcripts considered in sensitivity denominator.
    """

    good: int
    total: int
    ref: int


def parse_tmap_classifications(tmap_path: str | Path) -> dict[str, str]:
    """Parse a gffcompare ``.tmap`` file into query_id -> class_code mapping.

    Parameters
    ----------
    tmap_path : str | pathlib.Path
        Path to a gffcompare ``.tmap`` file.

    Returns
    -------
    dict[str, str]
        Mapping from query transcript id (``qry_id``) to class code.

    Raises
    ------
    FileNotFoundError
        If ``tmap_path`` does not exist.
    ValueError
        If required columns are missing.
    """

    path = Path(tmap_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header: list[str] | None = None
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                # Header may be commented.
                row[0] = row[0].lstrip("#")
            header = [cell.strip() for cell in row]
            break

        if header is None:
            raise ValueError(f"Empty tmap file: {path}")

        try:
            qry_idx = header.index("qry_id")
            code_idx = header.index("class_code")
        except ValueError as exc:
            raise ValueError(
                "tmap header missing required columns: qry_id, class_code"
            ) from exc

        mapping: dict[str, str] = {}
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) <= max(qry_idx, code_idx):
                continue
            qry_id = row[qry_idx].strip()
            class_code = row[code_idx].strip()
            if not qry_id:
                continue
            mapping[qry_id] = class_code

    return mapping


def compute_good_total(
    class_by_qry_id: dict[str, str],
    *,
    exclude_class_codes: set[str] | None = None,
    good_class_code: str = "=",
) -> tuple[int, int]:
    """Compute good/total counts from class mapping.

    Parameters
    ----------
    class_by_qry_id : dict[str, str]
        Mapping from query transcript id to gffcompare class code.
    exclude_class_codes : set[str] | None, default=None
        Codes excluded from the evaluation set. If None, defaults to {'c'} to
        match ``evaluate_scores.py`` filtering.
    good_class_code : str, default="="
        Code considered "good".

    Returns
    -------
    tuple[int, int]
        (good, total) counts.
    """

    excluded = exclude_class_codes if exclude_class_codes is not None else {"c"}
    total = 0
    good = 0
    for code in class_by_qry_id.values():
        if code in excluded:
            continue
        total += 1
        if code == good_class_code:
            good += 1
    return good, total


_REF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*Reference\s+mRNAs\s*:\s*(\d+)\b"),
    re.compile(r"^\s*Reference\s+transcripts\s*:\s*(\d+)\b"),
)


def parse_ref_count_from_stats(stats_path: str | Path) -> int:
    """Parse reference transcript count from a gffcompare ``.stats`` file.

    Parameters
    ----------
    stats_path : str | pathlib.Path
        Path to a gffcompare ``.stats`` file.

    Returns
    -------
    int
        Parsed reference transcript count.

    Raises
    ------
    FileNotFoundError
        If the stats file does not exist.
    ValueError
        If reference count cannot be found.
    """

    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r") as f:
        for line in f:
            for pat in _REF_PATTERNS:
                m = pat.search(line)
                if m:
                    return int(m.group(1))

    raise ValueError(f"Reference count not found in stats file: {path}")


def write_transcript_class_file(
    output_path: str | Path,
    class_by_qry_id: dict[str, str],
) -> None:
    """Write ``transcript_class.txt`` compatible with this repository.

    Parameters
    ----------
    output_path : str | pathlib.Path
        Output file path.
    class_by_qry_id : dict[str, str]
        Mapping from query transcript id to class code.
    """

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for tid in sorted(class_by_qry_id.keys()):
            f.write(f"{tid} {class_by_qry_id[tid]}\n")


def write_counts_file(output_path: str | Path, counts: GffCompareCounts) -> None:
    """Write a simple key-value count file for shell consumption."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"good\t{counts.good}\n")
        f.write(f"total\t{counts.total}\n")
        f.write(f"ref\t{counts.ref}\n")
