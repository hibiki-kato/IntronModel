"""Shared utilities for unique-intron assets.

This module centralizes file naming and parsing helpers used by:
- ``src/tools/build_unique_intron_assets.py``
- ``src/run_model.py`` map-back logic
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping

UNIQUE_TRANSCRIPTS_TSV_NAME: str = "transcripts.unique.tsv"
UNIQUE_MAP_TSV_NAME: str = "transcripts.unique.map.tsv"
UNIQUE_LABELED_INTRON_TSV_NAME: str = "intron_eval_flank10.unique.tsv"
UNIQUE_INTRON_CATALOG_TSV_NAME: str = "intron_unique_catalog.tsv"


@dataclass(frozen=True)
class UniqueMapMember:
    """One original intron member row mapped from one unique intron."""

    transcript_id: str
    intron_index: int


def set_csv_field_limit_max() -> None:
    """Set CSV parser field-size limit to the largest supported value."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def unique_asset_paths(species_dir: Path) -> dict[str, Path]:
    """Return canonical unique-intron asset paths under one species directory.

    Parameters
    ----------
    species_dir : Path
        Species directory, such as ``data/Dmel``.

    Returns
    -------
    dict[str, Path]
        Mapping with keys:
        ``transcripts_unique``, ``unique_map``, ``labeled_unique``, ``catalog``.
    """
    processed_dir = species_dir / "processed"
    return {
        "transcripts_unique": processed_dir / UNIQUE_TRANSCRIPTS_TSV_NAME,
        "unique_map": processed_dir / UNIQUE_MAP_TSV_NAME,
        "labeled_unique": processed_dir / UNIQUE_LABELED_INTRON_TSV_NAME,
        "catalog": processed_dir / UNIQUE_INTRON_CATALOG_TSV_NAME,
    }


def _parse_row_int(
    raw_row: Mapping[str, str],
    key: str,
    path: Path,
    line_no: int,
) -> int:
    """Parse one required integer field from a CSV row."""
    raw_value = str(raw_row.get(key, "")).strip()
    if raw_value == "":
        raise ValueError(f"Empty '{key}' at {path}:{line_no}")
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid '{key}' at {path}:{line_no}: {raw_value}") from exc


def parse_bool_flag(raw_value: str) -> int:
    """Parse one bool-like token to ``0`` or ``1``.

    Parameters
    ----------
    raw_value : str
        Raw text value.

    Returns
    -------
    int
        ``1`` for true-like values and ``0`` for false-like values.

    Raises
    ------
    ValueError
        If the token cannot be interpreted as bool-like.
    """
    value = raw_value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return 1
    if value in {"0", "false", "f", "no", "n", ""}:
        return 0
    raise ValueError(f"Invalid boolean-like flag: {raw_value}")


def load_unique_map(
    path: Path,
) -> dict[tuple[str, int], list[UniqueMapMember]]:
    """Load unique-to-original intron mapping from TSV.

    Parameters
    ----------
    path : Path
        TSV path for ``transcripts.unique.map.tsv``.

    Returns
    -------
    dict[tuple[str, int], list[UniqueMapMember]]
        Mapping from ``(unique_transcript_id, unique_intron_index)`` to ordered
        original transcript members.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the TSV schema or row values are invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Unique intron map TSV not found: {path}")

    set_csv_field_limit_max()
    required = {
        "unique_transcript_id",
        "unique_intron_index",
        "transcript_id",
        "intron_index",
    }
    mapping: dict[tuple[str, int], list[UniqueMapMember]] = {}
    seen_pairs: set[tuple[str, int, str, int]] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Unique map TSV must include columns: "
                "unique_transcript_id, unique_intron_index, transcript_id, "
                "intron_index"
            )

        for line_no, raw_row in enumerate(reader, start=2):
            unique_transcript_id = str(raw_row["unique_transcript_id"]).strip()
            if unique_transcript_id == "":
                raise ValueError(f"Empty unique_transcript_id at {path}:{line_no}")
            unique_intron_index = _parse_row_int(
                raw_row=raw_row,
                key="unique_intron_index",
                path=path,
                line_no=line_no,
            )
            transcript_id = str(raw_row["transcript_id"]).strip()
            if transcript_id == "":
                raise ValueError(f"Empty transcript_id at {path}:{line_no}")
            intron_index = _parse_row_int(
                raw_row=raw_row,
                key="intron_index",
                path=path,
                line_no=line_no,
            )

            pair_key = (
                unique_transcript_id,
                unique_intron_index,
                transcript_id,
                intron_index,
            )
            if pair_key in seen_pairs:
                raise ValueError(
                    "Duplicate mapping row in unique map TSV: "
                    f"{pair_key} at {path}:{line_no}"
                )
            seen_pairs.add(pair_key)
            mapping.setdefault(
                (unique_transcript_id, unique_intron_index),
                [],
            ).append(
                UniqueMapMember(
                    transcript_id=transcript_id,
                    intron_index=intron_index,
                )
            )

    if not mapping:
        raise ValueError(f"No valid rows in unique map TSV: {path}")
    return mapping


def invert_unique_map(
    unique_map: Mapping[tuple[str, int], Iterable[UniqueMapMember]],
) -> Dict[tuple[str, int], tuple[str, int]]:
    """Build original-to-unique reverse lookup from a unique map.

    Parameters
    ----------
    unique_map : Mapping[tuple[str, int], Iterable[UniqueMapMember]]
        Forward unique-to-original mapping.

    Returns
    -------
    dict[tuple[str, int], tuple[str, int]]
        Reverse mapping keyed by original ``(transcript_id, intron_index)``.

    Raises
    ------
    ValueError
        If one original key maps to multiple unique keys.
    """
    reverse_map: dict[tuple[str, int], tuple[str, int]] = {}
    for unique_key, members in unique_map.items():
        for member in members:
            original_key = (member.transcript_id, member.intron_index)
            previous = reverse_map.get(original_key)
            if previous is not None and previous != unique_key:
                raise ValueError(
                    "One original intron maps to multiple unique introns: "
                    f"{original_key} -> {previous} and {unique_key}"
                )
            reverse_map[original_key] = unique_key
    return reverse_map
