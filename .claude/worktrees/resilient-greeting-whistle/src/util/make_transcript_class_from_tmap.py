#!/usr/bin/env python3
"""Parse gffcompare output and write ``transcript_class.txt``.

The output file contains one line per query transcript::

    <qry_id> <class_code>

where ``qry_id`` is the transcript_id from the input GTF (e.g.
``MSTRG_00000483:2:3.453160``) and ``class_code`` is the gffcompare
classification (e.g. ``=``, ``j``, ``c``, ``u``).

Two input formats are supported:

**``.tracking`` file** (gffcompare default output with a single query GTF;
use ``--tracking``).  Tab-separated, no header, 5 columns (0-indexed)::

    0  TCONS_id   e.g. ``TCONS_00000001|7|2971``
    1  XLOC_id    e.g. ``XLOC_000001``
    2  ref        ``ref_gene|ref_id`` or ``-`` for novel
    3  class_code e.g. ``j``, ``=``, ``u``
    4  q1_info    ``q1:<xloc>|<qry_id>|<num_exons>|<FPKM>|<TPM>|<cov>|<len>``

**``.tmap`` file** (use ``--tmap``).  Tab-separated with header row,
12 columns (0-indexed)::

    0  ref_gene_id
    1  ref_id
    2  class_code
    3  qry_gene_id
    4  qry_id
    5  num_exons
    6  FPKM
    7  TPM
    8  cov
    9  len
    10 major_iso_id
    11 ref_match_len
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


_TMAP_CLASS_CODE_COL = 2
_TMAP_QRY_ID_COL = 4
_TMAP_HEADER_PREFIX = "ref_gene_id"

# gffcompare .tracking column indices (no header row)
_TRACKING_CLASS_CODE_COL = 3
_TRACKING_Q1_INFO_COL = 4
_TRACKING_MIN_COLS = 5
# Within the q1 info field (pipe-separated), the qry_id is at index 1:
#   q1:<xloc>|<qry_id>|<num_exons>|<FPKM>|<TPM>|<cov>|<len>
_TRACKING_QRY_ID_PIPE_IDX = 1


def parse_tmap(tmap_path: Path) -> list[tuple[str, str]]:
    """Parse a gffcompare ``.tmap`` file into (qry_id, class_code) pairs.

    Each query transcript appears exactly once in the ``.tmap`` file with
    the single best-match class code assigned by gffcompare.

    Parameters
    ----------
    tmap_path : Path
        Path to the ``.tmap`` file produced by gffcompare.

    Returns
    -------
    list[tuple[str, str]]
        Ordered list of ``(qry_id, class_code)`` pairs, one per query
        transcript.

    Raises
    ------
    FileNotFoundError
        If ``tmap_path`` does not exist.
    ValueError
        If the file does not contain the expected header or columns.
    """
    if not tmap_path.is_file():
        raise FileNotFoundError(f"tmap file not found: {tmap_path}")

    records: list[tuple[str, str]] = []
    with tmap_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"tmap file is empty: {tmap_path}")
        if not header[0].startswith(_TMAP_HEADER_PREFIX):
            raise ValueError(
                f"Unexpected tmap header (expected first column to start with "
                f"'{_TMAP_HEADER_PREFIX}'): {header[0]!r}"
            )
        min_cols = max(_TMAP_CLASS_CODE_COL, _TMAP_QRY_ID_COL) + 1
        for lineno, row in enumerate(reader, start=2):
            if len(row) < min_cols:
                raise ValueError(
                    f"tmap line {lineno} has {len(row)} columns "
                    f"(expected >= {min_cols}): {row!r}"
                )
            class_code = row[_TMAP_CLASS_CODE_COL].strip()
            qry_id = row[_TMAP_QRY_ID_COL].strip()
            if not qry_id or not class_code:
                continue
            records.append((qry_id, class_code))

    return records


def parse_tracking(tracking_path: Path) -> list[tuple[str, str]]:
    """Parse a gffcompare ``.tracking`` file into (qry_id, class_code) pairs.

    The ``.tracking`` file is created by default when gffcompare is run with
    a single query GTF and a reference annotation (``-r``).  Each line
    corresponds to one query transcript (TCONS entry).

    Column layout (0-indexed, tab-separated, no header row)::

        0  TCONS_id   -- e.g. ``TCONS_00000001|7|2971``
        1  XLOC_id    -- e.g. ``XLOC_000001``
        2  ref        -- ``ref_gene|ref_id`` or ``-`` for novel transcripts
        3  class_code -- gffcompare classification (``=``, ``j``, ``u``, ...)
        4  q1_info    -- ``q1:<xloc>|<qry_id>|<num_exons>|<FPKM>|...``

    The ``qry_id`` is extracted from the pipe-delimited ``q1_info`` field at
    index 1 (after splitting on ``|``).

    Parameters
    ----------
    tracking_path : Path
        Path to the ``.tracking`` file produced by gffcompare.

    Returns
    -------
    list[tuple[str, str]]
        Ordered list of ``(qry_id, class_code)`` pairs, one per query
        transcript.

    Raises
    ------
    FileNotFoundError
        If ``tracking_path`` does not exist.
    ValueError
        If the file is empty or a data row has fewer than the required
        columns.
    """
    if not tracking_path.is_file():
        raise FileNotFoundError(f"tracking file not found: {tracking_path}")

    with tracking_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        all_rows = list(reader)

    if not all_rows:
        raise ValueError(f"tracking file is empty: {tracking_path}")

    records: list[tuple[str, str]] = []
    for lineno, row in enumerate(all_rows, start=1):
        if len(row) < _TRACKING_MIN_COLS:
            raise ValueError(
                f"tracking line {lineno} has {len(row)} columns "
                f"(expected >= {_TRACKING_MIN_COLS}): {row!r}"
            )
        class_code = row[_TRACKING_CLASS_CODE_COL].strip()
        q1_info = row[_TRACKING_Q1_INFO_COL].strip()
        parts = q1_info.split("|")
        if len(parts) <= _TRACKING_QRY_ID_PIPE_IDX:
            continue
        qry_id = parts[_TRACKING_QRY_ID_PIPE_IDX].strip()
        if not qry_id or not class_code:
            continue
        records.append((qry_id, class_code))

    return records


def write_transcript_class(
    records: list[tuple[str, str]],
    out_path: Path,
) -> None:
    """Write ``transcript_class.txt`` from (qry_id, class_code) pairs.

    Parameters
    ----------
    records : list[tuple[str, str]]
        Sequence of ``(qry_id, class_code)`` pairs as returned by
        :func:`parse_tmap` or :func:`parse_tracking`.
    out_path : Path
        Destination file path.  Parent directories are created automatically.

    Raises
    ------
    ValueError
        If ``records`` is empty.
    """
    if not records:
        raise ValueError("No records to write; transcript_class.txt would be empty.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for qry_id, class_code in records:
            handle.write(f"{qry_id} {class_code}\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse gffcompare output (.tracking or .tmap) and write "
            "transcript_class.txt."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tracking",
        type=Path,
        help="Path to gffcompare .tracking file (preferred).",
    )
    source.add_argument(
        "--tmap",
        type=Path,
        help="Path to gffcompare .tmap file (legacy).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output path for transcript_class.txt.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for CLI usage.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    out_path: Path = args.out

    if args.tracking is not None:
        input_path: Path = args.tracking
        print(f"[make_transcript_class_from_tmap] tracking={input_path}")
        print(f"[make_transcript_class_from_tmap] out={out_path}")
        records = parse_tracking(input_path)
    else:
        input_path = args.tmap
        print(f"[make_transcript_class_from_tmap] tmap={input_path}")
        print(f"[make_transcript_class_from_tmap] out={out_path}")
        records = parse_tmap(input_path)

    print(f"[make_transcript_class_from_tmap] records={len(records)}")
    write_transcript_class(records, out_path)
    print(f"[make_transcript_class_from_tmap] wrote: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
