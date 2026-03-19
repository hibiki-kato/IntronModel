#!/usr/bin/env python3
"""Extract transcript class labels from gffcompare annotated GTF output."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_TRANSCRIPT_ID_PATTERN = re.compile(r'transcript_id "([^"]+)"')
_CLASS_CODE_PATTERN = re.compile(r'class_code "([^"]+)"')


def parse_annotated_gtf(gtf_path: Path) -> list[tuple[str, str]]:
    """Parse transcript_id and class_code pairs from an annotated GTF file.

    Parameters
    ----------
    gtf_path : Path
        Path to gffcompare ``<prefix>.annotated.gtf``.

    Returns
    -------
    list[tuple[str, str]]
        Ordered ``(transcript_id, class_code)`` records from transcript rows.

    Raises
    ------
    FileNotFoundError
        If ``gtf_path`` does not exist.
    ValueError
        If a non-comment row has fewer than 9 tab-delimited columns.
    """
    if not gtf_path.is_file():
        raise FileNotFoundError(f"annotated gtf file not found: {gtf_path}")

    records: list[tuple[str, str]] = []
    with gtf_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                raise ValueError(
                    f"annotated gtf line {lineno} has {len(fields)} columns "
                    "(expected >= 9)"
                )
            if fields[2] != "transcript":
                continue

            attrs = fields[8]
            transcript_match = _TRANSCRIPT_ID_PATTERN.search(attrs)
            class_match = _CLASS_CODE_PATTERN.search(attrs)
            if transcript_match is None or class_match is None:
                continue

            transcript_id = transcript_match.group(1).strip()
            class_code = class_match.group(1).strip()
            if not transcript_id or not class_code:
                continue
            records.append((transcript_id, class_code))

    return records


def write_transcript_class(records: list[tuple[str, str]], out_path: Path) -> None:
    """Write transcript class records to ``transcript_class.txt`` format.

    Parameters
    ----------
    records : list[tuple[str, str]]
        ``(transcript_id, class_code)`` records to write.
    out_path : Path
        Output path.

    Raises
    ------
    ValueError
        If ``records`` is empty.
    """
    if not records:
        raise ValueError("No records to write; transcript_class.txt would be empty.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for transcript_id, class_code in records:
            handle.write(f"{transcript_id} {class_code}\n")


def extract_transcript_classes(gtf_path: Path, out_path: Path) -> int:
    """Extract transcript classes from annotated GTF and write output file.

    Parameters
    ----------
    gtf_path : Path
        Path to annotated GTF file.
    out_path : Path
        Destination output path.

    Returns
    -------
    int
        Number of written records.
    """
    records = parse_annotated_gtf(gtf_path)
    write_transcript_class(records, out_path)
    return len(records)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract transcript_id/class_code pairs from gffcompare "
            "annotated GTF and write transcript_class.txt."
        )
    )
    parser.add_argument(
        "--gtf",
        required=True,
        type=Path,
        help="Path to gffcompare <prefix>.annotated.gtf file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output path for transcript_class.txt.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run command-line interface."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    records_count = extract_transcript_classes(gtf_path=args.gtf, out_path=args.out)
    print(f"[make_transcript_class_from_annotated_gtf] gtf={args.gtf}")
    print(f"[make_transcript_class_from_annotated_gtf] out={args.out}")
    print(f"[make_transcript_class_from_annotated_gtf] records={records_count}")


if __name__ == "__main__":
    main()
