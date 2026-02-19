"""Compute evaluation counts from gffcompare outputs.

This script reads a gffcompare ``.tmap`` and ``.stats`` pair and produces:

- a ``transcript_class.txt`` (query_id -> class_code)
- a small counts file containing ``good``, ``total``, and ``ref``

The counts are intended to be consumed by ``run_model.py`` during evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from util.gffcompare_util import (
    GffCompareCounts,
    compute_good_total,
    parse_ref_count_from_stats,
    parse_tmap_classifications,
    write_counts_file,
    write_transcript_class_file,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    p = argparse.ArgumentParser(
        description="Parse gffcompare outputs and write good/total/ref counts",
    )
    p.add_argument("--tmap", required=True, help="Path to gffcompare .tmap")
    p.add_argument("--stats", required=True, help="Path to gffcompare .stats")
    p.add_argument(
        "--exclude",
        default="c",
        help="Comma-separated class codes excluded from total (default: c)",
    )
    p.add_argument(
        "--out-class",
        default=None,
        help="Output transcript_class.txt path (optional)",
    )
    p.add_argument(
        "--out-counts",
        required=True,
        help="Output counts file path (good/total/ref)",
    )
    return p


def main() -> None:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    class_by_qry = parse_tmap_classifications(args.tmap)
    excluded = {c.strip() for c in str(args.exclude).split(",") if c.strip()}
    good, total = compute_good_total(class_by_qry, exclude_class_codes=excluded)
    ref = parse_ref_count_from_stats(args.stats)
    counts = GffCompareCounts(good=good, total=total, ref=ref)

    if args.out_class not in (None, "", "None"):
        write_transcript_class_file(Path(args.out_class), class_by_qry)
    write_counts_file(Path(args.out_counts), counts)


if __name__ == "__main__":
    main()
