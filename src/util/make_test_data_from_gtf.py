#!/usr/bin/env python3
"""Generate transcript splice-site test data from genome FASTA and GTF.

This script extracts donor/acceptor-centered sequence windows for introns and
writes a transcript-level TSV suitable for downstream model scoring.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from Bio import SeqIO

DONOR_EXON_BP = 3
ACCEPTOR_EXON_BP = 3


# ----------------------------
# Helpers
# ----------------------------
def parse_gtf_attributes(attr_str: str) -> dict[str, str]:
    """
    Parse GTF attributes column into a dict.
    Example: transcript_id "tx1"; gene_id "g1";
    """
    out: dict[str, str] = {}
    parts = [p.strip() for p in attr_str.strip().strip(";").split(";")]
    for p in parts:
        if not p:
            continue
        # Usually: key "value"
        if " " not in p:
            continue
        key, rest = p.split(" ", 1)
        val = rest.strip().strip('"')
        out[key] = val
    return out


def revcomp(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]


@dataclass
class Exon:
    chrom: str
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive
    strand: str
    transcript_id: str
    gene_id: Optional[str] = None


def fetch_interval(genome_record: object, start_1based: int, end_1based: int) -> str:
    """
    Fetch genomic sequence [start,end] inclusive (1-based) from a SeqRecord.
    """
    # python slice is 0-based and end-exclusive
    return str(genome_record.seq[start_1based - 1 : end_1based]).upper()


# ----------------------------
# Core extraction logic
# ----------------------------
def donor_coords_plus(
    intron_start: int,
    left_len: int,
    right_len: int,
) -> tuple[int, int]:
    """Return 1-based inclusive donor window on ``+`` strand."""
    # intron_start is the first intronic base (1-based)
    start = intron_start - left_len
    end = intron_start + right_len - 1
    return start, end


def acceptor_coords_plus(
    exon_start: int,
    left_len: int,
    right_len: int,
) -> tuple[int, int]:
    """Return 1-based inclusive acceptor window on ``+`` strand."""
    # exon_start is the first exonic base of downstream exon (1-based)
    start = exon_start - left_len
    end = exon_start + right_len - 1
    return start, end


def coords_minus(boundary: int, left_len: int, right_len: int) -> tuple[int, int]:
    """
    For minus strand, after we reverse-complement, we want the same left/right layout
    in transcript orientation. A convenient way:

    Extract genomic interval [boundary-(right_len-1), boundary+left_len] (1-based inclusive),
    then reverse-complement.

    Here `boundary` is:
      - donor: intron_start base in transcript orientation (genomic coordinate)
      - acceptor: exon_start base in transcript orientation (genomic coordinate)
    """
    start = boundary - (right_len - 1)
    end = boundary + left_len
    return start, end


def _resolve_intronic_context_lengths(
    intron_length: int,
    donor_intronic_len: int,
    acceptor_intronic_len: int,
    clip_short_intron: bool,
) -> tuple[int, int]:
    """Resolve intronic context lengths for donor/acceptor windows.

    Parameters
    ----------
    intron_length : int
        Intron length in base pairs.
    donor_intronic_len : int
        Requested donor intronic span.
    acceptor_intronic_len : int
        Requested acceptor intronic span.
    clip_short_intron : bool
        If ``True``, cap intronic span by ``intron_length``.

    Returns
    -------
    tuple[int, int]
        Effective donor and acceptor intronic spans.

    Raises
    ------
    ValueError
        If any input is negative.
    """
    if intron_length < 0:
        raise ValueError("intron_length must be >= 0")
    if donor_intronic_len < 0:
        raise ValueError("donor_intronic_len must be >= 0")
    if acceptor_intronic_len < 0:
        raise ValueError("acceptor_intronic_len must be >= 0")

    if not clip_short_intron:
        return donor_intronic_len, acceptor_intronic_len

    return (
        min(donor_intronic_len, intron_length),
        min(acceptor_intronic_len, intron_length),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="Genome FASTA (.fna)")
    ap.add_argument("--gtf", required=True, help="Annotations GTF")
    ap.add_argument("--out_tsv", required=True, help="Output TSV for model scoring")

    # Defaults match training data layout:
    # donor boundary offset: left=3
    # acceptor boundary offset: right=3
    ap.add_argument("--donor_len", type=int, default=15)
    ap.add_argument("--acceptor_len", type=int, default=30)
    ap.add_argument(
        "--clip-short-intron",
        action="store_true",
        help=(
            "When intron is shorter than requested intronic context, clip to the "
            "intron length instead of crossing into the opposite exon."
        ),
    )

    ap.add_argument(
        "--feature",
        default="exon",
        help="Which GTF feature to use (default: exon)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: limit number of rows written (0 = no limit)",
    )
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    """Run the test-data generation CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Optional CLI tokens. If ``None``, parse from ``sys.argv``.
    """
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    # Derived
    donor_boundary_offset = DONOR_EXON_BP
    acceptor_boundary_offset = ACCEPTOR_EXON_BP
    if args.donor_len < donor_boundary_offset:
        raise ValueError("--donor_len must be >= 3 for fixed donor boundary offset.")
    if args.acceptor_len < acceptor_boundary_offset:
        raise ValueError(
            "--acceptor_len must be >= 3 for fixed acceptor boundary offset."
        )
    donor_intronic_len = args.donor_len - donor_boundary_offset
    acceptor_intronic_len = args.acceptor_len - acceptor_boundary_offset

    # Index FASTA for random access (creates an index file alongside FASTA)
    genome = SeqIO.index(str(Path(args.fasta)), "fasta")

    # Read exons grouped by transcript_id
    tx_exons: dict[str, list[Exon]] = {}

    with open(args.gtf, "r", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            (
                chrom,
                _source,
                feature,
                start_s,
                end_s,
                _score,
                strand,
                _frame,
                attrs,
            ) = fields
            if feature != args.feature:
                continue
            if strand not in ["+", "-"]:
                continue

            start = int(start_s)
            end = int(end_s)
            ad = parse_gtf_attributes(attrs)
            tid = ad.get("transcript_id", None)
            if tid is None:
                continue
            gid = ad.get("gene_id", None)

            ex = Exon(
                chrom=chrom,
                start=start,
                end=end,
                strand=strand,
                transcript_id=tid,
                gene_id=gid,
            )
            tx_exons.setdefault(tid, []).append(ex)

    # Write output
    # TSV columns chosen to be convenient later:
    # transcript_id, gene_id, site_type, intron_index, chrom, strand, boundary_pos, seq
    out = open(args.out_tsv, "w", encoding="utf-8")
    print(
        "\t".join(
            [
                "transcript_id",
                "gene_id",
                "site_type",
                "intron_index",
                "chrom",
                "strand",
                "boundary_pos",
                "seq",
            ]
        ),
        file=out,
    )

    written = 0
    skipped_oob = 0
    skipped_contig = 0
    skipped_invalid_intron = 0

    for tid, exons in tx_exons.items():
        # Assume all exons share chrom/strand for a transcript
        strand = exons[0].strand
        chrom = exons[0].chrom
        gid = exons[0].gene_id if exons[0].gene_id is not None else ""

        if chrom not in genome:
            skipped_contig += 1
            continue

        # Order exons in transcript order (5'->3')
        if strand == "+":
            exons_sorted = sorted(exons, key=lambda e: (e.start, e.end))
        else:
            # minus strand transcript runs from high->low coords
            exons_sorted = sorted(exons, key=lambda e: (e.start, e.end), reverse=True)

        # For each intron between consecutive exons in transcript order:
        # donor = 5' splice site = intron start
        # acceptor = 3' splice site = exon start of downstream exon (in transcript orientation)
        for j in range(len(exons_sorted) - 1):
            up = exons_sorted[j]
            dn = exons_sorted[j + 1]

            if strand == "+":
                intron_start = up.end + 1         # first intronic base
                intron_end = dn.start - 1         # last intronic base
                intron_length = intron_end - intron_start + 1
                if intron_length <= 0:
                    skipped_invalid_intron += 1
                    continue
                # first exonic base of downstream exon
                exon_start = dn.start

                donor_right, acceptor_left = _resolve_intronic_context_lengths(
                    intron_length=intron_length,
                    donor_intronic_len=donor_intronic_len,
                    acceptor_intronic_len=acceptor_intronic_len,
                    clip_short_intron=args.clip_short_intron,
                )

                # donor window around intron_start
                d_start, d_end = donor_coords_plus(
                    intron_start, donor_boundary_offset, donor_right
                )
                # acceptor window around exon_start
                a_start, a_end = acceptor_coords_plus(
                    exon_start, acceptor_left, acceptor_boundary_offset
                )

                # bounds check
                chr_len = len(genome[chrom].seq)
                if d_start < 1 or d_end > chr_len or a_start < 1 or a_end > chr_len:
                    skipped_oob += 1
                    continue

                donor_seq = fetch_interval(genome[chrom], d_start, d_end)
                acceptor_seq = fetch_interval(genome[chrom], a_start, a_end)

            else:
                # minus strand example:
                # upstream exon is "earlier" in transcript (higher coords)
                # intron starts at up.start-1 (toward decreasing coords)
                intron_start = up.start - 1
                intron_length = up.start - dn.end - 1
                if intron_length <= 0:
                    skipped_invalid_intron += 1
                    continue

                # Downstream exon in transcript is lower coords, so transcript
                # starts the exon at dn.end.
                exon_start = dn.end

                donor_right, acceptor_left = _resolve_intronic_context_lengths(
                    intron_length=intron_length,
                    donor_intronic_len=donor_intronic_len,
                    acceptor_intronic_len=acceptor_intronic_len,
                    clip_short_intron=args.clip_short_intron,
                )

                d_start, d_end = coords_minus(
                    intron_start, donor_boundary_offset, donor_right
                )
                a_start, a_end = coords_minus(
                    exon_start, acceptor_left, acceptor_boundary_offset
                )

                chr_len = len(genome[chrom].seq)
                if d_start < 1 or d_end > chr_len or a_start < 1 or a_end > chr_len:
                    skipped_oob += 1
                    continue

                donor_seq = revcomp(fetch_interval(genome[chrom], d_start, d_end))
                acceptor_seq = revcomp(fetch_interval(genome[chrom], a_start, a_end))

            intron_idx = j + 1  # 1-based intron index within transcript

            # Donor row
            print(
                "\t".join(
                    [
                        tid,
                        gid,
                        "donor",
                        str(intron_idx),
                        chrom,
                        strand,
                        str(intron_start),
                        donor_seq,
                    ]
                ),
                file=out,
            )
            written += 1
            if args.limit and written >= args.limit:
                break

            # Acceptor row
            print(
                "\t".join(
                    [
                        tid,
                        gid,
                        "acceptor",
                        str(intron_idx),
                        chrom,
                        strand,
                        str(exon_start),
                        acceptor_seq,
                    ]
                ),
                file=out,
            )
            written += 1
            if args.limit and written >= args.limit:
                break

        if args.limit and written >= args.limit:
            break

    out.close()

    print(f"Done. Wrote {written} rows to {args.out_tsv}", file=sys.stderr)
    if skipped_contig:
        print(
            "Skipped "
            f"{skipped_contig} transcripts due to missing contig in FASTA index.",
            file=sys.stderr,
        )
    if skipped_oob:
        print(
            f"Skipped {skipped_oob} introns due to out-of-bounds window extraction.",
            file=sys.stderr,
        )
    if skipped_invalid_intron:
        print(
            "Skipped "
            f"{skipped_invalid_intron} introns with non-positive intron length.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
