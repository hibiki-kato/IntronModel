"""Build fixed-context site training ERR files from raw PWM ERR records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.make_intron_training_data_from_err import (  # noqa: E402
    FastaIndexedReader,
    reverse_complement,
)


_VALID_SITE_TYPES = {"donor", "acceptor", "pair"}


def _find_fasta(raw_dir: Path) -> Path | None:
    candidates = sorted(raw_dir.glob("*.fna"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _extract_site_window(
    reader: FastaIndexedReader,
    *,
    site_type: str,
    strand: str,
    chrom: str,
    pos: int,
    upstream_bp: int,
    downstream_bp: int,
) -> str:
    """Fetch one site window in transcript orientation."""
    if site_type not in {"donor", "acceptor"}:
        raise ValueError("site_type must be donor or acceptor")
    if strand not in {"+", "-"}:
        raise ValueError("strand must be + or -")

    if strand == "+":
        start = pos - upstream_bp + 1
        end = pos + downstream_bp
    else:
        start = pos - downstream_bp + 1
        end = pos + upstream_bp

    seq = reader.fetch_interval(chrom, start, end)
    return seq if strand == "+" else reverse_complement(seq)


def _extract_site_window_calibrated(
    reader: FastaIndexedReader,
    *,
    site_type: str,
    strand: str,
    chrom: str,
    pos: int,
    raw_seq: str,
    upstream_bp: int,
    downstream_bp: int,
) -> str:
    """Fetch a site window, correcting strand when raw 102bp sequence proves it."""
    candidate_strands = [strand]
    if strand in {"+", "-"}:
        candidate_strands.append("-" if strand == "+" else "+")

    if len(raw_seq) == 102:
        raw_upper = raw_seq.upper()
        for candidate_strand in candidate_strands:
            try:
                candidate = _extract_site_window(
                    reader,
                    site_type=site_type,
                    strand=candidate_strand,
                    chrom=chrom,
                    pos=pos,
                    upstream_bp=50,
                    downstream_bp=52,
                )
            except (KeyError, ValueError):
                continue
            if candidate == raw_upper:
                return _extract_site_window(
                    reader,
                    site_type=site_type,
                    strand=candidate_strand,
                    chrom=chrom,
                    pos=pos,
                    upstream_bp=upstream_bp,
                    downstream_bp=downstream_bp,
                )

    return _extract_site_window(
        reader,
        site_type=site_type,
        strand=strand,
        chrom=chrom,
        pos=pos,
        upstream_bp=upstream_bp,
        downstream_bp=downstream_bp,
    )


def _convert_record(
    line: str,
    reader: FastaIndexedReader,
    *,
    upstream_bp: int,
    downstream_bp: int,
) -> str | None:
    tokens = line.split()
    if len(tokens) < 3 or tokens[0] != "DEBUG" or tokens[1] not in _VALID_SITE_TYPES:
        return None

    record_type = tokens[1]
    if record_type in {"donor", "acceptor"}:
        if len(tokens) < 7:
            return None
        strand = tokens[3]
        chrom = tokens[5]
        try:
            pos = int(tokens[6])
        except ValueError:
            return None
        seq = _extract_site_window(
            reader,
            site_type=record_type,
            strand=strand,
            chrom=chrom,
            pos=pos,
            upstream_bp=upstream_bp,
            downstream_bp=downstream_bp,
        )
        return " ".join([tokens[0], record_type, seq, *tokens[3:]])

    if record_type == "pair":
        if len(tokens) < 8:
            return None
        strand = tokens[4]
        chrom = tokens[6]
        try:
            donor_pos = int(tokens[7])
            acceptor_pos = int(tokens[8]) if len(tokens) >= 9 else donor_pos
        except ValueError:
            return None
        donor_seq = _extract_site_window_calibrated(
            reader,
            site_type="donor",
            strand=strand,
            chrom=chrom,
            pos=donor_pos,
            raw_seq=tokens[2],
            upstream_bp=upstream_bp,
            downstream_bp=downstream_bp,
        )
        acceptor_seq = _extract_site_window_calibrated(
            reader,
            site_type="acceptor",
            strand=strand,
            chrom=chrom,
            pos=acceptor_pos,
            raw_seq=tokens[3],
            upstream_bp=upstream_bp,
            downstream_bp=downstream_bp,
        )
        return " ".join([tokens[0], record_type, donor_seq, acceptor_seq, *tokens[4:]])

    return None


def _process_one_file(
    src_path: Path,
    dst_path: Path,
    reader: FastaIndexedReader,
    *,
    label: str,
    upstream_bp: int,
    downstream_bp: int,
) -> tuple[int, int]:
    accepted = 0
    skipped = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", errors="ignore") as fin, dst_path.open("w") as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                out = _convert_record(
                    line,
                    reader,
                    upstream_bp=upstream_bp,
                    downstream_bp=downstream_bp,
                )
            except (KeyError, ValueError):
                out = None
            if out is None:
                skipped += 1
                continue
            fout.write(out + "\n")
            accepted += 1
    print(
        f"  [{label}] {src_path.name} -> {dst_path.name}: "
        f"{accepted} records ({skipped} skipped)",
        flush=True,
    )
    return accepted, skipped


def process_species(
    species: str,
    data_root: Path,
    pos_suffix: str,
    neg_suffix: str,
    out_pos_name: str,
    out_neg_name: str,
    upstream_bp: int,
    downstream_bp: int,
) -> None:
    raw_dir = data_root / species / "raw"
    processed_dir = data_root / species / "processed"
    fasta_path = _find_fasta(raw_dir)
    if fasta_path is None:
        print(f"[{species}] skipped: expected exactly one *.fna in {raw_dir}")
        return

    pos_files = sorted(raw_dir.glob(f"*{pos_suffix}"))
    neg_files = sorted(raw_dir.glob(f"*{neg_suffix}"))
    if not pos_files:
        print(f"[{species}] No positive files matching *{pos_suffix} in {raw_dir}")
        return
    if not neg_files:
        print(f"[{species}] No negative files matching *{neg_suffix} in {raw_dir}")
        return
    if len(pos_files) > 1:
        print(f"[{species}] Warning: using first positive file: {pos_files[0].name}")
    if len(neg_files) > 1:
        print(f"[{species}] Warning: using first negative file: {neg_files[0].name}")

    print(f"[{species}] FASTA={fasta_path.name}", flush=True)
    with FastaIndexedReader(fasta_path) as reader:
        _process_one_file(
            pos_files[0],
            processed_dir / out_pos_name,
            reader,
            label="pos",
            upstream_bp=upstream_bp,
            downstream_bp=downstream_bp,
        )
        _process_one_file(
            neg_files[0],
            processed_dir / out_neg_name,
            reader,
            label="neg",
            upstream_bp=upstream_bp,
            downstream_bp=downstream_bp,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build processed donor/acceptor site ERR files from raw "
            "*.coding.pwm.err / *.neg.pwm.err and reference FASTA coordinates."
        )
    )
    parser.add_argument("--species", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--pos-suffix", default=".coding.pwm.err")
    parser.add_argument("--neg-suffix", default=".neg.pwm.err")
    parser.add_argument("--out-pos-name", default="site_flank100.coding.err")
    parser.add_argument("--out-neg-name", default="site_flank100.neg.err")
    parser.add_argument("--upstream-bp", type=int, default=100)
    parser.add_argument("--downstream-bp", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.upstream_bp <= 0 or args.downstream_bp <= 0:
        raise ValueError("--upstream-bp and --downstream-bp must be > 0")
    data_root = Path(args.data_root)
    for species in [s.strip() for s in args.species.split(",") if s.strip()]:
        process_species(
            species=species,
            data_root=data_root,
            pos_suffix=args.pos_suffix,
            neg_suffix=args.neg_suffix,
            out_pos_name=args.out_pos_name,
            out_neg_name=args.out_neg_name,
            upstream_bp=args.upstream_bp,
            downstream_bp=args.downstream_bp,
        )


if __name__ == "__main__":
    main()
