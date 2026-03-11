from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
import re


_POSITIVE_PAIR_PATTERN = re.compile(
    r"^DEBUG donor ([ACGTNacgtn]+) acceptor ([ACGTNacgtn]+) ([+-]) (\S+) (-?\d+)$"
)
_NEGATIVE_PAIR_PATTERN = re.compile(
    r"^DEBUG pair ([ACGTNacgtn]+) ([ACGTNacgtn]+) ([+-]) (-?\d+)$"
)


@dataclass(frozen=True)
class PositivePairRecord:
    """One positive pair record from ``100bp.err``.

    Attributes
    ----------
    donor_seq : str
        Donor-side sequence.
    acceptor_seq : str
        Acceptor-side sequence.
    strand : str
        Strand token, ``+`` or ``-``.
    transcript_id : str
        Transcript identifier token.
    intron_half_length : int
        Right-most numeric column.
    source_line_no : int
        1-based source line number.
    """

    donor_seq: str
    acceptor_seq: str
    strand: str
    transcript_id: str
    intron_half_length: int
    source_line_no: int


@dataclass(frozen=True)
class NegativePairRecord:
    """One negative ``DEBUG pair`` record from ``100bp.neg.err``.

    Attributes
    ----------
    donor_seq : str
        Donor-side sequence.
    acceptor_seq : str
        Acceptor-side sequence.
    strand : str
        Strand token, ``+`` or ``-``.
    intron_half_length : int
        Right-most numeric column.
    source_line_no : int
        1-based source line number.
    """

    donor_seq: str
    acceptor_seq: str
    strand: str
    intron_half_length: int
    source_line_no: int


@dataclass(frozen=True)
class SpeciesTrimStats:
    """Summary statistics for one species trimming run."""

    species: str
    positive_rows: int
    negative_pair_rows: int
    positive_trimmed_min_len: int
    positive_trimmed_median_len: int
    positive_trimmed_max_len: int
    negative_trimmed_min_len: int
    negative_trimmed_median_len: int
    negative_trimmed_max_len: int


@dataclass(frozen=True)
class TrimmedPair:
    """Trimmed donor/acceptor pair sequence."""

    donor_seq: str
    acceptor_seq: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None, optional
        CLI tokens. ``None`` uses ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed argument object.

    Raises
    ------
    ValueError
        If numeric arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create variable-length trimmed pair datasets from "
            "100bp.err and 100bp.neg.err."
        )
    )
    parser.add_argument(
        "--species",
        required=True,
        help="Comma-separated species list (e.g., Dmel,Mmus,Athal).",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root directory containing <species>/raw.",
    )
    parser.add_argument(
        "--pos-input-name",
        default="100bp.err",
        help="Positive source filename in raw directory.",
    )
    parser.add_argument(
        "--neg-input-name",
        default="100bp.neg.err",
        help="Negative source filename in raw directory.",
    )
    parser.add_argument(
        "--out-pos-name",
        default="100bp_trimmed.err",
        help="Trimmed positive output filename in processed directory.",
    )
    parser.add_argument(
        "--out-neg-name",
        default="100bp_trimmed.neg.err",
        help="Trimmed negative output filename in processed directory.",
    )
    parser.add_argument(
        "--exon-context-bp",
        type=int,
        default=3,
        help=(
            "Boundary context to keep on each side. Effective keep length is "
            "min(original_len, intron_half_length + exon_context_bp)."
        ),
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Fail on malformed positive records or malformed DEBUG pair rows.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Skip malformed rows when possible.",
    )
    parser.add_argument(
        "--pad-with-n",
        action="store_true",
        help=(
            "Keep original sequence lengths by replacing trimmed-out regions "
            "with 'N'."
        ),
    )

    args = parser.parse_args(argv)
    if args.exon_context_bp < 0:
        raise ValueError("--exon-context-bp must be >= 0")

    return args


def _species_list(species_csv: str) -> list[str]:
    """Split species CSV string into a clean list."""
    tokens = [item.strip() for item in species_csv.split(",")]
    return [item for item in tokens if item]


def _trim_pair(
    donor_seq: str,
    acceptor_seq: str,
    intron_half_length: int,
    exon_context_bp: int,
    pad_with_n: bool = False,
) -> TrimmedPair:
    """Trim donor and acceptor sequences using intron half-length.

    Parameters
    ----------
    donor_seq : str
        Original donor sequence.
    acceptor_seq : str
        Original acceptor sequence.
    intron_half_length : int
        Half intron length token from source data.
    exon_context_bp : int
        Fixed context bp near splice boundary.
    pad_with_n : bool, default=False
        If ``True``, preserve original sequence lengths by replacing trimmed-out
        donor tail and acceptor head with ``N`` characters.

    Returns
    -------
    TrimmedPair
        Trimmed donor (prefix) and acceptor (suffix), optionally N-padded.

    Raises
    ------
    ValueError
        If intron_half_length is negative.

    Notes
    -----
    The core idea is to retain approximately one intron half from each side,
    plus boundary context. Runtime is ``O(L)`` per record where ``L`` is the
    original sequence length.
    """
    if intron_half_length < 0:
        raise ValueError(f"intron_half_length must be >= 0, got {intron_half_length}")

    keep_len = intron_half_length + exon_context_bp
    donor_keep = min(len(donor_seq), keep_len)
    acceptor_keep = min(len(acceptor_seq), keep_len)

    donor_trimmed = donor_seq[:donor_keep].upper()
    acceptor_trimmed = acceptor_seq[-acceptor_keep:].upper()

    if not pad_with_n:
        return TrimmedPair(
            donor_seq=donor_trimmed,
            acceptor_seq=acceptor_trimmed,
        )

    donor_padded = donor_trimmed + ("N" * (len(donor_seq) - donor_keep))
    acceptor_padded = ("N" * (len(acceptor_seq) - acceptor_keep)) + acceptor_trimmed
    return TrimmedPair(
        donor_seq=donor_padded,
        acceptor_seq=acceptor_padded,
    )


def _read_positive_pairs(path: Path, strict: bool) -> list[PositivePairRecord]:
    """Read strict-format positive pair records.

    Parameters
    ----------
    path : Path
        Positive source path.
    strict : bool
        Strict parsing mode.

    Returns
    -------
    list[PositivePairRecord]
        Parsed positive records.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If strict mode is enabled and malformed lines are found.
    """
    if not path.exists():
        raise FileNotFoundError(f"Positive input not found: {path}")

    records: list[PositivePairRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = _POSITIVE_PAIR_PATTERN.match(line)
            if match is None:
                if strict:
                    raise ValueError(
                        f"Malformed positive pair record at {path}:{line_no}: {line}"
                    )
                continue
            records.append(
                PositivePairRecord(
                    donor_seq=match.group(1),
                    acceptor_seq=match.group(2),
                    strand=match.group(3),
                    transcript_id=match.group(4),
                    intron_half_length=int(match.group(5)),
                    source_line_no=line_no,
                )
            )
    return records


def _read_negative_pairs(path: Path, strict: bool) -> list[NegativePairRecord]:
    """Read ``DEBUG pair`` rows from negative source.

    Parameters
    ----------
    path : Path
        Negative source path.
    strict : bool
        Strict parsing mode for malformed ``DEBUG pair`` rows.

    Returns
    -------
    list[NegativePairRecord]
        Parsed negative pair rows.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If strict mode is enabled and malformed ``DEBUG pair`` rows are found.
    """
    if not path.exists():
        raise FileNotFoundError(f"Negative input not found: {path}")

    records: list[NegativePairRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or not line.startswith("DEBUG"):
                continue
            if not line.startswith("DEBUG pair"):
                continue

            match = _NEGATIVE_PAIR_PATTERN.match(line)
            if match is None:
                if strict:
                    raise ValueError(
                        f"Malformed negative DEBUG pair at {path}:{line_no}: {line}"
                    )
                continue
            records.append(
                NegativePairRecord(
                    donor_seq=match.group(1),
                    acceptor_seq=match.group(2),
                    strand=match.group(3),
                    intron_half_length=int(match.group(4)),
                    source_line_no=line_no,
                )
            )
    return records


def _write_trimmed_positive(
    path: Path,
    records: list[PositivePairRecord],
    exon_context_bp: int,
    pad_with_n: bool,
) -> list[int]:
    """Write trimmed positive pair records.

    Parameters
    ----------
    path : Path
        Output path.
    records : list[PositivePairRecord]
        Positive pair records.
    exon_context_bp : int
        Boundary context parameter for trimming.
    pad_with_n : bool
        If ``True``, preserve original length with ``N`` padding.

    Returns
    -------
    list[int]
        Retained donor lengths for summary statistics.
    """
    retained_lengths: list[int] = []
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            trimmed = _trim_pair(
                donor_seq=record.donor_seq,
                acceptor_seq=record.acceptor_seq,
                intron_half_length=record.intron_half_length,
                exon_context_bp=exon_context_bp,
                pad_with_n=pad_with_n,
            )
            retained_lengths.append(len(trimmed.donor_seq))
            handle.write(
                "DEBUG donor "
                f"{trimmed.donor_seq} "
                f"acceptor {trimmed.acceptor_seq} "
                f"{record.strand} {record.transcript_id} {record.intron_half_length}\n"
            )
    return retained_lengths


def _write_trimmed_negative(
    path: Path,
    records: list[NegativePairRecord],
    exon_context_bp: int,
    pad_with_n: bool,
) -> list[int]:
    """Write trimmed negative ``DEBUG pair`` records.

    Parameters
    ----------
    path : Path
        Output path.
    records : list[NegativePairRecord]
        Negative pair records.
    exon_context_bp : int
        Boundary context parameter for trimming.
    pad_with_n : bool
        If ``True``, preserve original length with ``N`` padding.

    Returns
    -------
    list[int]
        Retained donor lengths for summary statistics.
    """
    retained_lengths: list[int] = []
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            trimmed = _trim_pair(
                donor_seq=record.donor_seq,
                acceptor_seq=record.acceptor_seq,
                intron_half_length=record.intron_half_length,
                exon_context_bp=exon_context_bp,
                pad_with_n=pad_with_n,
            )
            retained_lengths.append(len(trimmed.donor_seq))
            handle.write(
                "DEBUG pair "
                f"{trimmed.donor_seq} "
                f"{trimmed.acceptor_seq} "
                f"{record.strand} {record.intron_half_length}\n"
            )
    return retained_lengths


def _length_summary(lengths: list[int]) -> tuple[int, int, int]:
    """Return min/median/max length summary."""
    if not lengths:
        return 0, 0, 0
    return min(lengths), int(statistics.median(lengths)), max(lengths)


def process_species(
    species: str,
    data_root: Path,
    pos_input_name: str,
    neg_input_name: str,
    out_pos_name: str,
    out_neg_name: str,
    exon_context_bp: int,
    pad_with_n: bool,
    strict: bool,
) -> SpeciesTrimStats:
    """Generate trimmed pair datasets for one species.

    Parameters
    ----------
    species : str
        Species name.
    data_root : Path
        Data root directory.
    pos_input_name : str
        Positive source filename under raw directory.
    neg_input_name : str
        Negative source filename under raw directory.
    out_pos_name : str
        Positive output filename under processed directory.
    out_neg_name : str
        Negative output filename under processed directory.
    exon_context_bp : int
        Boundary context parameter.
    pad_with_n : bool
        If ``True``, preserve original sequence lengths with ``N`` padding.
    strict : bool
        Strict parsing mode.

    Returns
    -------
    SpeciesTrimStats
        Per-species trimming summary.

    Raises
    ------
    FileNotFoundError
        If required input files are missing.
    ValueError
        If strict validation fails.
    """
    raw_dir = data_root / species / "raw"
    processed_dir = data_root / species / "processed"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    processed_dir.mkdir(parents=True, exist_ok=True)

    pos_path = raw_dir / pos_input_name
    neg_path = raw_dir / neg_input_name
    out_pos_path = processed_dir / out_pos_name
    out_neg_path = processed_dir / out_neg_name

    positive_records = _read_positive_pairs(pos_path, strict=strict)
    negative_records = _read_negative_pairs(neg_path, strict=strict)

    if strict and not positive_records:
        raise ValueError(f"No valid positive pair records found in: {pos_path}")

    positive_lengths = _write_trimmed_positive(
        out_pos_path,
        positive_records,
        exon_context_bp=exon_context_bp,
        pad_with_n=pad_with_n,
    )
    negative_lengths = _write_trimmed_negative(
        out_neg_path,
        negative_records,
        exon_context_bp=exon_context_bp,
        pad_with_n=pad_with_n,
    )

    pos_min, pos_med, pos_max = _length_summary(positive_lengths)
    neg_min, neg_med, neg_max = _length_summary(negative_lengths)

    return SpeciesTrimStats(
        species=species,
        positive_rows=len(positive_records),
        negative_pair_rows=len(negative_records),
        positive_trimmed_min_len=pos_min,
        positive_trimmed_median_len=pos_med,
        positive_trimmed_max_len=pos_max,
        negative_trimmed_min_len=neg_min,
        negative_trimmed_median_len=neg_med,
        negative_trimmed_max_len=neg_max,
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entrypoint."""
    args = parse_args(argv)
    species_list = _species_list(args.species)
    if not species_list:
        raise ValueError("--species must contain at least one species")

    data_root = Path(args.data_root)
    for species in species_list:
        stats = process_species(
            species=species,
            data_root=data_root,
            pos_input_name=args.pos_input_name,
            neg_input_name=args.neg_input_name,
            out_pos_name=args.out_pos_name,
            out_neg_name=args.out_neg_name,
            exon_context_bp=args.exon_context_bp,
            pad_with_n=args.pad_with_n,
            strict=args.strict,
        )
        print(
            f"[{species}] pos_rows={stats.positive_rows} "
            f"neg_pair_rows={stats.negative_pair_rows} "
            f"pos_trim_len(min/med/max)="
            f"{stats.positive_trimmed_min_len}/"
            f"{stats.positive_trimmed_median_len}/"
            f"{stats.positive_trimmed_max_len} "
            f"neg_trim_len(min/med/max)="
            f"{stats.negative_trimmed_min_len}/"
            f"{stats.negative_trimmed_median_len}/"
            f"{stats.negative_trimmed_max_len}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
