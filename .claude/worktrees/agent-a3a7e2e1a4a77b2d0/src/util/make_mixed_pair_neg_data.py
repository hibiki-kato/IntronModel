from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Literal

from util.data_proc import ParsedTrainingRecord, parse_debug_training_record

MixMode = Literal["both", "donor_pos", "acceptor_pos"]


@dataclass(frozen=True)
class PositivePairRecord:
    """One positive pair record from training data.

    Attributes
    ----------
    donor_seq : str
        Donor-side sequence.
    acceptor_seq : str
        Acceptor-side sequence.
    strand : str
        Strand sign, ``+`` or ``-``.
    intron_half_length : int
        Intron half-length token used in pair rows.
    source_line_no : int
        One-based source line number.
    """

    donor_seq: str
    acceptor_seq: str
    strand: str
    intron_half_length: int
    source_line_no: int


@dataclass(frozen=True)
class NegativeSiteRecord:
    """One donor-only or acceptor-only negative site sequence."""

    sequence: str
    source_line_no: int


@dataclass(frozen=True)
class SpeciesMixStats:
    """Summary statistics for one species mixing run."""

    species: str
    positive_pairs: int
    negative_donor_sites: int
    negative_acceptor_sites: int
    generated_pairs: int
    output_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options.

    Parameters
    ----------
    argv : list[str] | None, optional
        CLI tokens. ``None`` uses ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.

    Raises
    ------
    ValueError
        If numeric options are invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create shuffled mixed-negative pair data with one positive side "
            "and one false side."
        )
    )
    parser.add_argument(
        "--species",
        required=True,
        help="Comma-separated species list (e.g., Mmus or Athal,Dmel,Mmus).",
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
        "--output-name",
        default="100bp_mixed_one_side.neg.err",
        help="Output filename in processed directory.",
    )
    parser.add_argument(
        "--mix-mode",
        choices=("both", "donor_pos", "acceptor_pos"),
        default="both",
        help=(
            "Mixing mode: donor_pos (true donor + false acceptor), "
            "acceptor_pos (false donor + true acceptor), or both."
        ),
    )
    parser.add_argument(
        "--samples-per-negative",
        type=int,
        default=1,
        help="Number of generated rows per negative and per enabled mix side.",
    )
    parser.add_argument(
        "--samples-per-positive",
        dest="samples_per_negative",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for deterministic sampling and shuffling.",
    )
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=True,
        help="Shuffle generated rows before writing (default: enabled).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Keep deterministic generation order without final shuffle.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Fail on malformed DEBUG rows.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Skip malformed DEBUG rows when possible.",
    )

    args = parser.parse_args(argv)
    if args.samples_per_negative <= 0:
        raise ValueError("--samples-per-negative must be a positive integer.")
    return args


def _species_list(species_csv: str) -> list[str]:
    """Split a comma-separated species string."""
    tokens = [token.strip() for token in species_csv.split(",")]
    return [token for token in tokens if token]


def _parse_debug_line_or_raise(
    *,
    path: Path,
    line_no: int,
    line: str,
    strict: bool,
) -> ParsedTrainingRecord | None:
    """Parse one DEBUG line with strict/non-strict behavior."""
    parsed = parse_debug_training_record(line)
    if parsed is None and strict:
        raise ValueError(f"Malformed DEBUG line at {path}:{line_no}: {line}")
    return parsed


def _read_positive_pairs(path: Path, *, strict: bool) -> list[PositivePairRecord]:
    """Read positive pair rows from one training file.

    Parameters
    ----------
    path : Path
        Positive source file path.
    strict : bool
        If ``True``, malformed ``DEBUG`` lines raise ``ValueError``.

    Returns
    -------
    list[PositivePairRecord]
        Parsed positive pair records.
    """
    rows: list[PositivePairRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "" or not line.startswith("DEBUG"):
                continue
            parsed = _parse_debug_line_or_raise(
                path=path,
                line_no=line_no,
                line=line,
                strict=strict,
            )
            if parsed is None or parsed.record_type != "pair":
                continue
            if parsed.donor_seq is None or parsed.acceptor_seq is None:
                continue
            strand = parsed.strand if parsed.strand in {"+", "-"} else "+"
            half = parsed.intron_half_length
            rows.append(
                PositivePairRecord(
                    donor_seq=parsed.donor_seq.upper(),
                    acceptor_seq=parsed.acceptor_seq.upper(),
                    strand=strand,
                    intron_half_length=0 if half is None else int(half),
                    source_line_no=line_no,
                )
            )
    return rows


def _read_negative_site_pools(
    path: Path,
    *,
    strict: bool,
) -> tuple[list[NegativeSiteRecord], list[NegativeSiteRecord]]:
    """Read donor-only and acceptor-only negative sequences.

    Parameters
    ----------
    path : Path
        Negative source file path.
    strict : bool
        If ``True``, malformed ``DEBUG`` lines raise ``ValueError``.

    Returns
    -------
    tuple[list[NegativeSiteRecord], list[NegativeSiteRecord]]
        Donor-site pool and acceptor-site pool, respectively.
    """
    donor_sites: list[NegativeSiteRecord] = []
    acceptor_sites: list[NegativeSiteRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "" or not line.startswith("DEBUG"):
                continue
            parsed = _parse_debug_line_or_raise(
                path=path,
                line_no=line_no,
                line=line,
                strict=strict,
            )
            if parsed is None:
                continue
            if parsed.record_type == "donor" and parsed.donor_seq is not None:
                donor_sites.append(
                    NegativeSiteRecord(
                        sequence=parsed.donor_seq.upper(),
                        source_line_no=line_no,
                    )
                )
            elif (
                parsed.record_type == "acceptor"
                and parsed.acceptor_seq is not None
            ):
                acceptor_sites.append(
                    NegativeSiteRecord(
                        sequence=parsed.acceptor_seq.upper(),
                        source_line_no=line_no,
                    )
                )
    return donor_sites, acceptor_sites


def _sample_positive_for_false_acceptor(
    *,
    rng: random.Random,
    false_acceptor: str,
    positive_pool: list[PositivePairRecord],
    positive_pairs: set[tuple[str, str]],
) -> PositivePairRecord:
    """Sample one positive anchor that stays negative with false acceptor."""
    if not positive_pool:
        raise ValueError("Positive pair pool is empty.")
    max_tries = max(32, len(positive_pool) * 2)
    for _ in range(max_tries):
        positive = positive_pool[rng.randrange(len(positive_pool))]
        pair_key = (positive.donor_seq, false_acceptor)
        if false_acceptor != positive.acceptor_seq and pair_key not in positive_pairs:
            return positive
    raise ValueError(
        "Failed to sample positive donor anchor for false acceptor."
    )


def _sample_positive_for_false_donor(
    *,
    rng: random.Random,
    false_donor: str,
    positive_pool: list[PositivePairRecord],
    positive_pairs: set[tuple[str, str]],
) -> PositivePairRecord:
    """Sample one positive anchor that stays negative with false donor."""
    if not positive_pool:
        raise ValueError("Positive pair pool is empty.")
    max_tries = max(32, len(positive_pool) * 2)
    for _ in range(max_tries):
        positive = positive_pool[rng.randrange(len(positive_pool))]
        pair_key = (false_donor, positive.acceptor_seq)
        if false_donor != positive.donor_seq and pair_key not in positive_pairs:
            return positive
    raise ValueError(
        "Failed to sample positive acceptor anchor for false donor."
    )


def _format_pair_line(
    donor_seq: str,
    acceptor_seq: str,
    strand: str,
    intron_half_length: int,
) -> str:
    """Format one mixed negative row in ``DEBUG pair`` style."""
    return (
        f"DEBUG pair {donor_seq.upper()} {acceptor_seq.upper()} "
        f"{strand} {intron_half_length}"
    )


def generate_mixed_negative_lines(
    *,
    positive_pairs: list[PositivePairRecord],
    negative_donor_pool: list[NegativeSiteRecord],
    negative_acceptor_pool: list[NegativeSiteRecord],
    mix_mode: MixMode,
    samples_per_negative: int,
    seed: int,
    shuffle: bool,
) -> list[str]:
    """Generate one-sided mixed negative pair rows.

    Parameters
    ----------
    positive_pairs : list[PositivePairRecord]
        Positive pair rows used as anchors.
    negative_donor_pool : list[NegativeSiteRecord]
        Donor-side false candidates from negative non-pair rows.
    negative_acceptor_pool : list[NegativeSiteRecord]
        Acceptor-side false candidates from negative non-pair rows.
    mix_mode : {"both", "donor_pos", "acceptor_pos"}
        Which one-sided patterns to emit.
    samples_per_negative : int
        Number of generated rows per negative and per enabled mix side.
    seed : int
        Random seed.
    shuffle : bool
        Whether to shuffle output lines.

    Returns
    -------
    list[str]
        Generated ``DEBUG pair ...`` rows.

    Raises
    ------
    ValueError
        If inputs are invalid or candidate pools are insufficient.

    Notes
    -----
    Core idea: keep one side of a true pair fixed and replace the other side
    with a site sampled from negative non-pair rows. Sampling rejects exact
    collisions with known positive donor/acceptor combinations. Complexity is
    ``O((Nd + Na) * K)`` expected, where ``Nd``/``Na`` are negative donor and
    acceptor counts and ``K`` is samples per negative.
    """
    if samples_per_negative <= 0:
        raise ValueError("samples_per_negative must be positive.")
    if not positive_pairs:
        raise ValueError("No positive pair rows found.")
    if mix_mode not in {"both", "donor_pos", "acceptor_pos"}:
        raise ValueError(f"Unsupported mix_mode: {mix_mode}")
    if mix_mode in {"both", "acceptor_pos"} and not negative_donor_pool:
        raise ValueError("Negative donor pool is empty.")
    if mix_mode in {"both", "donor_pos"} and not negative_acceptor_pool:
        raise ValueError("Negative acceptor pool is empty.")

    rng = random.Random(seed)
    known_positive_pairs = {
        (item.donor_seq, item.acceptor_seq)
        for item in positive_pairs
    }
    lines: list[str] = []
    for _ in range(samples_per_negative):
        if mix_mode in {"both", "donor_pos"}:
            for false_acceptor_item in negative_acceptor_pool:
                positive = _sample_positive_for_false_acceptor(
                    rng=rng,
                    false_acceptor=false_acceptor_item.sequence,
                    positive_pool=positive_pairs,
                    positive_pairs=known_positive_pairs,
                )
                lines.append(
                    _format_pair_line(
                        donor_seq=positive.donor_seq,
                        acceptor_seq=false_acceptor_item.sequence,
                        strand=positive.strand,
                        intron_half_length=positive.intron_half_length,
                    )
                )
        if mix_mode in {"both", "acceptor_pos"}:
            for false_donor_item in negative_donor_pool:
                positive = _sample_positive_for_false_donor(
                    rng=rng,
                    false_donor=false_donor_item.sequence,
                    positive_pool=positive_pairs,
                    positive_pairs=known_positive_pairs,
                )
                lines.append(
                    _format_pair_line(
                        donor_seq=false_donor_item.sequence,
                        acceptor_seq=positive.acceptor_seq,
                        strand=positive.strand,
                        intron_half_length=positive.intron_half_length,
                    )
                )
    if shuffle:
        rng.shuffle(lines)
    return lines


def process_species(
    *,
    species: str,
    data_root: Path,
    pos_input_name: str,
    neg_input_name: str,
    output_name: str,
    mix_mode: MixMode,
    samples_per_negative: int,
    seed: int,
    shuffle: bool,
    strict: bool,
) -> SpeciesMixStats:
    """Generate and write mixed one-sided negative pair rows for one species.

    Parameters
    ----------
    species : str
        Species directory under ``data_root``.
    data_root : Path
        Data root path.
    pos_input_name : str
        Positive source filename in ``raw`` directory.
    neg_input_name : str
        Negative source filename in ``raw`` directory.
    output_name : str
        Output filename in ``processed`` directory.
    mix_mode : {"both", "donor_pos", "acceptor_pos"}
        One-sided mixing mode.
    samples_per_negative : int
        Number of outputs per negative and per enabled side.
    seed : int
        Random seed.
    shuffle : bool
        Whether to shuffle output lines.
    strict : bool
        Strict parse mode.

    Returns
    -------
    SpeciesMixStats
        Run summary.
    """
    raw_dir = data_root / species / "raw"
    processed_dir = data_root / species / "processed"
    pos_path = raw_dir / pos_input_name
    neg_path = raw_dir / neg_input_name
    out_path = processed_dir / output_name

    if not pos_path.exists():
        raise FileNotFoundError(f"Positive source file not found: {pos_path}")
    if not neg_path.exists():
        raise FileNotFoundError(f"Negative source file not found: {neg_path}")
    processed_dir.mkdir(parents=True, exist_ok=True)

    positives = _read_positive_pairs(pos_path, strict=strict)
    donor_pool, acceptor_pool = _read_negative_site_pools(neg_path, strict=strict)
    lines = generate_mixed_negative_lines(
        positive_pairs=positives,
        negative_donor_pool=donor_pool,
        negative_acceptor_pool=acceptor_pool,
        mix_mode=mix_mode,
        samples_per_negative=samples_per_negative,
        seed=seed,
        shuffle=shuffle,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")

    return SpeciesMixStats(
        species=species,
        positive_pairs=len(positives),
        negative_donor_sites=len(donor_pool),
        negative_acceptor_sites=len(acceptor_pool),
        generated_pairs=len(lines),
        output_path=out_path,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    data_root = Path(args.data_root)
    species_list = _species_list(args.species)
    if not species_list:
        raise ValueError("--species must contain at least one species name.")

    for species in species_list:
        stats = process_species(
            species=species,
            data_root=data_root,
            pos_input_name=args.pos_input_name,
            neg_input_name=args.neg_input_name,
            output_name=args.output_name,
            mix_mode=args.mix_mode,
            samples_per_negative=args.samples_per_negative,
            seed=args.seed,
            shuffle=args.shuffle,
            strict=args.strict,
        )
        print(
            "[make_mixed_pair_neg_data] "
            f"species={stats.species} pos={stats.positive_pairs} "
            f"neg_donor={stats.negative_donor_sites} "
            f"neg_acceptor={stats.negative_acceptor_sites} "
            f"generated={stats.generated_pairs} "
            f"out={stats.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
