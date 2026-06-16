"""Build precomputed sequence variants of the unique transcript TSV.

Reads ``data/<species>/processed/transcripts.unique.tsv`` and
``data/<species>/processed/transcripts.unique.map.tsv``, then writes:

- ``transcripts.unique.tsv``      -- source file updated: ``intron_half_length``
                                     column filled from intron coordinates.
- ``transcripts.unique.mask.tsv`` -- N-padded outside the intron region
                                     (``mask_outside_intron_n`` applied offline).
- ``transcripts.unique.trunc.tsv``-- sequences truncated to
                                     ``intron_half_length + exon_context_bp``
                                     from the splice boundary (variable length).

``intron_half_length`` is derived as
``(intron_end - intron_start + 1) // 2`` from the coordinate map.

Generating these variants once avoids per-run transform computation and
eliminates the float-drift issue that arises when unique-collapsing isoforms
scored after per-isoform transforms.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from util.unique_intron import (
    UNIQUE_MAP_TSV_NAME,
    UNIQUE_TRANSCRIPTS_MASK_TSV_NAME,
    UNIQUE_TRANSCRIPTS_TRUNC_TSV_NAME,
    UNIQUE_TRANSCRIPTS_TSV_NAME,
    set_csv_field_limit_max,
)

TRANSCRIPT_TSV_FIELDS: tuple[str, ...] = (
    "transcript_id",
    "gene_id",
    "site_type",
    "intron_index",
    "chrom",
    "strand",
    "boundary_pos",
    "seq",
    "intron_half_length",
)

DEFAULT_EXON_CONTEXT_BP: int = 3


@dataclass(frozen=True)
class _MapCoord:
    """Genomic coordinates for one unique intron from the map file."""

    intron_start: int
    intron_end: int


def _load_half_length_map(map_path: Path) -> dict[tuple[str, int], int]:
    """Load intron half-lengths keyed by (unique_transcript_id, unique_intron_index).

    Parameters
    ----------
    map_path : Path
        Path to ``transcripts.unique.map.tsv``.

    Returns
    -------
    dict[tuple[str, int], int]
        Half-length in bp: ``(intron_end - intron_start + 1) // 2``.

    Raises
    ------
    FileNotFoundError
        If ``map_path`` does not exist.
    ValueError
        If required columns are missing or values are invalid.
    """
    if not map_path.is_file():
        raise FileNotFoundError(f"Unique map TSV not found: {map_path}")

    required = {
        "unique_transcript_id",
        "unique_intron_index",
        "intron_start",
        "intron_end",
    }
    result: dict[tuple[str, int], int] = {}

    with map_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Map TSV missing required columns {required}: {map_path}")
        for line_no, raw in enumerate(reader, start=2):
            uid = str(raw["unique_transcript_id"]).strip()
            if uid == "":
                raise ValueError(f"Empty unique_transcript_id at {map_path}:{line_no}")
            try:
                uidx = int(raw["unique_intron_index"])
                istart = int(raw["intron_start"])
                iend = int(raw["intron_end"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer field at {map_path}:{line_no}"
                ) from exc
            if iend < istart:
                raise ValueError(f"intron_end < intron_start at {map_path}:{line_no}")
            half_len = (iend - istart + 1) // 2
            key: tuple[str, int] = (uid, uidx)
            # Use the first occurrence (all rows for the same unique intron share
            # the same coordinates).
            result.setdefault(key, half_len)

    return result


@dataclass(frozen=True)
class _SiteRow:
    """One site row from ``transcripts.unique.tsv``."""

    transcript_id: str
    gene_id: str
    site_type: str
    intron_index: int
    chrom: str
    strand: str
    boundary_pos: int
    seq: str
    intron_half_length: int | None


def _read_unique_site_rows(path: Path) -> list[_SiteRow]:
    """Read all rows from ``transcripts.unique.tsv``.

    Parameters
    ----------
    path : Path
        Source TSV path.

    Returns
    -------
    list[_SiteRow]
        Parsed site rows.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If required columns are missing or values are malformed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Unique transcript TSV not found: {path}")

    required = {
        "transcript_id",
        "gene_id",
        "site_type",
        "intron_index",
        "chrom",
        "strand",
        "boundary_pos",
        "seq",
    }
    rows: list[_SiteRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Unique transcript TSV missing required columns {required}: {path}"
            )
        for line_no, raw in enumerate(reader, start=2):
            half_raw = str(raw.get("intron_half_length", "")).strip()
            half_len: int | None = None
            if half_raw != "":
                try:
                    half_len = int(half_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid intron_half_length at {path}:{line_no}"
                    ) from exc
            try:
                rows.append(
                    _SiteRow(
                        transcript_id=str(raw["transcript_id"]).strip(),
                        gene_id=str(raw["gene_id"]).strip(),
                        site_type=str(raw["site_type"]).strip().lower(),
                        intron_index=int(raw["intron_index"]),
                        chrom=str(raw["chrom"]).strip(),
                        strand=str(raw["strand"]).strip(),
                        boundary_pos=int(raw["boundary_pos"]),
                        seq=str(raw["seq"]).strip(),
                        intron_half_length=half_len,
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Malformed row at {path}:{line_no}: {exc}") from exc

    return rows


def _apply_mask_transform(
    seq: str,
    *,
    site_type: str,
    intron_half_length: int,
    exon_context_bp: int,
) -> str:
    """Return the sequence with exonic context replaced by ``N``.

    Parameters
    ----------
    seq : str
        Original site sequence in transcript orientation.
    site_type : str
        ``donor`` or ``acceptor``.
    intron_half_length : int
        Number of bp to keep from the intron side.
    exon_context_bp : int
        Number of bp to keep from the exon side.

    Returns
    -------
    str
        Sequence of the same length with positions outside the splice region
        replaced by ``N``.

    Notes
    -----
    Donor convention: intron is on the RIGHT end (sequence = exon | intron).
    Acceptor convention: intron is on the LEFT end (sequence = intron | exon).
    Keep region = ``exon_context_bp + intron_half_length`` from the splice site.
    Complexity: O(L) where L = len(seq).
    """
    keep = min(len(seq), intron_half_length + exon_context_bp)
    seq_upper = seq.upper()
    if site_type == "donor":
        return seq_upper[:keep] + "N" * (len(seq) - keep)
    else:  # acceptor
        tail_start = len(seq) - keep
        return "N" * tail_start + seq_upper[tail_start:]


def _apply_trunc_transform(
    seq: str,
    *,
    site_type: str,
    intron_half_length: int,
    exon_context_bp: int,
) -> str:
    """Return the sequence truncated to the splice region.

    Parameters
    ----------
    seq : str
        Original site sequence.
    site_type : str
        ``donor`` or ``acceptor``.
    intron_half_length : int
        Number of bp to keep from the intron side.
    exon_context_bp : int
        Number of bp to keep from the exon side.

    Returns
    -------
    str
        Clipped sequence of length ``min(len(seq), intron_half_length +
        exon_context_bp)``, uppercased.
    """
    keep = min(len(seq), intron_half_length + exon_context_bp)
    seq_upper = seq.upper()
    if site_type == "donor":
        return seq_upper[:keep]
    else:  # acceptor
        return seq_upper[len(seq) - keep :]


def _write_site_rows(
    path: Path,
    rows: Iterable[dict[str, object]],
    *,
    overwrite: bool = True,
) -> int:
    """Write site rows to a TSV file.

    Parameters
    ----------
    path : Path
        Output path.
    rows : Iterable[dict[str, object]]
        Rows to write.
    overwrite : bool, default=True
        If False, raise ``FileExistsError`` when the file already exists.

    Returns
    -------
    int
        Number of rows written.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is False.
    """
    if not overwrite and path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(TRANSCRIPT_TSV_FIELDS),
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _build_output_rows(
    site_rows: list[_SiteRow],
    half_map: dict[tuple[str, int], int],
    *,
    transform: str,
    exon_context_bp: int,
) -> tuple[list[dict[str, object]], int]:
    """Build output rows by applying a sequence transform.

    Parameters
    ----------
    site_rows : list[_SiteRow]
        Source rows from ``transcripts.unique.tsv``.
    half_map : dict[tuple[str, int], int]
        Mapping from ``(unique_transcript_id, unique_intron_index)`` to
        intron half-length in bp.
    transform : str
        One of ``"none"``, ``"mask"``, ``"trunc"``.
    exon_context_bp : int
        Context bp to retain on the exon side.

    Returns
    -------
    tuple[list[dict[str, object]], int]
        ``(output_rows, missing_count)`` where ``missing_count`` is the number
        of rows whose unique key had no entry in ``half_map``.

    Raises
    ------
    ValueError
        If ``transform`` is not one of the accepted values.
    """
    if transform not in {"none", "mask", "trunc"}:
        raise ValueError(f"transform must be none|mask|trunc, got: {transform}")

    output_rows: list[dict[str, object]] = []
    missing_count = 0

    for row in site_rows:
        key: tuple[str, int] = (row.transcript_id, row.intron_index)
        half_len = half_map.get(key)
        if half_len is None:
            missing_count += 1
            half_len = row.intron_half_length  # fall back to source value if any

        if transform == "none":
            transformed_seq = row.seq.upper()
        elif transform == "mask":
            if half_len is not None:
                transformed_seq = _apply_mask_transform(
                    row.seq,
                    site_type=row.site_type,
                    intron_half_length=half_len,
                    exon_context_bp=exon_context_bp,
                )
            else:
                transformed_seq = row.seq.upper()
        else:  # trunc
            if half_len is not None:
                transformed_seq = _apply_trunc_transform(
                    row.seq,
                    site_type=row.site_type,
                    intron_half_length=half_len,
                    exon_context_bp=exon_context_bp,
                )
            else:
                transformed_seq = row.seq.upper()

        output_rows.append(
            {
                "transcript_id": row.transcript_id,
                "gene_id": row.gene_id,
                "site_type": row.site_type,
                "intron_index": row.intron_index,
                "chrom": row.chrom,
                "strand": row.strand,
                "boundary_pos": row.boundary_pos,
                "seq": transformed_seq,
                "intron_half_length": "" if half_len is None else str(half_len),
            }
        )

    return output_rows, missing_count


def _build_for_species(
    species_dir: Path,
    *,
    exon_context_bp: int,
    overwrite: bool,
) -> None:
    """Build all sequence variant TSV files for one species directory.

    Parameters
    ----------
    species_dir : Path
        Species data directory (e.g., ``data/Athal``).
    exon_context_bp : int
        Context bp to retain on the exon side of each splice boundary.
    overwrite : bool
        If True, overwrite existing output files.

    Raises
    ------
    FileNotFoundError
        If required input files are not found.
    FileExistsError
        If an output file exists and ``overwrite`` is False.
    """
    species = species_dir.name
    processed_dir = species_dir / "processed"
    src_path = processed_dir / UNIQUE_TRANSCRIPTS_TSV_NAME
    map_path = processed_dir / UNIQUE_MAP_TSV_NAME
    mask_path = processed_dir / UNIQUE_TRANSCRIPTS_MASK_TSV_NAME
    trunc_path = processed_dir / UNIQUE_TRANSCRIPTS_TRUNC_TSV_NAME

    set_csv_field_limit_max()
    half_map = _load_half_length_map(map_path)
    site_rows = _read_unique_site_rows(src_path)

    # 1. Update source file with intron_half_length filled in
    plain_rows, missing = _build_output_rows(
        site_rows, half_map, transform="none", exon_context_bp=exon_context_bp
    )
    written = _write_site_rows(src_path, plain_rows, overwrite=True)
    if missing:
        print(
            f"[build_unique_sequence_variants] {species}: "
            f"{missing} rows not found in half_map (kept empty intron_half_length)"
        )
    print(
        f"[build_unique_sequence_variants] {species}: "
        f"updated {src_path.name} rows={written} (intron_half_length filled)"
    )

    # 2. mask variant
    mask_rows, _ = _build_output_rows(
        site_rows, half_map, transform="mask", exon_context_bp=exon_context_bp
    )
    written_mask = _write_site_rows(mask_path, mask_rows, overwrite=overwrite)
    print(
        f"[build_unique_sequence_variants] {species}: "
        f"wrote {mask_path.name} rows={written_mask}"
    )

    # 3. trunc variant
    trunc_rows, _ = _build_output_rows(
        site_rows, half_map, transform="trunc", exon_context_bp=exon_context_bp
    )
    written_trunc = _write_site_rows(trunc_path, trunc_rows, overwrite=overwrite)
    print(
        f"[build_unique_sequence_variants] {species}: "
        f"wrote {trunc_path.name} rows={written_trunc}"
    )


def _parse_species_csv(raw: str) -> list[str]:
    """Parse comma-separated species string."""
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    return tokens


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate precomputed mask and trunc variants of "
            "transcripts.unique.tsv. "
            "Also fills intron_half_length in the source file."
        )
    )
    parser.add_argument(
        "--species",
        default="Dmel,Mmus,Athal,Hsap",
        help="Comma-separated species list.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Data root directory.",
    )
    parser.add_argument(
        "--exon-context-bp",
        type=int,
        default=DEFAULT_EXON_CONTEXT_BP,
        help="Exon-side context bp to retain in mask/trunc transforms.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask/trunc output files.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``build_unique_sequence_variants``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_root = args.data_root
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    species_list = _parse_species_csv(args.species)
    if not species_list:
        raise ValueError("--species must contain at least one species name.")

    for species in species_list:
        species_dir = data_root / species
        if not species_dir.is_dir():
            print(
                f"[build_unique_sequence_variants] skipping {species}: "
                f"directory not found: {species_dir}",
                file=sys.stderr,
            )
            continue
        _build_for_species(
            species_dir,
            exon_context_bp=args.exon_context_bp,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
