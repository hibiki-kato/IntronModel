"""Build precomputed unique-intron assets from transcript and label datasets.

Outputs are written under ``data/<species>/processed``:
- ``transcripts.unique.tsv``
- ``transcripts.unique.map.tsv``
- ``intron_eval_flank10.unique.tsv``
- ``intron_unique_catalog.tsv``
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from util.data_proc import parse_debug_training_record
from util.unique_intron import (
    UNIQUE_INTRON_CATALOG_TSV_NAME,
    UNIQUE_LABELED_INTRON_TSV_NAME,
    UNIQUE_MAP_TSV_NAME,
    UNIQUE_TRANSCRIPTS_TSV_NAME,
    set_csv_field_limit_max,
)

CoordKey = tuple[str, str, int, int]
MemberKey = tuple[str, int]


@dataclass(frozen=True)
class TranscriptSiteRow:
    """One site row read from ``processed/transcripts.tsv``."""

    transcript_id: str
    gene_id: str
    site_type: str
    intron_index: int
    chrom: str
    strand: str
    boundary_pos: int
    seq: str
    intron_half_length: int | None


@dataclass(frozen=True)
class TranscriptIntronRecord:
    """One intron record assembled from donor/acceptor transcript rows."""

    transcript_id: str
    intron_index: int
    gene_id: str
    chrom: str
    strand: str
    intron_start: int
    intron_end: int
    donor_seq: str
    acceptor_seq: str
    donor_boundary_pos: int
    acceptor_boundary_pos: int
    intron_half_length: int | None


@dataclass(frozen=True)
class LabeledIntronRecord:
    """One coordinate-keyed intron label record."""

    species: str
    label: int
    donor_label: int | None
    acceptor_label: int | None


@dataclass(frozen=True)
class UniqueIntronRecord:
    """One unique intron merged from potentially many transcript introns."""

    unique_transcript_id: str
    unique_intron_index: int
    chrom: str
    strand: str
    intron_start: int
    intron_end: int
    gene_id: str
    donor_seq: str
    acceptor_seq: str
    intron_half_length: int | None
    label: int
    donor_label: int | None
    acceptor_label: int | None
    member_keys: tuple[MemberKey, ...]
    seen_train_pos_coord: int
    seen_train_neg_seq: int
    train_leak: int


def _parse_species_csv(raw_species: str) -> list[str]:
    """Parse optional comma-separated species values."""
    tokens = [token.strip() for token in raw_species.split(",")]
    return [token for token in tokens if token != ""]


def _resolve_species_dirs(data_root: Path, raw_species: str) -> list[Path]:
    """Resolve ordered species directories under ``data_root``."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    requested = _parse_species_csv(raw_species)
    if not requested:
        return sorted(path for path in data_root.iterdir() if path.is_dir())

    resolved: list[Path] = []
    for token in requested:
        direct = data_root / token
        if direct.is_dir():
            resolved.append(direct)
            continue
        matches = [
            path
            for path in data_root.iterdir()
            if path.is_dir() and path.name.lower() == token.lower()
        ]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        if not matches:
            raise FileNotFoundError(
                f"Species directory not found under {data_root}: {token}"
            )
        names = ", ".join(path.name for path in sorted(matches))
        raise ValueError(f"Ambiguous species token '{token}': {names}")
    return resolved


def _parse_optional_int(raw_value: str) -> int | None:
    """Parse one optional integer token."""
    value = raw_value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value: {raw_value}") from exc


def _read_transcript_site_rows(path: Path) -> list[TranscriptSiteRow]:
    """Read site rows from ``processed/transcripts.tsv``."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing transcript TSV: {path}")

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
    rows: list[TranscriptSiteRow] = []
    set_csv_field_limit_max()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Transcript TSV must include columns: "
                "transcript_id, gene_id, site_type, intron_index, chrom, strand, "
                "boundary_pos, seq"
            )

        for line_no, raw_row in enumerate(reader, start=2):
            transcript_id = str(raw_row["transcript_id"]).strip()
            if transcript_id == "":
                raise ValueError(f"Empty transcript_id at {path}:{line_no}")
            site_type = str(raw_row["site_type"]).strip().lower()
            if site_type not in {"donor", "acceptor"}:
                raise ValueError(
                    f"Unsupported site_type at {path}:{line_no}: {site_type}"
                )
            strand = str(raw_row["strand"]).strip()
            if strand not in {"+", "-"}:
                raise ValueError(f"Invalid strand at {path}:{line_no}: {strand}")
            try:
                intron_index = int(str(raw_row["intron_index"]).strip())
                boundary_pos = int(str(raw_row["boundary_pos"]).strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer field at {path}:{line_no}"
                ) from exc
            rows.append(
                TranscriptSiteRow(
                    transcript_id=transcript_id,
                    gene_id=str(raw_row["gene_id"]).strip(),
                    site_type=site_type,
                    intron_index=intron_index,
                    chrom=str(raw_row["chrom"]).strip(),
                    strand=strand,
                    boundary_pos=boundary_pos,
                    seq=str(raw_row["seq"]).strip().upper(),
                    intron_half_length=_parse_optional_int(
                        str(raw_row.get("intron_half_length", "")).strip()
                    ),
                )
            )
    if not rows:
        raise ValueError(f"No transcript rows found: {path}")
    return rows


def _assemble_transcript_introns(
    site_rows: Iterable[TranscriptSiteRow],
) -> dict[MemberKey, TranscriptIntronRecord]:
    """Assemble donor/acceptor rows into per-intron records."""
    grouped: dict[MemberKey, dict[str, TranscriptSiteRow]] = {}
    for row in site_rows:
        key = (row.transcript_id, row.intron_index)
        bucket = grouped.setdefault(key, {})
        if row.site_type in bucket:
            raise ValueError(
                "Duplicate site row for transcript intron: "
                f"{row.transcript_id}:{row.intron_index}:{row.site_type}"
            )
        bucket[row.site_type] = row

    introns: dict[MemberKey, TranscriptIntronRecord] = {}
    for key in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        transcript_id, intron_index = key
        bucket = grouped[key]
        donor_row = bucket.get("donor")
        acceptor_row = bucket.get("acceptor")
        if donor_row is None or acceptor_row is None:
            raise ValueError(
                "Each transcript intron must include donor and acceptor rows: "
                f"{transcript_id}:{intron_index}"
            )
        if donor_row.chrom != acceptor_row.chrom:
            raise ValueError(
                "Mismatched contig between donor/acceptor rows: "
                f"{transcript_id}:{intron_index}"
            )
        if donor_row.strand != acceptor_row.strand:
            raise ValueError(
                "Mismatched strand between donor/acceptor rows: "
                f"{transcript_id}:{intron_index}"
            )
        strand = donor_row.strand
        if strand == "+":
            intron_start = donor_row.boundary_pos
            intron_end = acceptor_row.boundary_pos - 1
        else:
            intron_start = acceptor_row.boundary_pos + 1
            intron_end = donor_row.boundary_pos
        if intron_end < intron_start:
            raise ValueError(
                "Invalid intron coordinates from boundary rows: "
                f"{transcript_id}:{intron_index} start={intron_start} "
                f"end={intron_end}"
            )
        intron_half_length = donor_row.intron_half_length
        if (
            intron_half_length is not None
            and acceptor_row.intron_half_length is not None
            and intron_half_length != acceptor_row.intron_half_length
        ):
            raise ValueError(
                "Mismatched intron_half_length between donor/acceptor rows: "
                f"{transcript_id}:{intron_index}"
            )
        if intron_half_length is None:
            intron_half_length = acceptor_row.intron_half_length
        introns[key] = TranscriptIntronRecord(
            transcript_id=transcript_id,
            intron_index=intron_index,
            gene_id=donor_row.gene_id if donor_row.gene_id != "" else acceptor_row.gene_id,
            chrom=donor_row.chrom,
            strand=strand,
            intron_start=intron_start,
            intron_end=intron_end,
            donor_seq=donor_row.seq,
            acceptor_seq=acceptor_row.seq,
            donor_boundary_pos=donor_row.boundary_pos,
            acceptor_boundary_pos=acceptor_row.boundary_pos,
            intron_half_length=intron_half_length,
        )
    return introns


def _read_labeled_introns_by_coord(path: Path) -> dict[CoordKey, LabeledIntronRecord]:
    """Read coordinate-keyed labels from labeled intron TSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing labeled intron TSV: {path}")
    required = {
        "species",
        "chrom",
        "strand",
        "intron_start",
        "intron_end",
        "label",
    }
    labels: dict[CoordKey, LabeledIntronRecord] = {}
    set_csv_field_limit_max()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Labeled intron TSV must include columns: "
                "species, chrom, strand, intron_start, intron_end, label"
            )
        for line_no, raw_row in enumerate(reader, start=2):
            chrom = str(raw_row["chrom"]).strip()
            strand = str(raw_row["strand"]).strip()
            if strand not in {"+", "-"}:
                raise ValueError(f"Invalid strand at {path}:{line_no}: {strand}")
            try:
                intron_start = int(str(raw_row["intron_start"]).strip())
                intron_end = int(str(raw_row["intron_end"]).strip())
                label = int(str(raw_row["label"]).strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label/coordinate field at {path}:{line_no}"
                ) from exc
            if label not in {0, 1}:
                raise ValueError(f"Label must be 0/1 at {path}:{line_no}: {label}")
            donor_label = _parse_optional_int(str(raw_row.get("donor_label", "")))
            acceptor_label = _parse_optional_int(
                str(raw_row.get("acceptor_label", ""))
            )
            if donor_label is not None and donor_label not in {0, 1}:
                raise ValueError(
                    f"donor_label must be 0/1 at {path}:{line_no}: {donor_label}"
                )
            if acceptor_label is not None and acceptor_label not in {0, 1}:
                raise ValueError(
                    "acceptor_label must be 0/1 "
                    f"at {path}:{line_no}: {acceptor_label}"
                )
            record = LabeledIntronRecord(
                species=str(raw_row["species"]).strip(),
                label=label,
                donor_label=donor_label,
                acceptor_label=acceptor_label,
            )
            key = (chrom, strand, intron_start, intron_end)
            previous = labels.get(key)
            if previous is not None and previous != record:
                raise ValueError(
                    "Conflicting labels for coordinate key in labeled TSV: "
                    f"{key}"
                )
            labels[key] = record
    if not labels:
        raise ValueError(f"No labeled rows found: {path}")
    return labels


def _read_train_positive_coords(path: Path) -> set[CoordKey]:
    """Read coordinate keys from positive intron training TSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing train-positive TSV: {path}")
    required = {"chrom", "strand", "intron_start", "intron_end"}
    coords: set[CoordKey] = set()
    set_csv_field_limit_max()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Train-positive TSV must include columns: "
                "chrom, strand, intron_start, intron_end"
            )
        for line_no, raw_row in enumerate(reader, start=2):
            strand = str(raw_row["strand"]).strip()
            if strand not in {"+", "-"}:
                raise ValueError(f"Invalid strand at {path}:{line_no}: {strand}")
            try:
                intron_start = int(str(raw_row["intron_start"]).strip())
                intron_end = int(str(raw_row["intron_end"]).strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid intron coordinate at {path}:{line_no}"
                ) from exc
            coords.add(
                (
                    str(raw_row["chrom"]).strip(),
                    strand,
                    intron_start,
                    intron_end,
                )
            )
    return coords


def _read_train_negative_pair_sequences(path: Path) -> set[tuple[str, str]]:
    """Read donor/acceptor sequence pairs from ``100bp.neg.err``."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing train-negative ERR: {path}")
    seq_pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parsed = parse_debug_training_record(raw_line.strip())
            if parsed is None or parsed.record_type != "pair":
                continue
            if parsed.donor_seq is None or parsed.acceptor_seq is None:
                continue
            seq_pairs.add((parsed.donor_seq.upper(), parsed.acceptor_seq.upper()))
    return seq_pairs


def _default_train_pos_path(species_dir: Path) -> Path:
    """Return default train-positive coordinate TSV path."""
    return species_dir / "processed" / "intron_full_flank10.pos.tsv"


def _default_train_neg_path(species_dir: Path) -> Path:
    """Return default train-negative ERR path."""
    return species_dir / "raw" / "100bp.neg.err"


def _resolve_species_template(raw_value: str, species: str) -> str:
    """Resolve ``{species}`` templates inside one path string."""
    return (
        raw_value.replace("{species}", species)
        .replace("{SPECIES}", species)
        .replace("${SPECIES}", species)
    )


def _resolve_train_input_path(
    species_dir: Path,
    raw_configured_path: str,
    default_path: Path,
) -> Path:
    """Resolve one train-input path from configured value and default."""
    if raw_configured_path.strip() == "":
        return default_path
    replaced = _resolve_species_template(raw_configured_path.strip(), species_dir.name)
    candidate = Path(replaced)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _build_unique_introns(
    *,
    species: str,
    transcript_introns: Mapping[MemberKey, TranscriptIntronRecord],
    labels_by_coord: Mapping[CoordKey, LabeledIntronRecord],
    train_pos_coords: set[CoordKey],
    train_neg_seq_pairs: set[tuple[str, str]],
) -> list[UniqueIntronRecord]:
    """Merge transcript introns by coordinate and attach labels/seen flags."""
    grouped_by_coord: dict[CoordKey, list[TranscriptIntronRecord]] = {}
    for record in transcript_introns.values():
        key = (record.chrom, record.strand, record.intron_start, record.intron_end)
        grouped_by_coord.setdefault(key, []).append(record)

    unique_records: list[UniqueIntronRecord] = []
    sorted_coords = sorted(grouped_by_coord.keys(), key=lambda item: item)
    for index, coord_key in enumerate(sorted_coords, start=1):
        members = grouped_by_coord[coord_key]
        representative = members[0]
        donor_seq = representative.donor_seq.upper()
        acceptor_seq = representative.acceptor_seq.upper()
        intron_half_length = representative.intron_half_length
        for member in members[1:]:
            if member.donor_seq.upper() != donor_seq:
                raise ValueError(
                    "Conflicting donor_seq among duplicated intron coordinates: "
                    f"{coord_key}"
                )
            if member.acceptor_seq.upper() != acceptor_seq:
                raise ValueError(
                    "Conflicting acceptor_seq among duplicated intron coordinates: "
                    f"{coord_key}"
                )
            if member.intron_half_length != intron_half_length:
                intron_half_length = None

        label_record = labels_by_coord.get(coord_key)
        if label_record is None:
            raise ValueError(
                "Missing label for unique intron coordinate. "
                f"species={species} coord={coord_key}"
            )
        seen_train_pos_coord = int(coord_key in train_pos_coords)
        seen_train_neg_seq = int((donor_seq, acceptor_seq) in train_neg_seq_pairs)
        member_keys = tuple(
            sorted(
                ((member.transcript_id, member.intron_index) for member in members),
                key=lambda item: (item[0], item[1]),
            )
        )
        unique_records.append(
            UniqueIntronRecord(
                unique_transcript_id=f"uintron_{index:08d}",
                unique_intron_index=1,
                chrom=coord_key[0],
                strand=coord_key[1],
                intron_start=coord_key[2],
                intron_end=coord_key[3],
                gene_id=representative.gene_id,
                donor_seq=donor_seq,
                acceptor_seq=acceptor_seq,
                intron_half_length=intron_half_length,
                label=label_record.label,
                donor_label=label_record.donor_label,
                acceptor_label=label_record.acceptor_label,
                member_keys=member_keys,
                seen_train_pos_coord=seen_train_pos_coord,
                seen_train_neg_seq=seen_train_neg_seq,
                train_leak=int(seen_train_pos_coord == 1 or seen_train_neg_seq == 1),
            )
        )
    return unique_records


def _write_transcripts_unique_tsv(path: Path, rows: Iterable[UniqueIntronRecord]) -> None:
    """Write ``transcripts.unique.tsv`` for inference input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "transcript_id",
        "gene_id",
        "site_type",
        "intron_index",
        "chrom",
        "strand",
        "boundary_pos",
        "seq",
        "intron_half_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            if row.strand == "+":
                donor_boundary = row.intron_start
                acceptor_boundary = row.intron_end + 1
            else:
                donor_boundary = row.intron_end
                acceptor_boundary = row.intron_start - 1
            intron_half_length_text = (
                "" if row.intron_half_length is None else str(row.intron_half_length)
            )
            writer.writerow(
                {
                    "transcript_id": row.unique_transcript_id,
                    "gene_id": row.gene_id,
                    "site_type": "donor",
                    "intron_index": row.unique_intron_index,
                    "chrom": row.chrom,
                    "strand": row.strand,
                    "boundary_pos": donor_boundary,
                    "seq": row.donor_seq,
                    "intron_half_length": intron_half_length_text,
                }
            )
            writer.writerow(
                {
                    "transcript_id": row.unique_transcript_id,
                    "gene_id": row.gene_id,
                    "site_type": "acceptor",
                    "intron_index": row.unique_intron_index,
                    "chrom": row.chrom,
                    "strand": row.strand,
                    "boundary_pos": acceptor_boundary,
                    "seq": row.acceptor_seq,
                    "intron_half_length": intron_half_length_text,
                }
            )


def _write_unique_map_tsv(path: Path, rows: Iterable[UniqueIntronRecord]) -> None:
    """Write ``transcripts.unique.map.tsv`` mapping rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "unique_transcript_id",
        "unique_intron_index",
        "transcript_id",
        "intron_index",
        "chrom",
        "strand",
        "intron_start",
        "intron_end",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            for transcript_id, intron_index in row.member_keys:
                writer.writerow(
                    {
                        "unique_transcript_id": row.unique_transcript_id,
                        "unique_intron_index": row.unique_intron_index,
                        "transcript_id": transcript_id,
                        "intron_index": intron_index,
                        "chrom": row.chrom,
                        "strand": row.strand,
                        "intron_start": row.intron_start,
                        "intron_end": row.intron_end,
                    }
                )


def _write_unique_labeled_tsv(
    path: Path,
    rows: Iterable[UniqueIntronRecord],
    species: str,
) -> None:
    """Write ``intron_eval_flank10.unique.tsv`` for unique intron evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "species",
        "transcript_id",
        "intron_index",
        "chrom",
        "strand",
        "intron_start",
        "intron_end",
        "intron_length",
        "label",
        "donor_label",
        "acceptor_label",
        "member_count",
        "seen_train_pos_coord",
        "seen_train_neg_seq",
        "train_leak",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "species": species,
                    "transcript_id": row.unique_transcript_id,
                    "intron_index": row.unique_intron_index,
                    "chrom": row.chrom,
                    "strand": row.strand,
                    "intron_start": row.intron_start,
                    "intron_end": row.intron_end,
                    "intron_length": row.intron_end - row.intron_start + 1,
                    "label": row.label,
                    "donor_label": (
                        "" if row.donor_label is None else str(row.donor_label)
                    ),
                    "acceptor_label": (
                        "" if row.acceptor_label is None else str(row.acceptor_label)
                    ),
                    "member_count": len(row.member_keys),
                    "seen_train_pos_coord": row.seen_train_pos_coord,
                    "seen_train_neg_seq": row.seen_train_neg_seq,
                    "train_leak": row.train_leak,
                }
            )


def _write_catalog_tsv(path: Path, rows: Iterable[UniqueIntronRecord]) -> None:
    """Write ``intron_unique_catalog.tsv`` summary rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "unique_transcript_id",
        "unique_intron_index",
        "chrom",
        "strand",
        "intron_start",
        "intron_end",
        "member_count",
        "seen_train_pos_coord",
        "seen_train_neg_seq",
        "train_leak",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "unique_transcript_id": row.unique_transcript_id,
                    "unique_intron_index": row.unique_intron_index,
                    "chrom": row.chrom,
                    "strand": row.strand,
                    "intron_start": row.intron_start,
                    "intron_end": row.intron_end,
                    "member_count": len(row.member_keys),
                    "seen_train_pos_coord": row.seen_train_pos_coord,
                    "seen_train_neg_seq": row.seen_train_neg_seq,
                    "train_leak": row.train_leak,
                }
            )


def _assert_writable_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    """Validate output overwrite policy."""
    if overwrite:
        return
    conflicts = [path for path in paths if path.exists()]
    if conflicts:
        names = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            "Output files already exist. Re-run with --overwrite to replace: "
            f"{names}"
        )


def _build_for_species(
    *,
    species_dir: Path,
    labeled_name: str,
    train_pos_path: str,
    train_neg_path: str,
    overwrite: bool,
) -> None:
    """Build all unique-intron assets for one species directory."""
    species = species_dir.name
    processed_dir = species_dir / "processed"
    raw_dir = species_dir / "raw"

    source_transcripts_tsv = processed_dir / "transcripts.tsv"
    source_labeled_tsv = processed_dir / labeled_name
    resolved_train_pos_path = _resolve_train_input_path(
        species_dir=species_dir,
        raw_configured_path=train_pos_path,
        default_path=_default_train_pos_path(species_dir),
    )
    resolved_train_neg_path = _resolve_train_input_path(
        species_dir=species_dir,
        raw_configured_path=train_neg_path,
        default_path=_default_train_neg_path(species_dir),
    )

    output_transcripts_unique = processed_dir / UNIQUE_TRANSCRIPTS_TSV_NAME
    output_unique_map = processed_dir / UNIQUE_MAP_TSV_NAME
    output_labeled_unique = processed_dir / UNIQUE_LABELED_INTRON_TSV_NAME
    output_catalog = processed_dir / UNIQUE_INTRON_CATALOG_TSV_NAME

    _assert_writable_outputs(
        [
            output_transcripts_unique,
            output_unique_map,
            output_labeled_unique,
            output_catalog,
        ],
        overwrite=overwrite,
    )
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    site_rows = _read_transcript_site_rows(source_transcripts_tsv)
    transcript_introns = _assemble_transcript_introns(site_rows)
    labels_by_coord = _read_labeled_introns_by_coord(source_labeled_tsv)
    train_pos_coords = _read_train_positive_coords(resolved_train_pos_path)
    train_neg_seq_pairs = _read_train_negative_pair_sequences(resolved_train_neg_path)

    unique_introns = _build_unique_introns(
        species=species,
        transcript_introns=transcript_introns,
        labels_by_coord=labels_by_coord,
        train_pos_coords=train_pos_coords,
        train_neg_seq_pairs=train_neg_seq_pairs,
    )
    _write_transcripts_unique_tsv(output_transcripts_unique, unique_introns)
    _write_unique_map_tsv(output_unique_map, unique_introns)
    _write_unique_labeled_tsv(output_labeled_unique, unique_introns, species=species)
    _write_catalog_tsv(output_catalog, unique_introns)

    train_leak_count = sum(row.train_leak for row in unique_introns)
    print(
        "[build_unique_intron_assets] "
        f"species={species} unique_introns={len(unique_introns)} "
        f"train_leak={train_leak_count} "
        f"outputs={output_transcripts_unique},{output_unique_map},"
        f"{output_labeled_unique},{output_catalog}"
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build precomputed unique-intron assets under data/<species>/processed."
        )
    )
    parser.add_argument(
        "--species",
        default="",
        help="Comma-separated species list. Empty means all species.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Data root path.",
    )
    parser.add_argument(
        "--labeled-name",
        default="intron_eval_flank10.tsv",
        help="Source labeled intron TSV name under processed/.",
    )
    parser.add_argument(
        "--train-pos-path",
        default="",
        help=(
            "Optional positive-train coordinate TSV path template. "
            "Supports {species}. Default: processed/intron_full_flank10.pos.tsv"
        ),
    )
    parser.add_argument(
        "--train-neg-path",
        default="",
        help=(
            "Optional negative-train ERR path template. Supports {species}. "
            "Default: raw/100bp.neg.err"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the unique-intron asset builder CLI."""
    args = _build_parser().parse_args(argv)
    species_dirs = _resolve_species_dirs(
        data_root=Path(args.data_root).resolve(),
        raw_species=str(args.species),
    )
    if not species_dirs:
        raise ValueError(f"No species directories found under {args.data_root}")

    for species_dir in species_dirs:
        _build_for_species(
            species_dir=species_dir,
            labeled_name=str(args.labeled_name),
            train_pos_path=str(args.train_pos_path),
            train_neg_path=str(args.train_neg_path),
            overwrite=bool(args.overwrite),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
