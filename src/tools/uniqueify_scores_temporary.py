"""Temporarily rewrite site/intron score TSV files to unique intron keys.

This tool rewrites existing ``data/<species>/site_score/*.tsv`` and
``data/<species>/intron_score/*.tsv`` files in-place:

- Replace ``(transcript_id, intron_index)`` with unique intron keys.
- Collapse duplicate original introns mapped to one unique intron.
- Attach ``train_leak`` and representative source metadata columns.

The script is intentionally strict and fails fast when inputs are inconsistent.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from util.unique_intron import (
    UNIQUE_INTRON_CATALOG_TSV_NAME,
    UNIQUE_MAP_TSV_NAME,
    UniqueMapMember,
    invert_unique_map,
    load_unique_map,
    parse_bool_flag,
    set_csv_field_limit_max,
)

OriginalKey = tuple[str, int]
UniqueKey = tuple[str, int]
SCORE_COLUMNS: tuple[str, ...] = ("donor_score", "acceptor_score", "score")
ADDED_COLUMNS: tuple[str, ...] = (
    "train_leak",
    "source_transcript_id",
    "source_intron_index",
    "member_count",
)


@dataclass(frozen=True)
class CatalogRecord:
    """One unique intron record loaded from ``intron_unique_catalog.tsv``."""

    train_leak: int
    member_count: int


@dataclass(frozen=True)
class FileResult:
    """Summary counts for one transformed TSV file."""

    rows_in: int
    rows_out: int


@dataclass(frozen=True)
class SpeciesResult:
    """Summary counts for one species directory."""

    site_files: int
    intron_files: int
    rows_in: int
    rows_out: int


@dataclass
class ScoreDriftAccumulator:
    """Running score statistics for one score column."""

    has_empty: bool
    has_non_empty: bool
    min_value: float | None
    max_value: float | None


@dataclass
class GroupAccumulator:
    """Running aggregation state for one unique intron group."""

    first_row: dict[str, str]
    representative_row: dict[str, str] | None
    label_values: set[str]
    score_stats: dict[str, ScoreDriftAccumulator]


def _parse_species_csv(raw_species: Optional[str]) -> list[str]:
    """Parse one optional comma-separated species list."""
    if raw_species is None:
        return []
    tokens = [token.strip() for token in raw_species.split(",")]
    return [token for token in tokens if token != ""]


def _iter_species_dirs(data_root: Path, include_species: list[str]) -> list[Path]:
    """Return sorted species directories under ``data_root``."""
    if include_species:
        candidates = [data_root / species for species in include_species]
    else:
        candidates = [path for path in data_root.iterdir() if path.is_dir()]
    return sorted(path for path in candidates if path.is_dir())


def _parse_required_int(
    *,
    raw: dict[str, str],
    key: str,
    path: Path,
    line_no: int,
) -> int:
    """Parse one required integer field from one TSV row.

    Parameters
    ----------
    raw : dict[str, str]
        Source row from ``csv.DictReader``.
    key : str
        Target field name.
    path : Path
        Source TSV path.
    line_no : int
        One-based line number in ``path``.

    Returns
    -------
    int
        Parsed integer value.

    Raises
    ------
    ValueError
        If the field is empty or cannot be parsed as integer.
    """
    value = str(raw.get(key, "")).strip()
    if value == "":
        raise ValueError(f"Empty '{key}' at {path}:{line_no}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid '{key}' at {path}:{line_no}: {value}") from exc


def _parse_optional_float(
    *,
    raw_value: str,
    path: Path,
    line_no: int,
    column: str,
) -> float | None:
    """Parse one optional float field.

    Parameters
    ----------
    raw_value : str
        Raw string value to parse.
    path : Path
        Source TSV path.
    line_no : int
        One-based line number in ``path``.
    column : str
        Column name.

    Returns
    -------
    float | None
        ``None`` when value is empty, otherwise parsed float.

    Raises
    ------
    ValueError
        If non-empty ``raw_value`` is not a valid finite float.
    """
    value = raw_value.strip()
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid float in column '{column}' at {path}:{line_no}: {raw_value}"
        ) from exc
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(
            f"Non-finite float in column '{column}' at {path}:{line_no}: {raw_value}"
        )
    return parsed


def _load_catalog(path: Path) -> dict[UniqueKey, CatalogRecord]:
    """Load unique intron catalog records keyed by unique intron id.

    Parameters
    ----------
    path : Path
        Path to ``intron_unique_catalog.tsv``.

    Returns
    -------
    dict[tuple[str, int], CatalogRecord]
        Catalog map keyed by ``(unique_transcript_id, unique_intron_index)``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If schema or values are invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Unique intron catalog TSV not found: {path}")

    required_base = {
        "unique_transcript_id",
        "unique_intron_index",
        "member_count",
    }
    catalog: dict[UniqueKey, CatalogRecord] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        if not required_base.issubset(fieldnames):
            raise ValueError(
                "Unique intron catalog TSV must include columns: "
                "unique_transcript_id, unique_intron_index, member_count"
            )
        leak_column = "train_leak"
        if leak_column not in fieldnames:
            raise ValueError(
                "Unique intron catalog TSV must include column: train_leak"
            )
        for line_no, raw in enumerate(reader, start=2):
            unique_transcript_id = str(raw["unique_transcript_id"]).strip()
            if unique_transcript_id == "":
                raise ValueError(f"Empty unique_transcript_id at {path}:{line_no}")
            unique_intron_index = _parse_required_int(
                raw=raw,
                key="unique_intron_index",
                path=path,
                line_no=line_no,
            )
            member_count = _parse_required_int(
                raw=raw,
                key="member_count",
                path=path,
                line_no=line_no,
            )
            train_leak = parse_bool_flag(str(raw[leak_column]))
            unique_key = (unique_transcript_id, unique_intron_index)
            if unique_key in catalog:
                raise ValueError(
                    "Duplicate unique key in catalog: "
                    f"{unique_key} at {path}:{line_no}"
                )
            catalog[unique_key] = CatalogRecord(
                train_leak=train_leak,
                member_count=member_count,
            )

    if not catalog:
        raise ValueError(f"No valid rows in unique intron catalog TSV: {path}")
    return catalog


def _build_representative_map(
    unique_map: dict[UniqueKey, list[UniqueMapMember]],
) -> dict[UniqueKey, OriginalKey]:
    """Pick one deterministic representative original key per unique key.

    Parameters
    ----------
    unique_map : dict[tuple[str, int], list[UniqueMapMember]]
        Unique-to-original mapping loaded from ``transcripts.unique.map.tsv``.

    Returns
    -------
    dict[tuple[str, int], tuple[str, int]]
        Representative original key map using lexicographic minimum
        ``(transcript_id, intron_index)``.

    Raises
    ------
    ValueError
        If any unique key has no member rows.
    """
    representatives: dict[UniqueKey, OriginalKey] = {}
    for unique_key, members in unique_map.items():
        member_keys = sorted(
            (member.transcript_id, member.intron_index) for member in members
        )
        if not member_keys:
            raise ValueError(f"Unique key has no members in map: {unique_key}")
        representatives[unique_key] = member_keys[0]
    return representatives


def _init_score_stats(score_columns: list[str]) -> dict[str, ScoreDriftAccumulator]:
    """Build empty score-drift accumulators for one group."""
    return {
        column: ScoreDriftAccumulator(
            has_empty=False,
            has_non_empty=False,
            min_value=None,
            max_value=None,
        )
        for column in score_columns
    }


def _update_label_accumulator(
    *,
    row: dict[str, str],
    line_no: int,
    path: Path,
    label_values: set[str],
) -> None:
    """Update one group label accumulator from one row."""
    if "label" not in row:
        return
    value = str(row["label"]).strip()
    if value != "" and value not in {"0", "1"}:
        raise ValueError(
            f"Label must be 0/1 or empty at {path}:{line_no}, got: {row['label']}"
        )
    if value != "":
        label_values.add(value)


def _update_score_accumulators(
    *,
    row: dict[str, str],
    line_no: int,
    path: Path,
    score_stats: dict[str, ScoreDriftAccumulator],
) -> None:
    """Update score drift accumulators for one group from one row."""
    for column, stats in score_stats.items():
        parsed = _parse_optional_float(
            raw_value=str(row.get(column, "")),
            path=path,
            line_no=line_no,
            column=column,
        )
        if parsed is None:
            stats.has_empty = True
            continue
        stats.has_non_empty = True
        if stats.min_value is None or parsed < stats.min_value:
            stats.min_value = parsed
        if stats.max_value is None or parsed > stats.max_value:
            stats.max_value = parsed


def _resolve_label_value(
    *,
    label_values: set[str],
    path: Path,
    unique_key: UniqueKey,
) -> str:
    """Resolve one output label from collected group labels."""
    if len(label_values) > 1:
        raise ValueError(
            "Conflicting labels in one unique intron group: "
            f"{unique_key} labels={sorted(label_values)} file={path}"
        )
    if not label_values:
        return ""
    return sorted(label_values)[0]


def _validate_score_stats(
    *,
    score_stats: dict[str, ScoreDriftAccumulator],
    tolerance: float,
    path: Path,
    unique_key: UniqueKey,
    strict_score_drift: bool,
) -> None:
    """Validate one group score accumulators."""
    for column, stats in score_stats.items():
        if stats.has_empty and stats.has_non_empty:
            raise ValueError(
                "Mixed empty/non-empty values in one unique intron group: "
                f"file={path} key={unique_key} column={column}"
            )
        if (
            stats.min_value is None
            or stats.max_value is None
            or not strict_score_drift
        ):
            continue
        diff = stats.max_value - stats.min_value
        if diff > tolerance:
            raise ValueError(
                "Score drift exceeds tolerance in one unique intron group: "
                f"file={path} key={unique_key} column={column} diff={diff:.8g} "
                f"tolerance={tolerance:.8g}"
            )


def _build_output_fieldnames(fieldnames: list[str]) -> list[str]:
    """Return output field names with required metadata columns appended."""
    output = list(fieldnames)
    for column in ADDED_COLUMNS:
        if column not in output:
            output.append(column)
    return output


def _transform_one_file(
    *,
    path: Path,
    original_to_unique: dict[OriginalKey, UniqueKey],
    representative_map: dict[UniqueKey, OriginalKey],
    catalog: dict[UniqueKey, CatalogRecord],
    tolerance: float,
    strict_score_drift: bool,
    dry_run: bool,
) -> FileResult:
    """Transform one score TSV file to unique keys and overwrite in-place.

    Parameters
    ----------
    path : Path
        Input/output TSV path.
    original_to_unique : dict[tuple[str, int], tuple[str, int]]
        Reverse mapping from original to unique intron keys.
    representative_map : dict[tuple[str, int], tuple[str, int]]
        Representative original member per unique intron.
    catalog : dict[tuple[str, int], CatalogRecord]
        Catalog metadata keyed by unique intron.
    tolerance : float
        Maximum allowed score drift in one unique group.
    dry_run : bool
        If ``True``, validate and count only without writing output.

    Returns
    -------
    FileResult
        Input and output row counts for this file.

    Raises
    ------
    ValueError
        If schema is invalid, keys are unmapped, labels conflict, or
        score drift exceeds tolerance.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in TSV: {path}")
        fieldnames = [str(name) for name in reader.fieldnames]
        required = {"transcript_id", "intron_index"}
        if not required.issubset(set(fieldnames)):
            raise ValueError(
                "Score TSV must include transcript_id and intron_index: "
                f"{path}"
            )

        score_columns = [column for column in SCORE_COLUMNS if column in fieldnames]
        grouped_rows: dict[UniqueKey, GroupAccumulator] = {}
        total_rows = 0
        for line_no, raw in enumerate(reader, start=2):
            total_rows += 1
            row = {name: str(raw.get(name, "")) for name in fieldnames}
            transcript_id = row["transcript_id"].strip()
            if transcript_id == "":
                raise ValueError(f"Empty transcript_id at {path}:{line_no}")
            intron_index = _parse_required_int(
                raw=row,
                key="intron_index",
                path=path,
                line_no=line_no,
            )
            original_key = (transcript_id, intron_index)
            unique_key = original_to_unique.get(original_key)
            if unique_key is None:
                # Some score files may already be rewritten to unique keys.
                if original_key in representative_map:
                    unique_key = original_key
                else:
                    raise ValueError(
                        "Unmapped original intron key in score TSV: "
                        f"{original_key} at {path}:{line_no}"
                    )
            representative_key = representative_map.get(unique_key)
            if representative_key is None:
                raise ValueError(f"Missing representative key for: {unique_key}")

            group = grouped_rows.get(unique_key)
            if group is None:
                group = GroupAccumulator(
                    first_row=row,
                    representative_row=None,
                    label_values=set(),
                    score_stats=_init_score_stats(score_columns),
                )
                grouped_rows[unique_key] = group

            if original_key == representative_key:
                group.representative_row = row
            _update_label_accumulator(
                row=row,
                line_no=line_no,
                path=path,
                label_values=group.label_values,
            )
            _update_score_accumulators(
                row=row,
                line_no=line_no,
                path=path,
                score_stats=group.score_stats,
            )

    output_fieldnames = _build_output_fieldnames(fieldnames)
    output_rows: list[dict[str, str]] = []
    for unique_key in sorted(grouped_rows.keys()):
        group = grouped_rows[unique_key]
        catalog_record = catalog.get(unique_key)
        if catalog_record is None:
            raise ValueError(f"Missing catalog record for unique key: {unique_key}")
        representative_key = representative_map.get(unique_key)
        if representative_key is None:
            raise ValueError(f"Missing representative key for: {unique_key}")
        label_value = _resolve_label_value(
            label_values=group.label_values,
            path=path,
            unique_key=unique_key,
        )
        _validate_score_stats(
            score_stats=group.score_stats,
            tolerance=tolerance,
            path=path,
            unique_key=unique_key,
            strict_score_drift=strict_score_drift,
        )
        representative_row = (
            group.representative_row
            if group.representative_row is not None
            else group.first_row
        )

        output_row = {name: representative_row.get(name, "") for name in fieldnames}
        output_row["transcript_id"] = unique_key[0]
        output_row["intron_index"] = str(unique_key[1])
        if "label" in output_row:
            output_row["label"] = label_value
        output_row["train_leak"] = str(catalog_record.train_leak)
        output_row["source_transcript_id"] = representative_key[0]
        output_row["source_intron_index"] = str(representative_key[1])
        output_row["member_count"] = str(catalog_record.member_count)
        output_rows.append(output_row)

    if not dry_run:
        tmp_path = path.with_suffix(".tmp.tsv")
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=output_fieldnames,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in output_rows:
                writer.writerow(row)
        tmp_path.replace(path)

    return FileResult(rows_in=total_rows, rows_out=len(output_rows))


def _transform_path_group(
    *,
    paths: Iterable[Path],
    original_to_unique: dict[OriginalKey, UniqueKey],
    representative_map: dict[UniqueKey, OriginalKey],
    catalog: dict[UniqueKey, CatalogRecord],
    tolerance: float,
    strict_score_drift: bool,
    dry_run: bool,
    label: str,
) -> tuple[int, int, int]:
    """Transform one group of score files and return summary counts."""
    file_count = 0
    rows_in = 0
    rows_out = 0
    for path in sorted(paths):
        if not path.is_file():
            continue
        file_count += 1
        result = _transform_one_file(
            path=path,
            original_to_unique=original_to_unique,
            representative_map=representative_map,
            catalog=catalog,
            tolerance=tolerance,
            strict_score_drift=strict_score_drift,
            dry_run=dry_run,
        )
        rows_in += result.rows_in
        rows_out += result.rows_out
        action = "validated" if dry_run else "wrote"
        print(
            f"[uniqueify_scores_temporary] {action}: {path} "
            f"type={label} rows_in={result.rows_in} rows_out={result.rows_out}"
        )
    return file_count, rows_in, rows_out


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the temporary unique updater."""
    parser = argparse.ArgumentParser(
        description=(
            "Temporarily rewrite data/<species>/site_score and intron_score TSV "
            "files to unique intron keys in-place."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--site-pattern", type=str, default="*.tsv")
    parser.add_argument("--intron-pattern", type=str, default="*.tsv")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--strict-score-drift",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--dry-run", type=int, choices=[0, 1], default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the temporary unique score updater CLI.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Optional CLI argument sequence. When ``None``, parse from ``sys.argv``.

    Returns
    -------
    int
        Process exit code (always ``0`` on success).

    Raises
    ------
    FileNotFoundError
        If required map/catalog files are missing.
    ValueError
        If file schema or score consistency checks fail.
    """
    args = build_arg_parser().parse_args(argv)
    set_csv_field_limit_max()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    tolerance = float(args.tolerance)
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    strict_score_drift = bool(args.strict_score_drift)
    dry_run = bool(args.dry_run)
    include_species = _parse_species_csv(args.species)

    total_site_files = 0
    total_intron_files = 0
    total_rows_in = 0
    total_rows_out = 0

    for species_dir in _iter_species_dirs(data_root, include_species):
        species = species_dir.name
        processed_dir = species_dir / "processed"
        unique_map_path = processed_dir / UNIQUE_MAP_TSV_NAME
        catalog_path = processed_dir / UNIQUE_INTRON_CATALOG_TSV_NAME
        unique_map = load_unique_map(unique_map_path)
        catalog = _load_catalog(catalog_path)
        representative_map = _build_representative_map(unique_map)
        original_to_unique = invert_unique_map(unique_map)

        for unique_key, members in unique_map.items():
            catalog_record = catalog.get(unique_key)
            if catalog_record is None:
                raise ValueError(f"Missing catalog record for unique key: {unique_key}")
            if catalog_record.member_count != len(members):
                raise ValueError(
                    "Catalog member_count mismatch for unique key: "
                    f"{unique_key} catalog={catalog_record.member_count} "
                    f"map={len(members)}"
                )

        site_dir = species_dir / "site_score"
        intron_dir = species_dir / "intron_score"
        site_paths = site_dir.glob(str(args.site_pattern)) if site_dir.is_dir() else []
        intron_paths = (
            intron_dir.glob(str(args.intron_pattern)) if intron_dir.is_dir() else []
        )

        site_files, site_rows_in, site_rows_out = _transform_path_group(
            paths=site_paths,
            original_to_unique=original_to_unique,
            representative_map=representative_map,
            catalog=catalog,
            tolerance=tolerance,
            strict_score_drift=strict_score_drift,
            dry_run=dry_run,
            label="site",
        )
        intron_files, intron_rows_in, intron_rows_out = _transform_path_group(
            paths=intron_paths,
            original_to_unique=original_to_unique,
            representative_map=representative_map,
            catalog=catalog,
            tolerance=tolerance,
            strict_score_drift=strict_score_drift,
            dry_run=dry_run,
            label="intron",
        )
        species_result = SpeciesResult(
            site_files=site_files,
            intron_files=intron_files,
            rows_in=site_rows_in + intron_rows_in,
            rows_out=site_rows_out + intron_rows_out,
        )
        total_site_files += species_result.site_files
        total_intron_files += species_result.intron_files
        total_rows_in += species_result.rows_in
        total_rows_out += species_result.rows_out

        print(
            "[uniqueify_scores_temporary] species="
            f"{species} site_files={species_result.site_files} "
            f"intron_files={species_result.intron_files} "
            f"rows_in={species_result.rows_in} rows_out={species_result.rows_out} "
            f"dry_run={dry_run}"
        )

    print(
        "[uniqueify_scores_temporary] done "
        f"site_files={total_site_files} intron_files={total_intron_files} "
        f"rows_in={total_rows_in} rows_out={total_rows_out} dry_run={dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
