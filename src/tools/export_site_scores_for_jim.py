"""Export site_score TSV files into Jim-compatible TSV files.

This utility reads ``data/<species>/site_score/*.tsv`` and writes
``data/<species>/jim/<model_name>.tsv``. Output rows use a simple
1-based ``ID`` index and remove ``transcript_id`` and ``intron_index``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

IntronKey = tuple[str, str]


def _configure_csv_field_size_limit() -> None:
    """Set CSV field size limit to a large safe value.

    This avoids ``csv.Error: field larger than field limit`` when reading
    processed tables that contain long sequence fields.
    """
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size = max_size // 10


def _parse_species_csv(raw_species: Optional[str]) -> list[str]:
    """Parse one optional comma-separated species list."""
    if raw_species is None:
        return []
    species = [token.strip() for token in raw_species.split(",")]
    return [token for token in species if token != ""]


def _iter_species_dirs(data_root: Path, include_species: list[str]) -> list[Path]:
    """Return sorted species directories under ``data_root``."""
    if include_species:
        candidates = [data_root / species for species in include_species]
    else:
        candidates = [path for path in data_root.iterdir() if path.is_dir()]
    return sorted(path for path in candidates if path.is_dir())


def _is_canonical_site_score_file(path: Path) -> bool:
    """Return whether one site_score file should be converted."""
    name = path.name
    if not name.endswith(".tsv"):
        return False
    return (
        ".legacy." not in name
        and ".widev" not in name
        and ".legacy.tsv" not in name
    )


def _read_label_lookup(site_score_paths: list[Path]) -> dict[IntronKey, str]:
    """Build one label lookup table from available site_score TSV files.

    Parameters
    ----------
    site_score_paths : list[Path]
        Candidate site_score TSV files from one species.

    Returns
    -------
    dict[IntronKey, str]
        Mapping from ``(transcript_id, intron_index)`` to non-empty ``label``.
    """
    label_lookup: dict[IntronKey, str] = {}
    for site_path in site_score_paths:
        with site_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                continue
            if not {"transcript_id", "intron_index", "label"}.issubset(
                set(reader.fieldnames)
            ):
                continue
            for row in reader:
                transcript_id = str(row.get("transcript_id", ""))
                intron_index = str(row.get("intron_index", ""))
                label = str(row.get("label", ""))
                if label == "":
                    continue
                label_lookup[(transcript_id, intron_index)] = label
    return label_lookup


def _read_processed_label_lookup(species_dir: Path) -> dict[IntronKey, str]:
    """Build one label lookup table from processed evaluation data.

    Parameters
    ----------
    species_dir : Path
        One species directory under ``data``.

    Returns
    -------
    dict[IntronKey, str]
        Mapping from ``(transcript_id, intron_index)`` to non-empty ``label``.
        Returns one empty mapping when the processed file is unavailable.
    """
    processed_path = species_dir / "processed" / "intron_eval_flank10.tsv"
    if not processed_path.is_file():
        return {}

    label_lookup: dict[IntronKey, str] = {}
    with processed_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return {}
        if not {"transcript_id", "intron_index", "label"}.issubset(
            set(reader.fieldnames)
        ):
            return {}
        for row in reader:
            transcript_id = str(row.get("transcript_id", ""))
            intron_index = str(row.get("intron_index", ""))
            label = str(row.get("label", ""))
            if label == "":
                continue
            label_lookup[(transcript_id, intron_index)] = label
    return label_lookup


def _convert_one_file(
    input_path: Path,
    output_path: Path,
    label_lookup: Optional[dict[IntronKey, str]] = None,
) -> int:
    """Convert one site_score TSV into one Jim TSV.

    Parameters
    ----------
    input_path : Path
        Input site-score TSV path.
    output_path : Path
        Output Jim TSV path.
    label_lookup : dict[IntronKey, str] | None, optional
        Optional fallback label lookup keyed by
        ``(transcript_id, intron_index)``. Used when one input row has an
        empty ``label`` value.

    Returns
    -------
    int
        Number of rows written.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {input_path}")

        fieldnames = list(reader.fieldnames)
        required = {"transcript_id", "intron_index"}
        if not required.issubset(set(fieldnames)):
            raise ValueError(
                "Input must contain transcript_id and intron_index: "
                f"{input_path}"
            )
        non_label_columns = [
            column
            for column in fieldnames
            if column not in {"transcript_id", "intron_index", "label"}
        ]
        output_columns = [*non_label_columns, "label"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.writer(output_handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["ID", *output_columns])
            written_rows = 0
            for row_index, row in enumerate(reader, start=1):
                transcript_id = str(row["transcript_id"])
                intron_index = str(row["intron_index"])
                row_label = str(row.get("label", ""))
                if row_label == "" and label_lookup is not None:
                    row_label = label_lookup.get(
                        (transcript_id, intron_index), ""
                    )
                row["label"] = row_label
                output_row = [str(row_index)]
                output_row.extend(str(row.get(col, "")) for col in output_columns)
                writer.writerow(output_row)
                written_rows += 1
    return written_rows


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for the Jim export utility.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export data/<species>/site_score/*.tsv to "
            "data/<species>/jim/<model_name>.tsv with ID=1..N."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*.tsv")
    parser.add_argument("--dry-run", type=int, choices=[0, 1], default=0)
    return parser


def main() -> int:
    """Run the Jim export command-line utility.

    Returns
    -------
    int
        Process exit code. Returns ``0`` on successful completion.

    Raises
    ------
    FileNotFoundError
        If the provided data root directory does not exist.
    """
    args = build_arg_parser().parse_args()
    _configure_csv_field_size_limit()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    include_species = _parse_species_csv(args.species)
    dry_run = bool(args.dry_run)
    scanned = 0
    written = 0
    skipped = 0

    for species_dir in _iter_species_dirs(data_root, include_species):
        species = species_dir.name
        site_score_dir = species_dir / "site_score"
        jim_dir = species_dir / "jim"
        if not site_score_dir.is_dir():
            continue

        site_paths = [
            site_path
            for site_path in sorted(site_score_dir.glob(str(args.pattern)))
            if site_path.is_file() and _is_canonical_site_score_file(site_path)
        ]
        label_lookup = _read_label_lookup(site_paths)
        if not label_lookup:
            label_lookup = _read_processed_label_lookup(species_dir)

        for site_path in site_paths:
            scanned += 1
            output_path = jim_dir / site_path.name
            try:
                if dry_run:
                    print(
                        "[export_site_scores_for_jim] dry-run: "
                        f"{site_path} -> {output_path}"
                    )
                    written += 1
                    continue
                row_count = _convert_one_file(
                    site_path,
                    output_path,
                    label_lookup=label_lookup,
                )
                print(
                    "[export_site_scores_for_jim] wrote: "
                    f"{output_path} rows={row_count} species={species}"
                )
                written += 1
            except ValueError as error:
                skipped += 1
                print(
                    "[export_site_scores_for_jim] skip: "
                    f"{site_path} reason={error}"
                )

    print(
        "[export_site_scores_for_jim] done "
        f"scanned={scanned} written={written} skipped={skipped} dry_run={dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
