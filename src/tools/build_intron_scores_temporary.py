"""Build intron_score TSV files from existing site_score TSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from util.transcript_eval import (
    build_intron_scores,
    read_site_scores,
    write_intron_scores,
)


def _parse_species_csv(raw_species: Optional[str]) -> list[str]:
    """Parse one optional comma-separated species list."""
    if raw_species is None:
        return []
    species = [token.strip() for token in raw_species.split(",")]
    return [token for token in species if token != ""]


def _set_csv_field_limit_max() -> None:
    """Set CSV field-size limit to the largest supported value."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _load_optional_labels(
    *,
    species_dir: Path,
    labeled_name: str,
) -> dict[tuple[str, int], int]:
    """Load optional intron labels from ``processed/<labeled_name>``."""
    labeled_path = species_dir / "processed" / labeled_name
    if not labeled_path.is_file():
        return {}

    labels: dict[tuple[str, int], int] = {}
    with labeled_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"transcript_id", "intron_index", "label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            return {}
        for raw in reader:
            transcript_id = str(raw["transcript_id"]).strip()
            if transcript_id == "":
                continue
            try:
                intron_index = int(str(raw["intron_index"]))
                label = int(str(raw["label"]))
            except ValueError:
                continue
            if label not in {0, 1}:
                continue
            labels[(transcript_id, intron_index)] = label
    return labels


def _iter_species_dirs(data_root: Path, include_species: list[str]) -> list[Path]:
    """Return sorted species directories under ``data_root``."""
    if include_species:
        candidates = [data_root / species for species in include_species]
    else:
        candidates = [path for path in data_root.iterdir() if path.is_dir()]
    return sorted(path for path in candidates if path.is_dir())


def _is_canonical_site_score_file(path: Path) -> bool:
    """Return whether one site_score file should be included for conversion."""
    name = path.name
    if not name.endswith(".tsv"):
        return False
    return ".legacy." not in name and ".widev" not in name and ".legacy.tsv" not in name


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Temporary utility to build data/<species>/intron_score/*.tsv "
            "from existing site_score outputs."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*.tsv")
    parser.add_argument(
        "--labeled-name",
        type=str,
        default="intron_eval_flank10.tsv",
        help="Label TSV name under data/<species>/processed.",
    )
    parser.add_argument(
        "--intron-score-op",
        type=str,
        default="+",
        choices=["+", "*", "harmonic", "min"],
    )
    parser.add_argument("--dry-run", type=int, choices=[0, 1], default=0)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    _set_csv_field_limit_max()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    include_species = _parse_species_csv(args.species)
    dry_run = bool(args.dry_run)
    scanned = 0
    written = 0
    skipped = 0

    for species_dir in _iter_species_dirs(data_root, include_species):
        site_score_dir = species_dir / "site_score"
        intron_score_dir = species_dir / "intron_score"
        if not site_score_dir.is_dir():
            continue
        labels = _load_optional_labels(
            species_dir=species_dir,
            labeled_name=str(args.labeled_name),
        )

        for site_path in sorted(site_score_dir.glob(str(args.pattern))):
            if not site_path.is_file():
                continue
            if not _is_canonical_site_score_file(site_path):
                continue
            scanned += 1
            try:
                site_rows = read_site_scores(str(site_path))
            except ValueError:
                skipped += 1
                print(f"[build_intron_scores_temporary] skip: {site_path}")
                continue
            intron_rows = build_intron_scores(
                site_score_rows=site_rows,
                intron_score_op=str(args.intron_score_op),
            )
            output_path = intron_score_dir / site_path.name
            if dry_run:
                written += 1
                print(
                    "[build_intron_scores_temporary] dry-run: "
                    f"{output_path} rows={len(intron_rows)}"
                )
                continue
            write_intron_scores(
                str(output_path),
                intron_rows,
                labels=labels,
            )
            written += 1
            print(
                "[build_intron_scores_temporary] wrote: "
                f"{output_path} rows={len(intron_rows)} labels={len(labels)}"
            )

    print(
        "[build_intron_scores_temporary] done "
        f"scanned={scanned} written={written} skipped={skipped} dry_run={dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
