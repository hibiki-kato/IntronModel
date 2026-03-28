"""Temporarily reformat legacy site_score TSV files to a wide schema.

This script is intended for existing datasets that already contain legacy
``site_score/*.tsv`` outputs in long format:
``transcript_id, intron_index, site_type, score``.

It rewrites each file to:
``transcript_id, intron_index, donor_score, acceptor_score, label, _score_space``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from util.transcript_eval import (
    SCORE_OUTPUT_PRECISION,
    SCORE_SPACE_FIELD,
    SCORE_SPACE_LOG10,
    probability_to_log10_score,
)


def _format_score(value: float) -> str:
    """Format a probability score in log10 space for TSV output."""
    log_score = probability_to_log10_score(value)
    if log_score == float("-inf"):
        return "-inf"
    return f"{log_score:.{SCORE_OUTPUT_PRECISION}f}"


@dataclass(frozen=True)
class ParsedSiteScores:
    """Grouped donor/acceptor/pair scores per intron key."""

    scores: dict[tuple[str, int], dict[str, float]]
    converted_rows: int


def _parse_species_csv(raw_species: Optional[str]) -> list[str]:
    """Parse one optional comma-separated species list."""
    if raw_species is None:
        return []
    species = [token.strip() for token in raw_species.split(",")]
    return [token for token in species if token != ""]


def _set_csv_field_limit_max() -> None:
    """Set CSV field size limit to the largest supported value."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _load_labeled_introns(
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
        for line_no, raw in enumerate(reader, start=2):
            transcript_id = str(raw["transcript_id"]).strip()
            if transcript_id == "":
                raise ValueError(f"Empty transcript_id at {labeled_path}:{line_no}")
            intron_index = int(str(raw["intron_index"]))
            label = int(str(raw["label"]))
            if label not in {0, 1}:
                raise ValueError(
                    f"Label must be 0/1 at {labeled_path}:{line_no}, got {label}"
                )
            labels[(transcript_id, intron_index)] = label
    return labels


def _read_source_site_scores(path: Path) -> ParsedSiteScores | None:
    """Read one convertible-format site-score TSV.

    Returns ``None`` when the file is already in target format or an unknown
    schema. Supported input schemas:
    - legacy long: ``transcript_id, intron_index, site_type, score``
    - prior wide: ``Transcript number, donor score, acceptor score, label``
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return None
        fieldnames = set(reader.fieldnames)
        legacy_required = {"transcript_id", "intron_index", "site_type", "score"}
        wide_required = {
            "transcript_id",
            "intron_index",
            "donor_score",
            "acceptor_score",
        }
        prior_wide_required = {"Transcript number", "donor score", "acceptor score"}
        if wide_required.issubset(fieldnames):
            return None
        has_legacy = legacy_required.issubset(fieldnames)
        has_prior_wide = prior_wide_required.issubset(fieldnames)
        if not has_legacy and not has_prior_wide:
            return None

        grouped: dict[tuple[str, int], dict[str, float]] = {}
        converted_rows = 0
        if has_legacy:
            for line_no, raw in enumerate(reader, start=2):
                transcript_id = str(raw["transcript_id"]).strip()
                if transcript_id == "":
                    raise ValueError(f"Empty transcript_id at {path}:{line_no}")
                intron_index = int(str(raw["intron_index"]))
                site_type = str(raw["site_type"]).strip().lower()
                if site_type not in {"donor", "acceptor", "pair"}:
                    raise ValueError(
                        f"Unsupported site_type '{site_type}' at {path}:{line_no}"
                    )
                score = float(str(raw["score"]))
                grouped.setdefault((transcript_id, intron_index), {})[site_type] = score
                converted_rows += 1
        else:
            for line_no, raw in enumerate(reader, start=2):
                token = str(raw["Transcript number"]).strip()
                parts = token.rsplit(":", 2)
                if len(parts) != 3:
                    raise ValueError(
                        "Transcript number must be "
                        "'<transcript_id>:<intron_index>:<combined_score>' "
                        f"at {path}:{line_no}"
                    )
                transcript_id = parts[0].strip()
                intron_index = int(parts[1].strip())
                combined_text = parts[2].strip()
                combined_score = (
                    float(combined_text) if combined_text != "" else None
                )
                donor_raw = str(raw["donor score"]).strip()
                acceptor_raw = str(raw["acceptor score"]).strip()
                donor_score = float(donor_raw) if donor_raw != "" else None
                acceptor_score = float(acceptor_raw) if acceptor_raw != "" else None
                per_site = grouped.setdefault((transcript_id, intron_index), {})
                if donor_score is not None:
                    per_site["donor"] = donor_score
                if acceptor_score is not None:
                    per_site["acceptor"] = acceptor_score
                if (
                    donor_score is None
                    and acceptor_score is None
                    and combined_score is not None
                ):
                    per_site["pair"] = combined_score
                converted_rows += 1
    return ParsedSiteScores(scores=grouped, converted_rows=converted_rows)


def _write_wide_site_scores(
    *,
    output_path: Path,
    grouped_scores: dict[tuple[str, int], dict[str, float]],
    labels: dict[tuple[str, int], int],
) -> int:
    """Write new-format site_score TSV and return written intron count."""
    written_rows = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "transcript_id",
                "intron_index",
                "donor_score",
                "acceptor_score",
                "label",
                SCORE_SPACE_FIELD,
            ]
        )
        for key in sorted(grouped_scores.keys()):
            transcript_id, intron_index = key
            per_site = grouped_scores[key]
            donor_score = per_site.get("donor")
            acceptor_score = per_site.get("acceptor")
            pair_score = per_site.get("pair")
            if (
                donor_score is None
                and acceptor_score is None
                and pair_score is None
            ):
                continue
            donor_text = "" if donor_score is None else _format_score(
                float(donor_score)
            )
            acceptor_text = (
                ""
                if acceptor_score is None
                else _format_score(float(acceptor_score))
            )
            label = labels.get(key)
            label_text = "" if label is None else str(label)
            writer.writerow(
                [
                    transcript_id,
                    str(intron_index),
                    donor_text,
                    acceptor_text,
                    label_text,
                    SCORE_SPACE_LOG10,
                ]
            )
            written_rows += 1
    return written_rows


def _iter_species_dirs(data_root: Path, include_species: list[str]) -> list[Path]:
    """Return sorted species directories under ``data_root``."""
    if include_species:
        dirs = [data_root / species for species in include_species]
    else:
        dirs = [path for path in data_root.iterdir() if path.is_dir()]
    return sorted(path for path in dirs if path.is_dir())


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Temporarily reformat legacy data/<species>/site_score/*.tsv."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*.tsv")
    parser.add_argument(
        "--labeled-name",
        type=str,
        default="intron_eval_flank10.tsv",
        help="Processed label TSV filename under data/<species>/processed.",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".legacy",
        help="Suffix inserted before .tsv when writing backups.",
    )
    parser.add_argument("--dry-run", type=int, choices=[0, 1], default=0)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    _set_csv_field_limit_max()
    include_species = _parse_species_csv(args.species)
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    dry_run = bool(args.dry_run)
    scanned_files = 0
    converted_files = 0
    skipped_files = 0

    for species_dir in _iter_species_dirs(data_root, include_species):
        species = species_dir.name
        site_score_dir = species_dir / "site_score"
        if not site_score_dir.is_dir():
            continue
        labels = _load_labeled_introns(
            species_dir=species_dir,
            labeled_name=str(args.labeled_name),
        )
        for site_score_tsv in sorted(site_score_dir.glob(str(args.pattern))):
            if not site_score_tsv.is_file():
                continue
            scanned_files += 1
            parsed = _read_source_site_scores(site_score_tsv)
            if parsed is None:
                skipped_files += 1
                print(f"[reformat_site_scores_temporary] skip: {site_score_tsv}")
                continue

            backup_path = site_score_tsv.with_name(
                f"{site_score_tsv.stem}{args.backup_suffix}.tsv"
            )
            tmp_path = site_score_tsv.with_suffix(".tmp.tsv")
            if dry_run:
                print(
                    "[reformat_site_scores_temporary] dry-run: "
                    f"{site_score_tsv} rows={parsed.converted_rows}"
                )
                converted_files += 1
                continue

            _ = _write_wide_site_scores(
                output_path=tmp_path,
                grouped_scores=parsed.scores,
                labels=labels,
            )
            if backup_path.exists():
                backup_path.unlink()
            site_score_tsv.replace(backup_path)
            tmp_path.replace(site_score_tsv)
            converted_files += 1
            print(
                "[reformat_site_scores_temporary] converted: "
                f"{site_score_tsv} backup={backup_path} labels={len(labels)}"
            )
            print(f"[reformat_site_scores_temporary] species={species}")

    print(
        "[reformat_site_scores_temporary] done "
        f"scanned={scanned_files} converted={converted_files} skipped={skipped_files} "
        f"dry_run={dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
