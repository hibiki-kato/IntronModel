"""Build random intron and transcript score TSV files per species.

This utility uses the canonical unique-intron assets under ``data/<species>``
to generate deterministic random intron scores and then aggregates them back
to transcript-level scores. The output is intended as a baseline comparison
against model-derived scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from util.transcript_eval import aggregate_pair_transcript_scores
from util.transcript_eval import write_intron_scores
from util.transcript_eval import write_transcript_scores
from util.unique_intron import UNIQUE_MAP_TSV_NAME
from util.unique_intron import UniqueMapMember
from util.unique_intron import set_csv_field_limit_max
from util.unique_intron import load_unique_map


@dataclass(frozen=True)
class UniqueIntronRecord:
    """One unique intron row used as the source for random scores."""

    transcript_id: str
    intron_index: int
    label: int


@dataclass(frozen=True)
class SpeciesOutputSummary:
    """Summary for one generated species output pair."""

    species: str
    intron_rows: int
    transcript_rows: int
    intron_score_tsv: Path
    trans_score_tsv: Path


def _parse_species_csv(raw_species: str | None) -> list[str]:
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
        candidates = [
            path
            for path in data_root.iterdir()
            if path.is_dir() and (path / "processed").is_dir()
        ]
    return sorted(path for path in candidates if path.is_dir())


def _derive_species_seed(base_seed: int, species: str) -> int:
    """Derive one deterministic per-species random seed."""
    digest = hashlib.sha256(f"{base_seed}:{species}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _load_unique_intron_records(species_dir: Path) -> list[UniqueIntronRecord]:
    """Load unique intron labels from ``intron_eval_flank10.unique.tsv``."""
    path = species_dir / "processed" / "intron_eval_flank10.unique.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Unique intron TSV not found: {path}")

    required_columns = {"transcript_id", "intron_index", "label"}
    rows: list[UniqueIntronRecord] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                "Unique intron TSV must include columns: "
                "transcript_id, intron_index, label"
            )
        for line_no, raw in enumerate(reader, start=2):
            transcript_id = str(raw["transcript_id"]).strip()
            if transcript_id == "":
                raise ValueError(f"Empty transcript_id at {path}:{line_no}")
            try:
                intron_index = int(str(raw["intron_index"]))
                label = int(str(raw["label"]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid intron_index/label at {path}:{line_no}"
                ) from exc
            if label not in {0, 1}:
                raise ValueError(
                    f"Invalid label at {path}:{line_no}: {label}"
                )
            rows.append(
                UniqueIntronRecord(
                    transcript_id=transcript_id,
                    intron_index=intron_index,
                    label=label,
                )
            )

    if not rows:
        raise ValueError(f"No valid rows in unique intron TSV: {path}")
    return rows


def _expand_unique_rows(
    *,
    unique_rows: list[dict[str, object]],
    unique_map: dict[tuple[str, int], list[UniqueMapMember]],
) -> list[dict[str, object]]:
    """Expand unique intron rows back to original transcript introns."""
    expanded_rows: list[dict[str, object]] = []
    missing_keys: list[tuple[str, int]] = []

    for row in unique_rows:
        unique_key = (str(row["transcript_id"]), int(row["intron_index"]))
        mapped_members = unique_map.get(unique_key)
        if mapped_members is None:
            missing_keys.append(unique_key)
            continue
        for member in mapped_members:
            copied = dict(row)
            copied["transcript_id"] = member.transcript_id
            copied["intron_index"] = member.intron_index
            expanded_rows.append(copied)

    if missing_keys:
        examples = ", ".join(
            f"{transcript_id}:{intron_index}"
            for transcript_id, intron_index in sorted(missing_keys)[:5]
        )
        raise ValueError(
            "Unique intron rows contain unmapped keys. "
            f"examples={examples}"
        )
    return expanded_rows


def _validate_output_paths(intron_score_tsv: Path, trans_score_tsv: Path) -> None:
    """Fail early if one of the target files already exists."""
    existing = [
        path for path in (intron_score_tsv, trans_score_tsv) if path.exists()
    ]
    if existing:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output file already exists: {existing_text}")


def _build_species_outputs(
    *,
    species_dir: Path,
    output_stem: str,
    base_seed: int,
) -> SpeciesOutputSummary:
    """Generate random intron and transcript score TSV files for one species."""
    species = species_dir.name
    unique_rows = _load_unique_intron_records(species_dir)
    unique_map_path = species_dir / "processed" / UNIQUE_MAP_TSV_NAME
    unique_map = load_unique_map(unique_map_path)

    rng = random.Random(_derive_species_seed(base_seed, species))
    labels: dict[tuple[str, int], int] = {}
    score_rows: list[dict[str, object]] = []
    for row in unique_rows:
        score = rng.random()
        score_rows.append(
            {
                "transcript_id": row.transcript_id,
                "intron_index": row.intron_index,
                "score": score,
            }
        )
        labels[(row.transcript_id, row.intron_index)] = row.label

    intron_score_tsv = species_dir / "intron_score" / f"{output_stem}.tsv"
    trans_score_tsv = species_dir / "trans_score" / f"{output_stem}.tsv"
    _validate_output_paths(intron_score_tsv, trans_score_tsv)

    write_intron_scores(str(intron_score_tsv), score_rows, labels=labels)

    expanded_rows = _expand_unique_rows(
        unique_rows=score_rows,
        unique_map=unique_map,
    )
    transcript_rows = aggregate_pair_transcript_scores(
        site_score_rows=expanded_rows,
        transcript_score_agg="min",
    )
    write_transcript_scores(str(trans_score_tsv), transcript_rows)

    return SpeciesOutputSummary(
        species=species,
        intron_rows=len(score_rows),
        transcript_rows=len(transcript_rows),
        intron_score_tsv=intron_score_tsv,
        trans_score_tsv=trans_score_tsv,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic random intron scores and transcript scores "
            "for one or more species."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--output-stem", type=str, default="random")
    parser.add_argument("--seed", type=int, default=20260327)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    set_csv_field_limit_max()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    include_species = _parse_species_csv(args.species)
    species_dirs = _iter_species_dirs(data_root, include_species)
    if not species_dirs:
        raise ValueError(f"No species directories found under: {data_root}")

    summaries: list[SpeciesOutputSummary] = []
    for species_dir in species_dirs:
        summary = _build_species_outputs(
            species_dir=species_dir,
            output_stem=str(args.output_stem),
            base_seed=int(args.seed),
        )
        summaries.append(summary)
        print(
            "[build_random_intron_and_trans_scores] "
            f"species={summary.species} intron_rows={summary.intron_rows} "
            f"transcript_rows={summary.transcript_rows}"
        )
        print(
            "[build_random_intron_and_trans_scores] "
            f"intron_score={summary.intron_score_tsv}"
        )
        print(
            "[build_random_intron_and_trans_scores] "
            f"trans_score={summary.trans_score_tsv}"
        )

    total_species = len(summaries)
    total_intron_rows = sum(summary.intron_rows for summary in summaries)
    total_transcript_rows = sum(summary.transcript_rows for summary in summaries)
    print(
        "[build_random_intron_and_trans_scores] done "
        f"species={total_species} intron_rows={total_intron_rows} "
        f"transcript_rows={total_transcript_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
