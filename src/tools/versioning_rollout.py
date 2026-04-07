"""Audit and seed versioned publication coverage across species and models.

This CLI helps evaluate whether each ``data/<species>/tuning/<model>`` directory
has the minimum best-config layout required for versioned publication and can
optionally seed missing ``.01`` publication history entries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
import sys

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from util.model_task_paths import checkpoint_tasks_for_model  # noqa: E402
from util.path_format import relativize_path_fields  # noqa: E402
from util.versioned_artifacts import ensure_publication_seed  # noqa: E402
from util.versioned_artifacts import read_version_history  # noqa: E402
from util.versioned_artifacts import resolve_version_history_path  # noqa: E402
from util.versioned_artifacts import write_version_history  # noqa: E402


@dataclass(frozen=True)
class RolloutRow:
    """One rollout audit row for one species/model pair."""

    species: str
    model_name: str
    tasks: tuple[str, ...]
    ready: bool
    history_exists: bool
    missing_paths: tuple[Path, ...]
    seed_result: str
    repair_result: str


@dataclass(frozen=True)
class RolloutReport:
    """Aggregate rollout report for all evaluated rows."""

    rows: tuple[RolloutRow, ...]

    @property
    def total(self) -> int:
        """Return the number of scanned rows."""
        return len(self.rows)

    @property
    def ready_count(self) -> int:
        """Return the number of rows with complete best-config layout."""
        return sum(1 for row in self.rows if row.ready)

    @property
    def missing_count(self) -> int:
        """Return the number of rows missing required best-config files."""
        return sum(1 for row in self.rows if not row.ready)

    @property
    def history_count(self) -> int:
        """Return the number of rows with existing version history."""
        return sum(1 for row in self.rows if row.history_exists)

    @property
    def seeded_count(self) -> int:
        """Return the number of rows seeded in this run."""
        return sum(1 for row in self.rows if row.seed_result == "seeded")

    @property
    def repaired_count(self) -> int:
        """Return the number of rows whose metadata was rewritten."""
        return sum(1 for row in self.rows if row.repair_result == "repaired")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit versioning readiness across data/<species>/tuning/<model> and "
            "optionally seed missing publication history entries."
        )
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path("."),
        help="Project root used by versioned publication utilities.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="Data root containing species/tuning directories.",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="",
        help="Optional comma-separated species whitelist.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional comma-separated model-name whitelist.",
    )
    parser.add_argument(
        "--apply_seed",
        type=int,
        choices=[0, 1],
        default=0,
        help="When 1, seed version history for ready rows without history.",
    )
    parser.add_argument(
        "--allow_seed_with_history",
        type=int,
        choices=[0, 1],
        default=0,
        help="When 1, also call seed on rows that already have history.",
    )
    parser.add_argument(
        "--repair_paths",
        type=int,
        choices=[0, 1],
        default=0,
        help="When 1, rewrite versioned metadata path fields repo-relatively.",
    )
    return parser


def _parse_filter(raw_value: str) -> set[str]:
    """Parse one comma-separated filter string into a normalized set."""
    values: set[str] = set()
    for token in raw_value.split(","):
        normalized = token.strip()
        if normalized != "":
            values.add(normalized)
    return values


def _required_best_configs(model_dir: Path, tasks: tuple[str, ...]) -> tuple[Path, ...]:
    """Return required best-config paths for one task signature."""
    if tasks == ("pair",):
        return (model_dir / "pair" / "best_config.json",)
    if tasks == ("donor", "acceptor"):
        return (
            model_dir / "donor" / "best_config.json",
            model_dir / "acceptor" / "best_config.json",
        )
    return ()


def _scan_rollout_rows(
    *,
    data_root: Path,
    species_filter: set[str],
    model_filter: set[str],
) -> tuple[RolloutRow, ...]:
    """Scan versioning readiness rows from one data root."""
    rows: list[RolloutRow] = []
    for species_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        species = species_dir.name
        if species_filter and species not in species_filter:
            continue
        tuning_root = species_dir / "tuning"
        if not tuning_root.is_dir():
            continue
        for model_dir in sorted(
            path for path in tuning_root.iterdir() if path.is_dir()
        ):
            model_name = model_dir.name
            if model_filter and model_name not in model_filter:
                continue
            tasks = checkpoint_tasks_for_model(model_name)
            required_paths = _required_best_configs(model_dir, tasks)
            missing_paths = tuple(path for path in required_paths if not path.is_file())
            ready = len(required_paths) > 0 and len(missing_paths) == 0
            history_exists = resolve_version_history_path(
                data_root=data_root,
                species=species,
                public_model_name=model_name,
            ).is_file()
            rows.append(
                RolloutRow(
                    species=species,
                    model_name=model_name,
                    tasks=tasks,
                    ready=ready,
                    history_exists=history_exists,
                    missing_paths=missing_paths,
                    seed_result="skipped",
                    repair_result="skipped",
                )
            )
    return tuple(rows)


def _seed_row_if_requested(
    *,
    row: RolloutRow,
    apply_seed: bool,
    allow_seed_with_history: bool,
    project_root: Path,
) -> RolloutRow:
    """Seed one row when requested and eligible."""
    if not apply_seed:
        return row
    if not row.ready:
        return replace(row, seed_result="missing-best-config")
    if row.history_exists and not allow_seed_with_history:
        return replace(row, seed_result="history-exists")

    published_name = ensure_publication_seed(
        project_root=project_root,
        species=row.species,
        model_name=row.model_name,
    )
    if published_name is None:
        return replace(row, seed_result="seed-failed")
    if row.history_exists:
        return replace(row, seed_result="history-kept")
    return replace(row, seed_result="seeded")


def _required_metadata_paths(
    model_dir: Path, tasks: tuple[str, ...]
) -> tuple[Path, ...]:
    """Return publication metadata files that should be path-normalized."""
    paths: list[Path] = []
    for path in _required_best_configs(model_dir, tasks):
        if path.is_file():
            paths.append(path)
    versions_dir = model_dir / "versions"
    if versions_dir.is_dir():
        for version_path in sorted(versions_dir.glob("*.json")):
            if version_path.is_file():
                paths.append(version_path)
    return tuple(paths)


def _normalize_json_file(path: Path) -> bool:
    """Rewrite one JSON file with repository-relative path fields."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = relativize_path_fields(payload, project_root=PROJECT_ROOT)
    if normalized == payload:
        return False
    path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    return True


def _repair_row_paths(
    *,
    row: RolloutRow,
    data_root: Path,
) -> RolloutRow:
    """Repair versioned metadata paths for one ready rollout row."""
    if not row.ready:
        return replace(row, repair_result="missing-best-config")

    model_dir = data_root / row.species / "tuning" / row.model_name
    changed = False
    history_path = resolve_version_history_path(
        data_root=data_root,
        species=row.species,
        public_model_name=row.model_name,
    )
    if history_path.is_file():
        history = read_version_history(data_root, row.species, row.model_name)
        write_version_history(data_root, row.species, row.model_name, history)
        changed = True

    for metadata_path in _required_metadata_paths(model_dir, row.tasks):
        changed = _normalize_json_file(metadata_path) or changed

    return replace(row, repair_result="repaired" if changed else "unchanged")


def run_rollout(
    *,
    project_root: Path,
    data_root: Path,
    species_filter: set[str],
    model_filter: set[str],
    apply_seed: bool,
    allow_seed_with_history: bool,
    repair_paths: bool,
) -> RolloutReport:
    """Run rollout audit and optional seed pass."""
    raw_rows = _scan_rollout_rows(
        data_root=data_root,
        species_filter=species_filter,
        model_filter=model_filter,
    )
    rows: list[RolloutRow] = []
    for row in raw_rows:
        rows.append(
            _seed_row_if_requested(
                row=row,
                apply_seed=apply_seed,
                allow_seed_with_history=allow_seed_with_history,
                project_root=project_root,
            )
        )
    repaired_rows: list[RolloutRow] = []
    for row in rows:
        if repair_paths:
            repaired_rows.append(_repair_row_paths(row=row, data_root=data_root))
        else:
            repaired_rows.append(row)
    return RolloutReport(rows=tuple(repaired_rows))


def _format_tasks(tasks: tuple[str, ...]) -> str:
    """Format one task signature for report output."""
    if len(tasks) == 0:
        return "none"
    return ",".join(tasks)


def _format_missing_paths(missing_paths: tuple[Path, ...]) -> str:
    """Format missing-path list for report output."""
    if len(missing_paths) == 0:
        return ""
    return ";".join(str(path) for path in missing_paths)


def main() -> None:
    """Run CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    species_filter = _parse_filter(args.species)
    model_filter = _parse_filter(args.model)

    report = run_rollout(
        project_root=project_root,
        data_root=data_root,
        species_filter=species_filter,
        model_filter=model_filter,
        apply_seed=bool(args.apply_seed),
        allow_seed_with_history=bool(args.allow_seed_with_history),
        repair_paths=bool(args.repair_paths),
    )

    print(
        "versioning_rollout_summary: "
        f"total={report.total} "
        f"ready={report.ready_count} "
        f"missing={report.missing_count} "
        f"history={report.history_count} "
        f"seeded={report.seeded_count} "
        f"repaired={report.repaired_count} "
        f"apply_seed={int(bool(args.apply_seed))}"
    )
    print(
        "species\tmodel\ttasks\tready\thistory_exists\tseed_result\trepair_result\tmissing_best_configs"
    )
    for row in report.rows:
        print(
            f"{row.species}\t"
            f"{row.model_name}\t"
            f"{_format_tasks(row.tasks)}\t"
            f"{int(row.ready)}\t"
            f"{int(row.history_exists)}\t"
            f"{row.seed_result}\t"
            f"{row.repair_result}\t"
            f"{_format_missing_paths(row.missing_paths)}"
        )


if __name__ == "__main__":
    main()
