"""CLI entrypoint to prune tuning checkpoints from missing-rank entries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from util.tuning_rank_prune import prune_missing_rank_tuning_checkpoints


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prune .pt checkpoints referenced by top_trials entries "
            "without valid rank."
        )
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--model_root", type=Path, default=Path("model"))
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dry_run", type=int, choices=[0, 1], default=1)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    report = prune_missing_rank_tuning_checkpoints(
        data_root=args.data_root,
        model_root=args.model_root,
        species=args.species,
        model_name=args.model,
        dry_run=bool(args.dry_run),
    )
    print(
        "prune_missing_rank_checkpoints: "
        f"scanned_best_configs={report.scanned_best_configs} "
        f"missing_rank_entries={report.missing_rank_entries} "
        f"candidate_paths={report.candidate_paths} "
        f"deleted={report.deleted_count} "
        f"dry_run={report.dry_run}"
    )
    for path in report.deleted_paths:
        print(f"delete_candidate: {path}")


if __name__ == "__main__":
    main()
