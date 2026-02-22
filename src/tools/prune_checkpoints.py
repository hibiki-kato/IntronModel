"""CLI entrypoint for checkpoint pruning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from util.checkpoint_prune import prune_species_model_checkpoints


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune model checkpoints with signature-isolated top-k policy."
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--species", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--dry_run", type=int, choices=[0, 1], default=1)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    report = prune_species_model_checkpoints(
        data_root=args.data_root.resolve(),
        species=str(args.species),
        model_name=str(args.model),
        top_k=int(args.top_k),
        dry_run=bool(args.dry_run),
    )
    print(
        "prune_checkpoints: "
        f"species={args.species} model={args.model} "
        f"total={report.total_candidates} kept={report.kept_count} "
        f"deleted={report.deleted_count} dry_run={report.dry_run}"
    )
    for path in report.deleted_paths:
        print(f"delete_candidate: {path}")


if __name__ == "__main__":
    main()
