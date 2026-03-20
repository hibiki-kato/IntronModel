"""Split ``cnn_v2`` tuning bests into non-pair and pair model namespaces.

This utility creates/updates:

- ``data/<species>/tuning/cnn_v2/both/best_config.json`` for non-pair runs
- ``data/<species>/tuning/cnn_v2_pair/pair/best_config.json`` for pair runs

Selection policy:

- Non-pair donor/acceptor bests are selected independently from legacy sources.
- Pair best is selected from existing pair-capable sources.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Mapping, Sequence


NON_PAIR_SOURCES: tuple[str, ...] = ("cnn_v2", "cnn", "cnn_mask")
PAIR_SOURCES: tuple[str, ...] = ("cnn_v2_pair", "cnn_v2", "cnn_pair", "cnn_pair_mask")
NON_PAIR_TASKS: tuple[str, ...] = ("donor", "acceptor")

_INDEPENDENT_TASK_KEYS: frozenset[str] = frozenset(
    {
        "conv_channels",
        "kernel_sizes",
    }
)
_INDEPENDENT_GLOBAL_KEYS: frozenset[str] = frozenset(
    {
        "batch_size",
        "lr",
        "loss",
        "max_pool_size",
        "conv_stride",
        "head_type",
        "dropout",
        "fc_hidden",
        "weight_decay",
        "eta_min_ratio",
        "val_frac",
        "grad_clip",
        "pos_weight_cap",
        "focal_gamma",
        "focal_alpha_pos",
        "f1_lambda",
        "asym_gamma_pos",
        "asym_gamma_neg",
        "asym_alpha_pos",
    }
)


@dataclass(frozen=True)
class BestCandidate:
    """One loaded best_config candidate."""

    source_model: str
    task: str
    path: Path
    objective_score: float
    payload: dict[str, object]
    sampled_params: dict[str, object]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Split cnn_v2 best artifacts into cnn_v2 (non-pair) and "
            "cnn_v2_pair (pair)."
        )
    )
    parser.add_argument(
        "--species",
        default="Athal,Dmel,Hsap,Mmus",
        help="Comma-separated species list.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing files.",
    )
    return parser.parse_args(argv)


def _parse_csv(raw_value: str) -> list[str]:
    """Parse one comma-separated list with stable deduplication."""
    values: list[str] = []
    seen: set[str] = set()
    for token in raw_value.split(","):
        value = token.strip()
        if value == "" or value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("Expected at least one species value.")
    return values


def _read_json_object(path: Path) -> dict[str, object]:
    """Read one JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_finite_float(value: object) -> float | None:
    """Convert one scalar-like value to finite float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            result = float(stripped)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def _load_candidate(
    *,
    data_root: Path,
    species: str,
    source_model: str,
    task: str,
) -> BestCandidate | None:
    """Load one candidate best_config payload if valid."""
    best_path = data_root / species / "tuning" / source_model / task / "best_config.json"
    if not best_path.is_file():
        return None
    payload = _read_json_object(best_path)
    status = str(payload.get("status", "")).strip().lower()
    if status != "ok":
        return None
    sampled_params_raw = payload.get("sampled_params")
    if not isinstance(sampled_params_raw, dict):
        return None
    objective_score = _as_finite_float(payload.get("objective_score"))
    if objective_score is None:
        objective_score = float("-inf")
    return BestCandidate(
        source_model=source_model,
        task=task,
        path=best_path,
        objective_score=objective_score,
        payload=payload,
        sampled_params=dict(sampled_params_raw),
    )


def _pick_best_candidate(
    *,
    data_root: Path,
    species: str,
    task: str,
    source_models: Sequence[str],
) -> BestCandidate | None:
    """Pick one highest-scoring candidate from source models."""
    best: BestCandidate | None = None
    for source_model in source_models:
        candidate = _load_candidate(
            data_root=data_root,
            species=species,
            source_model=source_model,
            task=task,
        )
        if candidate is None:
            continue
        if best is None or candidate.objective_score > best.objective_score:
            best = candidate
    return best


def _format_cli_value(value: object) -> str:
    """Convert one value to stable CLI-like string form."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ",".join(_format_cli_value(item) for item in value)
    return str(value)


def _build_non_pair_sampled_params(
    donor_params: Mapping[str, object],
    acceptor_params: Mapping[str, object],
) -> dict[str, object]:
    """Build ``cnn_v2`` independent-mode sampled params for ``train_target=both``."""
    sampled: dict[str, object] = {
        "input_mode": "onehot",
        "pair_mode": "independent",
        "train_target": "both",
        "sequence_transform": "none",
    }
    donor_len = donor_params.get("donor_len")
    acceptor_len = donor_params.get("acceptor_len")
    if donor_len is None:
        donor_len = acceptor_params.get("donor_len")
    if acceptor_len is None:
        acceptor_len = acceptor_params.get("acceptor_len")
    if donor_len is not None:
        sampled["donor_len"] = donor_len
    if acceptor_len is not None:
        sampled["acceptor_len"] = acceptor_len

    donor_global: dict[str, object] = {}
    acceptor_global: dict[str, object] = {}
    for key, value in donor_params.items():
        if key in {"donor_len", "acceptor_len"} or value is None:
            continue
        if key in _INDEPENDENT_TASK_KEYS:
            sampled[f"donor_{key}"] = _format_cli_value(value)
        elif key in _INDEPENDENT_GLOBAL_KEYS:
            donor_global[key] = value
    for key, value in acceptor_params.items():
        if key in {"donor_len", "acceptor_len"} or value is None:
            continue
        if key in _INDEPENDENT_TASK_KEYS:
            sampled[f"acceptor_{key}"] = _format_cli_value(value)
        elif key in _INDEPENDENT_GLOBAL_KEYS:
            acceptor_global[key] = value
    for key in sorted(_INDEPENDENT_GLOBAL_KEYS):
        if key in donor_global:
            sampled[key] = donor_global[key]
        elif key in acceptor_global:
            sampled[key] = acceptor_global[key]
    return sampled


def _write_json(path: Path, payload: Mapping[str, object], *, dry_run: bool) -> None:
    """Write one JSON object to disk."""
    if dry_run:
        print(f"[dry-run] write: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    """Copy one file with parent creation."""
    if dry_run:
        print(f"[dry-run] copy: {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _split_one_species(
    *,
    data_root: Path,
    species: str,
    dry_run: bool,
) -> dict[str, object]:
    """Split best artifacts for one species."""
    donor_best = _pick_best_candidate(
        data_root=data_root,
        species=species,
        task="donor",
        source_models=NON_PAIR_SOURCES,
    )
    acceptor_best = _pick_best_candidate(
        data_root=data_root,
        species=species,
        task="acceptor",
        source_models=NON_PAIR_SOURCES,
    )
    pair_best = _pick_best_candidate(
        data_root=data_root,
        species=species,
        task="pair",
        source_models=PAIR_SOURCES,
    )

    if donor_best is None or acceptor_best is None:
        raise FileNotFoundError(
            "Missing donor/acceptor best candidates. "
            f"species={species} donor={donor_best is not None} "
            f"acceptor={acceptor_best is not None}"
        )
    if pair_best is None:
        raise FileNotFoundError(f"Missing pair best candidate. species={species}")

    non_pair_sampled = _build_non_pair_sampled_params(
        donor_params=donor_best.sampled_params,
        acceptor_params=acceptor_best.sampled_params,
    )
    non_pair_objective = (
        donor_best.objective_score + acceptor_best.objective_score
    ) / 2.0
    non_pair_payload: dict[str, object] = {
        "status": "ok",
        "objective_metric": "mean_pr_auc",
        "objective_score": non_pair_objective,
        "sampled_params": non_pair_sampled,
        "source_donor_model": donor_best.source_model,
        "source_donor_best_config": str(donor_best.path),
        "source_acceptor_model": acceptor_best.source_model,
        "source_acceptor_best_config": str(acceptor_best.path),
        "generated_by": "split_cnn_v2_best.py",
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    species_tuning_root = data_root / species / "tuning"
    non_pair_both_path = species_tuning_root / "cnn_v2" / "both" / "best_config.json"
    non_pair_shared_path = species_tuning_root / "cnn_v2" / "best_config.json"
    pair_path = species_tuning_root / "cnn_v2_pair" / "pair" / "best_config.json"
    pair_shared_path = species_tuning_root / "cnn_v2_pair" / "best_config.json"

    pair_payload = dict(pair_best.payload)
    pair_payload["source_model"] = pair_best.source_model
    pair_payload["source_best_config"] = str(pair_best.path)
    pair_payload["generated_by"] = "split_cnn_v2_best.py"
    pair_payload["generated_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    _write_json(non_pair_both_path, non_pair_payload, dry_run=dry_run)
    _write_json(non_pair_shared_path, non_pair_payload, dry_run=dry_run)
    _write_json(pair_path, pair_payload, dry_run=dry_run)
    _write_json(pair_shared_path, pair_payload, dry_run=dry_run)

    old_pair_search_space = (
        species_tuning_root / "cnn_v2" / "pair" / "search_space.json"
    )
    new_pair_search_space = (
        species_tuning_root / "cnn_v2_pair" / "pair" / "search_space.json"
    )
    if old_pair_search_space.is_file() and not new_pair_search_space.exists():
        _copy_file(old_pair_search_space, new_pair_search_space, dry_run=dry_run)

    return {
        "species": species,
        "non_pair": {
            "donor_source_model": donor_best.source_model,
            "donor_source_best_config": str(donor_best.path),
            "acceptor_source_model": acceptor_best.source_model,
            "acceptor_source_best_config": str(acceptor_best.path),
            "objective_score": non_pair_objective,
            "output_best_config": str(non_pair_both_path),
        },
        "pair": {
            "source_model": pair_best.source_model,
            "source_best_config": str(pair_best.path),
            "objective_score": pair_best.objective_score,
            "output_best_config": str(pair_path),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    project_root = Path(args.project_root).resolve()
    data_root = project_root / "data"
    species_list = _parse_csv(str(args.species))
    dry_run = bool(args.dry_run)

    summary_rows: list[dict[str, object]] = []
    for species in species_list:
        row = _split_one_species(data_root=data_root, species=species, dry_run=dry_run)
        summary_rows.append(row)
        print(
            "[split] "
            f"species={species} non_pair={row['non_pair']} pair={row['pair']}",
            flush=True,
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_path = (
        project_root
        / "data"
        / "migration_runs"
        / f"cnn_v2_split_{timestamp}"
        / "summary.json"
    )
    summary_payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_root": str(project_root),
        "dry_run": dry_run,
        "results": summary_rows,
    }
    _write_json(summary_path, summary_payload, dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] summary={summary_path}")
    else:
        print(f"[done] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
