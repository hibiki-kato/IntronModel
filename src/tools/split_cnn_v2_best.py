"""Split ``cnn_v2`` tuning bests into task-local and public pair namespaces.

This utility creates/updates:

- ``data/<species>/tuning/cnn_v2/donor/best_config.json`` for donor runs
- ``data/<species>/tuning/cnn_v2/acceptor/best_config.json`` for acceptor runs
- ``data/<species>/tuning/cnn_pair_v2/pair/best_config.json`` for pair runs

Selection policy:

- Donor and acceptor bests are selected independently from legacy sources.
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

from util.checkpoint_io import (
    extract_checkpoint_paths,
    resolve_existing_checkpoint_path,
)

NON_PAIR_SOURCES: tuple[str, ...] = ("cnn_v2", "cnn", "cnn_mask")
PAIR_SOURCES: tuple[str, ...] = (
    "cnn_v2_pair",
    "cnn_v2",
    "cnn_pair",
    "cnn_pair_mask",
)
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


def _mask_value_from_sequence_transform(value: object) -> str | None:
    """Convert a legacy sequence-transform value into ``mask``."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized == "none":
        return "off"
    if normalized in {"mask_outside_intron_n", "truncate_outside_intron"}:
        return "on"
    return None


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
    model_root_dir: Path,
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
    checkpoint_paths = extract_checkpoint_paths(
        payload,
        base_dir=best_path.parent,
        existing_only=False,
        tasks=(task,),
    )
    target_checkpoint = checkpoint_paths.get(task)
    if target_checkpoint is None:
        return None
    try:
        _ = resolve_existing_checkpoint_path(
            target_checkpoint,
            model_root_dir=model_root_dir,
        )
    except FileNotFoundError:
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
    model_root_dir: Path,
    species: str,
    task: str,
    source_models: Sequence[str],
) -> BestCandidate | None:
    """Pick one highest-scoring candidate from source models."""
    best: BestCandidate | None = None
    for source_model in source_models:
        candidate = _load_candidate(
            data_root=data_root,
            model_root_dir=model_root_dir,
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


def _build_target_payload(
    *,
    source_payload: Mapping[str, object],
    source_path: Path,
    target: str,
    model_root_dir: Path,
) -> dict[str, object]:
    """Build one donor or acceptor best payload from one source payload."""
    objective_metric = f"{target}_pr_auc"
    objective_value = _as_finite_float(source_payload.get(objective_metric))
    if objective_value is None:
        raise ValueError(
            f"Missing finite {objective_metric} in source payload: {source_path}"
        )

    sampled_params_raw = source_payload.get("sampled_params")
    if not isinstance(sampled_params_raw, Mapping):
        raise ValueError(f"Missing sampled_params object: {source_path}")

    payload: dict[str, object] = dict(source_payload)
    payload["objective_metric"] = objective_metric
    payload["objective_score"] = objective_value
    payload["selection_score"] = objective_value
    if target == "donor":
        payload["donor_pr_auc"] = objective_value
        payload["acceptor_pr_auc"] = None
    else:
        payload["donor_pr_auc"] = None
        payload["acceptor_pr_auc"] = objective_value
    payload["mean_pr_auc"] = None
    payload["sampled_params"] = {
        key: value
        for key, value in sampled_params_raw.items()
        if key not in {"train_target", "sequence_transform"}
        and not key.startswith("donor_")
        and not key.startswith("acceptor_")
    }
    payload["sampled_params"]["train_target"] = target
    payload["sampled_params"]["pair_mode"] = "independent"

    conv_key = f"{target}_conv_channels"
    kernel_key = f"{target}_kernel_sizes"
    if conv_key in sampled_params_raw and sampled_params_raw[conv_key] is not None:
        payload["sampled_params"]["conv_channels"] = sampled_params_raw[conv_key]
    if kernel_key in sampled_params_raw and sampled_params_raw[kernel_key] is not None:
        payload["sampled_params"]["kernel_sizes"] = sampled_params_raw[kernel_key]

    hparam_context_raw = source_payload.get("hparam_context")
    if isinstance(hparam_context_raw, Mapping):
        hparam_context: dict[str, object] = dict(hparam_context_raw)
        hparam_context["objective_metric"] = objective_metric
        fixed_run_args_raw = hparam_context.get("fixed_run_args")
        if isinstance(fixed_run_args_raw, Mapping):
            fixed_run_args: dict[str, object] = dict(fixed_run_args_raw)
            fixed_run_args["train_target"] = target
            fixed_run_args.pop("sequence_transform", None)
            hparam_context["fixed_run_args"] = fixed_run_args
        payload["hparam_context"] = hparam_context

    payload["source_best_config"] = str(source_path)
    payload["generated_by"] = "split_cnn_v2_best.py"
    payload["generated_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _canonicalize_checkpoint_fields(
        payload,
        base_dir=source_path.parent,
        model_root_dir=model_root_dir,
    )


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


def _canonicalize_checkpoint_fields(
    payload: dict[str, object],
    *,
    base_dir: Path,
    model_root_dir: Path,
) -> dict[str, object]:
    """Resolve checkpoint fields in one payload to local existing files."""
    checkpoint_paths = extract_checkpoint_paths(
        payload,
        base_dir=base_dir,
        existing_only=False,
    )
    for task, checkpoint_path in checkpoint_paths.items():
        try:
            resolved_path = resolve_existing_checkpoint_path(
                checkpoint_path,
                model_root_dir=model_root_dir,
            )
        except FileNotFoundError:
            continue
        payload[f"{task}_checkpoint_path"] = str(resolved_path)
    return payload


def _split_one_species(
    *,
    data_root: Path,
    model_root_dir: Path,
    species: str,
    dry_run: bool,
) -> dict[str, object]:
    """Split best artifacts for one species."""
    donor_best = _pick_best_candidate(
        data_root=data_root,
        model_root_dir=model_root_dir,
        species=species,
        task="donor",
        source_models=NON_PAIR_SOURCES,
    )
    acceptor_best = _pick_best_candidate(
        data_root=data_root,
        model_root_dir=model_root_dir,
        species=species,
        task="acceptor",
        source_models=NON_PAIR_SOURCES,
    )
    pair_best = _pick_best_candidate(
        data_root=data_root,
        model_root_dir=model_root_dir,
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

    species_tuning_root = data_root / species / "tuning"
    donor_path = species_tuning_root / "cnn_v2" / "donor" / "best_config.json"
    acceptor_path = species_tuning_root / "cnn_v2" / "acceptor" / "best_config.json"
    pair_path = species_tuning_root / "cnn_pair_v2" / "pair" / "best_config.json"

    donor_payload = _build_target_payload(
        source_payload=donor_best.payload,
        source_path=donor_best.path,
        target="donor",
        model_root_dir=model_root_dir,
    )
    acceptor_payload = _build_target_payload(
        source_payload=acceptor_best.payload,
        source_path=acceptor_best.path,
        target="acceptor",
        model_root_dir=model_root_dir,
    )

    pair_payload = _canonicalize_checkpoint_fields(
        dict(pair_best.payload),
        base_dir=pair_best.path.parent,
        model_root_dir=model_root_dir,
    )
    pair_sampled = pair_payload.get("sampled_params")
    if isinstance(pair_sampled, dict):
        normalized_pair_sampled = dict(pair_sampled)
        if "mask" not in normalized_pair_sampled:
            legacy_mask = _mask_value_from_sequence_transform(
                normalized_pair_sampled.get("sequence_transform")
            )
            if legacy_mask is not None:
                normalized_pair_sampled["mask"] = legacy_mask
                normalized_pair_sampled.pop("sequence_transform", None)
        pair_payload["sampled_params"] = normalized_pair_sampled
    pair_payload["source_model"] = pair_best.source_model
    pair_payload["source_best_config"] = str(pair_best.path)
    pair_payload["generated_by"] = "split_cnn_v2_best.py"
    pair_payload["generated_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    _write_json(donor_path, donor_payload, dry_run=dry_run)
    _write_json(acceptor_path, acceptor_payload, dry_run=dry_run)
    _write_json(pair_path, pair_payload, dry_run=dry_run)

    new_pair_search_space = (
        species_tuning_root / "cnn_pair_v2" / "pair" / "search_space.json"
    )

    return {
        "species": species,
        "site": {
            "donor_source_model": donor_best.source_model,
            "donor_source_best_config": str(donor_best.path),
            "acceptor_source_model": acceptor_best.source_model,
            "acceptor_source_best_config": str(acceptor_best.path),
            "output_donor_best_config": str(donor_path),
            "output_acceptor_best_config": str(acceptor_path),
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
    model_root_dir = (project_root / "model").resolve()
    species_list = _parse_csv(str(args.species))
    dry_run = bool(args.dry_run)

    summary_rows: list[dict[str, object]] = []
    for species in species_list:
        row = _split_one_species(
            data_root=data_root,
            model_root_dir=model_root_dir,
            species=species,
            dry_run=dry_run,
        )
        summary_rows.append(row)
        print(
            "[split] "
            f"species={species} site={row['site']} pair={row['pair']}",
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
