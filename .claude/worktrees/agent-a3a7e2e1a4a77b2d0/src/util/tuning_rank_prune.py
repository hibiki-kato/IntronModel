"""Prune tuning-generated checkpoints from missing-rank top-trial entries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from util.checkpoint_io import (
    extract_checkpoint_paths,
    normalize_checkpoint_path,
    read_json_object,
)


@dataclass(frozen=True)
class MissingRankPruneReport:
    """Result report for missing-rank checkpoint pruning."""

    scanned_best_configs: int
    missing_rank_entries: int
    candidate_paths: int
    deleted_count: int
    dry_run: bool
    deleted_paths: tuple[Path, ...]


def _is_path_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is located under ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_metrics_path(metrics_json: str, *, base_dir: Path) -> Path:
    """Resolve one metrics JSON path from a raw string."""
    return normalize_checkpoint_path(metrics_json, base_dir=base_dir)


def _collect_checkpoint_paths_from_entry(
    entry: dict[str, object],
    *,
    base_dir: Path,
) -> set[Path]:
    """Collect checkpoint paths referenced by one top-trial entry."""
    collected: set[Path] = set(
        extract_checkpoint_paths(
            entry,
            base_dir=base_dir,
            existing_only=False,
        ).values()
    )

    metrics_json_raw = entry.get("metrics_json")
    if isinstance(metrics_json_raw, str) and metrics_json_raw.strip():
        metrics_path = _resolve_metrics_path(metrics_json_raw, base_dir=base_dir)
        metrics_payload = read_json_object(metrics_path)
        if metrics_payload is not None:
            collected.update(
                extract_checkpoint_paths(
                    metrics_payload,
                    base_dir=metrics_path.parent,
                    existing_only=False,
                ).values()
            )

    phase = entry.get("phase")
    trial_id = entry.get("trial_id")
    if isinstance(phase, str) and isinstance(trial_id, int):
        fallback_metrics = base_dir / f"{phase}_trial_{trial_id:04d}.metrics.json"
        metrics_payload = read_json_object(fallback_metrics)
        if metrics_payload is not None:
            collected.update(
                extract_checkpoint_paths(
                    metrics_payload,
                    base_dir=fallback_metrics.parent,
                    existing_only=False,
                ).values()
            )

    return {path.resolve() for path in collected}


def _iter_filtered_best_configs(
    *,
    data_root: Path,
    species: Optional[str],
    model_name: Optional[str],
) -> list[Path]:
    """Collect best-config paths filtered by optional species/model."""
    candidates: list[Path] = []
    for best_config_path in sorted(data_root.rglob("best_config.json")):
        relative_parts = best_config_path.resolve().relative_to(data_root.resolve()).parts
        if "tuning" not in relative_parts:
            continue
        tuning_index = relative_parts.index("tuning")
        if tuning_index < 1 or tuning_index + 1 >= len(relative_parts):
            continue
        found_species = relative_parts[tuning_index - 1]
        found_model = relative_parts[tuning_index + 1]
        if species is not None and found_species != species:
            continue
        if model_name is not None and found_model != model_name:
            continue
        candidates.append(best_config_path.resolve())
    return candidates


def prune_missing_rank_tuning_checkpoints(
    *,
    data_root: Path,
    model_root: Path,
    species: Optional[str] = None,
    model_name: Optional[str] = None,
    dry_run: bool = True,
) -> MissingRankPruneReport:
    """Delete checkpoint files referenced only by missing-rank top-trial entries.

    Parameters
    ----------
    data_root : Path
        Root directory of data artifacts.
    model_root : Path
        Root directory where model checkpoints are stored.
    species : str | None, default=None
        Optional species filter.
    model_name : str | None, default=None
        Optional model-name filter.
    dry_run : bool, default=True
        If ``True``, report candidates without deleting files.

    Returns
    -------
    MissingRankPruneReport
        Structured pruning report.

    Raises
    ------
    ValueError
        If ``species`` or ``model_name`` is an empty string.
    """
    if species is not None and species.strip() == "":
        raise ValueError("species must be a non-empty string when provided.")
    if model_name is not None and model_name.strip() == "":
        raise ValueError("model_name must be a non-empty string when provided.")

    resolved_data_root = data_root.resolve()
    resolved_model_root = model_root.resolve()
    best_configs = _iter_filtered_best_configs(
        data_root=resolved_data_root,
        species=species,
        model_name=model_name,
    )

    protected_paths: set[Path] = set()
    candidate_paths: set[Path] = set()
    missing_rank_entries = 0
    scanned_best_configs = 0

    for best_config_path in best_configs:
        payload = read_json_object(best_config_path)
        if payload is None or payload.get("status") != "ok":
            continue
        scanned_best_configs += 1
        base_dir = best_config_path.parent

        protected_paths.update(
            extract_checkpoint_paths(
                payload,
                base_dir=base_dir,
                existing_only=False,
            ).values()
        )

        top_trials_obj = payload.get("top_trials")
        if not isinstance(top_trials_obj, list):
            continue

        for top_trial in top_trials_obj:
            if not isinstance(top_trial, dict):
                continue
            referenced_paths = _collect_checkpoint_paths_from_entry(
                top_trial,
                base_dir=base_dir,
            )
            rank = top_trial.get("rank")
            if isinstance(rank, int) and rank > 0:
                protected_paths.update(referenced_paths)
            else:
                missing_rank_entries += 1
                candidate_paths.update(referenced_paths)

    deletable_paths = tuple(
        sorted(
            path
            for path in candidate_paths
            if path.exists()
            and path.is_file()
            and _is_path_within(path, resolved_model_root)
            and path not in protected_paths
        )
    )

    if not dry_run:
        for path in deletable_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                continue

    return MissingRankPruneReport(
        scanned_best_configs=scanned_best_configs,
        missing_rank_entries=missing_rank_entries,
        candidate_paths=len(candidate_paths),
        deleted_count=len(deletable_paths),
        dry_run=dry_run,
        deleted_paths=deletable_paths,
    )
