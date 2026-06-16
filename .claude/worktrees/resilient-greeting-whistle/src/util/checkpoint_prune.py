"""Checkpoint pruning utilities with validation-signature-aware ranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from util.checkpoint_io import (
    extract_checkpoint_paths,
    normalize_checkpoint_path,
    read_json_object,
)
from util.validation_protocol import LEGACY_VALIDATION_SIGNATURE

_TASKS: tuple[str, ...] = ("donor", "acceptor", "pair")


@dataclass(frozen=True)
class SummaryCandidate:
    """One train-summary candidate representing one hyperparameter trial."""

    summary_path: Path
    species: str
    model_name: str
    validation_signature: str
    selection_score: float
    best_pr_auc: float
    modified_time: float
    checkpoint_paths: tuple[Path, ...]
    hyperparameters: dict[str, object]


@dataclass(frozen=True)
class PruneReport:
    """Result of a pruning pass."""

    total_candidates: int
    kept_count: int
    deleted_count: int
    dry_run: bool
    deleted_paths: tuple[Path, ...]


def _safe_float(value: object, default: float = float("-inf")) -> float:
    """Convert scalar to float with fallback."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _extract_model_name(payload: dict[str, object]) -> Optional[str]:
    """Extract model name from summary payload."""
    model_name = payload.get("model")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return None


def _extract_validation_signature(payload: dict[str, object]) -> str:
    """Read validation signature from summary payload."""
    raw = payload.get("validation_signature")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return LEGACY_VALIDATION_SIGNATURE


def _extract_selection_score(payload: dict[str, object]) -> float:
    """Extract run-level selection score."""
    by_task = payload.get("selection_score_by_task")
    if isinstance(by_task, dict):
        values = [
            _safe_float(by_task.get(task))
            for task in _TASKS
        ]
        finite = [value for value in values if value != float("-inf")]
        if finite:
            return sum(finite) / len(finite)

    direct_values = [
        _safe_float(payload.get(f"selection_score_{task}"))
        for task in _TASKS
    ]
    finite_direct = [value for value in direct_values if value != float("-inf")]
    if finite_direct:
        return sum(finite_direct) / len(finite_direct)

    task_scores: list[float] = []
    for task in _TASKS:
        task_payload = payload.get(task)
        if not isinstance(task_payload, dict):
            continue
        score = _safe_float(task_payload.get("best_score"))
        if score != float("-inf"):
            task_scores.append(score)
    if task_scores:
        return sum(task_scores) / len(task_scores)
    return float("-inf")


def _extract_mean_best_pr_auc(payload: dict[str, object]) -> float:
    """Extract mean best PR-AUC for tie-break ranking."""
    values: list[float] = []
    for task in _TASKS:
        task_payload = payload.get(task)
        if not isinstance(task_payload, dict):
            continue
        pr_auc = _safe_float(task_payload.get("best_pr_auc"))
        if pr_auc != float("-inf"):
            values.append(pr_auc)
            continue
        best_metric = task_payload.get("best_metric")
        best_score = _safe_float(task_payload.get("best_score"))
        if best_metric == "pr_auc" and best_score != float("-inf"):
            values.append(best_score)
    if values:
        return sum(values) / len(values)
    return float("-inf")


def _extract_hyperparameters(payload: dict[str, object]) -> dict[str, object]:
    """Extract compact hyperparameter block for leaderboard output."""
    raw = payload.get("task_hyperparameters")
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}

    keys = (
        "epochs",
        "batch_size",
        "lr",
        "loss",
        "weight_decay",
        "dropout",
        "val_frac",
        "seed",
    )
    extracted: dict[str, object] = {}
    for key in keys:
        value = payload.get(key)
        if value is not None:
            extracted[key] = value
    return extracted


def _collect_summary_candidates(
    *,
    data_root: Path,
    species: str,
    model_name: str,
) -> list[SummaryCandidate]:
    """Collect summary candidates for one species/model."""
    summary_dirs: list[Path] = []
    learning_metric_dir = data_root / species / "learning_metric"
    if learning_metric_dir.is_dir():
        summary_dirs.append(learning_metric_dir)
    legacy_site_score_dir = data_root / species / "site_score"
    if legacy_site_score_dir.is_dir():
        summary_dirs.append(legacy_site_score_dir)
    if not summary_dirs:
        return []

    candidates: list[SummaryCandidate] = []
    for summary_dir in summary_dirs:
        for summary_path in sorted(summary_dir.glob("*.train.json")):
            payload = read_json_object(summary_path)
            if payload is None:
                continue
            summary_model = _extract_model_name(payload)
            if summary_model != model_name:
                continue
            checkpoint_map = extract_checkpoint_paths(
                payload,
                base_dir=summary_path.parent,
                existing_only=True,
            )
            checkpoint_paths = tuple(sorted(set(checkpoint_map.values())))
            if not checkpoint_paths:
                continue
            modified_time = max(path.stat().st_mtime for path in checkpoint_paths)
            candidates.append(
                SummaryCandidate(
                    summary_path=summary_path,
                    species=species,
                    model_name=model_name,
                    validation_signature=_extract_validation_signature(payload),
                    selection_score=_extract_selection_score(payload),
                    best_pr_auc=_extract_mean_best_pr_auc(payload),
                    modified_time=modified_time,
                    checkpoint_paths=checkpoint_paths,
                    hyperparameters=_extract_hyperparameters(payload),
                )
            )
    return candidates


def _resolve_checkpoint_references_from_best_config(
    best_config_path: Path,
) -> set[Path]:
    """Resolve checkpoint paths referenced by one tuning best_config."""
    payload = read_json_object(best_config_path)
    if payload is None or payload.get("status") != "ok":
        return set()

    protected: set[Path] = set(
        extract_checkpoint_paths(
            payload,
            base_dir=best_config_path.parent,
            existing_only=False,
        ).values()
    )

    metrics_json_raw = payload.get("metrics_json")
    if isinstance(metrics_json_raw, str) and metrics_json_raw.strip():
        metrics_path = normalize_checkpoint_path(
            metrics_json_raw.strip(),
            base_dir=best_config_path.parent,
        )
        metrics_payload = read_json_object(metrics_path)
        if metrics_payload is not None:
            for path in extract_checkpoint_paths(
                metrics_payload,
                base_dir=metrics_path.parent,
                existing_only=False,
            ).values():
                protected.add(path)

    phase = payload.get("phase")
    trial_id = payload.get("trial_id")
    if not isinstance(phase, str) or not isinstance(trial_id, int):
        return protected

    metrics_path = (
        best_config_path.parent / f"{phase}_trial_{trial_id:04d}.metrics.json"
    )
    metrics_payload = read_json_object(metrics_path)
    if metrics_payload is None:
        return protected
    for path in extract_checkpoint_paths(
        metrics_payload,
        base_dir=metrics_path.parent,
        existing_only=False,
    ).values():
        protected.add(path)
    return protected


def _collect_protected_checkpoints(
    *,
    data_root: Path,
    species: str,
    model_name: str,
) -> set[Path]:
    """Collect checkpoint paths that must never be pruned."""
    tuning_root = data_root / species / "tuning" / model_name
    if not tuning_root.is_dir():
        return set()
    protected: set[Path] = set()
    for best_config_path in sorted(tuning_root.rglob("best_config.json")):
        protected.update(
            _resolve_checkpoint_references_from_best_config(best_config_path)
        )
    return protected


def _write_leaderboard(
    *,
    data_root: Path,
    species: str,
    model_name: str,
    top_k: int,
    ranked_rows: dict[str, list[SummaryCandidate]],
) -> None:
    """Write leaderboard JSON for one species/model."""
    out_dir = data_root / species / "tuning" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"checkpoint_prune_top{top_k}.json"

    signatures: list[dict[str, object]] = []
    for signature, rows in sorted(ranked_rows.items()):
        entries: list[dict[str, object]] = []
        for rank, row in enumerate(rows[:top_k], start=1):
            entries.append(
                {
                    "rank": rank,
                    "summary_path": str(row.summary_path),
                    "selection_score": row.selection_score,
                    "best_pr_auc": row.best_pr_auc,
                    "checkpoint_paths": [str(path) for path in row.checkpoint_paths],
                    "hyperparameters": row.hyperparameters,
                }
            )
        signatures.append(
            {
                "validation_signature": signature,
                "entries": entries,
            }
        )

    payload = {
        "species": species,
        "model": model_name,
        "top_k": top_k,
        "signatures": signatures,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prune_species_model_checkpoints(
    *,
    data_root: Path,
    species: str,
    model_name: str,
    top_k: int,
    dry_run: bool = False,
) -> PruneReport:
    """Prune checkpoints by top-k ranking for one species/model."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")

    candidates = _collect_summary_candidates(
        data_root=data_root,
        species=species,
        model_name=model_name,
    )
    if not candidates:
        return PruneReport(
            total_candidates=0,
            kept_count=0,
            deleted_count=0,
            dry_run=dry_run,
            deleted_paths=(),
        )

    protected = _collect_protected_checkpoints(
        data_root=data_root,
        species=species,
        model_name=model_name,
    )

    by_signature: dict[str, list[SummaryCandidate]] = {}
    for row in candidates:
        by_signature.setdefault(row.validation_signature, []).append(row)

    keep_paths: set[Path] = set()
    ranked_for_report: dict[str, list[SummaryCandidate]] = {}
    for signature, rows in by_signature.items():
        ranked_rows = sorted(
            rows,
            key=lambda item: (
                item.selection_score,
                item.best_pr_auc,
                item.modified_time,
            ),
            reverse=True,
        )
        ranked_for_report[signature] = ranked_rows
        for candidate in ranked_rows[:top_k]:
            keep_paths.update(candidate.checkpoint_paths)

    keep_paths.update(protected)

    unique_paths = {
        path
        for candidate in candidates
        for path in candidate.checkpoint_paths
    }
    delete_paths = tuple(
        sorted(path for path in unique_paths if path not in keep_paths)
    )
    if not dry_run:
        for path in delete_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                continue

    _write_leaderboard(
        data_root=data_root,
        species=species,
        model_name=model_name,
        top_k=top_k,
        ranked_rows=ranked_for_report,
    )

    return PruneReport(
        total_candidates=len(unique_paths),
        kept_count=len(unique_paths) - len(delete_paths),
        deleted_count=len(delete_paths),
        dry_run=dry_run,
        deleted_paths=delete_paths,
    )
