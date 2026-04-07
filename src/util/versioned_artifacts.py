"""Utilities for publishing versioned best artifacts for tuned models.

Publication happens when a canonical ``best_config.json`` improves. Site models
publish synchronized donor/acceptor checkpoint versions, so a donor-only update
still bumps the acceptor side to the same version number via carry-forward.
Pair models publish a single versioned pair checkpoint. Older published
versions move under ``archive/versioned_artifacts/`` only after the new
version's score and eval outputs are finalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import csv
import json
import os
from pathlib import Path
import shutil
from typing import Iterable, Mapping

import torch

from util.checkpoint_io import (
    extract_task_checkpoint_path,
    read_json_object,
    resolve_existing_checkpoint_path,
)
from models.registry import available_models
from util.model_task_paths import checkpoint_tasks_for_model
from util.path_format import relativize_path_fields, relativize_path_string

SITE_PUBLICATION_TASKS: tuple[str, ...] = ("donor", "acceptor")
PAIR_PUBLICATION_TASKS: tuple[str, ...] = ("pair",)
LEGACY_PUBLIC_OUTPUT_STEMS: dict[str, tuple[str, ...]] = {
    "cnn_pair_v2": ("cnn_v2_pair",),
}
KNOWN_MODEL_NAMES: frozenset[str] = frozenset(available_models())
VERSION_HISTORY_COLUMNS: tuple[str, ...] = (
    "version",
    "published_name",
    "published_at",
    "source_best_config",
    "objective_metric",
    "objective_score",
    "updated_side",
    "carry_forward_side",
    "donor_checkpoint_path",
    "acceptor_checkpoint_path",
    "pair_checkpoint_path",
    "metrics_json",
    "archive_status",
)
INITIAL_VERSION_NUMBER: int = 1


@dataclass(frozen=True)
class VersionHistoryEntry:
    """One row from ``version_history.tsv``."""

    version: int
    published_name: str
    published_at: str
    source_best_config: str
    objective_metric: str
    objective_score: str
    updated_side: str
    carry_forward_side: str
    donor_checkpoint_path: str
    acceptor_checkpoint_path: str
    pair_checkpoint_path: str
    metrics_json: str
    archive_status: str

    def to_row(self) -> dict[str, str]:
        """Serialize one history entry for TSV output."""
        return {
            "version": str(self.version),
            "published_name": self.published_name,
            "published_at": self.published_at,
            "source_best_config": self.source_best_config,
            "objective_metric": self.objective_metric,
            "objective_score": self.objective_score,
            "updated_side": self.updated_side,
            "carry_forward_side": self.carry_forward_side,
            "donor_checkpoint_path": self.donor_checkpoint_path,
            "acceptor_checkpoint_path": self.acceptor_checkpoint_path,
            "pair_checkpoint_path": self.pair_checkpoint_path,
            "metrics_json": self.metrics_json,
            "archive_status": self.archive_status,
        }


def normalize_public_model_name(model_name: str) -> str:
    """Normalize a runtime/public model name to the published public name."""
    return model_name.strip()


def publication_tasks_for_model(model_name: str) -> tuple[str, ...] | None:
    """Return the checkpoint-task signature supported by version publication."""
    public_model_name = normalize_public_model_name(model_name)
    if public_model_name == "":
        return None
    if public_model_name not in KNOWN_MODEL_NAMES:
        return None
    tasks = checkpoint_tasks_for_model(public_model_name)
    if tasks == SITE_PUBLICATION_TASKS:
        return SITE_PUBLICATION_TASKS
    if tasks == PAIR_PUBLICATION_TASKS:
        return PAIR_PUBLICATION_TASKS
    return None


def is_active_public_model(model_name: str) -> bool:
    """Return whether one model participates in version publication."""
    return publication_tasks_for_model(model_name) is not None


def is_independent_public_model(model_name: str) -> bool:
    """Return whether one model publishes synchronized donor/acceptor assets."""
    return publication_tasks_for_model(model_name) == SITE_PUBLICATION_TASKS


def format_published_name(public_model_name: str, version: int) -> str:
    """Format one published version name."""
    if version <= 0:
        raise ValueError("version must be positive.")
    return f"{public_model_name}.{version:02d}"


def resolve_version_history_path(
    data_root: Path,
    species: str,
    public_model_name: str,
) -> Path:
    """Return the version-history TSV path for one published model."""
    return (
        data_root
        / species
        / "tuning"
        / normalize_public_model_name(public_model_name)
        / "version_history.tsv"
    )


def resolve_versions_dir(
    data_root: Path,
    species: str,
    public_model_name: str,
) -> Path:
    """Return the snapshot directory for one published model."""
    return (
        data_root
        / species
        / "tuning"
        / normalize_public_model_name(public_model_name)
        / "versions"
    )


def read_version_history(
    data_root: Path,
    species: str,
    public_model_name: str,
) -> list[VersionHistoryEntry]:
    """Read one version-history TSV file."""
    history_path = resolve_version_history_path(data_root, species, public_model_name)
    if not history_path.is_file():
        return []

    entries: list[VersionHistoryEntry] = []
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row is None:
                continue
            raw_version = str(row.get("version", "")).strip()
            if raw_version == "":
                continue
            entry = VersionHistoryEntry(
                version=int(raw_version),
                published_name=str(row.get("published_name", "")).strip(),
                published_at=str(row.get("published_at", "")).strip(),
                source_best_config=str(row.get("source_best_config", "")).strip(),
                objective_metric=str(row.get("objective_metric", "")).strip(),
                objective_score=str(row.get("objective_score", "")).strip(),
                updated_side=str(row.get("updated_side", "")).strip(),
                carry_forward_side=str(row.get("carry_forward_side", "")).strip(),
                donor_checkpoint_path=str(row.get("donor_checkpoint_path", "")).strip(),
                acceptor_checkpoint_path=str(
                    row.get("acceptor_checkpoint_path", "")
                ).strip(),
                pair_checkpoint_path=str(row.get("pair_checkpoint_path", "")).strip(),
                metrics_json=str(row.get("metrics_json", "")).strip(),
                archive_status=str(row.get("archive_status", "")).strip() or "live",
            )
            entries.append(entry)
    entries.sort(key=lambda item: item.version)
    return entries


def write_version_history(
    data_root: Path,
    species: str,
    public_model_name: str,
    entries: Iterable[VersionHistoryEntry],
) -> None:
    """Write one version-history TSV file."""
    history_path = resolve_version_history_path(data_root, species, public_model_name)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(VERSION_HISTORY_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in entries:
            row = relativize_path_fields(entry.to_row())
            if not isinstance(row, dict):
                raise TypeError("Version-history row must serialize to one object.")
            writer.writerow(row)


def resolve_latest_published_name(
    data_root: Path,
    species: str,
    model_name: str,
) -> str | None:
    """Resolve the latest published name for one active model."""
    public_model_name = normalize_public_model_name(model_name)
    latest_entry = resolve_latest_live_history_entry(
        data_root=data_root,
        species=species,
        model_name=public_model_name,
    )
    if latest_entry is None:
        return None
    return latest_entry.published_name


def resolve_latest_live_history_entry(
    data_root: Path,
    species: str,
    model_name: str,
) -> VersionHistoryEntry | None:
    """Resolve the newest live published entry for one active public model."""
    public_model_name = normalize_public_model_name(model_name)
    history = read_version_history(data_root, species, public_model_name)
    if not history:
        return None
    for entry in reversed(history):
        if entry.archive_status == "live":
            return entry
    return history[-1]


def resolve_latest_published_run_assets(
    *,
    project_root: Path,
    species: str,
    model_name: str,
) -> dict[str, str] | None:
    """Resolve latest published checkpoints and output paths for one run."""
    return resolve_published_run_assets(
        project_root=project_root,
        species=species,
        model_name=model_name,
        published_name=None,
    )


def resolve_published_history_entry(
    *,
    data_root: Path,
    species: str,
    model_name: str,
    published_name: str | None,
) -> VersionHistoryEntry | None:
    """Resolve one published history entry by name or latest live version."""
    public_model_name = normalize_public_model_name(model_name)
    if publication_tasks_for_model(public_model_name) is None:
        return None

    if published_name is None or published_name.strip() == "":
        return resolve_latest_live_history_entry(
            data_root=data_root,
            species=species,
            model_name=public_model_name,
        )

    target_name = published_name.strip()
    history = read_version_history(data_root, species, public_model_name)
    for entry in reversed(history):
        if entry.published_name == target_name and entry.archive_status == "live":
            return entry
    for entry in reversed(history):
        if entry.published_name == target_name:
            return entry
    return None


def resolve_published_run_assets(
    *,
    project_root: Path,
    species: str,
    model_name: str,
    published_name: str | None,
    allow_missing_checkpoints: bool = False,
) -> dict[str, str] | None:
    """Resolve checkpoints and output paths for one published run identity."""
    public_model_name = normalize_public_model_name(model_name)
    publication_tasks = publication_tasks_for_model(public_model_name)
    if publication_tasks is None:
        return None

    data_root = _resolve_data_root(project_root)
    model_root = _resolve_model_root(project_root)
    latest_entry = resolve_published_history_entry(
        data_root=data_root,
        species=species,
        model_name=public_model_name,
        published_name=published_name,
    )
    if latest_entry is None:
        return None

    assets: dict[str, str] = {
        "published_name": latest_entry.published_name,
        "site_output_tsv": str(
            data_root / species / "site_score" / f"{latest_entry.published_name}.tsv"
        ),
        "intron_output_tsv": str(
            data_root / species / "intron_score" / f"{latest_entry.published_name}.tsv"
        ),
        "transcript_output_tsv": str(
            data_root / species / "trans_score" / f"{latest_entry.published_name}.tsv"
        ),
        "eval_output_txt": str(
            data_root / species / "eval_score" / f"{latest_entry.published_name}.txt"
        ),
        "metrics_json": str(
            data_root
            / species
            / "learning_metric"
            / f"{latest_entry.published_name}.train.json"
        ),
    }

    if publication_tasks == SITE_PUBLICATION_TASKS:
        donor_path = _resolve_version_history_checkpoint_path(
            raw_path=latest_entry.donor_checkpoint_path,
            model_root=model_root,
            allow_missing=allow_missing_checkpoints,
        )
        acceptor_path = _resolve_version_history_checkpoint_path(
            raw_path=latest_entry.acceptor_checkpoint_path,
            model_root=model_root,
            allow_missing=allow_missing_checkpoints,
        )
        assets["donor_checkpoint_path"] = str(donor_path)
        assets["acceptor_checkpoint_path"] = str(acceptor_path)
    else:
        pair_path = _resolve_version_history_checkpoint_path(
            raw_path=latest_entry.pair_checkpoint_path,
            model_root=model_root,
            allow_missing=allow_missing_checkpoints,
        )
        assets["pair_checkpoint_path"] = str(pair_path)
    return assets


def _is_loadable_torch_checkpoint(path: Path) -> bool:
    """Return whether a path can be loaded as a Torch checkpoint payload."""
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if isinstance(payload, dict):
        return True
    return isinstance(payload, torch.Tensor)


def _find_latest_valid_checkpoint_candidate(
    *,
    model_root: Path,
    species: str,
    task: str,
    exclude_name: str,
) -> Path | None:
    """Find the newest valid checkpoint candidate for one task directory."""
    task_dir = model_root / species / task
    if not task_dir.is_dir():
        return None
    candidates = sorted(
        (
            path
            for path in task_dir.glob("*.pt")
            if path.name != exclude_name and path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _is_loadable_torch_checkpoint(candidate):
            return candidate.resolve()
    return None


def normalize_published_run_checkpoints(
    *,
    project_root: Path,
    species: str,
    model_name: str,
    published_name: str,
) -> dict[str, str]:
    """Normalize corrupted published checkpoints in-place for one run identity.

    This function validates published checkpoint files referenced by version
    history. If one file is missing or not loadable by ``torch.load``, it is
    replaced by the newest valid ``.pt`` candidate under the same
    ``model/<species>/<task>/`` directory, and metadata pointers are rewritten
    to the canonical published paths.

    Raises
    ------
    ValueError
        If no valid replacement checkpoint can be found for a broken task.
    """
    public_model_name = normalize_public_model_name(model_name)
    publication_tasks = publication_tasks_for_model(public_model_name)
    if publication_tasks is None:
        return {}

    data_root = _resolve_data_root(project_root)
    model_root = _resolve_model_root(project_root)
    history = read_version_history(data_root, species, public_model_name)
    target_entry: VersionHistoryEntry | None = None
    target_index = -1
    for index, entry in enumerate(history):
        if entry.published_name == published_name:
            target_entry = entry
            target_index = index
    if target_entry is None:
        raise ValueError(
            "Published entry not found for normalization: "
            f"species={species}, model={public_model_name}, "
            f"published_name={published_name}."
        )

    replacements: dict[str, str] = {}
    for task in publication_tasks:
        published_path = model_root / species / task / f"{published_name}.pt"
        if _is_loadable_torch_checkpoint(published_path):
            continue
        replacement = _find_latest_valid_checkpoint_candidate(
            model_root=model_root,
            species=species,
            task=task,
            exclude_name=f"{published_name}.pt",
        )
        if replacement is None:
            raise ValueError(
                "No valid checkpoint candidate found while normalizing "
                f"published checkpoint: {published_path}"
            )
        published_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(replacement, published_path)
        replacements[task] = str(replacement)

    donor_path = str(
        (model_root / species / "donor" / f"{published_name}.pt").resolve()
    )
    acceptor_path = str(
        (model_root / species / "acceptor" / f"{published_name}.pt").resolve()
    )
    pair_path = str((model_root / species / "pair" / f"{published_name}.pt").resolve())
    rewritten_entry = VersionHistoryEntry(
        version=target_entry.version,
        published_name=target_entry.published_name,
        published_at=target_entry.published_at,
        source_best_config=target_entry.source_best_config,
        objective_metric=target_entry.objective_metric,
        objective_score=target_entry.objective_score,
        updated_side=target_entry.updated_side,
        carry_forward_side=target_entry.carry_forward_side,
        donor_checkpoint_path=(
            relativize_path_string(donor_path)
            if publication_tasks == SITE_PUBLICATION_TASKS
            else target_entry.donor_checkpoint_path
        ),
        acceptor_checkpoint_path=(
            relativize_path_string(acceptor_path)
            if publication_tasks == SITE_PUBLICATION_TASKS
            else target_entry.acceptor_checkpoint_path
        ),
        pair_checkpoint_path=(
            relativize_path_string(pair_path)
            if publication_tasks == PAIR_PUBLICATION_TASKS
            else target_entry.pair_checkpoint_path
        ),
        metrics_json=target_entry.metrics_json,
        archive_status=target_entry.archive_status,
    )
    history[target_index] = rewritten_entry
    write_version_history(data_root, species, public_model_name, history)

    best_config_paths = _resolve_best_config_paths(
        data_root=data_root,
        species=species,
        public_model_name=public_model_name,
        publication_tasks=publication_tasks,
    )
    for config_path in best_config_paths.values():
        payload = read_json_object(config_path)
        if payload is None:
            continue
        payload["published_name"] = published_name
        if publication_tasks == SITE_PUBLICATION_TASKS:
            payload["donor_checkpoint_path"] = donor_path
            payload["acceptor_checkpoint_path"] = acceptor_path
        else:
            payload["pair_checkpoint_path"] = pair_path
        _write_json_object(config_path, payload)

    return replacements


def refresh_published_version_if_improved(
    *,
    project_root: Path,
    species: str,
    model_name: str,
    published_name: str,
    task_payloads: Mapping[str, Mapping[str, object]],
    metrics_json: str | None = None,
) -> VersionHistoryEntry | None:
    """Refresh one published version in-place when a full run improves it.

    Parameters
    ----------
    project_root : Path
        Repository root.
    species : str
        Species key under ``data`` and ``model``.
    model_name : str
        Public model name such as ``cnn_v2`` or ``cnn_pair_v3``.
    published_name : str
        Existing published version stem to refresh.
    task_payloads : Mapping[str, Mapping[str, object]]
        Candidate per-task payloads containing checkpoint path and objective
        fields from the current full run.
    metrics_json : str | None, default=None
        Training summary path for the current run.

    Returns
    -------
    VersionHistoryEntry | None
        Updated history entry when at least one task improved, otherwise
        ``None``.
    """
    public_model_name = normalize_public_model_name(model_name)
    publication_tasks = publication_tasks_for_model(public_model_name)
    if publication_tasks is None:
        return None
    target_published_name = published_name.strip()
    if target_published_name == "":
        raise ValueError("published_name must not be empty.")

    data_root = _resolve_data_root(project_root)
    model_root = _resolve_model_root(project_root)
    history = read_version_history(data_root, species, public_model_name)
    entry = resolve_published_history_entry(
        data_root=data_root,
        species=species,
        model_name=public_model_name,
        published_name=target_published_name,
    )
    if entry is None or entry.archive_status != "live":
        return None

    best_config_paths = _resolve_best_config_paths(
        data_root=data_root,
        species=species,
        public_model_name=public_model_name,
        publication_tasks=publication_tasks,
    )
    best_payloads: dict[str, dict[str, object]] = {}
    improved_tasks: list[str] = []
    published_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics_json_value = metrics_json.strip() if metrics_json is not None else ""

    for task in publication_tasks:
        best_path = best_config_paths[task]
        payload = read_json_object(best_path)
        if payload is None or payload.get("status") != "ok":
            return None
        mutable_payload = dict(payload)
        best_payloads[task] = mutable_payload
        candidate_payload = task_payloads.get(task)
        if candidate_payload is None:
            continue
        candidate_metric = str(candidate_payload.get("objective_metric", "")).strip()
        current_metric = str(mutable_payload.get("objective_metric", "")).strip()
        if (
            candidate_metric != ""
            and current_metric != ""
            and candidate_metric != current_metric
        ):
            continue
        candidate_score = _coerce_score(candidate_payload.get("objective_score"))
        current_score = _coerce_score(mutable_payload.get("objective_score"))
        if candidate_score <= current_score:
            continue

        checkpoint_source = _resolve_task_checkpoint_from_payload(
            payload=candidate_payload,
            task=task,
            base_dir=project_root,
            model_root=model_root,
        )
        if checkpoint_source is None:
            continue
        destination = model_root / species / task / f"{target_published_name}.pt"
        _copy_checkpoint_file(checkpoint_source, destination)
        mutable_payload[f"{task}_checkpoint_path"] = str(destination.resolve())
        mutable_payload["published_name"] = target_published_name
        mutable_payload["published_at"] = published_at
        if candidate_metric != "":
            mutable_payload["objective_metric"] = candidate_metric
        mutable_payload["objective_score"] = candidate_score
        if metrics_json_value != "":
            mutable_payload["metrics_json"] = metrics_json_value
        improved_tasks.append(task)

    if not improved_tasks:
        return None

    if publication_tasks == SITE_PUBLICATION_TASKS:
        donor_destination = (
            model_root / species / "donor" / f"{target_published_name}.pt"
        )
        acceptor_destination = (
            model_root / species / "acceptor" / f"{target_published_name}.pt"
        )
        for task in SITE_PUBLICATION_TASKS:
            best_payloads[task]["donor_checkpoint_path"] = str(
                donor_destination.resolve()
            )
            best_payloads[task]["acceptor_checkpoint_path"] = str(
                acceptor_destination.resolve()
            )
            best_payloads[task]["published_name"] = target_published_name
            best_payloads[task]["published_at"] = published_at
            if metrics_json_value != "":
                best_payloads[task]["metrics_json"] = metrics_json_value
    else:
        pair_destination = model_root / species / "pair" / f"{target_published_name}.pt"
        best_payloads["pair"]["pair_checkpoint_path"] = str(pair_destination.resolve())
        best_payloads["pair"]["published_name"] = target_published_name
        best_payloads["pair"]["published_at"] = published_at
        if metrics_json_value != "":
            best_payloads["pair"]["metrics_json"] = metrics_json_value

    for task, best_path in best_config_paths.items():
        _write_json_object(best_path, best_payloads[task])

    improved_side = ",".join(improved_tasks)
    carry_forward_side = ",".join(
        task for task in publication_tasks if task not in improved_tasks
    )
    snapshot_payload = _build_snapshot_payload(
        public_model_name=public_model_name,
        published_name=target_published_name,
        published_at=published_at,
        updated_side=improved_side,
        carry_forward_side=carry_forward_side,
        best_payloads=best_payloads,
        publication_tasks=publication_tasks,
    )
    _write_snapshot(
        data_root=data_root,
        species=species,
        public_model_name=public_model_name,
        published_name=target_published_name,
        payload=snapshot_payload,
    )

    primary_task = improved_tasks[0]
    primary_payload = best_payloads[primary_task]
    source_best_config = str(best_config_paths[primary_task].resolve())
    metrics_value = (
        metrics_json_value
        if metrics_json_value != ""
        else str(primary_payload.get("metrics_json", "")).strip()
    )
    updated_entry = VersionHistoryEntry(
        version=_parse_version_number(target_published_name, public_model_name),
        published_name=target_published_name,
        published_at=published_at,
        source_best_config=relativize_path_string(source_best_config),
        objective_metric=str(primary_payload.get("objective_metric", "")).strip(),
        objective_score=_stringify_scalar(primary_payload.get("objective_score")),
        updated_side=improved_side,
        carry_forward_side=carry_forward_side,
        donor_checkpoint_path=(
            relativize_path_string(
                str(
                    (
                        model_root / species / "donor" / f"{target_published_name}.pt"
                    ).resolve()
                )
            )
            if publication_tasks == SITE_PUBLICATION_TASKS
            else ""
        ),
        acceptor_checkpoint_path=(
            relativize_path_string(
                str(
                    (
                        model_root
                        / species
                        / "acceptor"
                        / f"{target_published_name}.pt"
                    ).resolve()
                )
            )
            if publication_tasks == SITE_PUBLICATION_TASKS
            else ""
        ),
        pair_checkpoint_path=(
            relativize_path_string(
                str(
                    (
                        model_root / species / "pair" / f"{target_published_name}.pt"
                    ).resolve()
                )
            )
            if publication_tasks == PAIR_PUBLICATION_TASKS
            else ""
        ),
        metrics_json=relativize_path_string(metrics_value) if metrics_value else "",
        archive_status="live",
    )
    rewritten_history: list[VersionHistoryEntry] = []
    for row in history:
        if row.published_name == target_published_name:
            rewritten_history.append(updated_entry)
        else:
            rewritten_history.append(row)
    write_version_history(data_root, species, public_model_name, rewritten_history)
    return updated_entry


def ensure_publication_seed(
    *,
    project_root: Path,
    species: str,
    model_name: str,
) -> str | None:
    """Seed the initial ``.01`` publication when history does not exist yet."""
    public_model_name = normalize_public_model_name(model_name)
    if publication_tasks_for_model(public_model_name) is None:
        return None

    data_root = _resolve_data_root(project_root)
    history = read_version_history(data_root, species, public_model_name)
    if history:
        return history[-1].published_name
    published = publish_latest_best_version(
        project_root=project_root,
        species=species,
        model_name=public_model_name,
        updated_side="seed",
    )
    return published.published_name if published is not None else None


def publish_latest_best_version(
    *,
    project_root: Path,
    species: str,
    model_name: str,
    updated_side: str,
) -> VersionHistoryEntry | None:
    """Publish the current canonical best state as the next public version."""
    public_model_name = normalize_public_model_name(model_name)
    publication_tasks = publication_tasks_for_model(public_model_name)
    if publication_tasks is None:
        return None

    data_root = _resolve_data_root(project_root)
    model_root = _resolve_model_root(project_root)
    history = read_version_history(data_root, species, public_model_name)
    next_version = INITIAL_VERSION_NUMBER if not history else history[-1].version + 1
    published_name = format_published_name(public_model_name, next_version)
    published_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    if publication_tasks == SITE_PUBLICATION_TASKS:
        entry = _publish_independent_public_version(
            project_root=project_root,
            data_root=data_root,
            model_root=model_root,
            species=species,
            history=history,
            public_model_name=public_model_name,
            published_name=published_name,
            published_at=published_at,
            updated_side=updated_side,
        )
    else:
        entry = _publish_pair_public_version(
            project_root=project_root,
            data_root=data_root,
            model_root=model_root,
            species=species,
            history=history,
            public_model_name=public_model_name,
            published_name=published_name,
            published_at=published_at,
            updated_side=updated_side,
        )
    if entry is None:
        return None

    updated_history = [*history, entry]
    write_version_history(data_root, species, public_model_name, updated_history)
    return updated_history[-1]


def finalize_published_version_outputs(
    *,
    project_root: Path,
    species: str,
    model_name: str,
    published_name: str,
) -> VersionHistoryEntry | None:
    """Archive stale live versions after one published output set is complete.

    Parameters
    ----------
    project_root : Path
        Repository root.
    species : str
        Species key under ``data`` and ``model``.
    model_name : str
        Public model name such as ``cnn_v2`` or ``cnn_pair_v3``.
    published_name : str
        Published version whose score/eval outputs have completed.

    Returns
    -------
    VersionHistoryEntry | None
        Finalized live entry when publication participates in versioning,
        otherwise ``None``.

    Raises
    ------
    FileNotFoundError
        If the required score/eval outputs are missing.
    ValueError
        If ``published_name`` is empty.
    """
    public_model_name = normalize_public_model_name(model_name)
    publication_tasks = publication_tasks_for_model(public_model_name)
    if publication_tasks is None:
        return None

    target_published_name = published_name.strip()
    if target_published_name == "":
        raise ValueError("published_name must not be empty.")

    data_root = _resolve_data_root(project_root)
    model_root = _resolve_model_root(project_root)
    history = read_version_history(data_root, species, public_model_name)
    if not history:
        return None

    target_entry: VersionHistoryEntry | None = None
    for entry in history:
        if entry.published_name == target_published_name:
            target_entry = entry
    if target_entry is None:
        return None

    _validate_published_outputs_ready(
        data_root=data_root,
        model_root=model_root,
        species=species,
        public_model_name=public_model_name,
        published_name=target_published_name,
    )

    archive_root = (
        project_root / "archive" / "versioned_artifacts" / species / public_model_name
    )
    updated_entries: list[VersionHistoryEntry] = []
    for entry in history:
        archive_status = (
            "live" if entry.published_name == target_published_name else "archived"
        )
        if archive_status == "archived" and entry.archive_status != "archived":
            _archive_one_version(
                archive_root=archive_root / entry.published_name,
                data_root=data_root,
                model_root=model_root,
                species=species,
                public_model_name=public_model_name,
                published_name=entry.published_name,
            )
        updated_entry = VersionHistoryEntry(
            version=entry.version,
            published_name=entry.published_name,
            published_at=entry.published_at,
            source_best_config=entry.source_best_config,
            objective_metric=entry.objective_metric,
            objective_score=entry.objective_score,
            updated_side=entry.updated_side,
            carry_forward_side=entry.carry_forward_side,
            donor_checkpoint_path=entry.donor_checkpoint_path,
            acceptor_checkpoint_path=entry.acceptor_checkpoint_path,
            pair_checkpoint_path=entry.pair_checkpoint_path,
            metrics_json=entry.metrics_json,
            archive_status=archive_status,
        )
        updated_entries.append(updated_entry)
        if updated_entry.published_name == target_published_name:
            target_entry = updated_entry

    write_version_history(data_root, species, public_model_name, updated_entries)
    return target_entry


def finalize_ready_published_outputs_for_species(
    *,
    project_root: Path,
    species: str,
) -> list[VersionHistoryEntry]:
    """Finalize every ready published model version for one species.

    This helper is intended for maintenance entrypoints such as plotting where
    one caller wants score/eval directories cleaned up before aggregating files.
    Models whose latest live version does not yet have a complete published
    score/eval set are skipped without error.
    """
    data_root = _resolve_data_root(project_root)
    finalized_entries: list[VersionHistoryEntry] = []
    for model_name in sorted(KNOWN_MODEL_NAMES):
        if not is_active_public_model(model_name):
            continue
        latest_entry = resolve_latest_live_history_entry(
            data_root=data_root,
            species=species,
            model_name=model_name,
        )
        if latest_entry is None:
            continue
        try:
            finalized_entry = finalize_published_version_outputs(
                project_root=project_root,
                species=species,
                model_name=model_name,
                published_name=latest_entry.published_name,
            )
        except FileNotFoundError:
            continue
        if finalized_entry is not None:
            finalized_entries.append(finalized_entry)
    return finalized_entries


def _publish_independent_public_version(
    *,
    project_root: Path,
    data_root: Path,
    model_root: Path,
    species: str,
    history: list[VersionHistoryEntry],
    public_model_name: str,
    published_name: str,
    published_at: str,
    updated_side: str,
) -> VersionHistoryEntry | None:
    donor_path = (
        data_root
        / species
        / "tuning"
        / public_model_name
        / "donor"
        / "best_config.json"
    )
    acceptor_path = (
        data_root
        / species
        / "tuning"
        / public_model_name
        / "acceptor"
        / "best_config.json"
    )
    donor_payload = read_json_object(donor_path)
    acceptor_payload = read_json_object(acceptor_path)
    if donor_payload is None or acceptor_payload is None:
        return None
    if donor_payload.get("status") != "ok" or acceptor_payload.get("status") != "ok":
        return None

    previous_entry = history[-1] if history else None
    updated_side_normalized = updated_side.strip().lower()
    carry_forward_side = ""
    donor_source: Path | None = None
    acceptor_source: Path | None = None

    donor_destination = model_root / species / "donor" / f"{published_name}.pt"
    acceptor_destination = model_root / species / "acceptor" / f"{published_name}.pt"
    if updated_side_normalized == "seed":
        donor_source = _resolve_task_checkpoint_from_payload(
            payload=donor_payload,
            task="donor",
            base_dir=donor_path.parent,
            model_root=model_root,
        )
        acceptor_source = _resolve_task_checkpoint_from_payload(
            payload=acceptor_payload,
            task="acceptor",
            base_dir=acceptor_path.parent,
            model_root=model_root,
        )
        if donor_source is None or acceptor_source is None:
            return None
        _move_checkpoint_file(donor_source, donor_destination)
        _move_checkpoint_file(acceptor_source, acceptor_destination)
        source_best_config = str(donor_path.resolve())
        objective_metric = str(donor_payload.get("objective_metric", "")).strip()
        objective_score = _stringify_scalar(donor_payload.get("objective_score"))
        metrics_json = str(donor_payload.get("metrics_json", "")).strip()
    elif updated_side_normalized == "acceptor":
        carry_forward_side = "donor"
        donor_source = _resolve_carry_forward_or_payload_checkpoint(
            project_root=project_root,
            previous_entry=previous_entry,
            payload=donor_payload,
            task="donor",
            base_dir=donor_path.parent,
            model_root=model_root,
            species=species,
            public_model_name=public_model_name,
        )
        acceptor_source = _resolve_task_checkpoint_from_payload(
            payload=acceptor_payload,
            task="acceptor",
            base_dir=acceptor_path.parent,
            model_root=model_root,
        )
        if donor_source is None or acceptor_source is None:
            return None
        _copy_checkpoint_file(donor_source, donor_destination)
        _move_checkpoint_file(acceptor_source, acceptor_destination)
        source_best_config = str(acceptor_path.resolve())
        objective_metric = str(acceptor_payload.get("objective_metric", "")).strip()
        objective_score = _stringify_scalar(acceptor_payload.get("objective_score"))
        metrics_json = str(acceptor_payload.get("metrics_json", "")).strip()
    else:
        if updated_side_normalized == "donor":
            carry_forward_side = "acceptor"
        donor_source = _resolve_updated_or_payload_checkpoint(
            payload=donor_payload,
            task="donor",
            base_dir=donor_path.parent,
            model_root=model_root,
        )
        acceptor_source = _resolve_carry_forward_or_payload_checkpoint(
            project_root=project_root,
            previous_entry=previous_entry,
            payload=acceptor_payload,
            task="acceptor",
            base_dir=acceptor_path.parent,
            model_root=model_root,
            species=species,
            public_model_name=public_model_name,
        )
        if donor_source is None or acceptor_source is None:
            return None
        if updated_side_normalized == "donor":
            _move_checkpoint_file(donor_source, donor_destination)
        else:
            _copy_checkpoint_file(donor_source, donor_destination)
        _copy_checkpoint_file(acceptor_source, acceptor_destination)
        source_best_config = str(donor_path.resolve())
        objective_metric = str(donor_payload.get("objective_metric", "")).strip()
        objective_score = _stringify_scalar(donor_payload.get("objective_score"))
        metrics_json = str(donor_payload.get("metrics_json", "")).strip()

    donor_payload["donor_checkpoint_path"] = str(donor_destination.resolve())
    donor_payload["acceptor_checkpoint_path"] = str(acceptor_destination.resolve())
    donor_payload["published_name"] = published_name
    donor_payload["published_at"] = published_at
    acceptor_payload["donor_checkpoint_path"] = str(donor_destination.resolve())
    acceptor_payload["acceptor_checkpoint_path"] = str(acceptor_destination.resolve())
    acceptor_payload["published_name"] = published_name
    acceptor_payload["published_at"] = published_at
    _write_json_object(donor_path, donor_payload)
    _write_json_object(acceptor_path, acceptor_payload)

    _seed_unversioned_outputs_if_needed(
        data_root=data_root,
        species=species,
        public_model_name=public_model_name,
        published_name=published_name,
    )
    snapshot_payload = {
        "public_model": public_model_name,
        "published_name": published_name,
        "published_at": published_at,
        "updated_side": updated_side_normalized,
        "carry_forward_side": carry_forward_side,
        "best_configs": {
            "donor": donor_payload,
            "acceptor": acceptor_payload,
        },
    }
    _write_snapshot(
        data_root=data_root,
        species=species,
        public_model_name=public_model_name,
        published_name=published_name,
        payload=snapshot_payload,
    )
    return VersionHistoryEntry(
        version=_parse_version_number(published_name, public_model_name),
        published_name=published_name,
        published_at=published_at,
        source_best_config=relativize_path_string(source_best_config),
        objective_metric=objective_metric,
        objective_score=objective_score,
        updated_side=updated_side_normalized,
        carry_forward_side=carry_forward_side,
        donor_checkpoint_path=relativize_path_string(str(donor_destination.resolve())),
        acceptor_checkpoint_path=relativize_path_string(
            str(acceptor_destination.resolve())
        ),
        pair_checkpoint_path="",
        metrics_json=relativize_path_string(metrics_json) if metrics_json else "",
        archive_status="live",
    )


def _publish_pair_public_version(
    *,
    project_root: Path,
    data_root: Path,
    model_root: Path,
    species: str,
    history: list[VersionHistoryEntry],
    public_model_name: str,
    published_name: str,
    published_at: str,
    updated_side: str,
) -> VersionHistoryEntry | None:
    pair_path = _resolve_pair_best_config_path(
        data_root=data_root,
        species=species,
        public_model_name=public_model_name,
    )
    pair_payload = read_json_object(pair_path)
    if pair_payload is None or pair_payload.get("status") != "ok":
        return None

    pair_source = _resolve_task_checkpoint_from_payload(
        payload=pair_payload,
        task="pair",
        base_dir=pair_path.parent,
        model_root=model_root,
    )
    if pair_source is None:
        return None

    pair_destination = model_root / species / "pair" / f"{published_name}.pt"
    _move_checkpoint_file(pair_source, pair_destination)
    pair_payload["pair_checkpoint_path"] = str(pair_destination.resolve())
    pair_payload["published_name"] = published_name
    pair_payload["published_at"] = published_at
    public_pair_path = (
        data_root / species / "tuning" / public_model_name / "pair" / "best_config.json"
    )
    _write_json_object(public_pair_path, pair_payload)
    if pair_path != public_pair_path:
        _write_json_object(pair_path, pair_payload)

    _seed_unversioned_outputs_if_needed(
        data_root=data_root,
        species=species,
        public_model_name=public_model_name,
        published_name=published_name,
    )
    snapshot_payload = {
        "public_model": public_model_name,
        "published_name": published_name,
        "published_at": published_at,
        "updated_side": updated_side.strip().lower(),
        "carry_forward_side": "",
        "best_config": pair_payload,
    }
    _write_snapshot(
        data_root=data_root,
        species=species,
        public_model_name=public_model_name,
        published_name=published_name,
        payload=snapshot_payload,
    )
    return VersionHistoryEntry(
        version=_parse_version_number(published_name, public_model_name),
        published_name=published_name,
        published_at=published_at,
        source_best_config=relativize_path_string(str(pair_path.resolve())),
        objective_metric=str(pair_payload.get("objective_metric", "")).strip(),
        objective_score=_stringify_scalar(pair_payload.get("objective_score")),
        updated_side=updated_side.strip().lower(),
        carry_forward_side="",
        donor_checkpoint_path="",
        acceptor_checkpoint_path="",
        pair_checkpoint_path=relativize_path_string(str(pair_destination.resolve())),
        metrics_json=(
            relativize_path_string(str(pair_payload.get("metrics_json", "")).strip())
            if str(pair_payload.get("metrics_json", "")).strip()
            else ""
        ),
        archive_status="live",
    )


def _archive_one_version(
    *,
    archive_root: Path,
    data_root: Path,
    model_root: Path,
    species: str,
    public_model_name: str,
    published_name: str,
) -> None:
    for live_path, relative_path in _iter_public_artifact_paths(
        data_root=data_root,
        model_root=model_root,
        species=species,
        public_model_name=public_model_name,
        published_name=published_name,
    ):
        if not live_path.exists():
            continue
        destination = archive_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(live_path), str(destination))


def _validate_published_outputs_ready(
    *,
    data_root: Path,
    model_root: Path,
    species: str,
    public_model_name: str,
    published_name: str,
) -> None:
    """Validate that one published version has its core outputs in place."""
    required_paths = list(
        _iter_required_live_artifact_paths(
            data_root=data_root,
            model_root=model_root,
            species=species,
            public_model_name=public_model_name,
            published_name=published_name,
        )
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Published outputs are incomplete; refusing to archive older "
            f"versions: {missing_text}"
        )


def _iter_required_live_artifact_paths(
    *,
    data_root: Path,
    model_root: Path,
    species: str,
    public_model_name: str,
    published_name: str,
) -> Iterable[Path]:
    """Yield core artifacts that must exist before one version is finalized."""
    yield data_root / species / "site_score" / f"{published_name}.tsv"
    yield data_root / species / "intron_score" / f"{published_name}.tsv"
    yield data_root / species / "trans_score" / f"{published_name}.tsv"
    yield data_root / species / "eval_score" / f"{published_name}.txt"


def _seed_unversioned_outputs_if_needed(
    *,
    data_root: Path,
    species: str,
    public_model_name: str,
    published_name: str,
) -> None:
    for source_stem in _iter_public_output_stem_candidates(public_model_name):
        for src_path, dst_path in _iter_public_data_output_moves(
            data_root=data_root,
            species=species,
            source_stem=source_stem,
            destination_stem=published_name,
        ):
            if not src_path.exists() or dst_path.exists():
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))


def _build_snapshot_payload(
    *,
    public_model_name: str,
    published_name: str,
    published_at: str,
    updated_side: str,
    carry_forward_side: str,
    best_payloads: Mapping[str, Mapping[str, object]],
    publication_tasks: tuple[str, ...],
) -> dict[str, object]:
    """Build one snapshot payload for a published version."""
    if publication_tasks == SITE_PUBLICATION_TASKS:
        return {
            "public_model": public_model_name,
            "published_name": published_name,
            "published_at": published_at,
            "updated_side": updated_side,
            "carry_forward_side": carry_forward_side,
            "best_configs": {
                "donor": best_payloads["donor"],
                "acceptor": best_payloads["acceptor"],
            },
        }
    return {
        "public_model": public_model_name,
        "published_name": published_name,
        "published_at": published_at,
        "updated_side": updated_side,
        "carry_forward_side": carry_forward_side,
        "best_config": best_payloads["pair"],
    }


def _iter_public_output_stem_candidates(public_model_name: str) -> tuple[str, ...]:
    legacy_stems = LEGACY_PUBLIC_OUTPUT_STEMS.get(public_model_name, ())
    return (public_model_name, *legacy_stems)


def _iter_public_data_output_moves(
    *,
    data_root: Path,
    species: str,
    source_stem: str,
    destination_stem: str,
) -> Iterable[tuple[Path, Path]]:
    yield (
        data_root / species / "site_score" / f"{source_stem}.tsv",
        data_root / species / "site_score" / f"{destination_stem}.tsv",
    )
    yield (
        data_root / species / "intron_score" / f"{source_stem}.tsv",
        data_root / species / "intron_score" / f"{destination_stem}.tsv",
    )
    yield (
        data_root / species / "trans_score" / f"{source_stem}.tsv",
        data_root / species / "trans_score" / f"{destination_stem}.tsv",
    )
    yield (
        data_root / species / "eval_score" / f"{source_stem}.txt",
        data_root / species / "eval_score" / f"{destination_stem}.txt",
    )
    yield (
        data_root / species / "learning_metric" / f"{source_stem}.train.json",
        data_root / species / "learning_metric" / f"{destination_stem}.train.json",
    )
    yield (
        data_root / species / "learning_metric" / f"{source_stem}_learning_curve.png",
        data_root
        / species
        / "learning_metric"
        / f"{destination_stem}_learning_curve.png",
    )


def _write_snapshot(
    *,
    data_root: Path,
    species: str,
    public_model_name: str,
    published_name: str,
    payload: Mapping[str, object],
) -> None:
    snapshot_path = (
        resolve_versions_dir(data_root, species, public_model_name)
        / f"{published_name}.json"
    )
    _write_json_object(snapshot_path, payload)


def _resolve_pair_best_config_path(
    data_root: Path,
    species: str,
    public_model_name: str,
) -> Path:
    return (
        data_root
        / species
        / "tuning"
        / normalize_public_model_name(public_model_name)
        / "pair"
        / "best_config.json"
    )


def _resolve_best_config_paths(
    *,
    data_root: Path,
    species: str,
    public_model_name: str,
    publication_tasks: tuple[str, ...],
) -> dict[str, Path]:
    """Resolve canonical best-config paths for one published model."""
    if publication_tasks == SITE_PUBLICATION_TASKS:
        base_dir = data_root / species / "tuning" / public_model_name
        return {
            "donor": base_dir / "donor" / "best_config.json",
            "acceptor": base_dir / "acceptor" / "best_config.json",
        }
    return {
        "pair": _resolve_pair_best_config_path(
            data_root=data_root,
            species=species,
            public_model_name=public_model_name,
        )
    }


def _resolve_task_checkpoint_from_payload(
    *,
    payload: Mapping[str, object],
    task: str,
    base_dir: Path,
    model_root: Path,
) -> Path | None:
    raw_path = extract_task_checkpoint_path(payload, task=task, base_dir=base_dir)
    if raw_path is None:
        return None
    return resolve_existing_checkpoint_path(raw_path, model_root_dir=model_root)


def _resolve_updated_or_payload_checkpoint(
    *,
    payload: Mapping[str, object],
    task: str,
    base_dir: Path,
    model_root: Path,
) -> Path | None:
    """Resolve the checkpoint for the side updated in the current publication."""
    return _resolve_task_checkpoint_from_payload(
        payload=payload,
        task=task,
        base_dir=base_dir,
        model_root=model_root,
    )


def _resolve_carry_forward_or_payload_checkpoint(
    *,
    project_root: Path,
    previous_entry: VersionHistoryEntry | None,
    payload: Mapping[str, object],
    task: str,
    base_dir: Path,
    model_root: Path,
    species: str,
    public_model_name: str,
) -> Path | None:
    """Resolve a carried-forward checkpoint with payload fallback.

    Prefer the previously published version for the untouched side because the
    current best payload may contain a stale raw checkpoint path that was moved
    during an earlier publication. Fall back to the payload only when no live
    previous version exists.
    """
    if previous_entry is not None:
        raw_value = ""
        if task == "donor":
            raw_value = previous_entry.donor_checkpoint_path
        elif task == "acceptor":
            raw_value = previous_entry.acceptor_checkpoint_path
        elif task == "pair":
            raw_value = previous_entry.pair_checkpoint_path
        if raw_value != "":
            try:
                return resolve_existing_checkpoint_path(
                    Path(raw_value),
                    model_root_dir=model_root,
                )
            except FileNotFoundError:
                archived_candidate = _resolve_archived_public_checkpoint(
                    project_root=project_root,
                    species=species,
                    public_model_name=public_model_name,
                    published_name=previous_entry.published_name,
                    task=task,
                )
                if archived_candidate is not None:
                    return archived_candidate
    return _resolve_task_checkpoint_from_payload(
        payload=payload,
        task=task,
        base_dir=base_dir,
        model_root=model_root,
    )


def _resolve_archived_public_checkpoint(
    *,
    project_root: Path,
    species: str,
    public_model_name: str,
    published_name: str,
    task: str,
) -> Path | None:
    """Return one archived published checkpoint when it still exists."""
    if task not in {"donor", "acceptor", "pair"}:
        return None
    archived_path = (
        project_root
        / "archive"
        / "versioned_artifacts"
        / species
        / public_model_name
        / published_name
        / "model"
        / task
        / f"{published_name}.pt"
    )
    if archived_path.is_file():
        return archived_path.resolve()
    return None


def _resolve_carry_forward_checkpoint(
    *,
    previous_entry: VersionHistoryEntry | None,
    fallback: Path,
    task: str,
) -> Path:
    if previous_entry is None:
        return fallback
    raw_value = ""
    if task == "donor":
        raw_value = previous_entry.donor_checkpoint_path
    elif task == "acceptor":
        raw_value = previous_entry.acceptor_checkpoint_path
    elif task == "pair":
        raw_value = previous_entry.pair_checkpoint_path
    candidate = Path(raw_value) if raw_value != "" else fallback
    if candidate.exists():
        return candidate.resolve()
    return fallback.resolve()


def _move_checkpoint_file(source: Path, destination: Path) -> None:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved:
        destination.parent.mkdir(parents=True, exist_ok=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.move(str(source_resolved), str(destination))


def _copy_checkpoint_file(source: Path, destination: Path) -> None:
    source_resolved = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_resolved == destination.resolve():
        return
    if destination.exists():
        destination.unlink()
    try:
        os.link(source_resolved, destination)
    except OSError:
        shutil.copy2(source_resolved, destination)


def _iter_public_artifact_paths(
    *,
    data_root: Path,
    model_root: Path,
    species: str,
    public_model_name: str,
    published_name: str,
) -> Iterable[tuple[Path, Path]]:
    if is_independent_public_model(public_model_name):
        yield (
            model_root / species / "donor" / f"{published_name}.pt",
            Path("model") / "donor" / f"{published_name}.pt",
        )
        yield (
            model_root / species / "acceptor" / f"{published_name}.pt",
            Path("model") / "acceptor" / f"{published_name}.pt",
        )
    else:
        yield (
            model_root / species / "pair" / f"{published_name}.pt",
            Path("model") / "pair" / f"{published_name}.pt",
        )
    for directory_name, suffix in (
        ("site_score", ".tsv"),
        ("intron_score", ".tsv"),
        ("trans_score", ".tsv"),
        ("eval_score", ".txt"),
    ):
        yield (
            data_root / species / directory_name / f"{published_name}{suffix}",
            Path("data") / directory_name / f"{published_name}{suffix}",
        )
    yield (
        data_root / species / "learning_metric" / f"{published_name}.train.json",
        Path("data") / "learning_metric" / f"{published_name}.train.json",
    )
    yield (
        data_root
        / species
        / "learning_metric"
        / f"{published_name}_learning_curve.png",
        Path("data") / "learning_metric" / f"{published_name}_learning_curve.png",
    )


def _write_json_object(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = relativize_path_fields(dict(payload))
    path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


def _resolve_version_history_checkpoint_path(
    *,
    raw_path: str,
    model_root: Path,
    allow_missing: bool = False,
) -> Path:
    """Resolve one checkpoint path stored in version history."""
    stripped = raw_path.strip()
    if stripped == "":
        raise FileNotFoundError("Version history is missing a checkpoint path.")
    raw_checkpoint_path = Path(stripped)
    if not allow_missing:
        return resolve_existing_checkpoint_path(
            raw_checkpoint_path,
            model_root_dir=model_root,
        )

    try:
        return resolve_existing_checkpoint_path(
            raw_checkpoint_path,
            model_root_dir=model_root,
        )
    except FileNotFoundError:
        return _resolve_checkpoint_reference_path(
            checkpoint_path=raw_checkpoint_path,
            model_root=model_root,
        )


def _resolve_checkpoint_reference_path(
    *,
    checkpoint_path: Path,
    model_root: Path,
) -> Path:
    """Resolve one checkpoint path reference without requiring file existence."""
    if checkpoint_path.is_absolute():
        path_parts = checkpoint_path.parts
        if "model" in path_parts:
            model_index = path_parts.index("model")
            relative_parts = path_parts[model_index + 1 :]
            if relative_parts:
                return model_root.joinpath(*relative_parts)
        return checkpoint_path

    relative_parts = checkpoint_path.parts
    if "model" in relative_parts:
        model_index = relative_parts.index("model")
        model_relative = relative_parts[model_index + 1 :]
        if model_relative:
            return model_root.joinpath(*model_relative)

    resolved_path = checkpoint_path
    if not resolved_path.is_absolute():
        resolved_path = (model_root / checkpoint_path.name).resolve(strict=False)
    return resolved_path


def _parse_version_number(published_name: str, public_model_name: str) -> int:
    prefix = f"{public_model_name}."
    if not published_name.startswith(prefix):
        raise ValueError(
            f"Published name does not match public model '{public_model_name}': "
            f"{published_name}"
        )
    return int(published_name[len(prefix) :])


def _stringify_scalar(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_score(value: object) -> float:
    """Coerce one objective score to float for direct comparison."""
    if isinstance(value, bool):
        return float("-inf")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float("-inf")


def _resolve_data_root(project_root: Path) -> Path:
    return _resolve_root_from_env(
        project_root=project_root,
        env_name="INTRONMODEL_DATA_ROOT",
        default_dirname="data",
    )


def _resolve_model_root(project_root: Path) -> Path:
    return _resolve_root_from_env(
        project_root=project_root,
        env_name="INTRONMODEL_MODEL_ROOT",
        default_dirname="model",
    )


def _resolve_root_from_env(
    *,
    project_root: Path,
    env_name: str,
    default_dirname: str,
) -> Path:
    """Resolve a data/model root from one environment override when available."""
    raw_value = os.environ.get(env_name, "").strip()
    if raw_value == "":
        return (project_root / default_dirname).resolve()
    root_path = Path(raw_value)
    if not root_path.is_absolute():
        root_path = project_root / root_path
    return root_path.resolve()
