"""Utilities for publishing versioned best artifacts for active models.

This module manages a small public-artifact layer for the active wrappers:

- ``cnn_v2`` (shared donor/acceptor publication)
- ``cnn_v3`` (shared donor/acceptor publication)
- ``cnn_pair_v2`` (public pair publication)
- ``cnn_pair_v3`` (public pair publication)

Publication happens when a canonical ``best_config.json`` improves. The latest
and previous versions remain live under ``data/`` and ``model/`` while older
versions move under ``archive/versioned_artifacts/``.
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

from util.checkpoint_io import (
    extract_task_checkpoint_path,
    read_json_object,
    resolve_existing_checkpoint_path,
)
from util.path_format import relativize_path_fields, relativize_path_string

INDEPENDENT_PUBLIC_MODEL_NAMES: frozenset[str] = frozenset({"cnn_v2", "cnn_v3"})
PAIR_PUBLIC_MODEL_NAMES: frozenset[str] = frozenset(
    {"cnn_pair_v2", "cnn_pair_v3"}
)
ACTIVE_PUBLIC_MODEL_NAMES: frozenset[str] = frozenset(
    {*INDEPENDENT_PUBLIC_MODEL_NAMES, *PAIR_PUBLIC_MODEL_NAMES}
)
LEGACY_PUBLIC_OUTPUT_STEMS: dict[str, tuple[str, ...]] = {
    "cnn_pair_v2": ("cnn_v2_pair",),
}
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
LIVE_VERSION_KEEP_COUNT: int = 2
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


def is_active_public_model(model_name: str) -> bool:
    """Return whether one model participates in version publication."""
    return normalize_public_model_name(model_name) in ACTIVE_PUBLIC_MODEL_NAMES


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
                donor_checkpoint_path=str(
                    row.get("donor_checkpoint_path", "")
                ).strip(),
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
    public_model_name = normalize_public_model_name(model_name)
    if public_model_name not in ACTIVE_PUBLIC_MODEL_NAMES:
        return None

    data_root = _resolve_data_root(project_root)
    model_root = _resolve_model_root(project_root)
    latest_entry = resolve_latest_live_history_entry(
        data_root=data_root,
        species=species,
        model_name=public_model_name,
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

    if public_model_name in INDEPENDENT_PUBLIC_MODEL_NAMES:
        donor_path = _resolve_version_history_checkpoint_path(
            raw_path=latest_entry.donor_checkpoint_path,
            model_root=model_root,
        )
        acceptor_path = _resolve_version_history_checkpoint_path(
            raw_path=latest_entry.acceptor_checkpoint_path,
            model_root=model_root,
        )
        assets["donor_checkpoint_path"] = str(donor_path)
        assets["acceptor_checkpoint_path"] = str(acceptor_path)
    else:
        pair_path = _resolve_version_history_checkpoint_path(
            raw_path=latest_entry.pair_checkpoint_path,
            model_root=model_root,
        )
        assets["pair_checkpoint_path"] = str(pair_path)
    return assets


def ensure_publication_seed(
    *,
    project_root: Path,
    species: str,
    model_name: str,
) -> str | None:
    """Seed the initial ``.01`` publication when history does not exist yet."""
    public_model_name = normalize_public_model_name(model_name)
    if public_model_name not in ACTIVE_PUBLIC_MODEL_NAMES:
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
    if public_model_name not in ACTIVE_PUBLIC_MODEL_NAMES:
        return None

    data_root = _resolve_data_root(project_root)
    model_root = _resolve_model_root(project_root)
    history = read_version_history(data_root, species, public_model_name)
    next_version = (
        INITIAL_VERSION_NUMBER if not history else history[-1].version + 1
    )
    published_name = format_published_name(public_model_name, next_version)
    published_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    if public_model_name in INDEPENDENT_PUBLIC_MODEL_NAMES:
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
    _archive_old_versions(
        project_root=project_root,
        data_root=data_root,
        model_root=model_root,
        species=species,
        public_model_name=public_model_name,
        history_entries=updated_history,
    )
    return updated_history[-1]


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
        data_root / species / "tuning" / public_model_name / "donor" / "best_config.json"
    )
    acceptor_path = (
        data_root / species / "tuning" / public_model_name / "acceptor" / "best_config.json"
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
        data_root
        / species
        / "tuning"
        / public_model_name
        / "pair"
        / "best_config.json"
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


def _archive_old_versions(
    *,
    project_root: Path,
    data_root: Path,
    model_root: Path,
    species: str,
    public_model_name: str,
    history_entries: list[VersionHistoryEntry],
) -> None:
    if len(history_entries) <= LIVE_VERSION_KEEP_COUNT:
        write_version_history(data_root, species, public_model_name, history_entries)
        return

    live_entries = history_entries[-LIVE_VERSION_KEEP_COUNT:]
    live_names = {entry.published_name for entry in live_entries}
    archive_root = (
        project_root
        / "archive"
        / "versioned_artifacts"
        / species
        / public_model_name
    )
    updated_entries: list[VersionHistoryEntry] = []
    for entry in history_entries:
        archive_status = "live" if entry.published_name in live_names else "archived"
        if archive_status == "archived":
            _archive_one_version(
                archive_root=archive_root / entry.published_name,
                data_root=data_root,
                model_root=model_root,
                species=species,
                public_model_name=public_model_name,
                published_name=entry.published_name,
            )
        updated_entries.append(
            VersionHistoryEntry(
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
        )
    write_version_history(data_root, species, public_model_name, updated_entries)


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
    if public_model_name in INDEPENDENT_PUBLIC_MODEL_NAMES:
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
) -> Path:
    """Resolve one checkpoint path stored in version history."""
    stripped = raw_path.strip()
    if stripped == "":
        raise FileNotFoundError("Version history is missing a checkpoint path.")
    return resolve_existing_checkpoint_path(Path(stripped), model_root_dir=model_root)


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
