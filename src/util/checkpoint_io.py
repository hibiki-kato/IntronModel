"""Shared helpers for checkpoint path and JSON payload handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Mapping

TaskName = Literal["donor", "acceptor"]
TASK_NAMES: tuple[TaskName, ...] = ("donor", "acceptor")


def read_json_object(path: Path) -> dict[str, object] | None:
    """Read one JSON object file.

    Parameters
    ----------
    path : Path
        JSON file path.

    Returns
    -------
    dict[str, object] | None
        Parsed object when valid JSON object exists; otherwise ``None``.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def normalize_checkpoint_path(raw_path: str, *, base_dir: Path) -> Path:
    """Normalize one checkpoint path string to an absolute path.

    Parameters
    ----------
    raw_path : str
        Raw checkpoint path from JSON.
    base_dir : Path
        Base directory for resolving relative paths.

    Returns
    -------
    Path
        Absolute normalized path.
    """
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def extract_task_checkpoint_path(
    payload: Mapping[str, object],
    *,
    task: TaskName,
    base_dir: Path,
) -> Path | None:
    """Extract one task checkpoint path from one payload.

    Parameters
    ----------
    payload : Mapping[str, object]
        JSON-like payload containing checkpoint fields.
    task : TaskName
        Target task name.
    base_dir : Path
        Base directory for resolving relative paths.

    Returns
    -------
    Path | None
        Resolved path when found; otherwise ``None``.
    """
    key_name = f"{task}_checkpoint_path"
    raw_path = payload.get(key_name)
    if isinstance(raw_path, str) and raw_path.strip():
        return normalize_checkpoint_path(raw_path, base_dir=base_dir)

    task_payload = payload.get(task)
    if not isinstance(task_payload, dict):
        return None
    nested = task_payload.get("checkpoint")
    if isinstance(nested, str) and nested.strip():
        return normalize_checkpoint_path(nested, base_dir=base_dir)
    return None


def extract_checkpoint_paths(
    payload: Mapping[str, object],
    *,
    base_dir: Path,
    existing_only: bool = False,
) -> dict[TaskName, Path]:
    """Extract task checkpoint paths for donor and acceptor.

    Parameters
    ----------
    payload : Mapping[str, object]
        JSON-like payload containing checkpoint fields.
    base_dir : Path
        Base directory for resolving relative paths.
    existing_only : bool, default=False
        Whether to keep only existing files.

    Returns
    -------
    dict[TaskName, Path]
        Mapping from task name to resolved checkpoint path.
    """
    out: dict[TaskName, Path] = {}
    for task in TASK_NAMES:
        path = extract_task_checkpoint_path(payload, task=task, base_dir=base_dir)
        if path is None:
            continue
        if existing_only and not path.exists():
            continue
        out[task] = path
    return out
