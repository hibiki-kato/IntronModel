"""Shared helpers for checkpoint path and JSON payload handling."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Mapping, Sequence

TaskName = str
TASK_NAMES: tuple[TaskName, ...] = ("donor", "acceptor", "pair")
_HASHED_CHECKPOINT_RE = re.compile(r"^(?P<prefix>.+)_h[0-9a-f]+(?P<suffix>\.pt)$")


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
    task : str
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
    tasks: Sequence[TaskName] | None = None,
) -> dict[TaskName, Path]:
    """Extract task checkpoint paths.

    Parameters
    ----------
    payload : Mapping[str, object]
        JSON-like payload containing checkpoint fields.
    base_dir : Path
        Base directory for resolving relative paths.
    existing_only : bool, default=False
        Whether to keep only existing files.
    tasks : Sequence[str] | None, default=None
        Task names to scan. Default includes donor/acceptor/pair.

    Returns
    -------
    dict[str, Path]
        Mapping from task name to resolved checkpoint path.
    """
    task_names = tuple(tasks) if tasks is not None else TASK_NAMES
    out: dict[TaskName, Path] = {}
    for task in task_names:
        path = extract_task_checkpoint_path(payload, task=task, base_dir=base_dir)
        if path is None:
            continue
        if existing_only and not path.exists():
            continue
        out[task] = path
    return out


def resolve_existing_checkpoint_path(
    checkpoint_path: Path,
    *,
    model_root_dir: Path,
) -> Path:
    """Resolve one checkpoint path against the local model root.

    Parameters
    ----------
    checkpoint_path : Path
        Original checkpoint path from a JSON payload.
    model_root_dir : Path
        Local root directory that stores checkpoint files.

    Returns
    -------
    Path
        Resolved local checkpoint path.

    Raises
    ------
    FileNotFoundError
        If no matching local checkpoint file can be found.
    """
    if checkpoint_path.is_file():
        return checkpoint_path.resolve()

    search_roots: list[Path] = []
    scoped_root: Path | None = None
    path_parts = checkpoint_path.parts
    if "model" in path_parts:
        model_index = path_parts.index("model")
        relative_parts = path_parts[model_index + 1 :]
        if relative_parts:
            candidate = model_root_dir.joinpath(*relative_parts)
            if candidate.is_file():
                return candidate.resolve()
            if len(relative_parts) >= 2:
                scoped_root = model_root_dir.joinpath(relative_parts[0], relative_parts[1])
                search_roots.append(scoped_root)
    search_roots.append(model_root_dir)

    basename = checkpoint_path.name
    if basename != "":
        exact_match = _find_checkpoint_candidate(search_roots, basename)
        if exact_match is not None:
            return exact_match

        pattern = _build_relaxed_checkpoint_glob(basename)
        if pattern is not None:
            relaxed_roots = search_roots
            if scoped_root is not None:
                relaxed_roots = [scoped_root]
            relaxed_match = _find_checkpoint_candidate(relaxed_roots, pattern)
            if relaxed_match is not None:
                return relaxed_match

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def _find_checkpoint_candidate(
    search_roots: Sequence[Path],
    pattern: str,
) -> Path | None:
    """Return one deterministic checkpoint candidate that matches ``pattern``."""
    for root in search_roots:
        if not root.exists():
            continue
        candidates = sorted(
            root.rglob(pattern),
            key=lambda path: (len(path.parts), str(path)),
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def _build_relaxed_checkpoint_glob(basename: str) -> str | None:
    """Build one relaxed glob that ignores the trailing checkpoint hash."""
    match = _HASHED_CHECKPOINT_RE.match(basename)
    if match is None:
        return None
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    return f"{prefix}_h*{suffix}"
