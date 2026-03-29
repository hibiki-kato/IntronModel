"""Helpers for repository-relative path storage and resolution."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_HINTS: frozenset[str] = frozenset(
    {
        ".",
        "..",
        "analysis",
        "archive",
        "data",
        "model",
        "run",
        "src",
        "tests",
    }
)
_PATH_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "checkpoint",
        "log_file",
        "metrics_json",
        "output_dir",
        "path",
        "project_root",
        "source_best_config",
    }
)


def repository_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT


def relativize_path_string(
    raw_path: str,
    *,
    project_root: Path | None = None,
) -> str:
    """Return one path string relative to the repository root when possible."""
    stripped = raw_path.strip()
    if stripped == "":
        return ""
    path = Path(stripped)
    if not path.is_absolute():
        return os.path.normpath(str(path))
    root = repository_root() if project_root is None else project_root.resolve()
    absolute = path.resolve(strict=False)
    return os.path.relpath(absolute, root)


def resolve_path_string(
    raw_path: str,
    *,
    base_dir: Path,
    project_root: Path | None = None,
) -> Path:
    """Resolve one serialized path string from repository root or ``base_dir``."""
    stripped = raw_path.strip()
    if stripped == "":
        raise ValueError("Path string must not be empty.")

    path = Path(os.path.normpath(stripped))
    if path.is_absolute():
        return path.resolve(strict=False)

    root = repository_root() if project_root is None else project_root.resolve()
    root_candidate = (root / path).resolve(strict=False)
    base_candidate = (base_dir / path).resolve(strict=False)
    if root_candidate.exists():
        return root_candidate
    if base_candidate.exists():
        return base_candidate
    first_part = path.parts[0] if path.parts else ""
    if first_part in _REPOSITORY_ROOT_HINTS:
        return root_candidate
    return base_candidate


def relativize_path_fields(
    value: object,
    *,
    project_root: Path | None = None,
    field_name: str | None = None,
) -> object:
    """Recursively convert path-like fields to repository-relative strings."""
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            normalized[str(key)] = relativize_path_fields(
                item,
                project_root=project_root,
                field_name=str(key),
            )
        return normalized
    if isinstance(value, list):
        return [
            relativize_path_fields(
                item,
                project_root=project_root,
                field_name=field_name,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            relativize_path_fields(
                item,
                project_root=project_root,
                field_name=field_name,
            )
            for item in value
        ]
    if isinstance(value, str) and _looks_like_path_field(field_name):
        return relativize_path_string(value, project_root=project_root)
    return value


def _looks_like_path_field(field_name: str | None) -> bool:
    """Return whether one JSON field name should be normalized as a path."""
    if field_name is None:
        return False
    if field_name in _PATH_FIELD_NAMES:
        return True
    return field_name.endswith("_path") or field_name.endswith("_paths")
