"""Helpers for repository-relative path storage and resolution."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PathLike = str | Path
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
    raw_path: PathLike,
    *,
    project_root: Path | None = None,
) -> str:
    """Return one path string relative to the repository root when possible."""
    path = _coerce_path(raw_path)
    if str(path).strip() == "":
        return ""
    if not path.is_absolute():
        trimmed = _trim_embedded_repository_prefix(
            os.path.normpath(str(path)),
            project_root=project_root,
        )
        return trimmed
    root = repository_root() if project_root is None else project_root.resolve()
    absolute = path.resolve(strict=False)
    return os.path.relpath(absolute, root)


def resolve_path_string(
    raw_path: PathLike,
    *,
    base_dir: Path,
    project_root: Path | None = None,
) -> Path:
    """Resolve one serialized path string from repository root or ``base_dir``."""
    path = _coerce_path(raw_path)
    if str(path).strip() == "":
        raise ValueError("Path string must not be empty.")
    path = Path(os.path.normpath(str(path)))
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


def resolve_command_string(
    raw_command: PathLike,
    *,
    base_dir: Path,
    project_root: Path | None = None,
) -> str:
    """Resolve one executable setting without misclassifying command names.

    Parameters
    ----------
    raw_command:
        Serialized executable value from configuration. This may be an absolute
        path, a repository-relative path, or a bare command name such as
        ``python3``.
    base_dir:
        Directory used to resolve relative filesystem paths.
    project_root:
        Optional repository root override for repository-relative paths.

    Returns
    -------
    str
        Resolved executable path when the value is path-like, or the original
        command name when it is a PATH-resolved executable token.

    Raises
    ------
    ValueError
        If ``raw_command`` is empty after trimming whitespace.
    """
    if isinstance(raw_command, Path):
        return str(
            resolve_path_string(
                raw_command,
                base_dir=base_dir,
                project_root=project_root,
            )
        )

    stripped = raw_command.strip()
    if stripped == "":
        raise ValueError("Command string must not be empty.")
    if _looks_like_filesystem_path(stripped):
        return str(
            resolve_path_string(
                stripped,
                base_dir=base_dir,
                project_root=project_root,
            )
        )
    if shutil.which(stripped) is not None:
        return stripped
    return stripped


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


def _coerce_path(raw_path: PathLike) -> Path:
    """Normalize one path-like value into ``Path``."""
    if isinstance(raw_path, Path):
        return raw_path
    return Path(raw_path.strip())


def _looks_like_filesystem_path(raw_value: str) -> bool:
    """Return whether one serialized value should be treated as a path."""
    if raw_value in {".", ".."}:
        return True
    if raw_value.startswith("./") or raw_value.startswith("../"):
        return True
    if os.path.sep in raw_value:
        return True
    if os.path.altsep is not None and os.path.altsep in raw_value:
        return True
    return Path(raw_value).is_absolute()


def _trim_embedded_repository_prefix(
    raw_path: str,
    *,
    project_root: Path | None = None,
) -> str:
    """Trim one serialized path down to the repository-relative suffix.

    This handles legacy metadata strings such as
    ``../../../../export/<user>/intronmodel/data/...`` which are technically
    relative paths but clearly embed an old absolute repository root.
    """
    normalized = os.path.normpath(raw_path)
    path = Path(normalized)
    parts = path.parts
    if not parts:
        return normalized

    root = repository_root() if project_root is None else project_root.resolve()
    repo_markers = {root.name.lower(), "intronmodel"}
    lowered_parts = [part.lower() for part in parts]
    for index, part in enumerate(lowered_parts):
        if part not in repo_markers:
            continue
        suffix_parts = parts[index + 1 :]
        if suffix_parts:
            return os.path.normpath(str(Path(*suffix_parts)))

    for index, part in enumerate(parts):
        if part in _REPOSITORY_ROOT_HINTS and part not in {".", ".."}:
            suffix_parts = parts[index:]
            return os.path.normpath(str(Path(*suffix_parts)))
    return normalized
