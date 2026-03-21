"""Validation protocol metadata helpers.

This module defines a stable representation for validation protocol metadata
and computes a short signature for comparison-safe model selection.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

LEGACY_VALIDATION_SIGNATURE: str = "legacy_unknown"
VALIDATION_SIGNATURE_CHARS: int = 12


def _normalize_for_json(value: object) -> object:
    """Normalize values for deterministic JSON serialization."""
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key in sorted(value):
            normalized[str(key)] = _normalize_for_json(value[key])
        return normalized
    if isinstance(value, list):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, float):
        return float(format(value, ".12g"))
    return value


def canonical_json_dumps(payload: Mapping[str, object]) -> str:
    """Return deterministic JSON text for protocol hashing."""
    normalized = _normalize_for_json(dict(payload))
    return json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def compute_validation_signature(validation_protocol: Mapping[str, object]) -> str:
    """Compute a short SHA-1 signature from canonicalized protocol JSON."""
    canonical = canonical_json_dumps(validation_protocol)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[
        :VALIDATION_SIGNATURE_CHARS
    ]


def _normalize_source_path(path: str | None) -> str:
    """Normalize one optional path into a compact stable identifier."""
    if path is None:
        return ""
    stripped = path.strip()
    if not stripped:
        return ""
    return os.path.normpath(stripped)


def _build_file_signature(path: str) -> dict[str, object]:
    """Build one compact file-signature mapping for protocol comparison."""
    normalized_path = _normalize_source_path(path)
    resolved_path = os.path.realpath(normalized_path)
    signature: dict[str, object] = {"path": resolved_path}
    try:
        stat_result = Path(resolved_path).stat()
    except OSError:
        signature["exists"] = False
        return signature
    signature["exists"] = True
    signature["size_bytes"] = int(stat_result.st_size)
    signature["mtime_ns"] = int(stat_result.st_mtime_ns)
    return signature


def _resolve_pair_extra_negative_paths(
    *,
    train_pos_path: str | None,
    train_neg_path: str | None,
    include_pair_mixed_negatives: bool,
) -> tuple[str, ...]:
    """Resolve extra mixed-negative paths used by pair-task loaders."""
    if not include_pair_mixed_negatives:
        return tuple()
    normalized_pos = _normalize_source_path(train_pos_path)
    normalized_neg = _normalize_source_path(train_neg_path)
    if not normalized_pos or not normalized_neg:
        return tuple()
    try:
        from util.data_proc import discover_default_pair_extra_negative_paths
    except Exception:
        return tuple()
    try:
        return discover_default_pair_extra_negative_paths(
            pos_path=normalized_pos,
            neg_path=normalized_neg,
        )
    except Exception:
        return tuple()


def _build_train_source_signature(
    *,
    train_pos_path: str | None,
    train_neg_path: str | None,
    include_pair_mixed_negatives: bool,
) -> dict[str, object]:
    """Build one deterministic train-source signature payload."""
    normalized_pos = _normalize_source_path(train_pos_path)
    normalized_neg = _normalize_source_path(train_neg_path)

    signature: dict[str, object] = {
        "train_pos": (
            _build_file_signature(normalized_pos) if normalized_pos else {"path": ""}
        ),
        "train_neg": (
            _build_file_signature(normalized_neg) if normalized_neg else {"path": ""}
        ),
    }
    extra_negative_paths = _resolve_pair_extra_negative_paths(
        train_pos_path=normalized_pos,
        train_neg_path=normalized_neg,
        include_pair_mixed_negatives=include_pair_mixed_negatives,
    )
    signature["pair_extra_negatives"] = [
        _build_file_signature(path) for path in extra_negative_paths
    ]
    return signature


def build_validation_protocol(
    *,
    val_frac: float | None,
    seed: int | None,
    train_pos_path: str | None,
    train_neg_path: str | None,
    metric_primary: str,
    split_type: str = "stratified_site",
    include_pair_mixed_negatives: bool = False,
) -> dict[str, object]:
    """Build the standard validation protocol payload."""
    normalized_pos = _normalize_source_path(train_pos_path)
    normalized_neg = _normalize_source_path(train_neg_path)
    return {
        "split_type": split_type,
        "val_frac": None if val_frac is None else float(val_frac),
        "seed": None if seed is None else int(seed),
        "include_pair_mixed_negatives": include_pair_mixed_negatives,
        "train_source": {
            "train_pos_path": normalized_pos,
            "train_neg_path": normalized_neg,
        },
        "train_source_signature": _build_train_source_signature(
            train_pos_path=normalized_pos,
            train_neg_path=normalized_neg,
            include_pair_mixed_negatives=include_pair_mixed_negatives,
        ),
        "metric_primary": metric_primary,
    }
