"""Validation protocol metadata helpers.

This module defines a stable representation for validation protocol metadata
and computes a short signature for comparison-safe model selection.
"""

from __future__ import annotations

import hashlib
import json
import os
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


def build_validation_protocol(
    *,
    val_frac: float | None,
    seed: int | None,
    train_pos_path: str | None,
    train_neg_path: str | None,
    metric_primary: str,
    split_type: str = "stratified_site",
) -> dict[str, object]:
    """Build the standard validation protocol payload."""
    return {
        "split_type": split_type,
        "val_frac": None if val_frac is None else float(val_frac),
        "seed": None if seed is None else int(seed),
        "train_source": {
            "train_pos_path": _normalize_source_path(train_pos_path),
            "train_neg_path": _normalize_source_path(train_neg_path),
        },
        "metric_primary": metric_primary,
    }

