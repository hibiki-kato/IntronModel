from __future__ import annotations

import json
from pathlib import Path

from util.checkpoint_io import (
    extract_checkpoint_paths,
    extract_task_checkpoint_path,
    normalize_checkpoint_path,
    read_json_object,
)


def test_read_json_object_returns_object_for_valid_json(tmp_path: Path) -> None:
    payload = {"status": "ok", "value": 1}
    path = tmp_path / "valid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = read_json_object(path)

    assert loaded == payload


def test_read_json_object_returns_none_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{not-json}", encoding="utf-8")

    loaded = read_json_object(path)

    assert loaded is None


def test_extract_task_checkpoint_path_supports_relative_and_nested(
    tmp_path: Path,
) -> None:
    direct_path = tmp_path / "direct.pt"
    direct_path.write_bytes(b"direct")
    nested_path = tmp_path / "nested.pt"
    nested_path.write_bytes(b"nested")

    payload_direct = {"donor_checkpoint_path": "direct.pt"}
    payload_nested = {"acceptor": {"checkpoint": "nested.pt"}}

    direct = extract_task_checkpoint_path(
        payload_direct,
        task="donor",
        base_dir=tmp_path,
    )
    nested = extract_task_checkpoint_path(
        payload_nested,
        task="acceptor",
        base_dir=tmp_path,
    )

    assert direct == direct_path.resolve()
    assert nested == nested_path.resolve()


def test_extract_checkpoint_paths_respects_existing_only(tmp_path: Path) -> None:
    existing = tmp_path / "exists.pt"
    existing.write_bytes(b"x")
    payload = {
        "donor_checkpoint_path": "exists.pt",
        "acceptor_checkpoint_path": "missing.pt",
    }

    all_paths = extract_checkpoint_paths(
        payload,
        base_dir=tmp_path,
        existing_only=False,
    )
    existing_only_paths = extract_checkpoint_paths(
        payload,
        base_dir=tmp_path,
        existing_only=True,
    )

    assert set(all_paths.keys()) == {"donor", "acceptor"}
    assert set(existing_only_paths.keys()) == {"donor"}
    assert existing_only_paths["donor"] == existing.resolve()


def test_normalize_checkpoint_path_resolves_relative_path(tmp_path: Path) -> None:
    resolved = normalize_checkpoint_path("relative/file.pt", base_dir=tmp_path)
    assert resolved == (tmp_path / "relative" / "file.pt").resolve()


def test_extract_checkpoint_paths_supports_pair_task(tmp_path: Path) -> None:
    pair = tmp_path / "pair.pt"
    pair.write_bytes(b"pair")
    payload = {"pair_checkpoint_path": "pair.pt"}

    paths = extract_checkpoint_paths(payload, base_dir=tmp_path, existing_only=True)

    assert paths == {"pair": pair.resolve()}
