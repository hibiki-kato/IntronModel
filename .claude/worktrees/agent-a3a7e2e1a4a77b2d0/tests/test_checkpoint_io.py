from __future__ import annotations

import json
from pathlib import Path

from util.checkpoint_io import (
    extract_checkpoint_paths,
    extract_task_checkpoint_path,
    normalize_checkpoint_path,
    read_json_object,
    resolve_existing_checkpoint_path,
)
from util.path_format import relativize_path_string


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


def test_normalize_checkpoint_path_supports_repository_relative_model_path(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model" / "Dmel" / "donor" / "demo.pt"
    base_dir = tmp_path / "data" / "Dmel" / "tuning" / "cnn_v3" / "donor"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"checkpoint")

    resolved = normalize_checkpoint_path(
        relativize_path_string(str(model_path)),
        base_dir=base_dir,
    )

    assert resolved == model_path.resolve()


def test_normalize_checkpoint_path_accepts_path_objects(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "metrics.json"
    checkpoint_path.write_text("{}", encoding="utf-8")

    resolved = normalize_checkpoint_path(checkpoint_path, base_dir=tmp_path)

    assert resolved == checkpoint_path.resolve()


def test_resolve_existing_checkpoint_path_relaxes_trailing_hash(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    expected = model_root / "Dmel" / "donor" / "cnn_v2_demo_h123456789abc.pt"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"checkpoint")

    resolved = resolve_existing_checkpoint_path(
        Path("/export/hibiki/intronmodel/model/Dmel/donor/cnn_v2_demo_habcdef123456.pt"),
        model_root_dir=model_root,
    )

    assert resolved == expected.resolve()


def test_resolve_existing_checkpoint_path_prefers_task_scoped_match(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    dmel = model_root / "Dmel" / "donor" / "cnn_v2_demo_h123456789abc.pt"
    mmus = model_root / "Mmus" / "donor" / "cnn_v2_demo_h123456789abc.pt"
    dmel.parent.mkdir(parents=True, exist_ok=True)
    mmus.parent.mkdir(parents=True, exist_ok=True)
    dmel.write_bytes(b"dmel")
    mmus.write_bytes(b"mmus")

    resolved = resolve_existing_checkpoint_path(
        Path("/export/hibiki/intronmodel/model/Dmel/donor/cnn_v2_demo_habcdef123456.pt"),
        model_root_dir=model_root,
    )

    assert resolved == dmel.resolve()
