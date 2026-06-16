from __future__ import annotations

from pathlib import Path

import pytest

from models import dnabert


def test_resolve_pretrained_model_name_raises_for_missing_absolute_path() -> None:
    missing_path = "/tmp/intronmodel_test_missing_dnabert_pretrained"
    with pytest.raises(FileNotFoundError) as exc_info:
        dnabert._resolve_pretrained_model_name(missing_path)
    message = str(exc_info.value)
    assert "does not exist" in message
    assert missing_path in message


def test_resolve_pretrained_model_name_returns_existing_local_path(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "dnabert2-117m"
    checkpoint_dir.mkdir()
    resolved = dnabert._resolve_pretrained_model_name(str(checkpoint_dir))
    assert resolved == str(checkpoint_dir)


def test_resolve_pretrained_model_name_preserves_hf_repo_id() -> None:
    repo_id = "zhihan1996/DNABERT-2-117M"
    resolved = dnabert._resolve_pretrained_model_name(repo_id)
    assert resolved == repo_id


def test_resolve_pretrained_model_name_remaps_stale_repo_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "model" / "pretrained" / "dnabert2-117m"
    checkpoint_dir.mkdir(parents=True)
    monkeypatch.setattr(dnabert, "repository_root", lambda: tmp_path)

    resolved = dnabert._resolve_pretrained_model_name(
        "/export/hibiki/intronmodel/model/pretrained/dnabert2-117m"
    )

    assert resolved == str(checkpoint_dir)
