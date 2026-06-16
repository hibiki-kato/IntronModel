from __future__ import annotations

from pathlib import Path

import pytest

from util import path_format


def test_resolve_command_string_keeps_bare_executable_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        path_format.shutil,
        "which",
        lambda raw: "/usr/bin/python3" if raw == "python3" else None,
    )

    resolved = path_format.resolve_command_string(
        "python3",
        base_dir=tmp_path,
    )

    assert resolved == "python3"


def test_resolve_command_string_resolves_relative_path(
    tmp_path: Path,
) -> None:
    executable_path = tmp_path / "bin" / "python3"
    executable_path.parent.mkdir(parents=True, exist_ok=True)
    executable_path.write_text("", encoding="utf-8")

    resolved = path_format.resolve_command_string(
        "bin/python3",
        base_dir=tmp_path,
    )

    assert resolved == str(executable_path.resolve())
