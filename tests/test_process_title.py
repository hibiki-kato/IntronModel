from __future__ import annotations

import sys
import types

import pytest

from util.process_title import apply_process_title, apply_process_title_from_env


def test_apply_process_title_returns_false_for_blank_title() -> None:
    assert apply_process_title("") is False


def test_apply_process_title_from_env_uses_setproctitle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    fake_module = types.SimpleNamespace(
        setproctitle=lambda title: captured.setdefault("title", title)
    )
    monkeypatch.setenv("INTRONMODEL_PROCESS_TITLE", "visible title")
    monkeypatch.setitem(sys.modules, "setproctitle", fake_module)

    applied = apply_process_title_from_env()

    assert applied is True
    assert captured["title"] == "visible title"


def test_apply_process_title_from_env_falls_back_to_linux_process_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_PROCESS_TITLE", "visible title")
    monkeypatch.delitem(sys.modules, "setproctitle", raising=False)
    monkeypatch.setattr(
        "util.process_title._apply_linux_process_name",
        lambda title: title == "visible title",
    )

    original_import = __import__

    def _raising_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "setproctitle":
            raise ImportError("missing setproctitle")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raising_import)

    assert apply_process_title_from_env() is True


def test_apply_process_title_from_env_returns_false_when_no_backend_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_PROCESS_TITLE", "visible title")
    monkeypatch.delitem(sys.modules, "setproctitle", raising=False)
    monkeypatch.setattr("util.process_title._apply_linux_process_name", lambda _: False)

    original_import = __import__

    def _raising_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "setproctitle":
            raise ImportError("missing setproctitle")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raising_import)

    assert apply_process_title_from_env() is False
