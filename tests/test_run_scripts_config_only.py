from __future__ import annotations

import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cnn_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "cnn.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_tune_cnn_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_cnn.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_run_scripts_are_shellcheck_parsable() -> None:
    root = _project_root()
    cnn = subprocess.run(
        ["bash", "-n", str(root / "run" / "cnn.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_cnn.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cnn.returncode == 0, cnn.stderr
    assert tune.returncode == 0, tune.stderr
