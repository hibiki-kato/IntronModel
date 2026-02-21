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


def test_cnn_resdil_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "cnn_resdil.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_tcn_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tcn.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_bert_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "bert.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_dnabert_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "dnabert.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_reservoir_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "reservoir.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_tune_cnn_resdil_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_cnn_resdil.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_tune_tcn_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_tcn.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_tune_reservoir_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_reservoir.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_tune_bert_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_bert.sh"
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
    cnn_resdil = subprocess.run(
        ["bash", "-n", str(root / "run" / "cnn_resdil.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tcn = subprocess.run(
        ["bash", "-n", str(root / "run" / "tcn.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    bert = subprocess.run(
        ["bash", "-n", str(root / "run" / "bert.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    dnabert = subprocess.run(
        ["bash", "-n", str(root / "run" / "dnabert.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    reservoir = subprocess.run(
        ["bash", "-n", str(root / "run" / "reservoir.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_resdil = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_cnn_resdil.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_tcn = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_tcn.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_reservoir = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_reservoir.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_bert = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_bert.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_resdil_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_cnn_resdil_time.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_tcn_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_tcn_time.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cnn.returncode == 0, cnn.stderr
    assert tune.returncode == 0, tune.stderr
    assert cnn_resdil.returncode == 0, cnn_resdil.stderr
    assert tcn.returncode == 0, tcn.stderr
    assert bert.returncode == 0, bert.stderr
    assert dnabert.returncode == 0, dnabert.stderr
    assert reservoir.returncode == 0, reservoir.stderr
    assert tune_resdil.returncode == 0, tune_resdil.stderr
    assert tune_tcn.returncode == 0, tune_tcn.stderr
    assert tune_reservoir.returncode == 0, tune_reservoir.stderr
    assert tune_bert.returncode == 0, tune_bert.stderr
    assert tune_resdil_time.returncode == 0, tune_resdil_time.stderr
    assert tune_tcn_time.returncode == 0, tune_tcn_time.stderr
