from __future__ import annotations

import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cnn_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_cnn.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_cnn_pair_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_cnn_pair.sh"
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


def test_tune_cnn_pair_time_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_cnn_pair_time.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_cnn_resdil_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_cnn_resdil.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_tcn_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_tcn.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_bert_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_bert.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_dnabert_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_dnabert.sh"
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


def test_tune_bert_time_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_bert_time.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_tune_reservoir_time_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_reservoir_time.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_cnn_tuning_scripts_forward_val_frac() -> None:
    root = _project_root()
    script_names = (
        "tune_cnn.sh",
        "tune_cnn_time.sh",
        "tune_cnn_pair_time.sh",
        "tune_cnn_resdil.sh",
        "tune_cnn_resdil_time.sh",
        "tune_bert.sh",
        "tune_bert_time.sh",
        "tune_dnabert.sh",
        "tune_dnabert_time.sh",
        "tune_tcn.sh",
        "tune_tcn_time.sh",
        "tune_reservoir.sh",
        "tune_reservoir_time.sh",
    )
    for script_name in script_names:
        content = (root / "run" / script_name).read_text(encoding="utf-8")
        assert 'VAL_FRAC="0.1"' in content
        assert '"val_frac": ${VAL_FRAC}' in content


def test_run_scripts_are_shellcheck_parsable() -> None:
    root = _project_root()
    cnn = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_cnn.sh")],
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
    tune_pair_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_cnn_pair_time.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_cnn_time.sh")],
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
    cnn_resdil = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_cnn_resdil.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    cnn_pair = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_cnn_pair.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tcn = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_tcn.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    bert = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_bert.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    dnabert = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_dnabert.sh")],
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
    tune_reservoir_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_reservoir_time.sh")],
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
    tune_bert_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_bert_time.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_dnabert = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_dnabert.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_dnabert_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_dnabert_time.sh")],
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
    tune_bert_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_bert_time.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_reservoir_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_reservoir_time.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    eval_pair = subprocess.run(
        ["bash", "-n", str(root / "run" / "eval_trans_score_pair.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    plot_eval = subprocess.run(
        ["bash", "-n", str(root / "run" / "plot_eval.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cnn.returncode == 0, cnn.stderr
    assert tune.returncode == 0, tune.stderr
    assert tune_pair_time.returncode == 0, tune_pair_time.stderr
    assert tune_time.returncode == 0, tune_time.stderr
    assert tune_resdil_time.returncode == 0, tune_resdil_time.stderr
    assert cnn_resdil.returncode == 0, cnn_resdil.stderr
    assert cnn_pair.returncode == 0, cnn_pair.stderr
    assert tcn.returncode == 0, tcn.stderr
    assert bert.returncode == 0, bert.stderr
    assert dnabert.returncode == 0, dnabert.stderr
    assert reservoir.returncode == 0, reservoir.stderr
    assert tune_resdil.returncode == 0, tune_resdil.stderr
    assert tune_tcn.returncode == 0, tune_tcn.stderr
    assert tune_reservoir.returncode == 0, tune_reservoir.stderr
    assert tune_reservoir_time.returncode == 0, tune_reservoir_time.stderr
    assert tune_bert.returncode == 0, tune_bert.stderr
    assert tune_bert_time.returncode == 0, tune_bert_time.stderr
    assert tune_dnabert.returncode == 0, tune_dnabert.stderr
    assert tune_dnabert_time.returncode == 0, tune_dnabert_time.stderr
    assert tune_tcn_time.returncode == 0, tune_tcn_time.stderr
    assert tune_resdil_time.returncode == 0, tune_resdil_time.stderr
    assert tune_tcn_time.returncode == 0, tune_tcn_time.stderr
    assert tune_bert_time.returncode == 0, tune_bert_time.stderr
    assert tune_reservoir_time.returncode == 0, tune_reservoir_time.stderr
    assert eval_pair.returncode == 0, eval_pair.stderr
    assert plot_eval.returncode == 0, plot_eval.stderr
