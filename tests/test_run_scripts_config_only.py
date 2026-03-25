from __future__ import annotations

import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _has_top_level_assignment(content: str, name: str) -> bool:
    """Return True when one exact top-level assignment exists."""
    prefix = f"{name}="
    return any(line.startswith(prefix) for line in content.splitlines())


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


def test_bilstm_pair_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_bilstm_pair.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_run_bilstm_pair_sh_includes_tuned_auto_and_common_runtime_keys() -> None:
    content = (_project_root() / "run" / "run_bilstm_pair.sh").read_text(
        encoding="utf-8"
    )
    assert 'USE_TUNED_HPARAMS="auto"' in content
    assert 'COMPILE_MODE="auto"' in content
    assert 'MPS_MAX_BATCH_SIZE="2048"' in content
    assert "INFER_COMPILE=" not in content
    assert "INFER_COMPILE_MODE=" not in content


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


def test_run_cnn_v2_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_cnn_v2.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_run_cnn_v2_sh_leaves_mask_to_best_config() -> None:
    content = (_project_root() / "run" / "run_cnn_v2.sh").read_text(
        encoding="utf-8"
    )
    assert 'INTRONMODEL_AUTO_TMUX="on"' in content
    assert "intronmodel_enable_auto_tmux" in content
    assert 'source "${SCRIPT_DIR}/lib/tuned_config.sh"' in content
    for unwanted in (
        "DONOR_LEN",
        "ACCEPTOR_LEN",
        "TRAIN_TARGET",
        "EPOCHS",
        "MAX_EPOCHS",
        "BATCH_SIZE",
        "LR",
        "LOSS",
        "INPUT_MODE",
        "PAIR_MODE",
        "EMBEDDING_DIM",
        "BPE_PRETRAINED_MODEL_NAME",
        "DROPOUT",
        "WEIGHT_DECAY",
        "ETA_MIN_RATIO",
        "VAL_FRAC",
        "GRAD_CLIP",
        "POS_WEIGHT_CAP",
        "FOCAL_GAMMA",
        "F1_LAMBDA",
        "ASYM_GAMMA_POS",
        "ASYM_GAMMA_NEG",
        "SEED",
    ):
        assert not _has_top_level_assignment(content, unwanted)
    assert "--sequence_transform" not in content


def test_run_cnn_v2_pair_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_cnn_v2_pair.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_run_cnn_v2_pair_sh_leaves_mask_to_best_config() -> None:
    content = (_project_root() / "run" / "run_cnn_v2_pair.sh").read_text(
        encoding="utf-8"
    )
    assert 'INTRONMODEL_AUTO_TMUX="on"' in content
    assert "intronmodel_enable_auto_tmux" in content
    assert 'source "${SCRIPT_DIR}/lib/tuned_config.sh"' in content
    for unwanted in (
        "DONOR_LEN",
        "ACCEPTOR_LEN",
        "TRAIN_TARGET",
        "EPOCHS",
        "MAX_EPOCHS",
        "BATCH_SIZE",
        "LR",
        "LOSS",
        "INPUT_MODE",
        "PAIR_MODE",
        "FUSION_MODE",
        "EMBEDDING_DIM",
        "BPE_PRETRAINED_MODEL_NAME",
        "CONV_CHANNELS",
        "KERNEL_SIZES",
        "DONOR_CONV_CHANNELS",
        "ACCEPTOR_CONV_CHANNELS",
        "DONOR_KERNEL_SIZES",
        "ACCEPTOR_KERNEL_SIZES",
        "MAX_POOL_SIZE",
        "CONV_STRIDE",
        "HEAD_TYPE",
        "FC_HIDDEN",
        "DROPOUT",
        "WEIGHT_DECAY",
        "ETA_MIN_RATIO",
        "VAL_FRAC",
        "GRAD_CLIP",
        "POS_WEIGHT_CAP",
        "FOCAL_GAMMA",
        "F1_LAMBDA",
        "ASYM_GAMMA_POS",
        "ASYM_GAMMA_NEG",
        "SEED",
    ):
        assert not _has_top_level_assignment(content, unwanted)
    assert "--sequence_transform" not in content


def test_run_cnn_v3_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_cnn_v3.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_run_cnn_v3_sh_includes_auto_tmux() -> None:
    content = (_project_root() / "run" / "run_cnn_v3.sh").read_text(
        encoding="utf-8"
    )
    assert 'INTRONMODEL_AUTO_TMUX="on"' in content
    assert "intronmodel_enable_auto_tmux" in content


def test_tune_cnn_v2_time_omits_max_model_params_and_adds_input_mode() -> None:
    content = (_project_root() / "run" / "tune_cnn_v2_time.sh").read_text(
        encoding="utf-8"
    )
    assert "MAX_MODEL_PARAMS" not in content
    assert '"input_mode": {' in content
    assert '"mask": {' not in content
    assert '"sequence_transform": {' not in content
    assert "MASK_MODE" not in content
    assert "TAG=" not in content


def test_tune_cnn_v2_pair_time_omits_max_model_params() -> None:
    content = (_project_root() / "run" / "tune_cnn_v2_pair_time.sh").read_text(
        encoding="utf-8"
    )
    assert "MAX_MODEL_PARAMS" not in content
    assert '"mask": {' in content
    assert '"sequence_transform": {' not in content
    assert "MASK_MODE" not in content
    assert "TAG=" not in content


def test_tune_bilstm_pair_time_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_bilstm_pair_time.sh"
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


def test_dnabert_pair_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_dnabert_pair.sh"
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


def test_markov_xgboost_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_markov_xgboost.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_run_clean_pt_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_clean_pt.sh"
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


def test_tune_markov_xgboost_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_markov_xgboost.sh"
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


def test_tuning_scripts_forward_seed_and_val_frac() -> None:
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
        "tune_markov_xgboost.sh",
    )
    for script_name in script_names:
        content = (root / "run" / script_name).read_text(encoding="utf-8")
        assert 'VAL_FRAC="0.1"' in content
        assert '"val_frac": ${VAL_FRAC}' in content
        assert '"seed": ${' in content


def test_run_tcn_sh_includes_head_type_config() -> None:
    content = (_project_root() / "run" / "run_tcn.sh").read_text(encoding="utf-8")
    assert 'HEAD_TYPE="gap"' in content


def test_run_cnn_v2_sh_includes_gpu_parallel_config() -> None:
    content = (_project_root() / "run" / "run_cnn_v2.sh").read_text(
        encoding="utf-8"
    )
    assert 'GPU_IDS="auto"' in content
    assert 'MAX_PARALLEL_TRIALS="auto"' in content


def test_run_cnn_v2_pair_sh_includes_gpu_parallel_config() -> None:
    content = (_project_root() / "run" / "run_cnn_v2_pair.sh").read_text(
        encoding="utf-8"
    )
    assert 'GPU_IDS="auto"' in content
    assert 'MAX_PARALLEL_TRIALS="auto"' in content


def test_run_cnn_v3_sh_includes_gpu_config() -> None:
    content = (_project_root() / "run" / "run_cnn_v3.sh").read_text(
        encoding="utf-8"
    )
    assert 'GPU_IDS="auto"' in content


def test_run_dnabert_sh_sets_default_process_title() -> None:
    content = (_project_root() / "run" / "run_dnabert.sh").read_text(encoding="utf-8")
    assert 'PROCESS_TITLE="use? email me"' in content


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
    tune_bilstm_pair_time = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_bilstm_pair_time.sh")],
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
    bilstm_pair = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_bilstm_pair.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    cnn_v2 = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_cnn_v2.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    cnn_v2_pair = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_cnn_v2_pair.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    cnn_v3 = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_cnn_v3.sh")],
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
    dnabert_pair = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_dnabert_pair.sh")],
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
    markov_xgb = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_markov_xgboost.sh")],
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
    assert tune_bilstm_pair_time.returncode == 0, tune_bilstm_pair_time.stderr
    assert tune_time.returncode == 0, tune_time.stderr
    assert tune_resdil_time.returncode == 0, tune_resdil_time.stderr
    assert cnn_resdil.returncode == 0, cnn_resdil.stderr
    assert cnn_pair.returncode == 0, cnn_pair.stderr
    assert bilstm_pair.returncode == 0, bilstm_pair.stderr
    assert cnn_v2.returncode == 0, cnn_v2.stderr
    assert cnn_v2_pair.returncode == 0, cnn_v2_pair.stderr
    assert cnn_v3.returncode == 0, cnn_v3.stderr
    assert tcn.returncode == 0, tcn.stderr
    assert bert.returncode == 0, bert.stderr
    assert dnabert.returncode == 0, dnabert.stderr
    assert dnabert_pair.returncode == 0, dnabert_pair.stderr
    assert reservoir.returncode == 0, reservoir.stderr
    assert markov_xgb.returncode == 0, markov_xgb.stderr
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
