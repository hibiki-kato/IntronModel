from __future__ import annotations

import subprocess
import time
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


def test_run_cnn_v2_sh_trains_both_tasks_before_inference() -> None:
    content = (_project_root() / "run" / "run_cnn_v2.sh").read_text(
        encoding="utf-8"
    )
    assert 'INTRONMODEL_AUTO_TMUX=' in content
    assert 'DEVICE="auto"' in content
    assert 'GPU_IDS="auto"' in content
    assert 'MAX_PARALLEL_TRIALS="auto"' in content
    assert 'MODEL="cnn_v2"' in content
    assert 'DONOR_LEN="100"' in content
    assert 'ACCEPTOR_LEN="100"' in content
    assert 'VAL_FRAC="0.2"' in content
    assert 'VALIDATION_METRIC="max_f1"' in content
    assert 'SEED="1337"' in content
    assert 'INTRON_SCORE_OP="+"' in content
    assert 'VISUALIZE="true"' in content
    assert 'SKIP_TRAINING="0"' in content
    assert 'CONTINUE_TRAINING="0"' in content
    assert 'TRAIN_ONLY="0"' in content
    assert 'CHECKPOINT_TOP_K="3"' in content
    assert 'CHECKPOINT_PRUNE_DRY_RUN="0"' in content
    assert 'INFER_COMPILE="0"' in content
    assert 'INFER_COMPILE_MODE="auto"' in content
    assert "intronmodel_enable_auto_tmux" in content
    assert 'source "${SCRIPT_DIR}/lib/tuned_config.sh"' in content
    assert "--sequence_transform" not in content
    assert "--intron_score_op" in content
    assert "--visualize" in content
    assert '--train_target "both"' in content
    assert "--validation_metric" in content
    assert "--seed" in content
    assert "--checkpoint_top_k" in content
    assert 'for task_name in donor acceptor; do' not in content


def test_run_cnn_v2_sh_ignores_cnn_v2_only_tuned_keys() -> None:
    content = (_project_root() / "run" / "run_cnn_v2.sh").read_text(
        encoding="utf-8"
    )
    assert "run_model.py forces cnn_v2 into pair_mode=independent" in content
    assert '|input_mode | pair_mode | sequence_transform | embedding_dim \\' in content
    assert '|bpe_pretrained_model_name | bpe_pretrained_revision \\' in content
    assert 'printf \'%s\\n\' "shared"' not in content[
        content.index('|input_mode | pair_mode | sequence_transform | embedding_dim \\') :
    ]
    assert 'printf \'%s\\n\' "ignore"' in content[
        content.index('|input_mode | pair_mode | sequence_transform | embedding_dim \\') :
    ]
    assert 'mode=independent tasks=donor,acceptor' in content


def test_run_cnn_pair_v2_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_cnn_pair_v2.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_run_cnn_pair_v2_sh_leaves_mask_to_best_config() -> None:
    content = (_project_root() / "run" / "run_cnn_pair_v2.sh").read_text(
        encoding="utf-8"
    )
    assert "INTRONMODEL_AUTO_TMUX" in content
    assert 'MODEL="cnn_pair_v2"' in content
    assert 'DONOR_LEN="100"' in content
    assert 'ACCEPTOR_LEN="100"' in content
    assert 'TRAIN_TARGET="pair"' in content
    assert 'VAL_FRAC="0.25"' in content
    assert 'VALIDATION_METRIC="max_f1"' in content
    assert 'SEED="1337"' in content
    assert 'INTRON_SCORE_OP="+"' in content
    assert 'VISUALIZE="true"' in content
    assert 'SKIP_TRAINING="0"' in content
    assert 'CONTINUE_TRAINING="0"' in content
    assert 'TRAIN_ONLY="0"' in content
    assert 'CHECKPOINT_TOP_K="3"' in content
    assert 'CHECKPOINT_PRUNE_DRY_RUN="0"' in content
    assert 'INFER_COMPILE="0"' in content
    assert 'INFER_COMPILE_MODE="auto"' in content
    assert "intronmodel_enable_auto_tmux" in content
    assert 'source "${SCRIPT_DIR}/lib/tuned_config.sh"' in content
    assert "--sequence_transform" not in content
    assert "--intron_score_op" in content
    assert "--visualize" in content
    assert "--train_target" in content
    assert "--validation_metric" in content
    assert "--val_frac" in content
    assert "--seed" in content
    assert "--checkpoint_top_k" in content


def test_run_cnn_pair_v2_sh_omits_empty_optional_loss_alpha_args() -> None:
    content = (_project_root() / "run" / "run_cnn_pair_v2.sh").read_text(
        encoding="utf-8"
    )

    assert 'local use_wrapper_hparams="1"' in content
    assert 'use_wrapper_hparams="0"' in content
    assert 'if [[ "${use_wrapper_hparams}" == "1" ]]; then' in content
    assert 'FOCAL_ALPHA_POS=""' in content
    assert 'ASYM_ALPHA_POS=""' in content
    assert '--focal_alpha_pos "${FOCAL_ALPHA_POS}"' not in content
    assert '--asym_alpha_pos "${ASYM_ALPHA_POS}"' not in content
    assert 'append_arg_if_set "focal_alpha_pos" "${FOCAL_ALPHA_POS}"' in content
    assert 'append_arg_if_set "asym_alpha_pos" "${ASYM_ALPHA_POS}"' in content


def test_run_cnn_pair_v2_sh_uses_best_hparams_when_tuned_is_loaded() -> None:
    content = (_project_root() / "run" / "run_cnn_pair_v2.sh").read_text(
        encoding="utf-8"
    )

    assert 'if [[ "${use_wrapper_hparams}" == "1" ]]; then' in content
    assert '--batch_size "${BATCH_SIZE}"' in content[
        content.index('if [[ "${use_wrapper_hparams}" == "1" ]]; then') :
    ]
    assert '--lr "${LR}"' in content[
        content.index('if [[ "${use_wrapper_hparams}" == "1" ]]; then') :
    ]


def test_run_cnn_pair_v2_sh_uses_single_pair_tuning_namespace() -> None:
    content = (_project_root() / "run" / "run_cnn_pair_v2.sh").read_text(
        encoding="utf-8"
    )
    assert 'TAG=""' in content
    assert "intronmodel_resolve_pair_tuning_model_name" in content
    assert 'append_arg_if_set "tag" "${resolved_tag}"' in content
    assert 'append_arg_if_set "train_pos_path" "${resolved_train_pos_path}"' in content
    assert 'append_arg_if_set "train_neg_path" "${resolved_train_neg_path}"' in content
    assert 'if [[ "${tuned_key}" == "tag" ]]; then' in content
    assert "intronmodel_resolve_pair_best_config_filename" in content
    assert 'best_config_filename="$(' in content
    assert "intronmodel_resolve_tuned_config_path" in content
    assert "append_versioned_output_args" in content
    assert "SYNTHESIZE_MODE" not in content
    assert "cnn_pair_v2_synth" not in content


def test_run_cnn_v3_sh_exposes_resdil_wrapper_knobs() -> None:
    content = (_project_root() / "run" / "run_cnn_v3.sh").read_text(
        encoding="utf-8"
    )
    assert 'MODEL="cnn_v3"' in content
    assert 'BLOCK_DILATIONS="1,2,4,8"' in content
    assert 'RESIDUAL_CHANNELS="32,64,96,128"' in content
    assert 'POOL_EVERY="2"' in content
    assert '--block_dilations "${BLOCK_DILATIONS}"' in content
    assert '--residual_channels "${RESIDUAL_CHANNELS}"' in content
    assert '--pool_every "${POOL_EVERY}"' in content
    assert 'task_tuned_path="${DATA_ROOT}/${species}/tuning/cnn_v3/${task_name}/best_config.json"' in content
    assert 'mode=independent tasks=donor,acceptor' in content


def test_run_cnn_pair_v3_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "run_cnn_pair_v3.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_run_cnn_pair_v3_sh_uses_single_pair_tuning_namespace() -> None:
    content = (_project_root() / "run" / "run_cnn_pair_v3.sh").read_text(
        encoding="utf-8"
    )
    assert 'MODEL="cnn_pair_v3"' in content
    assert 'tuned_model_name="${MODEL}"' in content
    assert "intronmodel_resolve_pair_best_config_filename" in content
    assert "intronmodel_resolve_tuned_config_path" in content
    assert 'append_versioned_output_args "cnn_pair_v3.sh" "${species}" "${MODEL}"' in content


def test_run_cnn_pair_v3_sh_exposes_resdil_wrapper_knobs() -> None:
    content = (_project_root() / "run" / "run_cnn_pair_v3.sh").read_text(
        encoding="utf-8"
    )
    assert 'BLOCK_DILATIONS="1,2,4,8"' in content
    assert 'RESIDUAL_CHANNELS="32,64,96,128"' in content
    assert 'POOL_EVERY="2"' in content


def test_tune_cnn_v2_time_omits_max_model_params_and_adds_input_mode() -> None:
    content = (_project_root() / "run" / "tune_cnn_v2_time.sh").read_text(
        encoding="utf-8"
    )
    assert 'OBJECTIVE_METRIC="max_f1"' in content
    assert 'TRIAL_STREAM_MODE="errors"' in content
    assert 'ENABLE_PHASE_OVERLAP="1"' in content
    assert '"trial_stream_mode": "${TRIAL_STREAM_MODE}"' in content
    assert '"enable_phase_overlap": ${ENABLE_PHASE_OVERLAP_JSON}' in content
    assert '"gpu_release_events_path": "${gpu_release_events_path}"' in content
    assert 'cycle_stdout.log' in content
    assert "MAX_MODEL_PARAMS" not in content
    assert "CROSS_SPECIES_BEST_MODE" not in content
    assert "resolve_cross_species_best_seed" not in content
    assert '"input_mode": {' in content
    assert '"mask": {' not in content
    assert '"sequence_transform": {' not in content
    assert "MASK_MODE" not in content
    assert "TAG=" not in content


def test_tune_cnn_pair_v2_time_omits_max_model_params() -> None:
    content = (_project_root() / "run" / "tune_cnn_pair_v2_time.sh").read_text(
        encoding="utf-8"
    )
    assert 'OBJECTIVE_METRIC="max_f1"' in content
    assert 'TRIAL_STREAM_MODE="errors"' in content
    assert '"trial_stream_mode": "${TRIAL_STREAM_MODE}"' in content
    assert '"enable_phase_overlap": true' in content
    assert '"gpu_release_events_path": "${gpu_release_events_path}"' in content
    assert 'cycle_stdout.log' in content
    assert "MAX_MODEL_PARAMS" not in content
    assert '"mask": {' in content
    assert '"sequence_transform": {' not in content
    assert "MASK_MODE" not in content


def test_tune_cnn_pair_v2_time_uses_single_pair_tuning_namespace() -> None:
    content = (_project_root() / "run" / "tune_cnn_pair_v2_time.sh").read_text(
        encoding="utf-8"
    )
    assert 'TAG=""' in content
    assert "intronmodel_resolve_pair_tuning_model_name" in content
    assert "intronmodel_resolve_pair_best_config_path" in content
    assert 'TUNING_MODEL_NAME="$(' in content
    assert '"tag": "${resolved_tag}"' in content
    assert "CROSS_SPECIES_BEST_MODE" not in content
    assert "resolve_cross_species_best_seed" not in content
    assert "SYNTHESIZE_MODE" not in content
    assert "cnn_pair_v2_synth" not in content


def test_tune_cnn_v3_time_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "tune_cnn_v3_time.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_tune_cnn_v3_time_sh_uses_reinforce_search_defaults() -> None:
    content = (_project_root() / "run" / "tune_cnn_v3_time.sh").read_text(
        encoding="utf-8"
    )
    assert 'SEARCH_ALGO="reinforce"' in content
    assert 'REINFORCE_TEMPERATURE="0.75"' in content
    assert 'POOL_EVERY="2"' in content
    assert '"model": "cnn_v3"' in content
    assert '"reinforce_temperature": ${REINFORCE_TEMPERATURE}' in content
    assert '"arch_mutation_steps"' in content
    assert '"arch_add_block_prob"' in content
    assert '"pool_every": ${POOL_EVERY}' in content
    assert '"validation_metric": "${OBJECTIVE_METRIC}"' in content
    assert 'intronmodel_run_with_process_title \\' in content
    assert '"${RUNTIME_PROCESS_TITLE}" \\' in content


def test_tune_cnn_pair_v3_time_sh_uses_eta_process_title_for_scheduler() -> None:
    content = (_project_root() / "run" / "tune_cnn_pair_v3_time.sh").read_text(
        encoding="utf-8"
    )
    assert 'intronmodel_run_with_process_title \\' in content
    assert '"${RUNTIME_PROCESS_TITLE}" \\' in content


def test_cnn_v3_scripts_have_valid_bash_syntax() -> None:
    root = _project_root()
    run_result = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_cnn_v3.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    tune_result = subprocess.run(
        ["bash", "-n", str(root / "run" / "tune_cnn_v3_time.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert tune_result.returncode == 0, tune_result.stderr


def test_common_run_with_deadline_times_out() -> None:
    common_sh = _project_root() / "run" / "lib" / "common.sh"
    script = f"""
set -euo pipefail
source "{common_sh}"
deadline=$(( $(date +%s) + 1 ))
start=$SECONDS
if intronmodel_run_with_deadline "$deadline" 1 "" sleep 5; then
  rc=0
else
  rc=$?
fi
elapsed=$((SECONDS - start))
echo "rc=$rc elapsed=$elapsed"
"""
    start_time = time.monotonic()
    run = subprocess.run(
        ["bash", "-lc", script],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - start_time
    assert run.returncode == 0, run.stderr
    assert "rc=124" in run.stdout
    assert elapsed < 4.0


def test_common_prune_timeout_artifacts_removes_output_dir(tmp_path: Path) -> None:
    common_sh = _project_root() / "run" / "lib" / "common.sh"
    output_dir = tmp_path / "partial-run"
    output_dir.mkdir()
    (output_dir / "artifact.txt").write_text("tmp", encoding="utf-8")
    capture_path = tmp_path / "prune_args.txt"
    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text(
        f"""#!/usr/bin/env bash
printf '%s\\n' "$@" > "{capture_path}"
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = f"""
set -euo pipefail
source "{common_sh}"
intronmodel_prune_timeout_artifacts \
  "test-script" \
  "{fake_python}" \
  "{_project_root()}" \
  "{tmp_path}" \
  "{tmp_path}" \
  "Dmel" \
  "cnn" \
  "{output_dir}"
"""
    run = subprocess.run(
        ["bash", "-lc", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert not output_dir.exists()
    captured_args = capture_path.read_text(encoding="utf-8")
    assert "prune_missing_rank_checkpoints.py" in captured_args
    assert "--species" in captured_args
    assert "Dmel" in captured_args


def test_modified_tuning_scripts_do_not_use_cross_species_seed_fallback() -> None:
    root = _project_root()
    script_names = (
        "tune_bert.sh",
        "tune_bert_time.sh",
        "tune_bilstm_pair_time.sh",
        "tune_cnn_resdil.sh",
        "tune_cnn_resdil_time.sh",
        "tune_cnn_pair_v2_time.sh",
        "tune_cnn_v2_time.sh",
        "tune_dnabert.sh",
        "tune_dnabert_pair.sh",
        "tune_dnabert_pair_time.sh",
        "tune_dnabert_time.sh",
        "tune_reservoir.sh",
        "tune_reservoir_time.sh",
        "tune_tcn.sh",
        "tune_tcn_time.sh",
    )
    for script_name in script_names:
        content = (root / "run" / script_name).read_text(encoding="utf-8")
        assert "CROSS_SPECIES_BEST_MODE" not in content
        assert "resolve_cross_species_best_seed" not in content
        assert "tuning_cross_species_best" not in content


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


def test_sync_sh_pushes_run_scripts_with_checksum() -> None:
    content = (_project_root() / "sync.sh").read_text(encoding="utf-8")
    assert 'CHECKSUM_SYNC_PATHS=(' in content
    assert '"run/"' in content
    assert '"src/scripts/"' in content
    assert '--exclude "$sync_path"' in content
    assert "--size-only" in content
    assert "--checksum" in content


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


def test_run_dnabert_pair_sh_exposes_synthesize_mode() -> None:
    content = (_project_root() / "run" / "run_dnabert_pair.sh").read_text(
        encoding="utf-8"
    )
    assert 'SYNTHESIZE_MODE="off"' in content
    assert 'TAG=""' in content


def test_tune_dnabert_pair_scripts_expose_synthesize_mode() -> None:
    for script_name in ("tune_dnabert_pair.sh", "tune_dnabert_pair_time.sh"):
        content = (_project_root() / "run" / script_name).read_text(
            encoding="utf-8"
        )
        assert 'SYNTHESIZE_MODE="off"' in content
        assert "intronmodel_resolve_synth_tuning_model_name" in content
        assert "intronmodel_resolve_pair_best_config_filename" in content


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


def test_run_cnn_pair_v2_sh_includes_gpu_parallel_config() -> None:
    content = (_project_root() / "run" / "run_cnn_pair_v2.sh").read_text(
        encoding="utf-8"
    )
    assert 'GPU_IDS="auto"' in content
    assert 'MAX_PARALLEL_TRIALS="auto"' in content


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
    cnn_pair_v2 = subprocess.run(
        ["bash", "-n", str(root / "run" / "run_cnn_pair_v2.sh")],
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
    assert cnn_pair_v2.returncode == 0, cnn_pair_v2.stderr
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
