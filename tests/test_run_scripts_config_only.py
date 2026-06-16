from __future__ import annotations

import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_scripts() -> list[Path]:
    return sorted(
        path
        for path in (_project_root() / "run").glob("*.sh")
        if path.name != "tempCodeRunnerFile.sh"
    )


def test_current_run_scripts_have_valid_bash_syntax() -> None:
    for script_path in _run_scripts():
        run = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0, f"{script_path.name}: {run.stderr}"


def test_config_only_run_scripts_reject_cli_arguments() -> None:
    for script_path in _run_scripts():
        content = script_path.read_text(encoding="utf-8")
        if "config-only" not in content:
            continue
        run = subprocess.run(
            ["bash", str(script_path), "--dummy"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode != 0, script_path.name
        assert "config-only" in run.stderr, script_path.name


def test_current_training_wrappers_use_shared_runtime_helpers() -> None:
    wrapper_names = (
        "run_cnn_v2.sh",
        "run_cnn_v3.sh",
        "run_cnn_pair_v2.sh",
        "run_cnn_pair_v3.sh",
        "run_dnabert_pair.sh",
        "run_spliceformer_sc.sh",
    )
    for script_name in wrapper_names:
        content = (_project_root() / "run" / script_name).read_text(encoding="utf-8")
        assert "source \"${SCRIPT_DIR}/lib/common.sh\"" in content
        assert "intronmodel_append_arg_if_set args" in content
        assert "intronmodel_append_flag_if_truthy args" in content

    parallel_names = (
        "run_cnn_v2.sh",
        "run_cnn_v3.sh",
        "run_cnn_pair_v2.sh",
        "run_cnn_pair_v3.sh",
    )
    for script_name in parallel_names:
        content = (_project_root() / "run" / script_name).read_text(encoding="utf-8")
        assert "intronmodel_run_model_with_optional_gpu" in content
        assert "intronmodel_run_species_jobs" in content
        assert "wait -n -p completed_pid" not in content


def test_tune_time_wrappers_delegate_duplicate_helpers_to_common_runtime() -> None:
    tune_names = (
        "tune_cnn_v2_time.sh",
        "tune_cnn_v3_time.sh",
        "tune_cnn_pair_v2_time.sh",
        "tune_cnn_pair_v3_time.sh",
        "tune_dnabert_time.sh",
        "tune_dnabert_pair_time.sh",
    )
    forbidden_defs = (
        "normalize_json_object_file()",
        "run_double_descent_plot()",
        "append_unique_gpu_ids()",
        "remove_gpu_from_csv()",
    )
    for script_name in tune_names:
        content = (_project_root() / "run" / script_name).read_text(encoding="utf-8")
        assert "source \"${SCRIPT_DIR}/lib/common.sh\"" in content
        assert "intronmodel_normalize_json_object_file" in content
        assert "intronmodel_run_double_descent_plot" in content
        for function_def in forbidden_defs:
            assert function_def not in content


def test_active_pair_wrappers_keep_tuning_namespace_helpers() -> None:
    run_pair = (_project_root() / "run" / "run_dnabert_pair.sh").read_text(
        encoding="utf-8"
    )
    tune_pair = (_project_root() / "run" / "tune_dnabert_pair_time.sh").read_text(
        encoding="utf-8"
    )

    assert "SYNTHESIZE_MODE=\"off\"" in run_pair
    assert "intronmodel_resolve_pair_tuning_model_name" in run_pair
    assert "intronmodel_resolve_pair_best_config_filename" in run_pair
    assert "SYNTHESIZE_MODE=\"off\"" in tune_pair
    assert "intronmodel_resolve_synth_tuning_model_name" in tune_pair
    assert "intronmodel_resolve_pair_best_config_path" in tune_pair
