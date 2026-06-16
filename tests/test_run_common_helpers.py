from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_common_shell(command: str) -> subprocess.CompletedProcess[str]:
    common_path = _project_root() / "run" / "lib" / "common.sh"
    shell_command = f"source {shlex.quote(str(common_path))}\n{command}"
    return subprocess.run(
        ["bash", "-lc", shell_command],
        capture_output=True,
        text=True,
        check=False,
    )


def test_common_resolve_species_template_replaces_supported_patterns() -> None:
    run = _run_common_shell(
        "intronmodel_resolve_species_template "
        "'data/{species}/raw/${SPECIES}/{SPECIES}.err' 'Athal'"
    )

    assert run.returncode == 0
    assert run.stdout.strip() == "data/Athal/raw/Athal/Athal.err"


def test_common_json_string_or_null_returns_null_for_empty_value() -> None:
    run = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        'intronmodel_json_string_or_null "${PY_BIN}" ""'
    )

    assert run.returncode == 0
    assert run.stdout.strip() == "null"


def test_common_json_string_or_null_quotes_resolved_species_path() -> None:
    run = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        'RESOLVED_PATH="$('
        "intronmodel_resolve_species_template "
        "'data/{species}/raw/${SPECIES}.err' 'Dmel'"
        ')"\n'
        'intronmodel_json_string_or_null "${PY_BIN}" "${RESOLVED_PATH}"'
    )

    assert run.returncode == 0
    assert run.stdout.strip() == '"data/Dmel/raw/Dmel.err"'


def test_common_format_eta_epoch_uses_month_day_and_hour_minute() -> None:
    run = _run_common_shell(
        'TZ=UTC intronmodel_format_eta_epoch "1704072660"'
    )

    assert run.returncode == 0
    assert run.stdout.strip() == "01/01 1:31"


def test_common_build_eta_process_title_formats_eta_prefix() -> None:
    run = _run_common_shell(
        'intronmodel_build_eta_process_title "01/01 1:31"'
    )

    assert run.returncode == 0
    assert run.stdout.strip() == "ETA:01/01 1:31"


def test_common_resolve_dnabert_relative_path_supports_variant_s() -> None:
    run = _run_common_shell(
        'intronmodel_resolve_dnabert_relative_path '
        '"test_common.sh" "S" "pretrained/dnabert2" '
        '"pretrained/dnabert6" "pretrained/dnabert-s"'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "pretrained/dnabert-s"


def test_common_resolve_dnabert_pretrained_name_normalizes_model_prefix() -> None:
    run = _run_common_shell(
        'intronmodel_resolve_dnabert_pretrained_name '
        '"test_common.sh" "2" "" "/tmp/model-root" '
        '"model/pretrained/dnabert2" "pretrained/dnabert6" '
        '"pretrained/dnabert-s"'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "/tmp/model-root/pretrained/dnabert2"


def test_common_resolve_search_space_file_prefers_first_existing_candidate(
    tmp_path: Path,
) -> None:
    first_candidate = tmp_path / "first.json"
    second_candidate = tmp_path / "second.json"
    first_candidate.write_text("{}", encoding="utf-8")
    second_candidate.write_text("{}", encoding="utf-8")

    run = _run_common_shell(
        "intronmodel_resolve_search_space_file "
        '"test_common.sh" "" '
        f'"{first_candidate}" "{second_candidate}"'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == str(first_candidate)


def test_common_resolve_search_space_file_rejects_missing_explicit_path() -> None:
    run = _run_common_shell(
        'intronmodel_resolve_search_space_file '
        '"test_common.sh" "/tmp/does-not-exist.json" "/tmp/fallback.json"'
    )

    assert run.returncode == 2
    assert "SEARCH_SPACE_FILE not found" in run.stderr


def test_common_resolve_eta_scope_uses_gpu_when_slots_are_short(
) -> None:
    run = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        'intronmodel_resolve_eta_scope '
        '"test_common.sh" "0,1" "auto" "cuda" "4" "${PY_BIN}"'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "gpu"


def test_common_resolve_eta_scope_uses_species_when_gpus_cover_jobs(
) -> None:
    run = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        'intronmodel_resolve_eta_scope '
        '"test_common.sh" "0,1,2,3" "auto" "cuda" "2" "${PY_BIN}"'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "species"


def test_common_resolve_seed_list_defaults_to_base_seed() -> None:
    run = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        'intronmodel_resolve_seed_list "test_common.sh" "1337" "" "${PY_BIN}"'
    )

    assert run.returncode == 0
    assert run.stdout.strip().splitlines() == ["1337"]


def test_common_resolve_seed_list_normalizes_and_deduplicates_entries() -> None:
    run = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        'intronmodel_resolve_seed_list '
        '"test_common.sh" "1337" " 2024,1337,2024, 3407 " "${PY_BIN}"'
    )

    assert run.returncode == 0
    assert run.stdout.strip().splitlines() == ["1337"]
    assert "SEED_LIST is ignored" in run.stderr


def test_common_collect_gpu_release_ids_returns_only_new_events(
    tmp_path: Path,
) -> None:
    release_file = tmp_path / "gpu_release_events.jsonl"
    cursor_file = tmp_path / "gpu_release.cursor"
    release_file.write_text(
        "\n".join(
            [
                '{"event":"gpu_released","gpu_id":"2"}',
                '{"event":"gpu_released","gpu_id":"5"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_first = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        f'intronmodel_collect_gpu_release_ids "${{PY_BIN}}" '
        f'{shlex.quote(str(release_file))} {shlex.quote(str(cursor_file))}\n'
    )
    run_second = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        f'intronmodel_collect_gpu_release_ids "${{PY_BIN}}" '
        f'{shlex.quote(str(release_file))} {shlex.quote(str(cursor_file))}\n'
    )

    assert run_first.returncode == 0, run_first.stderr
    assert run_first.stdout.strip().splitlines() == ["2", "5"]
    assert run_second.returncode == 0, run_second.stderr
    assert run_second.stdout.strip() == ""


def test_common_run_with_process_title_preserves_python_executable() -> None:
    run = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        'intronmodel_run_with_process_title '
        '"test process title" '
        '"${PY_BIN}" - <<\'PY\'\n'
        "import sys\n"
        "print(sys.executable)\n"
        "PY"
    )

    assert run.returncode == 0
    assert run.stdout.strip() != ""


def test_common_init_paths_configures_runtime_cache_dirs(
    tmp_path: Path,
) -> None:
    user_name = os.environ.get("USER", "unknown")
    tmpdir = tmp_path / "tmp"
    script_path = _project_root() / "run" / "tune_cnn_v2_time.sh"
    run = _run_common_shell(
        f'export TMPDIR={shlex.quote(str(tmpdir))}\n'
        "unset XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME TRANSFORMERS_CACHE\n"
        "unset HF_MODULES_CACHE MPLCONFIGDIR TORCHINDUCTOR_CACHE_DIR\n"
        "unset TRITON_CACHE_DIR HF_HUB_OFFLINE TRANSFORMERS_OFFLINE\n"
        f'intronmodel_init_paths {shlex.quote(str(script_path))}\n'
        'printf "%s\\n" '
        '"${XDG_CACHE_HOME}" '
        '"${XDG_CONFIG_HOME}" '
        '"${MPLCONFIGDIR}" '
        '"${TORCHINDUCTOR_CACHE_DIR}" '
        '"${TRITON_CACHE_DIR}" '
        '"${HF_HUB_OFFLINE}" '
        '"${TRANSFORMERS_OFFLINE}"\n'
    )

    assert run.returncode == 0
    lines = run.stdout.strip().splitlines()
    expected_root = tmpdir / f"intronmodel-cache-{user_name}"
    assert lines == [
        str(expected_root),
        str(expected_root / "config"),
        str(expected_root / "config" / "matplotlib"),
        str(expected_root / "torchinductor"),
        str(expected_root / "triton"),
        "1",
        "1",
    ]
    for resolved_path in lines[:5]:
        assert Path(resolved_path).is_dir()


def test_common_resolve_pair_best_config_path_prefers_public_pair_tree(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    run = _run_common_shell(
        f'DATA_ROOT={shlex.quote(str(data_root))}\n'
        'printf "%s\\n" '
        '"$(intronmodel_resolve_pair_best_config_path "${DATA_ROOT}" '
        '"Dmel" "cnn_pair_v2")"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        str(data_root / "Dmel" / "tuning" / "cnn_pair_v2" / "pair" / "best_config.json")
    ]


def test_common_resolve_pair_best_config_path_supports_cnn_pair_v3(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    run = _run_common_shell(
        f'DATA_ROOT={shlex.quote(str(data_root))}\n'
        'printf "%s\\n" '
        '"$(intronmodel_resolve_pair_best_config_path "${DATA_ROOT}" '
        '"Dmel" "cnn_pair_v3")"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        str(
            data_root / "Dmel" / "tuning" / "cnn_pair_v3" / "pair" / "best_config.json"
        )
    ]


def test_common_resolve_pair_tuning_model_name_supports_active_pair_models() -> None:
    run = _run_common_shell(
        'printf "%s\\n%s\\n%s\\n" '
        '"$(intronmodel_resolve_pair_tuning_model_name cnn_pair_v2)" '
        '"$(intronmodel_resolve_pair_tuning_model_name cnn_pair_v3)" '
        '"$(intronmodel_resolve_pair_tuning_model_name dnabert2_pair)"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        "cnn_pair_v2",
        "cnn_pair_v3",
        "dnabert2_pair",
    ]


def test_common_resolve_latest_published_name_does_not_seed_when_history_missing(
    tmp_path: Path,
) -> None:
    project_root = _project_root()
    data_root = tmp_path / "external_data"
    model_root = tmp_path / "external_model"
    donor_checkpoint = model_root / "SpX" / "donor" / "donor_raw.pt"
    acceptor_checkpoint = model_root / "SpX" / "acceptor" / "acceptor_raw.pt"
    donor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    acceptor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    donor_checkpoint.write_bytes(b"donor")
    acceptor_checkpoint.write_bytes(b"acceptor")
    donor_best = (
        data_root / "SpX" / "tuning" / "cnn_v3" / "donor" / "best_config.json"
    )
    acceptor_best = (
        data_root / "SpX" / "tuning" / "cnn_v3" / "acceptor" / "best_config.json"
    )
    donor_best.parent.mkdir(parents=True, exist_ok=True)
    acceptor_best.parent.mkdir(parents=True, exist_ok=True)
    donor_best.write_text(
        '{"status":"ok","donor_checkpoint_path":"'
        + str(donor_checkpoint)
        + '"}\n',
        encoding="utf-8",
    )
    acceptor_best.write_text(
        '{"status":"ok","acceptor_checkpoint_path":"'
        + str(acceptor_checkpoint)
        + '"}\n',
        encoding="utf-8",
    )

    run = _run_common_shell(
        f'PROJECT_ROOT={shlex.quote(str(project_root))}\n'
        f'INTRONMODEL_DATA_ROOT={shlex.quote(str(data_root))}\n'
        f'INTRONMODEL_MODEL_ROOT={shlex.quote(str(model_root))}\n'
        'printf "%s\\n" '
        '"$(intronmodel_resolve_latest_published_name '
        '"test_common.sh" "SpX" "cnn_v3")"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "cnn_v3"
    assert not (
        data_root / "SpX" / "tuning" / "cnn_v3" / "version_history.tsv"
    ).exists()


def test_common_resolve_latest_published_name_uses_data_root_override(
    tmp_path: Path,
) -> None:
    project_root = _project_root()
    data_root = tmp_path / "external_data"
    tuning_root = data_root / "SpX" / "tuning" / "cnn_pair_v3"
    history_path = tuning_root / "version_history.tsv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "\t".join(
            [
                "version",
                "published_name",
                "published_at",
                "source_best_config",
                "objective_metric",
                "objective_score",
                "updated_side",
                "carry_forward_side",
                "donor_checkpoint_path",
                "acceptor_checkpoint_path",
                "pair_checkpoint_path",
                "metrics_json",
                "archive_status",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "2",
                "cnn_pair_v3.02",
                "2026-03-30T00:00:00Z",
                "data/SpX/tuning/cnn_pair_v3/pair/best_config.json",
                "pair_pr_auc",
                "0.95",
                "pair",
                "",
                "",
                "",
                "model/SpX/pair/cnn_pair_v3.02.pt",
                "data/SpX/learning_metric/cnn_pair_v3.02.train.json",
                "live",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run = _run_common_shell(
        f'PROJECT_ROOT={shlex.quote(str(project_root))}\n'
        f'DATA_ROOT={shlex.quote(str(data_root))}\n'
        'printf "%s\\n" '
        '"$(intronmodel_resolve_latest_published_name '
        '"test_common.sh" "SpX" "cnn_pair_v3")"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "cnn_pair_v3.02"


def test_common_is_active_public_model_supports_generic_models() -> None:
    run = _run_common_shell(
        'printf "%s\\n%s\\n%s\\n%s\\n%s\\n" '
        '"$(intronmodel_is_active_public_model test_common.sh cnn_resdil)" '
        '"$(intronmodel_is_active_public_model test_common.sh bilstm_pair)" '
        '"$(intronmodel_is_active_public_model test_common.sh reservoir)" '
        '"$(intronmodel_is_active_public_model test_common.sh markov_xgboost)" '
        '"$(intronmodel_is_active_public_model test_common.sh nonexistent_model)"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == ["1", "1", "0", "0", "0"]


def test_common_resolve_synth_tuning_model_name_switches_by_mode() -> None:
    run = _run_common_shell(
        'printf "%s\\n%s\\n" '
        '"$(intronmodel_resolve_synth_tuning_model_name dnabert2_pair off)" '
        '"$(intronmodel_resolve_synth_tuning_model_name dnabert2_pair on)"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        "dnabert2_pair",
        "dnabert2_pair_synth",
    ]


def test_wrapper_runtime_append_helpers_preserve_argument_boundaries() -> None:
    run = _run_common_shell(
        'args=()\n'
        'intronmodel_append_arg_if_set args label "value with spaces"\n'
        'intronmodel_append_arg_if_set args empty ""\n'
        'intronmodel_append_flag_if_truthy args use_feature " YES "\n'
        'intronmodel_append_flag_if_truthy args skip_feature "no"\n'
        'printf "<%s>\\n" "${args[@]}"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        "<--label>",
        "<value with spaces>",
        "<--use_feature>",
    ]


def test_wrapper_runtime_species_jobs_reports_first_failure() -> None:
    run = _run_common_shell(
        'seen=()\n'
        'runner() {\n'
        '  seen+=("$1:$2")\n'
        '  [[ "$1" != "bad" ]]\n'
        '}\n'
        'species=(good bad later)\n'
        'gpus=(0)\n'
        'intronmodel_run_species_jobs test_wrapper species gpus 1 runner\n'
        'code=$?\n'
        'printf "code=%s\\n" "${code}"\n'
        'printf "%s\\n" "${seen[@]}"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        "code=1",
        "good:0",
        "bad:0",
    ]


def test_wrapper_runtime_tuning_helpers_normalize_and_dedupe(
    tmp_path: Path,
) -> None:
    search_space = tmp_path / "search_space.json"
    search_space.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")

    run = _run_common_shell(
        'PY_BIN="$(intronmodel_resolve_python_bin test_common.sh)"\n'
        f'intronmodel_normalize_json_object_file '
        f'"${{PY_BIN}}" {shlex.quote(str(search_space))}\n'
        'values=(0 1)\n'
        'intronmodel_append_unique_values values 1 2 "" 0 3\n'
        'printf "%s\\n" "${values[*]}"\n'
        'intronmodel_remove_value_from_csv "0,1,2,3" "2"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        '{"b":2,"a":1}',
        "0 1 2 3",
        "0,1,3",
    ]
