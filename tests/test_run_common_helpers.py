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


def _run_cross_species_best_shell(
    command: str,
) -> subprocess.CompletedProcess[str]:
    common_path = _project_root() / "run" / "lib" / "common.sh"
    cross_species_path = (
        _project_root() / "run" / "lib" / "tuning_cross_species_best.sh"
    )
    shell_command = (
        f"source {shlex.quote(str(common_path))}\n"
        f"source {shlex.quote(str(cross_species_path))}\n"
        f"{command}"
    )
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
    assert run.stdout.strip().splitlines() == ["2024", "1337", "3407"]


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
        "unset TRITON_CACHE_DIR\n"
        f'intronmodel_init_paths {shlex.quote(str(script_path))}\n'
        'printf "%s\\n" '
        '"${XDG_CACHE_HOME}" '
        '"${XDG_CONFIG_HOME}" '
        '"${MPLCONFIGDIR}" '
        '"${TORCHINDUCTOR_CACHE_DIR}" '
        '"${TRITON_CACHE_DIR}"\n'
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
    ]
    for resolved_path in lines:
        assert Path(resolved_path).is_dir()


def test_common_resolve_pair_synthesize_defaults_appends_suffix_and_paths(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    run = _run_common_shell(
        f'DATA_ROOT={shlex.quote(str(data_root))}\n'
        'IFS=$\'\\t\' read -r TAG TRAIN_POS_PATH TRAIN_NEG_PATH <<< "$('
        'intronmodel_resolve_pair_synthesize_defaults '
        'Dmel on exp1 "" ""'
        ')"\n'
        'printf "%s\\n%s\\n%s\\n" "${TAG}" "${TRAIN_POS_PATH}" '
        '"${TRAIN_NEG_PATH}"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        "exp1_synth",
        str(data_root / "Dmel" / "raw" / "100bp.err"),
        str(data_root / "Dmel" / "processed" / "100bp_mixed_one_side.neg.err"),
    ]


def test_common_resolve_pair_best_config_path_switches_by_mode(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    run = _run_common_shell(
        f'DATA_ROOT={shlex.quote(str(data_root))}\n'
        'printf "%s\\n%s\\n" '
        '"$(intronmodel_resolve_pair_best_config_path "${DATA_ROOT}" '
        '"Dmel" "cnn_v2_pair" "off")" '
        '"$(intronmodel_resolve_pair_best_config_path "${DATA_ROOT}" '
        '"Dmel" "cnn_v2_pair" "on")"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        str(data_root / "Dmel" / "tuning" / "cnn_v2_pair" / "pair" / "best_config.json"),
        str(
            data_root
            / "Dmel"
            / "tuning"
            / "cnn_v2_pair"
            / "pair"
            / "best_synth_config.json"
        ),
    ]


def test_common_resolve_pair_tuning_model_name_switches_by_mode() -> None:
    run = _run_common_shell(
        'printf "%s\\n%s\\n" '
        '"$(intronmodel_resolve_pair_tuning_model_name off)" '
        '"$(intronmodel_resolve_pair_tuning_model_name on)"\n'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        "cnn_v2_pair",
        "cnn_v2_pair_synth",
    ]


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


def test_cross_species_best_seed_uses_synth_specific_filename(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    local_species = "Alpha"
    remote_species = "Beta"
    for species, score in ((local_species, 0.5), (remote_species, 0.9)):
        config_path = (
            data_root
            / species
            / "tuning"
            / "cnn_v2_pair"
            / "pair"
            / "best_synth_config.json"
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "{\"status\": \"ok\", \"objective_score\": %s}" % score,
            encoding="utf-8",
        )

    missing_local_best = (
        data_root
        / local_species
        / "tuning"
        / "cnn_v2_pair"
        / "pair"
        / "best_synth_config.json.missing"
    )
    run = _run_cross_species_best_shell(
        f'PY_BIN="$(intronmodel_resolve_python_bin test_cross_species.sh)"\n'
        f'resolve_cross_species_best_seed "test_cross_species.sh" '
        f'"${{PY_BIN}}" "{data_root}" "cnn_v2_pair" "{local_species}" "pair" '
        f'"{missing_local_best}" "auto" "" "" "best_synth_config.json"'
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == str(
        data_root
        / remote_species
        / "tuning"
        / "cnn_v2_pair"
        / "pair"
        / "best_synth_config.json"
    )
