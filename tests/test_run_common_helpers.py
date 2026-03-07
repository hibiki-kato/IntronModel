from __future__ import annotations

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
    assert run.stdout.strip() == "ETA: 01/01 1:31"


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
