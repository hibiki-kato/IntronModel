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
