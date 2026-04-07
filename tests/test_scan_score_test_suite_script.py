from __future__ import annotations

import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_scan_score_test_suite_sh_rejects_cli_arguments() -> None:
    script_path = _project_root() / "run" / "scan_score_test_suite.sh"
    run = subprocess.run(
        ["bash", str(script_path), "--dummy"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode != 0
    assert "config-only" in run.stderr


def test_scan_score_test_suite_sh_targets_students_outputs() -> None:
    content = (_project_root() / "run" / "scan_score_test_suite.sh").read_text(
        encoding="utf-8"
    )

    assert 'TAG="${TAG:-h}"' in content
    assert 'SUITE_ROOT="${PROJECT_ROOT}/score_test_suite"' in content
    assert 'STUDENTS_DIR="${SUITE_ROOT}/Students"' in content
    assert '--suite-root "${SUITE_ROOT}"' in content
    assert '--students-dir "${STUDENTS_DIR}"' in content
    assert '--tag "${TAG}"' in content
