from __future__ import annotations

import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_share_eval_to_chirag_rejects_absolute_source_root() -> None:
    script_path = _project_root() / "run" / "share_eval_to_chirag.sh"
    run = subprocess.run(
        [
            "bash",
            str(script_path),
            "--species",
            "Dmel",
            "--source-root",
            "/tmp/forbidden-source",
            "--dest-root",
            "external/Genomics_Plotting",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 2
    assert "--source-root must be relative to project root" in run.stderr


def test_share_eval_to_chirag_accepts_relative_source_root() -> None:
    script_path = _project_root() / "run" / "share_eval_to_chirag.sh"
    run = subprocess.run(
        [
            "bash",
            str(script_path),
            "--species",
            "Dmel",
            "--source-root",
            "run",
            "--dest-root",
            "external/Genomics_Plotting",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode != 2
    assert "--source-root must be relative to project root" not in run.stderr
