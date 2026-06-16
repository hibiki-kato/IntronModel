from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
        cwd=_project_root(),
    )


def test_plot_eval_interactive_multi_species_runs_in_parallel(
    tmp_path: Path,
) -> None:
    script_path = _project_root() / "run" / "plot_eval.sh"
    log_path = tmp_path / "parallel.log"
    command = "\n".join(
        [
            f"source {shlex.quote(str(script_path))}",
            'INTERACTIVE="1"',
            f'LOG_PATH={shlex.quote(str(log_path))}',
            "run_for_one_species() {",
            '  local sp="$1"',
            '  printf "%s start\\n" "${sp}" >> "${LOG_PATH}"',
            "  sleep 0.4",
            '  printf "%s end\\n" "${sp}" >> "${LOG_PATH}"',
            "}",
            'run_species_selection "Athal,Dmel,Mmus" ""',
        ]
    )

    started_at = time.monotonic()
    run = _run_bash(command)
    elapsed_seconds = time.monotonic() - started_at

    assert run.returncode == 0, run.stderr
    assert elapsed_seconds < 0.9

    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 6
    first_end_index = next(
        index for index, line in enumerate(log_lines) if line.endswith(" end")
    )
    assert first_end_index >= 2


def test_plot_eval_interactive_multi_species_propagates_failure() -> None:
    script_path = _project_root() / "run" / "plot_eval.sh"
    command = "\n".join(
        [
            f"source {shlex.quote(str(script_path))}",
            'INTERACTIVE="1"',
            "run_for_one_species() {",
            '  local sp="$1"',
            '  if [[ "${sp}" == "Dmel" ]]; then',
            "    return 1",
            "  fi",
            "  sleep 0.1",
            "}",
            'run_species_selection "Athal,Dmel,Mmus" ""',
        ]
    )

    run = _run_bash(command)

    assert run.returncode != 0
    assert "[plot_eval.sh] Failed for species Dmel" in run.stderr
