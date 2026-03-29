from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_executable(path: Path) -> None:
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR)


def _patch_assignment(script_text: str, name: str, value: str) -> str:
    pattern = re.compile(rf'^{re.escape(name)}="[^"]*"$', re.MULTILINE)
    replaced, count = pattern.subn(f'{name}="{value}"', script_text, count=1)
    if count != 1:
        raise AssertionError(f"Failed to patch assignment for {name}.")
    return replaced


def _write_fake_conda(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "conda() {",
                '  if [[ "${1-}" == "activate" ]]; then',
                "    return 0",
                "  fi",
                '  if [[ "${1-}" == "info" && "${2-}" == "--base" ]]; then',
                '    printf "%s\\n" "/tmp/fake-conda"',
                "    return 0",
                "  fi",
                "  return 0",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _make_executable(path)


def _write_fake_run_model_helper(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import json",
                "import os",
                "import re",
                "import sys",
                "import time",
                "from pathlib import Path",
                "",
                "",
                "def _parse_flag(argv: list[str], flag: str) -> str:",
                "    for index, token in enumerate(argv):",
                "        if token == flag and index + 1 < len(argv):",
                "            return argv[index + 1]",
                '    raise ValueError(f\"missing flag: {flag}\")',
                "",
                "",
                "def _write_trace(trace_path: Path, payload: dict[str, object]) -> None:",
                '    with trace_path.open(\"a\", encoding=\"utf-8\") as handle:',
                "        handle.write(json.dumps(payload, sort_keys=True))",
                '        handle.write(\"\\n\")',
                "",
                "",
                "def _scenario_sleep(",
                "    scenario: str,",
                "    cycle_index: int,",
                "    phase: str,",
                "    trial_id: int,",
                ") -> float:",
                '    if scenario == \"overlap\":',
                '        if cycle_index == 0 and phase == \"quick\":',
                "            return 0.05",
                '        if cycle_index == 0 and phase == \"full\":',
                "            return 0.30",
                "        return 0.25",
                '    if scenario == \"grow_slots\":',
                '        if cycle_index == 0 and phase == \"quick\":',
                "            return 0.05",
                '        if cycle_index == 0 and phase == \"full\":',
                "            return 0.10",
                "        return 0.25",
                '    if scenario == \"no_release\":',
                "        del trial_id",
                "        return 0.30",
                "    raise ValueError(f\"unknown TEST_SCENARIO: {scenario}\")",
                "",
                "",
                "def _scenario_score(",
                "    scenario: str,",
                "    cycle_index: int,",
                "    phase: str,",
                "    trial_id: int,",
                ") -> float:",
                "    del scenario",
                '    if phase == \"full\":',
                "        return 0.95 - (trial_id * 0.01)",
                "    if cycle_index == 0:",
                "        return 0.90 - (trial_id * 0.05)",
                "    return 0.70 - (trial_id * 0.01)",
                "",
                "",
                "def main() -> int:",
                "    argv = sys.argv[1:]",
                "    metrics_json = Path(_parse_flag(argv, \"--metrics_json\"))",
                "    metrics_json.parent.mkdir(parents=True, exist_ok=True)",
                "    phase_match = re.search(",
                '        r\"(quick|full)_trial_(\\d+)\\.metrics\\.json$\",',
                "        metrics_json.name,",
                "    )",
                "    if phase_match is None:",
                '        raise ValueError(\"phase/trial_id could not be parsed\")',
                "    phase = phase_match.group(1)",
                "    trial_id = int(phase_match.group(2))",
                "    cycle_match = re.search(r\"_c(\\d+)$\", metrics_json.parent.name)",
                "    if cycle_match is None:",
                '        raise ValueError(\"cycle suffix not found in metrics path\")',
                "    cycle_index = int(cycle_match.group(1))",
                '    trace_path = Path(os.environ[\"TEST_TRACE_FILE\"])',
                '    scenario = os.environ[\"TEST_SCENARIO\"]',
                '    gpu_id = os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"cpu\")',
                "    now = time.time()",
                "    _write_trace(",
                "        trace_path,",
                "        {",
                '            \"event\": \"trial_start\",',
                '            \"cycle\": cycle_index,',
                '            \"phase\": phase,',
                '            \"trial_id\": trial_id,',
                '            \"gpu_id\": gpu_id,',
                '            \"time\": now,',
                "        },",
                "    )",
                "    time.sleep(_scenario_sleep(scenario, cycle_index, phase, trial_id))",
                "    score = _scenario_score(scenario, cycle_index, phase, trial_id)",
                "    payload = {",
                '        \"validation_signature\": \"test-signature\",',
                '        \"validation_protocol\": {\"name\": \"test-protocol\"},',
                '        \"donor\": {',
                '            \"best_pr_auc\": score,',
                '            \"best_max_f1\": score,',
                '            \"best_metric\": \"max_f1\",',
                '            \"best_score\": score,',
                "        },",
                '        \"acceptor\": {',
                '            \"best_pr_auc\": score,',
                '            \"best_max_f1\": score,',
                '            \"best_metric\": \"max_f1\",',
                '            \"best_score\": score,',
                "        },",
                '        \"pair\": {',
                '            \"best_pr_auc\": score,',
                '            \"best_max_f1\": score,',
                '            \"best_metric\": \"max_f1\",',
                '            \"best_score\": score,',
                "        },",
                "    }",
                "    metrics_json.write_text(json.dumps(payload), encoding=\"utf-8\")",
                "    _write_trace(",
                "        trace_path,",
                "        {",
                '            \"event\": \"trial_end\",',
                '            \"cycle\": cycle_index,',
                '            \"phase\": phase,',
                '            \"trial_id\": trial_id,',
                '            \"gpu_id\": gpu_id,',
                '            \"time\": time.time(),',
                "        },",
                "    )",
                '    print(',
                '        f\"[fake_run_model] cycle={cycle_index} phase={phase} \"',
                '        f\"trial={trial_id:04d} gpu={gpu_id}\",',
                "        flush=True,",
                "    )",
                "    return 0",
                "",
                "",
                'if __name__ == \"__main__\":',
                "    raise SystemExit(main())",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fake_python_wrapper(path: Path, helper_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'REAL_PYTHON={json.dumps(sys.executable)}',
                f'HELPER={json.dumps(str(helper_path))}',
                'if [[ $# -ge 2 && "${1}" == "-u" && "${2}" == *"/src/run_model.py" ]]; then',
                "  shift 2",
                '  exec "${REAL_PYTHON}" "${HELPER}" "$@"',
                "fi",
                'if [[ $# -ge 1 && "${1}" == *"/src/run_model.py" ]]; then',
                "  shift 1",
                '  exec "${REAL_PYTHON}" "${HELPER}" "$@"',
                "fi",
                'exec "${REAL_PYTHON}" "$@"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _make_executable(path)


def _prepare_script_project(
    *,
    tmp_path: Path,
    script_name: str,
    patches: dict[str, str],
) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    (project_root / "run" / "lib").mkdir(parents=True)
    (project_root / "run").mkdir(exist_ok=True)
    (project_root / "src" / "tools").mkdir(parents=True)
    (project_root / "data" / "Dmel").mkdir(parents=True)

    common_src = _project_root() / "run" / "lib" / "common.sh"
    auto_tmux_src = _project_root() / "run" / "lib" / "auto_tmux.sh"
    tune_src = _project_root() / "run" / script_name
    scheduler_src = _project_root() / "src" / "tools" / "tune_time_scheduler.py"

    common_dst = project_root / "run" / "lib" / "common.sh"
    auto_tmux_dst = project_root / "run" / "lib" / "auto_tmux.sh"
    tune_dst = project_root / "run" / script_name
    scheduler_dst = project_root / "src" / "tools" / "tune_time_scheduler.py"

    common_dst.write_text(common_src.read_text(encoding="utf-8"), encoding="utf-8")
    auto_tmux_dst.write_text(
        auto_tmux_src.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    scheduler_dst.write_text(
        scheduler_src.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    tune_text = tune_src.read_text(encoding="utf-8")
    for name, value in patches.items():
        tune_text = _patch_assignment(tune_text, name, value)
    tune_dst.write_text(tune_text, encoding="utf-8")
    _make_executable(tune_dst)
    return project_root, tune_dst


def _run_with_fake_model(
    *,
    tmp_path: Path,
    project_root: Path,
    tune_dst: Path,
    scenario: str,
    timeout_seconds: float,
) -> tuple[str, str, Path]:
    fake_conda = tmp_path / "fake_conda.sh"
    _write_fake_conda(fake_conda)

    helper_path = tmp_path / "fake_run_model.py"
    _write_fake_run_model_helper(helper_path)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    _write_fake_python_wrapper(fake_python, helper_path)

    trace_path = tmp_path / "trace.jsonl"
    env = dict(os.environ)
    env["INTRONMODEL_CONDA_SH"] = str(fake_conda)
    env["INTRONMODEL_DATA_ROOT"] = str(project_root / "data")
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["TEST_TRACE_FILE"] = str(trace_path)
    env["TEST_SCENARIO"] = scenario
    env["INTRONMODEL_TRIAL_PYTHON_BIN"] = str(fake_python)
    env["PYTHONPATH"] = (
        f"{_project_root() / 'src'}:{env.get('PYTHONPATH', '')}"
        if env.get("PYTHONPATH")
        else str(_project_root() / "src")
    )

    try:
        run = subprocess.run(
            ["bash", str(tune_dst)],
            capture_output=True,
            text=True,
            check=False,
            cwd=project_root,
            env=env,
            timeout=timeout_seconds,
        )
        return run.stdout, run.stderr, trace_path
    except subprocess.TimeoutExpired as exc:
        if isinstance(exc.stdout, bytes):
            stdout = exc.stdout.decode()
        elif isinstance(exc.stdout, str):
            stdout = exc.stdout
        else:
            stdout = ""
        if isinstance(exc.stderr, bytes):
            stderr = exc.stderr.decode()
        elif isinstance(exc.stderr, str):
            stderr = exc.stderr
        else:
            stderr = ""
        return stdout, stderr, trace_path


def _load_trace_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_tune_cnn_pair_v3_time_overlaps_full_into_next_quick(
    tmp_path: Path,
) -> None:
    project_root, tune_dst = _prepare_script_project(
        tmp_path=tmp_path,
        script_name="tune_cnn_pair_v3_time.sh",
        patches={
            "TIME_BUDGET_MINUTES": "1",
            "INTRONMODEL_AUTO_TMUX": "off",
            "GPU_IDS": "0,1,2,3",
            "MAX_PARALLEL_TRIALS": "4",
            "QUICK_TRIALS": "4",
            "TOP_K": "2",
        },
    )
    stdout, stderr, trace_path = _run_with_fake_model(
        tmp_path=tmp_path,
        project_root=project_root,
        tune_dst=tune_dst,
        scenario="overlap",
        timeout_seconds=2.0,
    )

    assert stderr == ""
    trace_rows = _load_trace_rows(trace_path)
    cycle_zero_full_start = next(
        row
        for row in trace_rows
        if row["event"] == "trial_start"
        and row["cycle"] == 0
        and row["phase"] == "full"
    )
    cycle_zero_full_end = next(
        row
        for row in trace_rows
        if row["event"] == "trial_end"
        and row["cycle"] == 0
        and row["phase"] == "full"
    )
    cycle_one_quick_start = next(
        row
        for row in trace_rows
        if row["event"] == "trial_start"
        and row["cycle"] == 1
        and row["phase"] == "quick"
    )

    assert cycle_zero_full_start["time"] <= cycle_one_quick_start["time"]
    assert cycle_one_quick_start["time"] < cycle_zero_full_end["time"]
    assert "trial scheduler across GPUs: 0,1,2,3" in stdout
    assert "[hparam_search] quick trial 0000 started on gpu:0." in stdout
    assert "[hparam_search] full trial 0000 success" in stdout


def test_tune_cnn_pair_v3_time_grows_next_cycle_slot_budget_after_more_gpus_free(
    tmp_path: Path,
) -> None:
    project_root, tune_dst = _prepare_script_project(
        tmp_path=tmp_path,
        script_name="tune_cnn_pair_v3_time.sh",
        patches={
            "TIME_BUDGET_MINUTES": "1",
            "INTRONMODEL_AUTO_TMUX": "off",
            "GPU_IDS": "0,1,2,3",
            "MAX_PARALLEL_TRIALS": "4",
            "QUICK_TRIALS": "4",
            "TOP_K": "1",
        },
    )
    stdout, stderr, trace_path = _run_with_fake_model(
        tmp_path=tmp_path,
        project_root=project_root,
        tune_dst=tune_dst,
        scenario="grow_slots",
        timeout_seconds=2.0,
    )

    assert stderr == ""
    trace_rows = _load_trace_rows(trace_path)
    cycle_zero_full_end = next(
        row
        for row in trace_rows
        if row["event"] == "trial_end"
        and row["cycle"] == 0
        and row["phase"] == "full"
    )
    cycle_one_quick_starts = sorted(
        [
            row
            for row in trace_rows
            if row["event"] == "trial_start"
            and row["cycle"] == 1
            and row["phase"] == "quick"
        ],
        key=lambda row: float(row["time"]),
    )

    assert len(cycle_one_quick_starts) >= 4
    assert cycle_one_quick_starts[2]["time"] < cycle_zero_full_end["time"]
    assert cycle_zero_full_end["time"] <= cycle_one_quick_starts[3]["time"]
    assert len({str(row["gpu_id"]) for row in cycle_one_quick_starts[:4]}) == 4
    assert "[hparam_search] quick trial 0000 started on gpu:0." in stdout
    assert "[hparam_search] quick trial 0003 started on gpu:3." in stdout


@pytest.mark.parametrize(
    ("script_name", "gpu_ids"),
    (
        ("tune_cnn_v3_time.sh", "0,4,6,7"),
        ("tune_cnn_pair_v3_time.sh", "4,5,6,7"),
    ),
)
def test_tune_time_v3_scripts_do_not_start_next_cycle_before_gpu_release(
    tmp_path: Path,
    script_name: str,
    gpu_ids: str,
) -> None:
    project_root, tune_dst = _prepare_script_project(
        tmp_path=tmp_path,
        script_name=script_name,
        patches={
            "TIME_BUDGET_MINUTES": "1",
            "INTRONMODEL_AUTO_TMUX": "off",
            "GPU_IDS": gpu_ids,
            "MAX_PARALLEL_TRIALS": "2",
            "QUICK_TRIALS": "4",
            "TOP_K": "2",
        },
    )
    stdout, stderr, trace_path = _run_with_fake_model(
        tmp_path=tmp_path,
        project_root=project_root,
        tune_dst=tune_dst,
        scenario="no_release",
        timeout_seconds=1.4,
    )

    assert stderr == ""
    trace_rows = _load_trace_rows(trace_path)
    quick_starts = [
        row
        for row in trace_rows
        if row["event"] == "trial_start" and row["phase"] == "quick"
    ]

    assert quick_starts
    assert {int(row["cycle"]) for row in quick_starts} == {0}
    assert "trial scheduler across GPUs" in stdout
