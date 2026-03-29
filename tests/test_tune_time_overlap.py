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


def test_tune_cnn_pair_v3_time_overlaps_full_into_next_quick(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "run" / "lib").mkdir(parents=True)
    (project_root / "run").mkdir(exist_ok=True)
    (project_root / "src" / "tools").mkdir(parents=True)
    (project_root / "data" / "Dmel").mkdir(parents=True)

    common_src = _project_root() / "run" / "lib" / "common.sh"
    auto_tmux_src = _project_root() / "run" / "lib" / "auto_tmux.sh"
    tune_src = _project_root() / "run" / "tune_cnn_pair_v3_time.sh"
    scheduler_src = _project_root() / "src" / "tools" / "tune_time_scheduler.py"
    common_dst = project_root / "run" / "lib" / "common.sh"
    auto_tmux_dst = project_root / "run" / "lib" / "auto_tmux.sh"
    tune_dst = project_root / "run" / "tune_cnn_pair_v3_time.sh"
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
    tune_text = _patch_assignment(tune_text, "TIME_BUDGET_MINUTES", "1")
    tune_text = _patch_assignment(tune_text, "INTRONMODEL_AUTO_TMUX", "off")
    tune_text = _patch_assignment(tune_text, "GPU_IDS", "0,1,2,3")
    tune_dst.write_text(tune_text, encoding="utf-8")
    _make_executable(tune_dst)

    fake_conda = tmp_path / "fake_conda.sh"
    fake_conda.write_text(
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
    _make_executable(fake_conda)

    trace_path = tmp_path / "trace.jsonl"
    fake_hparam_helper = tmp_path / "fake_hparam_search.py"
    fake_hparam_helper.write_text(
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
                "def _write_trace(path: Path, payload: dict[str, object]) -> None:",
                '    with path.open("a", encoding="utf-8") as handle:',
                "        handle.write(json.dumps(payload, sort_keys=True))",
                '        handle.write("\\n")',
                "",
                "",
                "def main() -> int:",
                "    args = sys.argv[1:]",
                '    config_index = args.index("--config")',
                "    config_path = Path(args[config_index + 1])",
                '    config = json.loads(config_path.read_text(encoding="utf-8"))',
                '    output_dir = Path(str(config["output_dir"]))',
                '    match = re.search(r"_c(\\d+)$", output_dir.name)',
                "    if match is None:",
                '        raise ValueError("cycle suffix not found in output_dir name")',
                "    cycle_index = int(match.group(1))",
                '    gpu_ids = [gpu for gpu in str(config["gpu_ids"]).split(",") if gpu]',
                '    trace_path = Path(os.environ["TEST_TRACE_FILE"])',
                "    start_payload = {",
                '        "event": "quick_start",',
                '        "cycle": cycle_index,',
                '        "gpu_ids": gpu_ids,',
                '        "time": time.time(),',
                "    }",
                "    _write_trace(trace_path, start_payload)",
                '    print(f"[fake_hparam_search] quick cycle={cycle_index} gpus={gpu_ids}", flush=True)',
                "    if cycle_index == 0:",
                "        time.sleep(0.10)",
                "        _write_trace(",
                "            trace_path,",
                "            {",
                '                "event": "full_start",',
                '                "cycle": cycle_index,',
                '                "gpu_ids": gpu_ids[:2],',
                '                "time": time.time(),',
                "            },",
                "        )",
                '        print("[fake_hparam_search] full overlap started", flush=True)',
                '        release_path_raw = config.get("gpu_release_events_path")',
                "        if isinstance(release_path_raw, str) and release_path_raw:",
                "            release_path = Path(release_path_raw)",
                '            release_path.parent.mkdir(parents=True, exist_ok=True)',
                '            with release_path.open("a", encoding="utf-8") as handle:',
                "                for gpu_id in gpu_ids[2:]:",
                "                    handle.write(",
                "                        json.dumps(",
                "                            {",
                '                                "event": "gpu_released",',
                '                                "gpu_id": gpu_id,',
                '                                "reason": "full_only_idle_slot",',
                '                                "timestamp": time.time(),',
                "                            },",
                "                            sort_keys=True,",
                "                        )",
                "                    )",
                '                    handle.write("\\n")',
                "        time.sleep(0.35)",
                "        _write_trace(",
                "            trace_path,",
                "            {",
                '                "event": "cycle_end",',
                '                "cycle": cycle_index,',
                '                "time": time.time(),',
                "            },",
                "        )",
                "        return 0",
                "    _write_trace(",
                "        trace_path,",
                "        {",
                '            "event": "cycle_end",',
                '            "cycle": cycle_index,',
                '            "time": time.time(),',
                "        },",
                "    )",
                "    return 124",
                "",
                "",
                'if __name__ == "__main__":',
                "    raise SystemExit(main())",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'REAL_PYTHON={json.dumps(sys.executable)}',
                f'HELPER={json.dumps(str(fake_hparam_helper))}',
                'if [[ $# -ge 1 && "${1}" == *"/src/tools/hparam_search.py" ]]; then',
                "  shift",
                '  exec "${REAL_PYTHON}" "${HELPER}" "$@"',
                "fi",
                'exec "${REAL_PYTHON}" "$@"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _make_executable(fake_python)

    env = dict(os.environ)
    env["INTRONMODEL_CONDA_SH"] = str(fake_conda)
    env["INTRONMODEL_DATA_ROOT"] = str(project_root / "data")
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["TEST_TRACE_FILE"] = str(trace_path)

    run = subprocess.run(
        ["bash", str(tune_dst)],
        capture_output=True,
        text=True,
        check=False,
        cwd=project_root,
        env=env,
        timeout=20,
    )

    assert run.returncode == 124, run.stderr
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quick_zero = next(
        row for row in trace_rows if row["event"] == "quick_start" and row["cycle"] == 0
    )
    full_zero = next(
        row for row in trace_rows if row["event"] == "full_start" and row["cycle"] == 0
    )
    quick_one = next(
        row for row in trace_rows if row["event"] == "quick_start" and row["cycle"] == 1
    )
    end_zero = next(
        row for row in trace_rows if row["event"] == "cycle_end" and row["cycle"] == 0
    )

    assert full_zero["time"] >= quick_zero["time"]
    assert quick_one["time"] < end_zero["time"]
    assert quick_one["gpu_ids"] == ["2", "3"]
    assert "cycle-parallel scheduler across GPUs: 0,1,2,3" in run.stdout
    assert "[fake_hparam_search] quick cycle=0" in run.stdout
    assert "[fake_hparam_search] quick cycle=1" in run.stdout


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
    tune_text = _patch_assignment(tune_text, "TIME_BUDGET_MINUTES", "1")
    tune_text = _patch_assignment(tune_text, "INTRONMODEL_AUTO_TMUX", "off")
    tune_text = _patch_assignment(tune_text, "GPU_IDS", gpu_ids)
    tune_text = _patch_assignment(tune_text, "MAX_PARALLEL_TRIALS", "2")
    tune_dst.write_text(tune_text, encoding="utf-8")
    _make_executable(tune_dst)

    fake_conda = tmp_path / "fake_conda.sh"
    fake_conda.write_text(
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
    _make_executable(fake_conda)

    trace_path = tmp_path / "trace_no_release.jsonl"
    fake_hparam_helper = tmp_path / "fake_hparam_search_no_release.py"
    fake_hparam_helper.write_text(
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
                "def _write_trace(path: Path, payload: dict[str, object]) -> None:",
                '    with path.open("a", encoding="utf-8") as handle:',
                "        handle.write(json.dumps(payload, sort_keys=True))",
                '        handle.write("\\n")',
                "",
                "",
                "def main() -> int:",
                "    args = sys.argv[1:]",
                '    config_index = args.index("--config")',
                "    config_path = Path(args[config_index + 1])",
                '    config = json.loads(config_path.read_text(encoding="utf-8"))',
                '    output_dir = Path(str(config["output_dir"]))',
                '    match = re.search(r"_c(\\d+)$", output_dir.name)',
                "    if match is None:",
                '        raise ValueError("cycle suffix not found in output_dir name")',
                "    cycle_index = int(match.group(1))",
                '    gpu_ids = [gpu for gpu in str(config["gpu_ids"]).split(",") if gpu]',
                '    trace_path = Path(os.environ["TEST_TRACE_FILE"])',
                "    _write_trace(",
                "        trace_path,",
                "        {",
                '            "event": "quick_start",',
                '            "cycle": cycle_index,',
                '            "gpu_ids": gpu_ids,',
                '            "time": time.time(),',
                "        },",
                "    )",
                '    print(f"[fake_hparam_search] quick cycle={cycle_index} gpus={gpu_ids}", flush=True)',
                "    time.sleep(0.15)",
                "    return 124",
                "",
                "",
                'if __name__ == "__main__":',
                "    raise SystemExit(main())",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'REAL_PYTHON={json.dumps(sys.executable)}',
                f'HELPER={json.dumps(str(fake_hparam_helper))}',
                'if [[ $# -ge 1 && "${1}" == *"/src/tools/hparam_search.py" ]]; then',
                "  shift",
                '  exec "${REAL_PYTHON}" "${HELPER}" "$@"',
                "fi",
                'exec "${REAL_PYTHON}" "$@"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _make_executable(fake_python)

    env = dict(os.environ)
    env["INTRONMODEL_CONDA_SH"] = str(fake_conda)
    env["INTRONMODEL_DATA_ROOT"] = str(project_root / "data")
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["TEST_TRACE_FILE"] = str(trace_path)

    run = subprocess.run(
        ["bash", str(tune_dst)],
        capture_output=True,
        text=True,
        check=False,
        cwd=project_root,
        env=env,
        timeout=20,
    )

    assert run.returncode == 124, run.stderr
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quick_starts = [row for row in trace_rows if row["event"] == "quick_start"]

    assert len(quick_starts) == 1
    assert quick_starts[0]["cycle"] == 0
