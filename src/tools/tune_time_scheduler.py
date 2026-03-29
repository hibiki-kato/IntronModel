from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any


@dataclass(frozen=True)
class CycleTemplate:
    """Static template for one repeating tuning cycle."""

    species: str
    target_name: str
    seed: int
    tuning_model_name: str
    template_config_path: Path
    output_parent_dir: Path
    plot_target_name: str | None


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for the centralized tune-time scheduler."""

    script_name: str
    project_root: Path
    data_root: Path
    model_root: Path
    python_bin: str
    hparam_search_path: Path
    time_budget_minutes: int
    timeout_grace_seconds: int
    selected_gpu_ids: list[str]
    parallel_slot_count: int
    start_epoch: str
    jobs: list[CycleTemplate]


@dataclass
class RunningCycle:
    """Mutable runtime state for one active tuning cycle."""

    cycle_index: int
    template: CycleTemplate
    assigned_gpu_ids: list[str]
    output_dir: Path
    config_path: Path
    stdout_log: Path
    release_file: Path
    process: subprocess.Popen[str]
    stream_thread: threading.Thread
    start_time: float
    release_cursor: int = 0
    owned_gpu_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.owned_gpu_ids is None:
            self.owned_gpu_ids = list(self.assigned_gpu_ids)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Centralized queue scheduler for tune_*_time wrappers."
    )
    parser.add_argument("--config", required=True, help="Path to scheduler JSON.")
    return parser.parse_args(argv)


def _require_str(raw: Any, field_name: str) -> str:
    """Return one required non-empty string."""
    if not isinstance(raw, str) or raw.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty string.")
    return raw


def _require_positive_int(raw: Any, field_name: str) -> int:
    """Return one required positive integer."""
    if not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return raw


def _require_non_negative_int(raw: Any, field_name: str) -> int:
    """Return one required non-negative integer."""
    if not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return raw


def _load_jobs(path: Path) -> list[CycleTemplate]:
    """Load cycle templates from one JSONL manifest."""
    jobs: list[CycleTemplate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text == "":
            continue
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("Each jobs.jsonl row must be a JSON object.")
        plot_target_raw = raw.get("plot_target_name")
        if plot_target_raw is not None and not isinstance(plot_target_raw, str):
            raise ValueError("plot_target_name must be null or a string.")
        jobs.append(
            CycleTemplate(
                species=_require_str(raw.get("species"), "jobs[].species"),
                target_name=_require_str(raw.get("target_name"), "jobs[].target_name"),
                seed=_require_non_negative_int(raw.get("seed"), "jobs[].seed"),
                tuning_model_name=_require_str(
                    raw.get("tuning_model_name"),
                    "jobs[].tuning_model_name",
                ),
                template_config_path=Path(
                    _require_str(
                        raw.get("template_config_path"),
                        "jobs[].template_config_path",
                    )
                ),
                output_parent_dir=Path(
                    _require_str(
                        raw.get("output_parent_dir"),
                        "jobs[].output_parent_dir",
                    )
                ),
                plot_target_name=plot_target_raw,
            )
        )
    if not jobs:
        raise ValueError("Scheduler manifest must contain at least one job.")
    return jobs


def _load_config(path: Path) -> SchedulerConfig:
    """Load one scheduler configuration JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Scheduler config root must be a JSON object.")
    selected_gpu_ids_raw = raw.get("selected_gpu_ids")
    if not isinstance(selected_gpu_ids_raw, list):
        raise ValueError("selected_gpu_ids must be a JSON array.")
    selected_gpu_ids = [
        _require_str(item, "selected_gpu_ids[]") for item in selected_gpu_ids_raw
    ]
    jobs_file = Path(_require_str(raw.get("jobs_file"), "jobs_file"))
    return SchedulerConfig(
        script_name=_require_str(raw.get("script_name"), "script_name"),
        project_root=Path(_require_str(raw.get("project_root"), "project_root")),
        data_root=Path(_require_str(raw.get("data_root"), "data_root")),
        model_root=Path(_require_str(raw.get("model_root"), "model_root")),
        python_bin=_require_str(raw.get("python_bin"), "python_bin"),
        hparam_search_path=Path(
            _require_str(raw.get("hparam_search_path"), "hparam_search_path")
        ),
        time_budget_minutes=_require_positive_int(
            raw.get("time_budget_minutes"),
            "time_budget_minutes",
        ),
        timeout_grace_seconds=_require_positive_int(
            raw.get("timeout_grace_seconds"),
            "timeout_grace_seconds",
        ),
        selected_gpu_ids=selected_gpu_ids,
        parallel_slot_count=_require_positive_int(
            raw.get("parallel_slot_count"),
            "parallel_slot_count",
        ),
        start_epoch=_require_str(raw.get("start_epoch"), "start_epoch"),
        jobs=_load_jobs(jobs_file),
    )


def _format_elapsed(seconds: int) -> str:
    """Return one compact ``HH:MM:SS`` or ``MM:SS`` duration string."""
    if seconds < 0:
        seconds = 0
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _append_unique_gpu_ids(target: list[str], candidates: list[str]) -> None:
    """Append unseen GPU ids while preserving order."""
    for candidate in candidates:
        if candidate == "" or candidate in target:
            continue
        target.append(candidate)


def _collect_released_gpu_ids(
    release_file: Path,
    cursor: int,
) -> tuple[list[str], int]:
    """Collect newly released GPU ids from one JSONL file cursor."""
    if not release_file.is_file():
        return [], cursor
    lines = release_file.read_text(encoding="utf-8").splitlines()
    released: list[str] = []
    for line in lines[cursor:]:
        text = line.strip()
        if text == "":
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            continue
        if payload.get("event") != "gpu_released":
            continue
        gpu_id = payload.get("gpu_id")
        if isinstance(gpu_id, str) and gpu_id != "":
            released.append(gpu_id)
    return released, len(lines)


def _write_cycle_config(
    *,
    template: CycleTemplate,
    cycle_index: int,
    assigned_gpu_ids: list[str],
    assigned_parallel_slots: int,
) -> tuple[Path, Path, Path]:
    """Materialize one per-cycle hparam_search config from a template."""
    payload = json.loads(template.template_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Template config must contain a JSON object.")
    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_stamp}_seed{template.seed}_c{cycle_index:03d}"
    output_dir = template.output_parent_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "hparam_search_config.json"
    release_file = output_dir / "gpu_release_events.jsonl"
    payload["output_dir"] = str(output_dir)
    payload["gpu_ids"] = ",".join(assigned_gpu_ids)
    payload["max_parallel_trials"] = str(assigned_parallel_slots)
    payload["gpu_release_events_path"] = str(release_file)
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_dir, config_path, release_file


def _build_cycle_prefix(script_name: str, cycle_index: int, template: CycleTemplate) -> str:
    """Build the stable log prefix for one running cycle."""
    return (
        f"[{script_name}][cycle={cycle_index}]"
        f"[species={template.species}]"
        f"[target={template.target_name}]"
        f"[seed={template.seed}]"
    )


def _stream_cycle_output(
    *,
    prefix: str,
    process: subprocess.Popen[str],
    stdout_log: Path,
) -> None:
    """Stream one subprocess's merged stdout/stderr with a cycle prefix."""
    assert process.stdout is not None
    with stdout_log.open("a", encoding="utf-8") as log_handle:
        for line in process.stdout:
            text = line.rstrip("\n")
            rendered = f"{prefix} {text}" if text != "" else prefix
            print(rendered, flush=True)
            log_handle.write(rendered + "\n")
            log_handle.flush()


def _launch_cycle(
    *,
    config: SchedulerConfig,
    cycle_index: int,
    template: CycleTemplate,
    assigned_gpu_ids: list[str],
    assigned_parallel_slots: int,
    elapsed_seconds: int,
    remaining_seconds: int,
) -> RunningCycle:
    """Launch one hparam_search subprocess for a scheduled cycle."""
    output_dir, config_path, release_file = _write_cycle_config(
        template=template,
        cycle_index=cycle_index,
        assigned_gpu_ids=assigned_gpu_ids,
        assigned_parallel_slots=assigned_parallel_slots,
    )
    stdout_log = output_dir / "cycle_stdout.log"
    remaining_hms = _format_elapsed(remaining_seconds)
    print(
        f"[{config.script_name}] cycle={cycle_index} "
        f"elapsed={_format_elapsed(elapsed_seconds)} "
        f"start={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"ETA:{remaining_hms} species={template.species} "
        f"target={template.target_name} seed={template.seed} "
        f"gpus={','.join(assigned_gpu_ids)} log={stdout_log}",
        flush=True,
    )
    env = os.environ.copy()
    pythonpath_items = [str(config.project_root / "src")]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath != "":
        pythonpath_items.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_items)
    process = subprocess.Popen(
        [config.python_bin, str(config.hparam_search_path), "--config", str(config_path)],
        cwd=config.project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    prefix = _build_cycle_prefix(config.script_name, cycle_index, template)
    stream_thread = threading.Thread(
        target=_stream_cycle_output,
        kwargs={
            "prefix": prefix,
            "process": process,
            "stdout_log": stdout_log,
        },
        daemon=True,
    )
    stream_thread.start()
    return RunningCycle(
        cycle_index=cycle_index,
        template=template,
        assigned_gpu_ids=list(assigned_gpu_ids),
        output_dir=output_dir,
        config_path=config_path,
        stdout_log=stdout_log,
        release_file=release_file,
        process=process,
        stream_thread=stream_thread,
        start_time=time.monotonic(),
    )


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: int,
) -> None:
    """Terminate one process group and escalate to SIGKILL if needed."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _prune_timeout_artifacts(
    *,
    config: SchedulerConfig,
    running_cycle: RunningCycle,
) -> None:
    """Remove partial timeout outputs and prune dangling rank checkpoints."""
    if running_cycle.output_dir.is_dir():
        shutil.rmtree(running_cycle.output_dir)
        print(
            f"[{config.script_name}] removed partial output dir: "
            f"{running_cycle.output_dir}",
            flush=True,
        )
    subprocess.run(
        [
            config.python_bin,
            str(config.project_root / "src" / "tools" / "prune_missing_rank_checkpoints.py"),
            "--data_root",
            str(config.data_root),
            "--model_root",
            str(config.model_root),
            "--species",
            running_cycle.template.species,
            "--model",
            running_cycle.template.tuning_model_name,
            "--dry_run",
            "0",
        ],
        cwd=config.project_root,
        check=False,
    )


def _run_cycle_plot(
    *,
    config: SchedulerConfig,
    running_cycle: RunningCycle,
) -> None:
    """Update the optional double-descent plot for one completed cycle."""
    plot_target_name = running_cycle.template.plot_target_name
    if plot_target_name is None:
        return
    command = [
        config.python_bin,
        str(config.project_root / "src" / "tools" / "plot_tuning_double_descent.py"),
        "--project_root",
        str(config.project_root),
        "--species",
        running_cycle.template.species,
        "--target",
        plot_target_name,
        "--model",
        running_cycle.template.tuning_model_name,
    ]
    with running_cycle.stdout_log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=config.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            handle.write(result.stdout)
        if result.stderr:
            handle.write(result.stderr)


def _should_dispatch_next_cycle(
    *,
    script_name: str,
    remaining_seconds: int,
    total_cycle_seconds: int,
    completed_cycles: int,
) -> bool:
    """Return whether one new cycle should be dispatched."""
    if completed_cycles <= 0:
        return True
    avg_cycle_seconds = total_cycle_seconds // completed_cycles
    if avg_cycle_seconds > 0 and remaining_seconds < avg_cycle_seconds:
        print(
            f"[{script_name}] stop before next cycle: "
            f"remaining={_format_elapsed(remaining_seconds)} "
            f"< avg_cycle={_format_elapsed(avg_cycle_seconds)}",
            flush=True,
        )
        return False
    return True


def _resolve_parallel_slots(max_parallel_trials: int, available_slots: int) -> int:
    """Cap requested parallel slots to currently available slots."""
    if available_slots <= 0:
        return 0
    return min(max_parallel_trials, available_slots)


def run_scheduler(config: SchedulerConfig) -> int:
    """Run the centralized cycle queue until the time budget is exhausted."""
    if not config.jobs:
        raise ValueError("At least one scheduler job is required.")
    start_time = time.monotonic()
    budget_seconds = config.time_budget_minutes * 60
    deadline = start_time + budget_seconds
    total_cycle_seconds = 0
    completed_cycles = 0
    cycle_index = 0
    first_error_code = 0
    max_active_cycles = 2
    running_cycles: list[RunningCycle] = []
    available_gpu_ids = list(config.selected_gpu_ids)
    if available_gpu_ids:
        print(
            f"[{config.script_name}] cycle-parallel scheduler across GPUs: "
            f"{','.join(available_gpu_ids[: config.parallel_slot_count])}",
            flush=True,
        )
    else:
        print(
            f"[{config.script_name}] cycle scheduler using CPU fallback.",
            flush=True,
        )
    stop_submitting = False
    while running_cycles or not stop_submitting:
        progress = False
        released_gpu_progress = False
        elapsed_seconds = int(time.monotonic() - start_time)
        remaining_seconds = max(0, int(deadline - time.monotonic()))
        min_dispatch_slots = 1
        if running_cycles and config.parallel_slot_count > 1:
            min_dispatch_slots = 2

        for running_cycle in running_cycles:
            released_gpu_ids, next_cursor = _collect_released_gpu_ids(
                running_cycle.release_file,
                running_cycle.release_cursor,
            )
            running_cycle.release_cursor = next_cursor
            if not released_gpu_ids:
                continue
            _append_unique_gpu_ids(available_gpu_ids, released_gpu_ids)
            released_gpu_progress = True
            running_cycle.owned_gpu_ids = [
                gpu_id
                for gpu_id in running_cycle.owned_gpu_ids or []
                if gpu_id not in released_gpu_ids
            ]
            progress = True

        if not stop_submitting and not _should_dispatch_next_cycle(
            script_name=config.script_name,
            remaining_seconds=remaining_seconds,
            total_cycle_seconds=total_cycle_seconds,
            completed_cycles=completed_cycles,
        ):
            stop_submitting = True

        if time.monotonic() >= deadline and running_cycles:
            stop_submitting = True
            if first_error_code == 0:
                first_error_code = 124
            for running_cycle in running_cycles:
                _terminate_process_group(
                    running_cycle.process,
                    grace_seconds=config.timeout_grace_seconds,
                )
                running_cycle.stream_thread.join(timeout=5.0)
                _prune_timeout_artifacts(
                    config=config,
                    running_cycle=running_cycle,
                )
            running_cycles.clear()
            break

        while True:
            can_launch_next_cycle = True
            if stop_submitting:
                can_launch_next_cycle = False
            elif running_cycles and not released_gpu_progress:
                can_launch_next_cycle = False
            elif available_gpu_ids and len(available_gpu_ids) < min_dispatch_slots:
                can_launch_next_cycle = False
            elif not available_gpu_ids and running_cycles:
                can_launch_next_cycle = False
            elif len(running_cycles) >= max_active_cycles:
                can_launch_next_cycle = False
            elif len(running_cycles) == 1:
                active_cycle = running_cycles[0].cycle_index
                if cycle_index != active_cycle + 1:
                    can_launch_next_cycle = False
            if not can_launch_next_cycle:
                break
            if available_gpu_ids:
                assigned_parallel_slots = _resolve_parallel_slots(
                    config.parallel_slot_count,
                    len(available_gpu_ids),
                )
                if assigned_parallel_slots <= 0:
                    break
                assigned_gpu_ids = available_gpu_ids[:assigned_parallel_slots]
                del available_gpu_ids[:assigned_parallel_slots]
            else:
                if running_cycles:
                    break
                assigned_parallel_slots = 1
                assigned_gpu_ids = []
            template = config.jobs[cycle_index % len(config.jobs)]
            running_cycles.append(
                _launch_cycle(
                    config=config,
                    cycle_index=cycle_index,
                    template=template,
                    assigned_gpu_ids=assigned_gpu_ids,
                    assigned_parallel_slots=assigned_parallel_slots,
                    elapsed_seconds=elapsed_seconds,
                    remaining_seconds=remaining_seconds,
                )
            )
            cycle_index += 1
            progress = True

        next_running_cycles: list[RunningCycle] = []
        for running_cycle in running_cycles:
            return_code = running_cycle.process.poll()
            if return_code is None:
                next_running_cycles.append(running_cycle)
                continue
            running_cycle.stream_thread.join(timeout=5.0)
            _append_unique_gpu_ids(
                available_gpu_ids,
                list(running_cycle.owned_gpu_ids or []),
            )
            cycle_duration_seconds = max(
                0,
                int(time.monotonic() - running_cycle.start_time),
            )
            if return_code in {124, 130}:
                if first_error_code == 0:
                    first_error_code = return_code
                stop_submitting = True
            else:
                total_cycle_seconds += cycle_duration_seconds
                completed_cycles += 1
                _run_cycle_plot(
                    config=config,
                    running_cycle=running_cycle,
                )
            avg_cycle_seconds = (
                total_cycle_seconds // completed_cycles if completed_cycles > 0 else 0
            )
            remaining_seconds = max(0, int(deadline - time.monotonic()))
            estimated_cycles_left = (
                remaining_seconds // avg_cycle_seconds if avg_cycle_seconds > 0 else 0
            )
            print(
                f"[{config.script_name}] cycle_done={running_cycle.cycle_index} "
                f"cycle_time={_format_elapsed(cycle_duration_seconds)} "
                f"avg_cycle={_format_elapsed(avg_cycle_seconds)} "
                f"ETA_cycles_left={estimated_cycles_left} "
                f"log={running_cycle.stdout_log} exit={return_code}",
                flush=True,
            )
            progress = True
        running_cycles = next_running_cycles

        if not running_cycles and stop_submitting:
            break
        if not progress:
            time.sleep(0.1)

    if first_error_code != 0:
        return first_error_code
    total_seconds = int(time.monotonic() - start_time)
    print(
        f"[{config.script_name}] done start={config.start_epoch} "
        f"end={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"elapsed={_format_elapsed(total_seconds)} cycles={cycle_index}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    actual_argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(actual_argv)
    config = _load_config(Path(args.config))
    try:
        return run_scheduler(config)
    except KeyboardInterrupt:
        print("[tune_time_scheduler] Interrupted by user.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
