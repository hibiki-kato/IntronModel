from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import TypeAlias

from tools import hparam_search
from util.process_title import apply_process_title_from_env

_ = apply_process_title_from_env()

Scalar: TypeAlias = hparam_search.Scalar
ArgValue: TypeAlias = hparam_search.ArgValue
TrialResult: TypeAlias = hparam_search.TrialResult
ScheduledTrialTask: TypeAlias = hparam_search.ScheduledTrialTask


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
    time_budget_minutes: int
    timeout_grace_seconds: int
    selected_gpu_ids: list[str]
    parallel_slot_count: int
    start_epoch: str
    jobs: list[CycleTemplate]


@dataclass
class CycleState:
    """Mutable state for one active tuning cycle."""

    cycle_index: int
    template: CycleTemplate
    config: hparam_search.SearchConfig
    output_dir: Path
    config_path: Path
    stdout_log: Path
    start_time: float
    baseline_validation_protocol: dict[str, object]
    current_hparam_context: dict[str, object]
    previous_global_best_score: float | None
    quick_overrides: dict[str, ArgValue]
    full_overrides: dict[str, ArgValue]
    full_epochs_value: int
    quick_params: list[dict[str, Scalar]]
    quick_pending_indices: list[int]
    quick_rows: list[TrialResult] = field(default_factory=list)
    full_rows: list[TrialResult] = field(default_factory=list)
    completed_quick_rows: list[TrialResult] = field(default_factory=list)
    pending_full_tasks: list[ScheduledTrialTask] = field(default_factory=list)
    full_consumed_keys: set[str] = field(default_factory=set)
    seed_best_params: dict[str, Scalar] | None = None
    seed_best_key: str | None = None
    seed_best_context_mismatch: bool = False
    global_best_recheck_params: dict[str, Scalar] | None = None
    global_best_recheck_context_mismatch: bool = False
    full_priority_params: dict[str, Scalar] | None = None
    full_priority_inserted: bool = False
    next_full_trial_id: int = 0
    skipped_same_best_epoch: int = 0
    skipped_seed_context_match: int = 0
    quick_running_count: int = 0
    full_running_count: int = 0
    full_queue_built: bool = False
    resolved_trial_stream_mode: str = "full"
    start_logged: bool = False
    finalized: bool = False
    exit_code: int = 0

    @property
    def prefix(self) -> str:
        """Return the stable log prefix for one active cycle."""
        return (
            f"[{self.config.base_args.get('script_name', 'tune_time_scheduler')}]"
            f"[cycle={self.cycle_index}]"
            f"[species={self.template.species}]"
            f"[target={self.template.target_name}]"
            f"[seed={self.template.seed}]"
        )

    @property
    def running_count(self) -> int:
        """Return the number of currently running trials for the cycle."""
        return self.quick_running_count + self.full_running_count

    @property
    def has_quick_work(self) -> bool:
        """Return whether the cycle still owns quick-phase work."""
        return bool(self.quick_pending_indices) or self.quick_running_count > 0


@dataclass(frozen=True)
class RunningTrial:
    """One globally scheduled active trial."""

    cycle_index: int
    task: ScheduledTrialTask
    assigned_gpu_id: str | None
    known_trial_count: int


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Centralized direct-trial scheduler for tune_*_time wrappers."
    )
    parser.add_argument("--config", required=True, help="Path to scheduler JSON.")
    return parser.parse_args(argv)


def _require_str(raw: object, field_name: str) -> str:
    """Return one required non-empty string."""
    if not isinstance(raw, str) or raw.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty string.")
    return raw


def _require_positive_int(raw: object, field_name: str) -> int:
    """Return one required positive integer."""
    if not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return raw


def _require_non_negative_int(raw: object, field_name: str) -> int:
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
    """Return one compact duration string."""
    if seconds < 0:
        seconds = 0
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _emit_scheduler_line(config: SchedulerConfig, text: str) -> None:
    """Emit one scheduler-scoped log line."""
    print(f"[{config.script_name}] {text}", flush=True)


def _emit_cycle_line(cycle: CycleState, text: str) -> None:
    """Emit one cycle-scoped log line and mirror it to the cycle log."""
    rendered = (
        f"[{cycle.config.base_args.get('script_name', 'tune_time_scheduler')}]"
        f"[cycle={cycle.cycle_index}]"
        f"[species={cycle.template.species}]"
        f"[target={cycle.template.target_name}]"
        f"[seed={cycle.template.seed}] {text}"
    )
    print(rendered, flush=True)
    with cycle.stdout_log.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")


def _cycle_log_prefix(cycle: CycleState) -> str:
    """Return the stable log prefix used for one cycle."""
    return (
        f"[{cycle.config.base_args.get('script_name', 'tune_time_scheduler')}]"
        f"[cycle={cycle.cycle_index}]"
        f"[species={cycle.template.species}]"
        f"[target={cycle.template.target_name}]"
        f"[seed={cycle.template.seed}]"
    )


def _emit_trial_start_line(
    *,
    cycle: CycleState,
    task: ScheduledTrialTask,
    assigned_gpu_id: str | None,
) -> None:
    """Emit one trial-start line using the historical hparam_search format."""
    if not hparam_search._should_print_trial_start(cycle.resolved_trial_stream_mode):
        return
    with hparam_search.trial_log_prefix(_cycle_log_prefix(cycle)):
        hparam_search._print_trial_start(
            phase=task.phase,
            trial_id=task.trial_id,
            assigned_gpu=assigned_gpu_id,
        )


def _emit_trial_result_line(
    *,
    cycle: CycleState,
    task: ScheduledTrialTask,
    result: TrialResult,
    known_trial_count: int,
) -> None:
    """Emit one trial-result line using the historical hparam_search format."""
    if not hparam_search._should_print_trial_result_line(
        cycle.resolved_trial_stream_mode
    ):
        return
    completed_count: int
    if task.phase == "quick":
        completed_count = len(cycle.quick_rows)
    else:
        completed_count = len(cycle.full_rows)
    with hparam_search.trial_log_prefix(_cycle_log_prefix(cycle)):
        hparam_search._print_trial_result(
            phase=task.phase,
            trial_count=max(1, known_trial_count),
            completed_count=completed_count,
            result=result,
        )


def _build_cycle_runtime_paths(
    *,
    template: CycleTemplate,
    cycle_index: int,
) -> tuple[Path, Path, Path]:
    """Create per-cycle output paths."""
    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_stamp}_seed{template.seed}_c{cycle_index:03d}"
    output_dir = template.output_parent_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "hparam_search_config.json"
    stdout_log = output_dir / "cycle_stdout.log"
    stdout_log.touch()
    return output_dir, config_path, stdout_log


def _write_materialized_cycle_config(
    *,
    template: CycleTemplate,
    scheduler_config: SchedulerConfig,
    cycle_index: int,
) -> tuple[hparam_search.SearchConfig, Path, Path, Path]:
    """Write one materialized per-cycle config for reproducibility."""
    output_dir, config_path, stdout_log = _build_cycle_runtime_paths(
        template=template,
        cycle_index=cycle_index,
    )
    payload = json.loads(template.template_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Template config must contain a JSON object.")
    payload["output_dir"] = str(output_dir)
    payload["gpu_ids"] = scheduler_config.selected_gpu_ids
    payload["max_parallel_trials"] = scheduler_config.parallel_slot_count
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cycle_config = hparam_search.load_config(config_path)
    cycle_config.base_args["script_name"] = scheduler_config.script_name
    return cycle_config, output_dir, config_path, stdout_log


def _prepare_cycle_state(
    *,
    scheduler_config: SchedulerConfig,
    template: CycleTemplate,
    cycle_index: int,
    total_slot_count: int,
) -> CycleState:
    """Initialize one cycle state from the materialized hparam_search config."""
    cycle_config, output_dir, config_path, stdout_log = _write_materialized_cycle_config(
        template=template,
        scheduler_config=scheduler_config,
        cycle_index=cycle_index,
    )
    if cycle_config.trial_process_mode != "subprocess":
        raise ValueError(
            "Direct tune-time scheduler currently requires "
            "trial_process_mode=subprocess."
        )

    baseline_validation_protocol = hparam_search._derive_validation_protocol_from_args(
        merged_args=dict(cycle_config.base_args),
        objective_metric=cycle_config.objective_metric,
    )
    full_overrides = dict(cycle_config.full_overrides)
    full_overrides.setdefault("epochs", cycle_config.full_epochs)
    full_overrides.setdefault("compile_mode", "auto")
    full_epochs_value = hparam_search._to_positive_int(full_overrides.get("epochs"))
    if full_epochs_value is None:
        full_epochs_value = cycle_config.full_epochs
    fixed_run_args = hparam_search._build_fixed_run_args_context(
        base_args=dict(cycle_config.base_args),
        full_overrides=full_overrides,
        search_space=cycle_config.search_space,
    )
    current_hparam_context = hparam_search._build_hparam_context(
        objective_metric=cycle_config.objective_metric,
        full_epochs=full_epochs_value,
        validation_protocol=baseline_validation_protocol,
        fixed_run_args=fixed_run_args,
    )
    previous_global_best_score = hparam_search._read_best_objective_score(
        cycle_config.global_best_config_path,
        cycle_config.objective_metric,
        expected_hparam_context=current_hparam_context,
    )

    history_trials: list[tuple[float, dict[str, Scalar]]] = []
    if cycle_config.search_algo in {"history_guided", "reinforce"}:
        history_trials = hparam_search.load_historical_trials(
            output_dir=cycle_config.output_dir,
            search_space=cycle_config.search_space,
            objective_metric=cycle_config.objective_metric,
            top_n=cycle_config.history_top_n,
            base_args=cycle_config.base_args,
        )

    seed_best_params: dict[str, Scalar] | None = None
    seed_best_key: str | None = None
    seed_best_context_mismatch = False
    if cycle_config.seed_best_config_path is not None:
        try:
            seed_best_config = hparam_search.load_seed_best_config(
                path=cycle_config.seed_best_config_path,
                search_space=cycle_config.search_space,
                base_args=cycle_config.base_args,
                default_objective_metric=cycle_config.objective_metric,
            )
        except ValueError as exc:
            _emit_scheduler_line(
                scheduler_config,
                (
                    "seed best config ignored due to parse error "
                    f"(cycle={cycle_index}): {exc}"
                ),
            )
        else:
            if seed_best_config is not None:
                seed_best_params = dict(seed_best_config.sampled_params)
                seed_best_key = json.dumps(
                    seed_best_params,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                seed_best_context_mismatch = not hparam_search._contexts_match(
                    seed_best_config.hparam_context,
                    current_hparam_context,
                )

    global_best_recheck_params: dict[str, Scalar] | None = None
    global_best_recheck_context_mismatch = False
    if (
        cycle_config.seed_best_config_path is None
        and cycle_config.global_best_config_path is not None
    ):
        try:
            global_best_config = hparam_search.load_seed_best_config(
                path=cycle_config.global_best_config_path,
                search_space=cycle_config.search_space,
                base_args=cycle_config.base_args,
                default_objective_metric=cycle_config.objective_metric,
            )
        except ValueError as exc:
            _emit_scheduler_line(
                scheduler_config,
                (
                    "global best config ignored due to parse error "
                    f"(cycle={cycle_index}): {exc}"
                ),
            )
        else:
            if (
                global_best_config is not None
                and global_best_config.hparam_context is not None
            ):
                global_best_recheck_context_mismatch = not hparam_search._contexts_match(
                    global_best_config.hparam_context,
                    current_hparam_context,
                )
                if global_best_recheck_context_mismatch:
                    global_best_recheck_params = dict(
                        global_best_config.sampled_params
                    )

    full_priority_params: dict[str, Scalar] | None = None
    next_full_trial_id = 0
    if seed_best_params is not None and seed_best_context_mismatch:
        full_priority_params = dict(seed_best_params)
        next_full_trial_id = 1
    elif (
        global_best_recheck_params is not None
        and global_best_recheck_context_mismatch
    ):
        full_priority_params = dict(global_best_recheck_params)
        next_full_trial_id = 1

    quick_params = hparam_search.build_trial_params(
        config=cycle_config,
        phase="quick",
        count=cycle_config.quick_trials,
        seed_offset=0,
        history_trials=history_trials,
    )
    if seed_best_params is not None and quick_params:
        quick_params[0] = dict(seed_best_params)

    quick_overrides = dict(cycle_config.quick_overrides)
    quick_overrides.setdefault("epochs", cycle_config.quick_epochs)
    quick_overrides.setdefault("compile_mode", "off")

    full_compile_mode = str(full_overrides.get("compile_mode", "auto")).strip().lower()
    if full_compile_mode == "auto" and scheduler_config.selected_gpu_ids:
        if hparam_search._find_cuda_header() is None:
            full_overrides["compile_mode"] = "off"

    resolved_trial_stream_mode = hparam_search._resolve_trial_stream_mode(
        cycle_config.trial_stream_mode,
        total_slot_count,
    )

    cycle = CycleState(
        cycle_index=cycle_index,
        template=template,
        config=cycle_config,
        output_dir=output_dir,
        config_path=config_path,
        stdout_log=stdout_log,
        start_time=time.monotonic(),
        baseline_validation_protocol=baseline_validation_protocol,
        current_hparam_context=current_hparam_context,
        previous_global_best_score=previous_global_best_score,
        quick_overrides=quick_overrides,
        full_overrides=full_overrides,
        full_epochs_value=full_epochs_value,
        quick_params=quick_params,
        quick_pending_indices=list(range(len(quick_params))),
        seed_best_params=seed_best_params,
        seed_best_key=seed_best_key,
        seed_best_context_mismatch=seed_best_context_mismatch,
        global_best_recheck_params=global_best_recheck_params,
        global_best_recheck_context_mismatch=global_best_recheck_context_mismatch,
        full_priority_params=full_priority_params,
        next_full_trial_id=next_full_trial_id,
        resolved_trial_stream_mode=resolved_trial_stream_mode,
    )
    return cycle


def _emit_cycle_start(
    *,
    scheduler_config: SchedulerConfig,
    cycle: CycleState,
    elapsed_seconds: int,
    remaining_seconds: int,
    worker_gpu_ids: list[str],
) -> None:
    """Emit the user-facing cycle-start summary and initialization details."""
    assigned_gpu_text = (
        ",".join(worker_gpu_ids)
        if worker_gpu_ids
        else "cpu-fallback"
    )
    parallel_count = len(worker_gpu_ids) if worker_gpu_ids else 1
    _emit_scheduler_line(
        scheduler_config,
        (
            f"cycle={cycle.cycle_index} elapsed={_format_elapsed(elapsed_seconds)} "
            f"start={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} "
            f"ETA:{_format_elapsed(remaining_seconds)} species={cycle.template.species} "
            f"target={cycle.template.target_name} seed={cycle.template.seed} "
            f"gpus={assigned_gpu_text} log={cycle.stdout_log}"
        ),
    )
    _emit_cycle_line(
        cycle,
        (
            "[hparam_search] Using GPU slots: "
            f"{assigned_gpu_text} (parallel={parallel_count})."
            if worker_gpu_ids
            else "[hparam_search] No GPU detected; using CPU (parallel=1)."
        ),
    )
    _emit_cycle_line(
        cycle,
        (
            "[hparam_search] Trial stdout stream mode: "
            f"{cycle.resolved_trial_stream_mode} "
            "(logs are still saved per trial)."
        ),
    )
    _emit_cycle_line(
        cycle,
        "[hparam_search] Trial process mode: subprocess "
        "(owned by tune_time_scheduler, direct trial dispatch).",
    )
    if cycle.previous_global_best_score is not None:
        _emit_cycle_line(
            cycle,
            (
                "[hparam_search] Reference global best "
                f"{cycle.config.objective_metric}="
                f"{cycle.previous_global_best_score:.6f}."
            ),
        )
    if cycle.config.search_algo == "history_guided":
        history_trials = hparam_search.load_historical_trials(
            output_dir=cycle.config.output_dir,
            search_space=cycle.config.search_space,
            objective_metric=cycle.config.objective_metric,
            top_n=cycle.config.history_top_n,
            base_args=cycle.config.base_args,
        )
        _emit_cycle_line(
            cycle,
            (
                "[hparam_search] Search algorithm: history_guided "
                f"(history_candidates={len(history_trials)}, "
                f"top_n={cycle.config.history_top_n}, "
                f"random_fraction={cycle.config.guided_random_fraction:.2f}, "
                f"mutation_rate={cycle.config.guided_mutation_rate:.2f})."
            ),
        )
    elif cycle.config.search_algo == "reinforce":
        history_trials = hparam_search.load_historical_trials(
            output_dir=cycle.config.output_dir,
            search_space=cycle.config.search_space,
            objective_metric=cycle.config.objective_metric,
            top_n=cycle.config.history_top_n,
            base_args=cycle.config.base_args,
        )
        _emit_cycle_line(
            cycle,
            (
                "[hparam_search] Search algorithm: reinforce "
                f"(history_candidates={len(history_trials)}, "
                f"top_n={cycle.config.history_top_n}, "
                f"random_fraction={cycle.config.guided_random_fraction:.2f}, "
                f"mutation_rate={cycle.config.guided_mutation_rate:.2f}, "
                f"temperature={cycle.config.reinforce_temperature:.2f})."
            ),
        )
    else:
        _emit_cycle_line(cycle, "[hparam_search] Search algorithm: random.")
    if cycle.seed_best_params is not None:
        _emit_cycle_line(
            cycle,
            (
                "[hparam_search] Loaded seed best sampled params from "
                f"{cycle.config.seed_best_config_path}."
            ),
        )
    if cycle.seed_best_context_mismatch and cycle.full_priority_params is not None:
        _emit_cycle_line(
            cycle,
            "[hparam_search] Seed best context changed. "
            "Schedule one full-phase recheck with stored seed.",
        )
    if (
        cycle.global_best_recheck_context_mismatch
        and cycle.global_best_recheck_params is not None
    ):
        _emit_cycle_line(
            cycle,
            "[hparam_search] Global best context changed. "
            "Schedule one full-phase recheck with stored best.",
        )
    _emit_cycle_line(
        cycle,
        (
            f"[hparam_search] Quick phase: {cycle.config.quick_trials} trials, "
            f"epochs={cycle.quick_overrides.get('epochs')}."
        ),
    )
    if cycle.config.skip_full_phase:
        _emit_cycle_line(
            cycle,
            "[hparam_search] Full phase skipped by config (skip_full_phase=true).",
        )
    elif cycle.config.enable_phase_overlap:
        _emit_cycle_line(
            cycle,
            "[hparam_search] Phase overlap enabled: quick and full share slots.",
        )


def _queue_full_task(cycle: CycleState, task: ScheduledTrialTask) -> None:
    """Insert one full-phase task by descending priority."""
    cycle.pending_full_tasks.append(task)
    cycle.pending_full_tasks.sort(
        key=lambda item: (-item.priority_score, item.trial_id)
    )


def _maybe_queue_priority_recheck(cycle: CycleState) -> None:
    """Queue the priority full-phase recheck once it becomes schedulable."""
    if cycle.full_priority_inserted or cycle.full_priority_params is None:
        return
    task = ScheduledTrialTask(
        phase="full",
        trial_id=0,
        priority_score=float("inf"),
        sampled_params=dict(cycle.full_priority_params),
        overrides=dict(cycle.full_overrides),
        metrics_json=str(cycle.output_dir / "full_trial_0000.metrics.json"),
        log_file=str(cycle.output_dir / "full_trial_0000.log.txt"),
    )
    _queue_full_task(cycle, task)
    cycle.full_priority_inserted = True
    cycle.next_full_trial_id = max(cycle.next_full_trial_id, 1)


def _maybe_promote_locked_rows(cycle: CycleState) -> None:
    """Promote mathematically locked quick rows into full tasks."""
    unfinished_quick_count = len(cycle.quick_pending_indices) + cycle.quick_running_count
    locked_rows = hparam_search._select_locked_quick_trials(
        completed_quick_rows=cycle.completed_quick_rows,
        unfinished_quick_count=unfinished_quick_count,
        top_k=cycle.config.top_k,
    )
    if locked_rows:
        _maybe_queue_priority_recheck(cycle)
    for row in locked_rows:
        row_key = json.dumps(
            row.sampled_params,
            sort_keys=True,
            separators=(",", ":"),
        )
        if row_key in cycle.full_consumed_keys:
            continue
        if (
            cycle.seed_best_key is not None
            and row_key == cycle.seed_best_key
            and not cycle.seed_best_context_mismatch
        ):
            cycle.skipped_seed_context_match += 1
            cycle.full_consumed_keys.add(row_key)
            continue
        quick_best_epoch = hparam_search._read_objective_best_epoch_from_metrics(
            metrics_json_path=Path(row.metrics_json),
            objective_metric=cycle.config.objective_metric,
        )
        if (
            quick_best_epoch is not None
            and quick_best_epoch == cycle.full_epochs_value
        ):
            cycle.skipped_same_best_epoch += 1
            cycle.full_consumed_keys.add(row_key)
            continue
        task = ScheduledTrialTask(
            phase="full",
            trial_id=cycle.next_full_trial_id,
            priority_score=(
                float(row.objective_score)
                if row.objective_score is not None
                else float("-inf")
            ),
            sampled_params=dict(row.sampled_params),
            overrides=dict(cycle.full_overrides),
            metrics_json=str(
                cycle.output_dir / f"full_trial_{cycle.next_full_trial_id:04d}.metrics.json"
            ),
            log_file=str(
                cycle.output_dir / f"full_trial_{cycle.next_full_trial_id:04d}.log.txt"
            ),
        )
        cycle.next_full_trial_id += 1
        cycle.full_consumed_keys.add(row_key)
        _queue_full_task(cycle, task)


def _build_non_overlap_full_queue(cycle: CycleState) -> None:
    """Build the full-phase queue after the quick phase fully finishes."""
    if cycle.full_queue_built:
        return
    cycle.full_queue_built = True
    if cycle.config.skip_full_phase:
        return
    ranked_quick = hparam_search.rank_successful_trials(cycle.quick_rows)
    selected_for_full: list[TrialResult] = []
    selected_for_full_keys: set[str] = set()
    for row in ranked_quick:
        row_key = json.dumps(
            row.sampled_params,
            sort_keys=True,
            separators=(",", ":"),
        )
        if row_key in selected_for_full_keys:
            continue
        if (
            cycle.seed_best_key is not None
            and row_key == cycle.seed_best_key
            and not cycle.seed_best_context_mismatch
        ):
            cycle.skipped_seed_context_match += 1
            continue
        quick_best_epoch = hparam_search._read_objective_best_epoch_from_metrics(
            metrics_json_path=Path(row.metrics_json),
            objective_metric=cycle.config.objective_metric,
        )
        if (
            quick_best_epoch is not None
            and quick_best_epoch == cycle.full_epochs_value
        ):
            cycle.skipped_same_best_epoch += 1
            continue
        selected_for_full.append(row)
        selected_for_full_keys.add(row_key)
        if len(selected_for_full) >= cycle.config.top_k:
            break
    if cycle.seed_best_params is not None and cycle.seed_best_context_mismatch:
        _maybe_queue_priority_recheck(cycle)
    elif (
        cycle.global_best_recheck_params is not None
        and cycle.global_best_recheck_context_mismatch
    ):
        _maybe_queue_priority_recheck(cycle)
    for row in selected_for_full:
        task = ScheduledTrialTask(
            phase="full",
            trial_id=cycle.next_full_trial_id,
            priority_score=(
                float(row.objective_score)
                if row.objective_score is not None
                else float("-inf")
            ),
            sampled_params=dict(row.sampled_params),
            overrides=dict(cycle.full_overrides),
            metrics_json=str(
                cycle.output_dir / f"full_trial_{cycle.next_full_trial_id:04d}.metrics.json"
            ),
            log_file=str(
                cycle.output_dir / f"full_trial_{cycle.next_full_trial_id:04d}.log.txt"
            ),
        )
        cycle.next_full_trial_id += 1
        _queue_full_task(cycle, task)
    _emit_cycle_line(
        cycle,
        (
            f"[hparam_search] Full phase: top_k={cycle.config.top_k}, "
            f"selected={len(selected_for_full)}, "
            f"skipped_same_best_epoch={cycle.skipped_same_best_epoch}, "
            "skipped_seed_context_match="
            f"{cycle.skipped_seed_context_match}, "
            f"injected_best_full_recheck={cycle.full_priority_inserted}, "
            f"epochs={cycle.full_overrides.get('epochs')}, "
            f"objective={cycle.config.objective_metric}, "
            "execution_mode=subprocess."
        ),
    )


def _ensure_post_quick_state(cycle: CycleState) -> None:
    """Ensure full-phase work is materialized once quick work fully drains."""
    if cycle.has_quick_work:
        return
    if cycle.config.skip_full_phase:
        cycle.full_queue_built = True
        return
    if cycle.config.enable_phase_overlap:
        _maybe_queue_priority_recheck(cycle)
        _maybe_promote_locked_rows(cycle)
        cycle.full_queue_built = True
        return
    _build_non_overlap_full_queue(cycle)


def _pop_next_task(cycle: CycleState) -> ScheduledTrialTask | None:
    """Pop the next schedulable task for one cycle."""
    _ensure_post_quick_state(cycle)
    if cycle.quick_pending_indices:
        trial_id = cycle.quick_pending_indices.pop(0)
        return ScheduledTrialTask(
            phase="quick",
            trial_id=trial_id,
            priority_score=0.0,
            sampled_params=dict(cycle.quick_params[trial_id]),
            overrides=dict(cycle.quick_overrides),
            metrics_json=str(cycle.output_dir / f"quick_trial_{trial_id:04d}.metrics.json"),
            log_file=str(cycle.output_dir / f"quick_trial_{trial_id:04d}.log.txt"),
        )
    if cycle.pending_full_tasks:
        return cycle.pending_full_tasks.pop(0)
    return None


def _run_trial_for_cycle(
    *,
    cycle: CycleState,
    task: ScheduledTrialTask,
    assigned_gpu_id: str | None,
) -> TrialResult:
    """Run one trial while attaching the cycle-scoped log prefix."""
    with hparam_search.trial_log_prefix(_cycle_log_prefix(cycle)):
        return hparam_search.run_trial(
            config=cycle.config,
            phase=task.phase,
            trial_id=task.trial_id,
            sampled_params=task.sampled_params,
            overrides=task.overrides,
            assigned_gpu_id=assigned_gpu_id,
            metrics_json=Path(task.metrics_json),
            log_file=Path(task.log_file),
        )


def _handle_completed_trial(
    cycle: CycleState,
    task: ScheduledTrialTask,
    result: TrialResult,
) -> None:
    """Update one cycle with a newly completed trial result."""
    if task.phase == "quick":
        cycle.quick_running_count -= 1
        cycle.quick_rows.append(result)
        if result.status == "success":
            cycle.completed_quick_rows.append(result)
        if cycle.config.enable_phase_overlap:
            _maybe_promote_locked_rows(cycle)
    else:
        cycle.full_running_count -= 1
        cycle.full_rows.append(result)
    _ensure_post_quick_state(cycle)


def _cycle_has_pending_work(cycle: CycleState) -> bool:
    """Return whether one cycle still has queued or running work."""
    _ensure_post_quick_state(cycle)
    return bool(
        cycle.quick_pending_indices
        or cycle.quick_running_count > 0
        or cycle.pending_full_tasks
        or cycle.full_running_count > 0
    )


def _snapshot_trial_count(cycle: CycleState, task: ScheduledTrialTask) -> int:
    """Return the best-known phase trial count at dispatch time."""
    if task.phase == "quick":
        return len(cycle.quick_params)
    return max(
        1,
        len(cycle.full_rows) + len(cycle.pending_full_tasks) + cycle.full_running_count + 1,
    )


def _pop_next_global_task(
    cycle_queue: list[CycleState],
) -> tuple[CycleState, ScheduledTrialTask] | None:
    """Pop the next immediately runnable task from the admitted cycle queue."""
    for cycle in cycle_queue:
        task = _pop_next_task(cycle)
        if task is not None:
            return cycle, task
    return None


def _should_dispatch_next_cycle(
    *,
    script_name: str,
    remaining_seconds: int,
    total_cycle_seconds: int,
    completed_cycles: int,
) -> bool:
    """Return whether one new cycle should be launched."""
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


def _append_next_cycle(
    *,
    config: SchedulerConfig,
    cycle_queue: list[CycleState],
    cycle_cursor: int,
    total_slot_count: int,
) -> int:
    """Append one new cycle to the admitted queue and return the next cursor."""
    template = config.jobs[cycle_cursor % len(config.jobs)]
    cycle = _prepare_cycle_state(
        scheduler_config=config,
        template=template,
        cycle_index=cycle_cursor,
        total_slot_count=total_slot_count,
    )
    if cycle_cursor == 0:
        hparam_search._set_active_trial_stream_mode(cycle.resolved_trial_stream_mode)
    cycle_queue.append(cycle)
    return cycle_cursor + 1


def _prune_timeout_artifacts(
    *,
    scheduler_config: SchedulerConfig,
    cycle: CycleState,
) -> None:
    """Remove partial timeout outputs and prune dangling rank checkpoints."""
    if cycle.output_dir.is_dir():
        shutil.rmtree(cycle.output_dir)
        _emit_scheduler_line(
            scheduler_config,
            f"removed partial output dir: {cycle.output_dir}",
        )

    subprocess.run(
        [
            scheduler_config.python_bin,
            str(
                scheduler_config.project_root
                / "src"
                / "tools"
                / "prune_missing_rank_checkpoints.py"
            ),
            "--data_root",
            str(scheduler_config.data_root),
            "--model_root",
            str(scheduler_config.model_root),
            "--species",
            cycle.template.species,
            "--model",
            cycle.template.tuning_model_name,
            "--dry_run",
            "0",
        ],
        cwd=scheduler_config.project_root,
        check=False,
    )


def _run_cycle_plot(
    *,
    scheduler_config: SchedulerConfig,
    cycle: CycleState,
) -> None:
    """Update the optional double-descent plot for one completed cycle."""
    plot_target_name = cycle.template.plot_target_name
    if plot_target_name is None:
        return

    command = [
        scheduler_config.python_bin,
        str(
            scheduler_config.project_root
            / "src"
            / "tools"
            / "plot_tuning_double_descent.py"
        ),
        "--project_root",
        str(scheduler_config.project_root),
        "--species",
        cycle.template.species,
        "--target",
        plot_target_name,
        "--model",
        cycle.template.tuning_model_name,
    ]
    with cycle.stdout_log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=scheduler_config.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            handle.write(result.stdout)
        if result.stderr:
            handle.write(result.stderr)


def _finalize_cycle(
    *,
    scheduler_config: SchedulerConfig,
    cycle: CycleState,
    gpu_ids: list[str],
) -> int:
    """Write cycle artifacts and return the cycle exit code."""
    hparam_search.write_trials_tsv(cycle.output_dir / "quick_trials.tsv", cycle.quick_rows)
    ranked_quick = hparam_search.rank_successful_trials(cycle.quick_rows)
    hparam_search.write_trials_tsv(cycle.output_dir / "full_trials.tsv", cycle.full_rows)

    excluded_ranking_param_keys: set[str] = set()
    if cycle.seed_best_params is not None and cycle.seed_best_context_mismatch:
        excluded_ranking_param_keys.add(
            hparam_search._sampled_params_key(cycle.seed_best_params)
        )
    elif (
        cycle.global_best_recheck_params is not None
        and cycle.global_best_recheck_context_mismatch
    ):
        excluded_ranking_param_keys.add(
            hparam_search._sampled_params_key(cycle.global_best_recheck_params)
        )

    ranked_full = hparam_search.rank_successful_trials(cycle.full_rows)
    ranked_quick_for_export = hparam_search._exclude_recheck_rows_from_ranking(
        ranked_quick,
        excluded_param_keys=excluded_ranking_param_keys,
    )
    ranked_full_for_export = hparam_search._exclude_recheck_rows_from_ranking(
        ranked_full,
        excluded_param_keys=excluded_ranking_param_keys,
    )
    if ranked_full_for_export:
        best_row = ranked_full_for_export[0]
        ranked_for_export = ranked_full_for_export
    elif ranked_quick_for_export:
        best_row = ranked_quick_for_export[0]
        ranked_for_export = ranked_quick_for_export
    else:
        best_row = None
        ranked_for_export = []

    hparam_search.write_best_config(
        cycle.output_dir / "best_config.json",
        best_row,
        top_rows=ranked_for_export,
        top_k=cycle.config.top_k,
        fallback_validation_protocol=cycle.baseline_validation_protocol,
        hparam_context=cycle.current_hparam_context,
    )
    hparam_search._write_tuning_leaderboard(
        config=cycle.config,
        ranked_rows=ranked_for_export,
        best_row=best_row,
    )
    pruned_count = hparam_search._prune_non_best_trial_checkpoints(
        project_root=cycle.config.project_root,
        trial_rows=cycle.quick_rows + cycle.full_rows,
        best_row=best_row,
        min_mtime_epoch=time.time(),
    )
    if pruned_count > 0:
        _emit_cycle_line(
            cycle,
            (
                "[hparam_search] Pruned non-best trial checkpoints: "
                f"deleted={pruned_count}."
            ),
        )
    hparam_search.maybe_update_global_best(
        global_best_path=cycle.config.global_best_config_path,
        best_row=best_row,
        fallback_validation_protocol=cycle.baseline_validation_protocol,
        hparam_context=cycle.current_hparam_context,
    )
    if cycle.config.enable_visualization:
        viz_path = cycle.output_dir / f"{cycle.config.species}_snpr.png"
        viz_error = hparam_search.write_visualization(
            viz_path,
            model_name=str(cycle.config.base_args.get("model", "cnn")),
            species=cycle.config.species,
            objective_metric=cycle.config.objective_metric,
            quick_rows=cycle.quick_rows,
            full_rows=cycle.full_rows,
            base_args=cycle.config.base_args,
        )
        if viz_error is None:
            _emit_cycle_line(
                cycle,
                f"[hparam_search] Wrote tuning visualization: {viz_path}",
            )
        else:
            _emit_cycle_line(
                cycle,
                f"[hparam_search] Visualization skipped: {viz_error}",
            )
    else:
        _emit_cycle_line(
            cycle,
            (
                "[hparam_search] Visualization disabled by config "
                "(enable_visualization=false)."
            ),
        )
    hparam_search.write_summary_markdown(
        cycle.output_dir / "run_summary.md",
        config=cycle.config,
        gpu_ids=gpu_ids,
        quick_rows=cycle.quick_rows,
        full_rows=cycle.full_rows,
        best_row=best_row,
        previous_global_best_score=cycle.previous_global_best_score,
    )
    if best_row is None:
        _emit_cycle_line(cycle, "[hparam_search] No successful trial found.")
        cycle.finalized = True
        cycle.exit_code = 1
        return 1
    if (
        cycle.previous_global_best_score is not None
        and best_row.objective_score is not None
    ):
        delta = best_row.objective_score - cycle.previous_global_best_score
        delta_text = f"+{delta:.6f}" if delta >= 0.0 else f"{delta:.6f}"
        _emit_cycle_line(
            cycle,
            f"[hparam_search] Comparison to previous global best: {delta_text}.",
        )
    _emit_cycle_line(
        cycle,
        (
            f"[hparam_search] Best {best_row.objective_metric} "
            f"{best_row.objective_score:.6f} from {best_row.phase} "
            f"trial {best_row.trial_id}."
        ),
    )
    cycle.finalized = True
    cycle.exit_code = 0
    return 0


def run_scheduler(config: SchedulerConfig) -> int:
    """Run the centralized direct-trial scheduler until the time budget ends."""
    if not config.jobs:
        raise ValueError("At least one scheduler job is required.")
    total_slot_count = max(
        1,
        min(
            config.parallel_slot_count,
            len(config.selected_gpu_ids) if config.selected_gpu_ids else 1,
        ),
    )
    if config.selected_gpu_ids:
        worker_gpu_ids = list(config.selected_gpu_ids[:total_slot_count])
        _emit_scheduler_line(
            config,
            (
                "trial scheduler across GPUs: "
                f"{','.join(worker_gpu_ids)}"
            ),
        )
    else:
        worker_gpu_ids = []
        _emit_scheduler_line(config, "trial scheduler using CPU fallback.")

    start_time = time.monotonic()
    deadline = start_time + (config.time_budget_minutes * 60)
    completed_cycles = 0
    total_cycle_seconds = 0
    cycle_cursor = 0
    first_error_code = 0
    max_queued_cycles = 3
    stop_submitting = False

    cycle_queue: list[CycleState] = []
    running_trials: dict[Future[TrialResult], RunningTrial] = {}
    executor = ThreadPoolExecutor(max_workers=total_slot_count)
    free_gpu_ids: list[str | None]
    if worker_gpu_ids:
        free_gpu_ids = list(worker_gpu_ids)
    else:
        free_gpu_ids = [None]

    previous_parallel = hparam_search._set_active_max_parallel_trials(total_slot_count)
    previous_stream = hparam_search._set_active_trial_stream_mode("errors")
    try:
        while not stop_submitting and len(cycle_queue) < max_queued_cycles:
            cycle_cursor = _append_next_cycle(
                config=config,
                cycle_queue=cycle_queue,
                cycle_cursor=cycle_cursor,
                total_slot_count=total_slot_count,
            )

        while cycle_queue or not stop_submitting:
            now = time.monotonic()
            elapsed_seconds = int(now - start_time)
            remaining_seconds = max(0, int(deadline - now))

            if (
                not stop_submitting
                and not _should_dispatch_next_cycle(
                    script_name=config.script_name,
                    remaining_seconds=remaining_seconds,
                    total_cycle_seconds=total_cycle_seconds,
                    completed_cycles=completed_cycles,
                )
            ):
                stop_submitting = True

            if now >= deadline:
                stop_submitting = True
                if first_error_code == 0:
                    first_error_code = 124
                hparam_search._interrupt_active_trial_processes(
                    wait_timeout_sec=float(config.timeout_grace_seconds)
                )
                for cycle in cycle_queue:
                    _prune_timeout_artifacts(
                        scheduler_config=config,
                        cycle=cycle,
                    )
                break

            cycle_queue = [cycle for cycle in cycle_queue if not cycle.finalized]

            scheduled_progress = False
            while free_gpu_ids:
                next_item = _pop_next_global_task(cycle_queue)
                if next_item is None:
                    break
                cycle, task = next_item
                if not cycle.start_logged:
                    _emit_cycle_start(
                        scheduler_config=config,
                        cycle=cycle,
                        elapsed_seconds=elapsed_seconds,
                        remaining_seconds=remaining_seconds,
                        worker_gpu_ids=worker_gpu_ids,
                    )
                    cycle.start_logged = True
                assigned_gpu_id = free_gpu_ids.pop(0)
                known_trial_count = _snapshot_trial_count(cycle, task)
                future = executor.submit(
                    _run_trial_for_cycle,
                    cycle=cycle,
                    task=task,
                    assigned_gpu_id=assigned_gpu_id,
                )
                running_trials[future] = RunningTrial(
                    cycle_index=cycle.cycle_index,
                    task=task,
                    assigned_gpu_id=assigned_gpu_id,
                    known_trial_count=known_trial_count,
                )
                if task.phase == "quick":
                    cycle.quick_running_count += 1
                else:
                    cycle.full_running_count += 1
                _emit_trial_start_line(
                    cycle=cycle,
                    task=task,
                    assigned_gpu_id=assigned_gpu_id,
                )
                scheduled_progress = True

            if not running_trials:
                if stop_submitting and not cycle_queue:
                    break
                if not scheduled_progress:
                    time.sleep(0.1)
                continue

            done, _ = wait(running_trials.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                running_trial = running_trials.pop(future)
                free_gpu_ids.append(running_trial.assigned_gpu_id)
                cycle = next(
                    candidate
                    for candidate in cycle_queue
                    if candidate.cycle_index == running_trial.cycle_index
                )
                result = future.result()
                _handle_completed_trial(cycle, running_trial.task, result)
                _emit_trial_result_line(
                    cycle=cycle,
                    task=running_trial.task,
                    result=result,
                    known_trial_count=running_trial.known_trial_count,
                )
                if not _cycle_has_pending_work(cycle):
                    cycle_duration_seconds = max(
                        0,
                        int(time.monotonic() - cycle.start_time),
                    )
                    cycle_exit_code = _finalize_cycle(
                        scheduler_config=config,
                        cycle=cycle,
                        gpu_ids=worker_gpu_ids,
                    )
                    if cycle_exit_code != 0 and first_error_code == 0:
                        first_error_code = cycle_exit_code
                    if cycle_exit_code == 0:
                        total_cycle_seconds += cycle_duration_seconds
                        completed_cycles += 1
                    _run_cycle_plot(
                        scheduler_config=config,
                        cycle=cycle,
                    )
                    avg_cycle_seconds = (
                        total_cycle_seconds // completed_cycles
                        if completed_cycles > 0
                        else 0
                    )
                    remaining_seconds = max(0, int(deadline - time.monotonic()))
                    estimated_cycles_left = (
                        remaining_seconds // avg_cycle_seconds
                        if avg_cycle_seconds > 0
                        else 0
                    )
                    _emit_scheduler_line(
                        config,
                        (
                            f"cycle_done={cycle.cycle_index} "
                            f"cycle_time={_format_elapsed(cycle_duration_seconds)} "
                            f"avg_cycle={_format_elapsed(avg_cycle_seconds)} "
                            f"ETA_cycles_left={estimated_cycles_left} "
                            f"log={cycle.stdout_log} exit={cycle_exit_code}"
                        ),
                    )
                    cycle_queue = [
                        queued_cycle
                        for queued_cycle in cycle_queue
                        if not queued_cycle.finalized
                    ]
                    while not stop_submitting and len(cycle_queue) < max_queued_cycles:
                        cycle_cursor = _append_next_cycle(
                            config=config,
                            cycle_queue=cycle_queue,
                            cycle_cursor=cycle_cursor,
                            total_slot_count=total_slot_count,
                        )
            cycle_queue = [cycle for cycle in cycle_queue if not cycle.finalized]
    except KeyboardInterrupt:
        hparam_search._interrupt_active_trial_processes(
            wait_timeout_sec=float(config.timeout_grace_seconds)
        )
        return 130
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        hparam_search._set_active_trial_stream_mode(previous_stream)
        hparam_search._set_active_max_parallel_trials(previous_parallel)

    if first_error_code != 0:
        return first_error_code
    total_seconds = int(time.monotonic() - start_time)
    _emit_scheduler_line(
        config,
        (
            f"done start={config.start_epoch} "
            f"end={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} "
            f"elapsed={_format_elapsed(total_seconds)} cycles={cycle_cursor}"
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    actual_argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(actual_argv)
    config = _load_config(Path(args.config))
    return run_scheduler(config)


if __name__ == "__main__":
    raise SystemExit(main())
