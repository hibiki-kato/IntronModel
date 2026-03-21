"""Random hyperparameter search runner with two-phase orchestration.

This tool executes two-phase tuning for model commands:
1) quick random sweep
2) full re-train on top-k quick candidates

It is intentionally model-agnostic and relies on command arguments specified
in a JSON configuration file.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import inspect
import io
import json
import math
import multiprocessing as mp
import os
from queue import Empty
import random
import shutil
import shlex
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Optional, Sequence

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from util.validation_protocol import (
    LEGACY_VALIDATION_SIGNATURE,
    build_validation_protocol,
)
from util.checkpoint_io import extract_checkpoint_paths, read_json_object
from util.process_title import apply_process_title_from_env

_ = apply_process_title_from_env()

Scalar = int | float | str | bool
ArgValue = Scalar | None

TRIAL_STREAM_MODE_CHOICES: set[str] = {"auto", "full", "errors", "silent"}
TRIAL_PROCESS_MODE_CHOICES: set[str] = {
    "subprocess",
    "persistent_quick",
    "persistent_all",
}
_ACTIVE_TRIAL_STREAM_MODE: str = "full"
_ACTIVE_MAX_PARALLEL_TRIALS: int = 1
_ACTIVE_TRIAL_PROCESSES: set[subprocess.Popen[str]] = set()
_ACTIVE_TRIAL_PROCESSES_LOCK = threading.Lock()
_DEFAULT_PHASE_EXECUTION_MODE: str = "subprocess"
_PERSISTENT_PHASE_EXECUTION_MODE: str = "persistent"
SUPPORTED_OBJECTIVE_METRIC_NAMES: tuple[str, ...] = (
    "mean_pr_auc",
    "donor_pr_auc",
    "acceptor_pr_auc",
    "pair_pr_auc",
    "mean_roc_auc",
    "donor_roc_auc",
    "acceptor_roc_auc",
    "pair_roc_auc",
    "mean_max_f1",
    "donor_max_f1",
    "acceptor_max_f1",
    "pair_max_f1",
    "test_pr_auc",
    "test_max_f1",
)
SUPPORTED_OBJECTIVE_METRICS: set[str] = set(SUPPORTED_OBJECTIVE_METRIC_NAMES)
_SITE_WINDOW_LEN_KEYS: tuple[str, str] = ("donor_len", "acceptor_len")
_SITE_WINDOW_LEN_DEFAULT: int = 100
_SITE_WINDOW_LEN_MIN: int = 40
_SITE_WINDOW_LEN_MAX: int = 100
_SITE_WINDOW_LEN_STEP: int = 10
_DNABERT_READOUT_CHOICES: tuple[str, ...] = ("cnn", "linear", "mlp")
_DNABERT_MODEL_PREFIX: str = "dnabert"
_DNABERT_CNN_ONLY_KEYS: frozenset[str] = frozenset({"readout_cnn_kernel_size"})
_DNABERT_MLP_ONLY_KEYS: frozenset[str] = frozenset(
    {"readout_mlp_hidden_dim", "readout_mlp_layers"}
)
_CONTEXT_ARG_IGNORE_KEYS: set[str] = {
    "allow_tf32",
    "amp_dtype",
    "checkpoint_prune_dry_run",
    "checkpoint_top_k",
    "compile_mode",
    "cudnn_benchmark",
    "deterministic",
    "device",
    "epochs",
    "max_oom_retries",
    "metrics_json",
    "min_batch_size",
    "name_fields",
    "num_workers",
    "persistent_workers",
    "pin_memory",
    "prefetch_factor",
    "tag",
    "train_only",
    "use_amp",
    "visualize",
}


@dataclass(frozen=True)
class TrialResult:
    """Result record for one trial."""

    phase: str
    trial_id: int
    status: str
    gpu_id: Optional[str]
    sampled_params: dict[str, Scalar]
    effective_batch_size: int
    oom_retries: int
    donor_pr_auc: Optional[float]
    acceptor_pr_auc: Optional[float]
    mean_pr_auc: Optional[float]
    objective_metric: str
    objective_score: Optional[float]
    error_message: Optional[str]
    return_code: int
    duration_sec: float
    metrics_json: str
    log_file: str
    validation_signature: str = LEGACY_VALIDATION_SIGNATURE
    validation_protocol: Optional[dict[str, object]] = None
    selection_score: Optional[float] = None


@dataclass(frozen=True)
class SearchConfig:
    """Validated hyperparameter search configuration."""

    project_root: Path
    species: str
    output_dir: Path
    quick_trials: int
    quick_epochs: int
    top_k: int
    full_epochs: int
    base_seed: int
    gpu_ids_setting: object
    max_parallel_trials_setting: object
    min_batch_size: int
    max_oom_retries: int
    max_model_params: Optional[int]
    objective_metric: str
    global_best_config_path: Optional[Path]
    seed_best_config_path: Optional[Path]
    base_args: dict[str, ArgValue]
    quick_overrides: dict[str, ArgValue]
    full_overrides: dict[str, ArgValue]
    search_space: dict[str, dict[str, object]]
    search_algo: str = "random"
    history_top_n: int = 64
    guided_random_fraction: float = 0.35
    guided_mutation_rate: float = 0.25
    surrogate_warmup_trials: int = 8
    surrogate_candidates_per_step: int = 128
    surrogate_min_observations: int = 8
    trial_stream_mode: str = "auto"
    trial_process_mode: str = "subprocess"
    skip_full_phase: bool = False
    enable_visualization: bool = True


@dataclass(frozen=True)
class SeedBestConfig:
    """Validated seed-best payload for optional injection into full trials."""

    sampled_params: dict[str, Scalar]
    objective_score: Optional[float]
    objective_metric: Optional[str]
    objective_best_epoch: Optional[int]
    hparam_context: Optional[dict[str, object]]


@dataclass(frozen=True)
class PersistentTrialTask:
    """One task payload submitted to a persistent trial worker."""

    trial_id: int
    sampled_params: dict[str, Scalar]
    metrics_json: str
    log_file: str


@dataclass(frozen=True)
class PersistentTrialOutcome:
    """One completed trial payload returned from a persistent worker."""

    slot_index: int
    result: TrialResult


def _validate_positive_int(value: object, name: str) -> int:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


def _validate_non_negative_int(value: object, name: str) -> int:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")
    return value


def _validate_unit_interval(value: object, name: str) -> float:
    """Validate that a value is numeric and in [0.0, 1.0]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric in [0.0, 1.0].")
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0].")
    return parsed


def _validate_trial_stream_mode(value: object, name: str) -> str:
    """Validate trial stream mode for subprocess output control."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be one of: {', '.join(sorted(TRIAL_STREAM_MODE_CHOICES))}."
        )
    parsed = value.strip().lower()
    if parsed not in TRIAL_STREAM_MODE_CHOICES:
        raise ValueError(
            f"{name} must be one of: {', '.join(sorted(TRIAL_STREAM_MODE_CHOICES))}."
        )
    return parsed


def _validate_trial_process_mode(value: object, name: str) -> str:
    """Validate trial process mode for phase execution backend selection."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be one of: {', '.join(sorted(TRIAL_PROCESS_MODE_CHOICES))}."
        )
    parsed = value.strip().lower()
    if parsed not in TRIAL_PROCESS_MODE_CHOICES:
        raise ValueError(
            f"{name} must be one of: {', '.join(sorted(TRIAL_PROCESS_MODE_CHOICES))}."
        )
    return parsed


def _validate_search_space(
    raw_space: object,
) -> dict[str, dict[str, object]]:
    """Validate and normalize mixed search-space specifications."""
    if not isinstance(raw_space, dict):
        raise ValueError("search_space must be an object.")
    normalized: dict[str, dict[str, object]] = {}
    for name, raw_spec in raw_space.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Each search-space key must be a non-empty string.")
        if not isinstance(raw_spec, dict):
            raise ValueError(f"search_space['{name}'] must be an object.")
        kind = raw_spec.get("type")
        if kind == "categorical":
            values = raw_spec.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"search_space['{name}'].values must be a non-empty list."
                )
            for value in values:
                if not isinstance(value, (int, float, str, bool)):
                    raise ValueError(
                        f"search_space['{name}'] categorical values must be scalars."
                    )
            normalized[name] = {"type": "categorical", "values": values}
            continue
        if kind == "float":
            min_value = raw_spec.get("min")
            max_value = raw_spec.get("max")
            scale = raw_spec.get("scale", "linear")
            if not isinstance(min_value, (int, float)):
                raise ValueError(f"search_space['{name}'].min must be numeric.")
            if not isinstance(max_value, (int, float)):
                raise ValueError(f"search_space['{name}'].max must be numeric.")
            if float(min_value) >= float(max_value):
                raise ValueError(
                    f"search_space['{name}'] requires min < max for float."
                )
            if scale not in {"linear", "log"}:
                raise ValueError(f"search_space['{name}'].scale must be linear or log.")
            if scale == "log" and float(min_value) <= 0.0:
                raise ValueError(f"search_space['{name}'] log scale requires min > 0.")
            normalized[name] = {
                "type": "float",
                "min": float(min_value),
                "max": float(max_value),
                "scale": str(scale),
            }
            continue
        if kind == "int":
            min_value = raw_spec.get("min")
            max_value = raw_spec.get("max")
            step = raw_spec.get("step", 1)
            if not isinstance(min_value, int) or not isinstance(max_value, int):
                raise ValueError(f"search_space['{name}'] int bounds must be integers.")
            if not isinstance(step, int) or step <= 0:
                raise ValueError(f"search_space['{name}'].step must be > 0 integer.")
            if min_value > max_value:
                raise ValueError(f"search_space['{name}'] requires min <= max for int.")
            normalized[name] = {
                "type": "int",
                "min": min_value,
                "max": max_value,
                "step": step,
            }
            continue
        raise ValueError(f"search_space['{name}'].type must be categorical|float|int.")
    if not normalized:
        raise ValueError("search_space must define at least one parameter.")
    return normalized


def _site_window_len_spec() -> dict[str, object]:
    """Return canonical search spec for donor/acceptor sequence window length."""
    return {
        "type": "int",
        "min": _SITE_WINDOW_LEN_MIN,
        "max": _SITE_WINDOW_LEN_MAX,
        "step": _SITE_WINDOW_LEN_STEP,
    }


def _resolve_site_window_len_search_keys(
    base_args: dict[str, ArgValue],
) -> tuple[str, ...]:
    """Resolve which site-window length keys should be searched."""
    raw_train_target = base_args.get("train_target", "both")
    train_target = str(raw_train_target).strip().lower()
    if train_target == "donor":
        return ("donor_len",)
    if train_target == "acceptor":
        return ("acceptor_len",)
    return _SITE_WINDOW_LEN_KEYS


def _inject_site_window_len_space(
    search_space: dict[str, dict[str, object]],
    base_args: dict[str, ArgValue],
) -> dict[str, dict[str, object]]:
    """Ensure donor_len and acceptor_len are always part of search space."""
    normalized = dict(search_space)
    for key in _resolve_site_window_len_search_keys(base_args):
        if key not in normalized:
            normalized[key] = _site_window_len_spec()
    return normalized


def load_config(path: Path) -> SearchConfig:
    """Load and validate search configuration from JSON."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be an object.")

    project_root_raw = raw.get("project_root")
    species = raw.get("species")
    output_dir_raw = raw.get("output_dir")
    base_args = raw.get("base_args")
    search_space = raw.get("search_space")

    if not isinstance(project_root_raw, str) or not project_root_raw:
        raise ValueError("project_root must be a non-empty string.")
    if not isinstance(species, str) or not species:
        raise ValueError("species must be a non-empty string.")
    if not isinstance(output_dir_raw, str) or not output_dir_raw:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(base_args, dict):
        raise ValueError("base_args must be an object.")
    model_name = base_args.get("model")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("base_args.model must be a non-empty string.")

    project_root = Path(project_root_raw).resolve()
    output_dir = Path(output_dir_raw).resolve()
    quick_trials = _validate_positive_int(raw.get("quick_trials"), "quick_trials")
    quick_epochs = _validate_positive_int(raw.get("quick_epochs"), "quick_epochs")
    top_k = _validate_positive_int(raw.get("top_k"), "top_k")
    full_epochs = _validate_positive_int(raw.get("full_epochs"), "full_epochs")
    base_seed = _validate_non_negative_int(raw.get("base_seed"), "base_seed")
    min_batch_size = _validate_positive_int(
        raw.get("min_batch_size", 64),
        "min_batch_size",
    )
    max_oom_retries = _validate_non_negative_int(
        raw.get("max_oom_retries", 8),
        "max_oom_retries",
    )
    max_model_params_raw = raw.get("max_model_params")
    max_model_params: Optional[int]
    if max_model_params_raw is None:
        max_model_params = None
    else:
        max_model_params = _validate_positive_int(
            max_model_params_raw,
            "max_model_params",
        )
    search_algo = str(raw.get("search_algo", "random")).strip()
    if search_algo not in {"random", "history_guided"}:
        raise ValueError("search_algo must be one of: random, history_guided.")
    history_top_n = _validate_positive_int(
        raw.get("history_top_n", 64),
        "history_top_n",
    )
    guided_random_fraction = _validate_unit_interval(
        raw.get("guided_random_fraction", 0.35),
        "guided_random_fraction",
    )
    guided_mutation_rate = _validate_unit_interval(
        raw.get("guided_mutation_rate", 0.25),
        "guided_mutation_rate",
    )
    trial_stream_mode = _validate_trial_stream_mode(
        raw.get("trial_stream_mode", "auto"),
        "trial_stream_mode",
    )
    trial_process_mode = _validate_trial_process_mode(
        raw.get("trial_process_mode", "subprocess"),
        "trial_process_mode",
    )
    objective_metric = str(raw.get("objective_metric", "mean_pr_auc"))
    if objective_metric not in SUPPORTED_OBJECTIVE_METRICS:
        raise ValueError(
            "objective_metric must be one of: "
            f"{', '.join(SUPPORTED_OBJECTIVE_METRIC_NAMES)}."
        )
    normalized_base_args: dict[str, ArgValue] = {
        str(key): value for key, value in base_args.items()
    }
    normalized_base_args["model"] = model_name.strip()
    normalized_base_args.setdefault("donor_len", _SITE_WINDOW_LEN_DEFAULT)
    normalized_base_args.setdefault("acceptor_len", _SITE_WINDOW_LEN_DEFAULT)

    normalized_space = _inject_site_window_len_space(
        _validate_search_space(search_space),
        normalized_base_args,
    )
    global_best_config_raw = raw.get("global_best_config_path")
    global_best_config_path: Optional[Path]
    if global_best_config_raw is None:
        global_best_config_path = None
    else:
        if not isinstance(global_best_config_raw, str) or not global_best_config_raw:
            raise ValueError("global_best_config_path must be a non-empty string.")
        global_best_config_path = Path(global_best_config_raw).resolve()
    seed_best_config_raw = raw.get("seed_best_config_path")
    seed_best_config_path: Optional[Path]
    if seed_best_config_raw is None:
        seed_best_config_path = None
    else:
        if not isinstance(seed_best_config_raw, str) or not seed_best_config_raw:
            raise ValueError("seed_best_config_path must be a non-empty string.")
        seed_best_config_path = Path(seed_best_config_raw).resolve()
    quick_overrides = raw.get("quick_overrides", {})
    full_overrides = raw.get("full_overrides", {})
    if not isinstance(quick_overrides, dict):
        raise ValueError("quick_overrides must be an object.")
    if not isinstance(full_overrides, dict):
        raise ValueError("full_overrides must be an object.")
    skip_full_phase = _to_bool(raw.get("skip_full_phase", False))
    enable_visualization = _to_bool(raw.get("enable_visualization", True))

    return SearchConfig(
        project_root=project_root,
        species=species,
        output_dir=output_dir,
        quick_trials=quick_trials,
        quick_epochs=quick_epochs,
        top_k=top_k,
        full_epochs=full_epochs,
        base_seed=base_seed,
        gpu_ids_setting=raw.get("gpu_ids", "auto"),
        max_parallel_trials_setting=raw.get("max_parallel_trials", "auto"),
        min_batch_size=min_batch_size,
        max_oom_retries=max_oom_retries,
        max_model_params=max_model_params,
        objective_metric=objective_metric,
        global_best_config_path=global_best_config_path,
        seed_best_config_path=seed_best_config_path,
        base_args=normalized_base_args,
        quick_overrides={str(k): v for k, v in quick_overrides.items()},
        full_overrides={str(k): v for k, v in full_overrides.items()},
        search_space=normalized_space,
        search_algo=search_algo,
        history_top_n=history_top_n,
        guided_random_fraction=guided_random_fraction,
        guided_mutation_rate=guided_mutation_rate,
        trial_stream_mode=trial_stream_mode,
        trial_process_mode=trial_process_mode,
        skip_full_phase=skip_full_phase,
        enable_visualization=enable_visualization,
    )


def _value_matches_spec(value: Scalar, spec: dict[str, object]) -> bool:
    """Return whether one scalar value is valid for a search-space spec."""
    kind = str(spec["type"])
    if kind == "categorical":
        values = spec["values"]
        if not isinstance(values, list):
            return False
        return any(value == candidate for candidate in values)
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        min_value = float(spec["min"])
        max_value = float(spec["max"])
        return min_value <= float(value) <= max_value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        min_value = int(spec["min"])
        max_value = int(spec["max"])
        step = int(spec["step"])
        if not (min_value <= value <= max_value):
            return False
        return (value - min_value) % step == 0
    return False


def _normalize_context_object(value: object) -> object:
    """Normalize one JSON-like value for deterministic context comparison."""
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key in sorted(value):
            normalized[str(key)] = _normalize_context_object(value[key])
        return normalized
    if isinstance(value, list):
        return [_normalize_context_object(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_context_object(item) for item in value]
    return value


def _context_digest(context: dict[str, object]) -> str:
    """Serialize one context mapping to a canonical digest string."""
    normalized = _normalize_context_object(context)
    if not isinstance(normalized, dict):
        raise ValueError("context must normalize to an object.")
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _contexts_match(
    left: Optional[dict[str, object]],
    right: Optional[dict[str, object]],
) -> bool:
    """Return whether two optional context mappings are equivalent."""
    if left is None or right is None:
        return False
    return _context_digest(left) == _context_digest(right)


def _build_hparam_context(
    *,
    objective_metric: str,
    full_epochs: int,
    validation_protocol: dict[str, object],
    fixed_run_args: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """Build comparison context used for global-best compatibility checks."""
    context: dict[str, object] = {
        "version": 2,
        "objective_metric": objective_metric,
        "full_epochs": full_epochs,
        "validation_protocol": _normalize_context_object(validation_protocol),
    }
    if fixed_run_args:
        context["fixed_run_args"] = _normalize_context_object(fixed_run_args)
    return context


def _build_fixed_run_args_context(
    *,
    base_args: dict[str, ArgValue],
    full_overrides: dict[str, ArgValue],
    search_space: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Build context for non-search fixed arguments that affect comparability."""
    merged_args: dict[str, ArgValue] = dict(base_args)
    for key, value in full_overrides.items():
        merged_args[key] = value

    search_keys = set(search_space)
    fixed_args: dict[str, object] = {}
    for key, value in merged_args.items():
        if key in search_keys or key in _CONTEXT_ARG_IGNORE_KEYS:
            continue
        if value is None:
            continue
        if isinstance(value, Path):
            text = str(value)
            if text != "":
                fixed_args[key] = text
            continue
        if isinstance(value, str):
            if value != "":
                fixed_args[key] = value
            continue
        if isinstance(value, (bool, int, float)):
            fixed_args[key] = value
            continue
        fixed_args[key] = str(value)
    normalized = _normalize_context_object(fixed_args)
    if not isinstance(normalized, dict):
        raise ValueError("fixed_run_args must normalize to an object.")
    return normalized


def _extract_hparam_context(raw: dict[str, object]) -> Optional[dict[str, object]]:
    """Extract optional HPO context from one best-config payload."""
    context_obj = raw.get("hparam_context")
    if not isinstance(context_obj, dict):
        return None
    normalized = _normalize_context_object(context_obj)
    if not isinstance(normalized, dict):
        return None
    return normalized


def _extract_sampled_params_from_best_config(
    *,
    raw: dict[str, object],
    search_space: dict[str, dict[str, object]],
    base_args: dict[str, ArgValue],
) -> dict[str, Scalar]:
    """Validate and normalize sampled params loaded from best_config.json."""
    sampled = raw.get("sampled_params")
    if not isinstance(sampled, dict):
        raise ValueError("Global best config missing sampled_params object.")

    normalized: dict[str, Scalar] = {}
    for key, value in sampled.items():
        if not isinstance(value, (int, float, str, bool)):
            raise ValueError(
                f"Global best sampled_params.{key} must be a scalar value."
            )
        if key in search_space and not _value_matches_spec(value, search_space[key]):
            raise ValueError(
                f"Global best sampled_params.{key}={value} is not in current "
                "search space."
            )
        normalized[key] = value

    for key, spec in search_space.items():
        if key in sampled:
            continue
        if key not in base_args:
            continue
        base_value = base_args[key]
        if not isinstance(base_value, (int, float, str, bool)):
            raise ValueError(f"Global best base_args.{key} must be a scalar value.")
        if not _value_matches_spec(base_value, spec):
            raise ValueError(
                f"Global best base_args.{key}={base_value} is not in current "
                "search space."
            )
    return normalized


def load_global_best_params(
    *,
    path: Optional[Path],
    search_space: dict[str, dict[str, object]],
    base_args: dict[str, ArgValue],
) -> Optional[dict[str, Scalar]]:
    """Load and validate previous best sampled params for forced inclusion."""
    if path is None or not path.exists():
        return None
    raw = read_json_object(path)
    if raw is None:
        raise ValueError(f"Invalid global best config JSON: {path}")
    if raw.get("status") != "ok":
        return None
    return _extract_sampled_params_from_best_config(
        raw=raw,
        search_space=search_space,
        base_args=base_args,
    )


def load_seed_best_config(
    *,
    path: Optional[Path],
    search_space: dict[str, dict[str, object]],
    base_args: dict[str, ArgValue],
    default_objective_metric: str,
) -> Optional[SeedBestConfig]:
    """Load rich seed-best metadata for context-aware full-phase injection."""
    if path is None or not path.exists():
        return None
    raw = read_json_object(path)
    if raw is None:
        raise ValueError(f"Invalid global best config JSON: {path}")
    if raw.get("status") != "ok":
        return None

    sampled_params = _extract_sampled_params_from_best_config(
        raw=raw,
        search_space=search_space,
        base_args=base_args,
    )
    objective_metric = raw.get("objective_metric")
    metric_name: Optional[str]
    if isinstance(objective_metric, str) and objective_metric.strip():
        metric_name = objective_metric.strip()
    else:
        metric_name = default_objective_metric

    objective_score: Optional[float] = None
    score_raw = raw.get("objective_score")
    if isinstance(score_raw, (int, float)):
        objective_score = float(score_raw)
    elif metric_name is not None:
        metric_score_raw = raw.get(metric_name)
        if isinstance(metric_score_raw, (int, float)):
            objective_score = float(metric_score_raw)

    objective_best_epoch = _to_positive_int(raw.get("objective_best_epoch"))
    if objective_best_epoch is None:
        metrics_json_raw = raw.get("metrics_json")
        if isinstance(metrics_json_raw, str) and metrics_json_raw.strip():
            objective_best_epoch = _read_objective_best_epoch_from_metrics(
                metrics_json_path=metrics_json_raw,
                objective_metric=metric_name or default_objective_metric,
            )

    hparam_context = _extract_hparam_context(raw)
    return SeedBestConfig(
        sampled_params=sampled_params,
        objective_score=objective_score,
        objective_metric=metric_name,
        objective_best_epoch=objective_best_epoch,
        hparam_context=hparam_context,
    )


def _parse_objective_metric_name(objective_metric: str) -> tuple[str, str]:
    """Split one objective metric into scope and metric name."""
    if objective_metric not in SUPPORTED_OBJECTIVE_METRICS:
        raise ValueError(f"Unsupported objective metric: {objective_metric}")
    scope, metric_name = objective_metric.split("_", maxsplit=1)
    return scope, metric_name


def _is_test_objective_metric(objective_metric: str) -> bool:
    """Return whether objective selection uses held-out test evaluation."""
    scope, _ = _parse_objective_metric_name(objective_metric)
    return scope == "test"


def _extract_max_f1_from_eval_output(path: Path) -> Optional[float]:
    """Extract maximum F1 score from ``evaluate_scores`` text output."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    best_f1: Optional[float] = None
    for raw_line in lines:
        line = raw_line.strip()
        if line == "":
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            f1_value = float(parts[5])
        except ValueError:
            continue
        if not math.isfinite(f1_value):
            continue
        if best_f1 is None or f1_value > best_f1:
            best_f1 = f1_value
    return best_f1


def _extract_test_objective_score(
    *,
    objective_metric: str,
    eval_output_path: Path,
) -> Optional[float]:
    """Extract one held-out test objective score from evaluation artifacts."""
    if objective_metric == "test_max_f1":
        return _extract_max_f1_from_eval_output(eval_output_path)
    return None


def _to_optional_int(value: object) -> Optional[int]:
    """Convert scalar-like values to an optional integer."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _resolve_hparam_data_root(project_root: Path) -> Path:
    """Resolve data root from env override or ``<project_root>/data``."""
    raw = os.environ.get("INTRONMODEL_DATA_ROOT", "").strip()
    if raw == "":
        return (project_root / "data").resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path.resolve()


def _resolve_test_pr_auc_score_source(train_target: str) -> str:
    """Resolve score-source mode from training target."""
    if train_target == "donor":
        return "donor"
    if train_target == "acceptor":
        return "acceptor"
    if train_target == "pair":
        return "pair"
    return "donor_acceptor"


def _multiply_donor_acceptor_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Convert donor/acceptor rows into pair rows by score multiplication."""
    donor_scores: dict[tuple[str, int], float] = {}
    acceptor_scores: dict[tuple[str, int], float] = {}
    for row in rows:
        row_site_type = str(row.get("site_type", "")).strip().lower()
        key = (str(row["transcript_id"]), int(row["intron_index"]))
        score = float(row["score"])
        if row_site_type == "donor":
            donor_scores[key] = score
        elif row_site_type == "acceptor":
            acceptor_scores[key] = score

    pair_rows: list[dict[str, object]] = []
    for key in sorted(set(donor_scores) & set(acceptor_scores)):
        pair_rows.append(
            {
                "transcript_id": key[0],
                "intron_index": key[1],
                "site_type": "pair",
                "score": donor_scores[key] * acceptor_scores[key],
            }
        )
    return pair_rows


def _score_site_rows_single_task_model(
    *,
    model_name: str,
    task: str,
    checkpoint_path: Path,
    site_rows: list[dict[str, object]],
    device: str,
    batch_size: int,
    sequence_transform: str,
) -> list[dict[str, object]]:
    """Score donor/acceptor site rows for one single-task model checkpoint."""
    from models.registry import load_model_module
    from util.sequence_transform import apply_site_sequence_transform

    model_module = load_model_module(model_name)
    if not hasattr(model_module, "load_task_model") or not hasattr(
        model_module, "score_sequences"
    ):
        raise ValueError(
            f"Model '{model_name}' does not support test_pr_auc site scoring."
        )

    transformed_rows: list[dict[str, object]] = []
    for row in site_rows:
        row_site_type = str(row["site_type"])
        if row_site_type != task:
            continue
        transformed_seq = apply_site_sequence_transform(
            str(row["seq"]),
            site_type=row_site_type,
            transform_mode=sequence_transform,
            intron_half_length=(
                int(row["intron_half_length"])
                if row.get("intron_half_length") is not None
                else None
            ),
        )
        transformed = dict(row)
        transformed["seq"] = transformed_seq
        transformed_rows.append(transformed)

    if not transformed_rows:
        return []

    model, checkpoint_payload = model_module.load_task_model(
        str(checkpoint_path), device
    )
    window_len = int(checkpoint_payload.get("window_len", 50))
    sequences = [str(row["seq"]) for row in transformed_rows]
    scores = model_module.score_sequences(
        model,
        sequences,
        window_len,
        device,
        batch_size=batch_size,
    )

    out_rows: list[dict[str, object]] = []
    for row, score in zip(transformed_rows, scores):
        out_rows.append(
            {
                "transcript_id": str(row["transcript_id"]),
                "intron_index": int(row["intron_index"]),
                "site_type": task,
                "score": float(score),
            }
        )
    return out_rows


def _score_site_rows_pair_model(
    *,
    model_name: str,
    checkpoint_path: Path,
    pair_rows: list[dict[str, object]],
    device: str,
    batch_size: int,
    sequence_transform: str,
) -> list[dict[str, object]]:
    """Score pair-site rows for one pair-model checkpoint."""
    if model_name == "cnn_pair":
        from models import cnn_pair as pair_module
    elif model_name in {"cnn_v2", "cnn_v2_pair"}:
        from models import cnn_v2 as pair_module
    else:
        raise ValueError(f"Unsupported pair model for scoring: {model_name}")

    return pair_module.infer_pair_site_scores(
        pair_rows=pair_rows,
        pair_model_path=str(checkpoint_path),
        device=device,
        batch_size=batch_size,
        sequence_transform=sequence_transform,
    )


def _compute_test_pr_auc_objective(
    *,
    config: SearchConfig,
    merged_args: dict[str, ArgValue],
    metrics_json: Path,
    trial_artifact_base: Path,
) -> Optional[float]:
    """Compute external-test PR-AUC objective for one trial."""
    from evaluate_intron_pr_auc import evaluate_labeled_introns
    from util.data_proc import (
        read_test_pair_rows,
        read_test_site_rows,
        resolve_test_tsv,
    )
    from util.transcript_eval import write_site_scores

    model_name = str(merged_args.get("model", "")).strip()
    species = str(merged_args.get("species", config.species)).strip()
    if model_name == "" or species == "":
        return None

    train_target_raw = merged_args.get("train_target", "both")
    train_target = str(train_target_raw).strip().lower() or "both"
    if train_target not in {"both", "donor", "acceptor", "pair"}:
        return None
    cnn_v2_pair_mode = str(merged_args.get("pair_mode", "pair")).strip().lower()
    if model_name == "cnn_v2_pair":
        cnn_v2_pair_mode = "pair"
    if model_name == "cnn_v2" and cnn_v2_pair_mode != "pair" and train_target == "pair":
        train_target = "both"

    donor_len = _to_optional_int(merged_args.get("donor_len"))
    acceptor_len = _to_optional_int(merged_args.get("acceptor_len"))
    sequence_transform = str(merged_args.get("sequence_transform", "none")).strip()
    if sequence_transform == "":
        sequence_transform = "none"
    batch_size = _to_optional_int(merged_args.get("batch_size"))
    if batch_size is None or batch_size <= 0:
        batch_size = 512
    device = str(merged_args.get("device", "auto")).strip() or "auto"

    test_tsv_override = merged_args.get("test_tsv")
    test_tsv_arg = (
        str(test_tsv_override) if isinstance(test_tsv_override, (str, Path)) else None
    )
    test_tsv = Path(resolve_test_tsv(species, test_tsv_arg))

    checkpoint_paths = _extract_checkpoint_paths_from_metrics(str(metrics_json))
    scored_rows: list[dict[str, object]] = []
    use_pair_model_scoring = model_name in {"cnn_pair", "cnn_v2_pair"} or (
        model_name == "cnn_v2" and cnn_v2_pair_mode == "pair"
    )
    if use_pair_model_scoring:
        pair_checkpoint_raw = checkpoint_paths.get("pair_checkpoint_path")
        if pair_checkpoint_raw is None:
            return None
        pair_rows, _skipped_short, _skipped_unpaired = read_test_pair_rows(
            str(test_tsv),
            donor_len,
            acceptor_len,
        )
        if not pair_rows:
            return None
        scored_rows = _score_site_rows_pair_model(
            model_name=model_name,
            checkpoint_path=Path(pair_checkpoint_raw),
            pair_rows=pair_rows,
            device=device,
            batch_size=batch_size,
            sequence_transform=sequence_transform,
        )
    else:
        site_rows, _skipped_short = read_test_site_rows(
            str(test_tsv),
            donor_len,
            acceptor_len,
        )
        tasks_to_score: tuple[str, ...]
        if train_target == "both":
            tasks_to_score = ("donor", "acceptor")
        elif train_target in {"donor", "acceptor"}:
            tasks_to_score = (train_target,)
        else:
            return None

        for task in tasks_to_score:
            checkpoint_key = f"{task}_checkpoint_path"
            checkpoint_raw = checkpoint_paths.get(checkpoint_key)
            if checkpoint_raw is None:
                return None
            task_rows = _score_site_rows_single_task_model(
                model_name=model_name,
                task=task,
                checkpoint_path=Path(checkpoint_raw),
                site_rows=site_rows,
                device=device,
                batch_size=batch_size,
                sequence_transform=sequence_transform,
            )
            scored_rows.extend(task_rows)

    if model_name == "cnn_v2" and cnn_v2_pair_mode != "pair":
        scored_rows = _multiply_donor_acceptor_rows(scored_rows)
        train_target = "pair"

    if not scored_rows:
        return None

    site_score_tsv = Path(str(trial_artifact_base) + ".site.tsv")
    write_site_scores(str(site_score_tsv), scored_rows)

    labeled_intron_raw = merged_args.get("labeled_intron_tsv")
    if isinstance(labeled_intron_raw, (str, Path)) and str(labeled_intron_raw).strip():
        labeled_intron_tsv = Path(str(labeled_intron_raw))
    else:
        data_root = _resolve_hparam_data_root(config.project_root)
        labeled_intron_tsv = (
            data_root / species / "processed" / "intron_eval_flank10.unique.tsv"
        )

    intron_score_op_raw = merged_args.get("intron_score_op", "*")
    intron_score_op = str(intron_score_op_raw).strip() or "*"
    score_source = _resolve_test_pr_auc_score_source(train_target)
    summary, rows = evaluate_labeled_introns(
        labeled_tsv=labeled_intron_tsv,
        site_score_tsv=site_score_tsv,
        intron_score_op=intron_score_op,
        score_source=score_source,
        strict_missing=False,
    )
    objective_summary_path = Path(str(trial_artifact_base) + ".test_pr_auc.json")
    objective_payload = {
        "pr_auc": float(summary.pr_auc),
        "used_introns": int(summary.used_introns),
        "positive_count": int(summary.positive_count),
        "negative_count": int(summary.negative_count),
        "score_source": summary.score_source,
        "site_score_tsv": str(site_score_tsv),
        "labeled_intron_tsv": str(labeled_intron_tsv),
        "rows_written": len(rows),
    }
    objective_summary_path.write_text(
        json.dumps(objective_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return float(summary.pr_auc)


def _select_objective_score(
    objective_metric: str,
    objective_values: dict[str, Optional[float]],
) -> Optional[float]:
    """Return one objective score from a normalized objective-value mapping."""
    _ = _parse_objective_metric_name(objective_metric)
    return objective_values.get(objective_metric)


def _compute_mean_metric(
    donor_metric: Optional[float],
    acceptor_metric: Optional[float],
    pair_metric: Optional[float],
    objective_metric: str,
    pair_objective_metric: str,
) -> Optional[float]:
    """Compute donor/acceptor mean metric with pair fallback for pair objectives."""
    if donor_metric is None or acceptor_metric is None:
        if objective_metric == pair_objective_metric:
            return pair_metric
        return None
    return (donor_metric + acceptor_metric) / 2.0


def _read_best_objective_score(
    path: Optional[Path],
    objective_metric: str,
    expected_hparam_context: Optional[dict[str, object]] = None,
) -> Optional[float]:
    """Read objective score from a best_config.json payload."""
    if path is None:
        return None
    raw = read_json_object(path)
    if raw is None:
        return None
    if expected_hparam_context is not None:
        existing_context = _extract_hparam_context(raw)
        if not _contexts_match(existing_context, expected_hparam_context):
            return None
    value = raw.get("objective_score")
    if isinstance(value, (int, float)):
        return float(value)
    value = raw.get(objective_metric)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_validation_protocol(
    summary: dict[str, object],
) -> Optional[dict[str, object]]:
    """Extract validation protocol from one metrics summary."""
    raw = summary.get("validation_protocol")
    if not isinstance(raw, dict):
        return None
    normalized: dict[str, object] = {}
    for key, value in raw.items():
        normalized[str(key)] = value
    return normalized


def _extract_validation_signature(summary: dict[str, object]) -> str:
    """Extract validation signature with legacy fallback."""
    raw = summary.get("validation_signature")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return LEGACY_VALIDATION_SIGNATURE


def _sample_value(spec: dict[str, object], rng: random.Random) -> Scalar:
    """Sample one value from a validated search-space spec."""
    kind = str(spec["type"])
    if kind == "categorical":
        values = spec["values"]
        if not isinstance(values, list):
            raise ValueError("categorical values must be a list.")
        selected = values[rng.randrange(len(values))]
        if not isinstance(selected, (int, float, str, bool)):
            raise ValueError("categorical values must be scalar.")
        return selected
    if kind == "float":
        min_value = float(spec["min"])
        max_value = float(spec["max"])
        scale = str(spec["scale"])
        if scale == "linear":
            return min_value + (max_value - min_value) * rng.random()
        log_min = math.log(min_value)
        log_max = math.log(max_value)
        return math.exp(log_min + (log_max - log_min) * rng.random())
    if kind == "int":
        min_value = int(spec["min"])
        max_value = int(spec["max"])
        step = int(spec["step"])
        count = ((max_value - min_value) // step) + 1
        return min_value + step * rng.randrange(count)
    raise ValueError(f"Unsupported search-space type: {kind}")


def _sample_trial_params_with_rng(
    search_space: dict[str, dict[str, object]],
    rng: random.Random,
) -> dict[str, Scalar]:
    """Sample one parameter set using an existing RNG."""
    sampled: dict[str, Scalar] = {}
    for key in sorted(search_space):
        sampled[key] = _sample_value(search_space[key], rng)
    return sampled


def sample_trial_params(
    search_space: dict[str, dict[str, object]],
    seed: int,
) -> dict[str, Scalar]:
    """Sample one parameter set deterministically from a seed."""
    rng = random.Random(seed)
    return _sample_trial_params_with_rng(search_space, rng)


def _parse_bool_text(value: str) -> Optional[bool]:
    """Parse one bool-like text value."""
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_history_param_value(
    *,
    raw_value: object,
    spec: dict[str, object],
) -> Optional[Scalar]:
    """Parse one parameter value loaded from historical TSV rows."""
    if not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    if not text:
        return None
    kind = str(spec["type"])
    if kind == "categorical":
        values = spec.get("values")
        if not isinstance(values, list):
            return None
        for candidate in values:
            if not isinstance(candidate, (int, float, str, bool)):
                continue
            if text == str(candidate):
                return candidate
            if isinstance(candidate, bool):
                parsed_bool = _parse_bool_text(text)
                if parsed_bool is not None and parsed_bool == candidate:
                    return candidate
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                try:
                    parsed_int = int(text)
                except ValueError:
                    continue
                if parsed_int == candidate:
                    return candidate
            if isinstance(candidate, float):
                try:
                    parsed_float = float(text)
                except ValueError:
                    continue
                if math.isfinite(parsed_float) and parsed_float == candidate:
                    return candidate
        return None
    if kind == "float":
        try:
            parsed = float(text)
        except ValueError:
            return None
        if not math.isfinite(parsed):
            return None
        min_value = float(spec["min"])
        max_value = float(spec["max"])
        if parsed < min_value or parsed > max_value:
            return None
        return parsed
    if kind == "int":
        try:
            parsed = int(text)
        except ValueError:
            return None
        min_value = int(spec["min"])
        max_value = int(spec["max"])
        step = int(spec["step"])
        if parsed < min_value or parsed > max_value:
            return None
        if (parsed - min_value) % step != 0:
            return None
        return parsed
    return None


def _is_dnabert_model_name(model_name: str) -> bool:
    """Return whether model name maps to one DNABERT variant."""
    return model_name.strip().lower().startswith(_DNABERT_MODEL_PREFIX)


def _normalize_dnabert_readout_type(raw_value: object) -> str:
    """Normalize DNABERT readout type with strict value validation."""
    normalized = str(raw_value).strip().lower()
    if normalized not in _DNABERT_READOUT_CHOICES:
        choices_text = ", ".join(_DNABERT_READOUT_CHOICES)
        raise ValueError(f"readout_type must be one of: {choices_text}.")
    return normalized


def _materialize_dnabert_readout_params(
    *,
    model_name: str,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> dict[str, Scalar]:
    """Drop inactive DNABERT readout params and fill active defaults."""
    out = dict(sampled_params)
    if not _is_dnabert_model_name(model_name):
        return out

    readout_raw = out.get("readout_type", base_args.get("readout_type", "cnn"))
    readout_type = _normalize_dnabert_readout_type(readout_raw)
    out["readout_type"] = readout_type

    active_keys: frozenset[str]
    if readout_type == "cnn":
        active_keys = _DNABERT_CNN_ONLY_KEYS
    elif readout_type == "mlp":
        active_keys = _DNABERT_MLP_ONLY_KEYS
    else:
        active_keys = frozenset()

    for key in active_keys:
        if key in out:
            continue
        candidate = base_args.get(key)
        if isinstance(candidate, (int, float, str, bool)):
            out[key] = candidate

    inactive_keys = (_DNABERT_CNN_ONLY_KEYS | _DNABERT_MLP_ONLY_KEYS) - active_keys
    for inactive_key in inactive_keys:
        out.pop(inactive_key, None)
    return out


def load_historical_trials(
    *,
    output_dir: Path,
    search_space: dict[str, dict[str, object]],
    objective_metric: str,
    top_n: int,
    base_args: Optional[dict[str, ArgValue]] = None,
) -> list[tuple[float, dict[str, Scalar]]]:
    """Load successful historical trials from sibling run directories."""
    model_name = ""
    resolved_base_args: dict[str, ArgValue]
    if base_args is None:
        resolved_base_args = {}
    else:
        resolved_base_args = dict(base_args)
        model_obj = resolved_base_args.get("model")
        if isinstance(model_obj, str):
            model_name = model_obj
    tuning_root = output_dir.parent
    if not tuning_root.exists():
        return []
    collected: list[tuple[float, dict[str, Scalar]]] = []
    for run_dir in sorted(tuning_root.iterdir()):
        if not run_dir.is_dir() or run_dir.resolve() == output_dir.resolve():
            continue
        for file_name in ("quick_trials.tsv", "full_trials.tsv"):
            tsv_path = run_dir / file_name
            if not tsv_path.exists():
                continue
            try:
                with tsv_path.open("r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    for row in reader:
                        if row.get("status") != "success":
                            continue
                        raw_score = row.get("objective_score", "")
                        if not raw_score:
                            raw_score = row.get(objective_metric, "")
                        try:
                            score = float(raw_score)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(score):
                            continue
                        params: dict[str, Scalar] = {}
                        valid = True
                        for key in sorted(search_space):
                            spec = search_space[key]
                            parsed = _parse_history_param_value(
                                raw_value=row.get(key, ""),
                                spec=spec,
                            )
                            if parsed is None:
                                fallback: Optional[Scalar] = None
                                if base_args is not None:
                                    base_raw = base_args.get(key)
                                    if isinstance(base_raw, (int, float, str, bool)):
                                        fallback = base_raw
                                if fallback is None and key in _SITE_WINDOW_LEN_KEYS:
                                    fallback = _SITE_WINDOW_LEN_DEFAULT
                                if fallback is not None and _value_matches_spec(
                                    fallback,
                                    spec,
                                ):
                                    parsed = fallback
                            if parsed is None:
                                valid = False
                                break
                            params[key] = parsed
                        if valid:
                            normalized = _materialize_dnabert_readout_params(
                                model_name=model_name,
                                sampled_params=params,
                                base_args=resolved_base_args,
                            )
                            collected.append((score, normalized))
            except OSError:
                continue
    if not collected:
        return []

    best_by_key: dict[str, tuple[float, dict[str, Scalar]]] = {}
    for score, params in collected:
        dedup_key = json.dumps(params, sort_keys=True)
        previous = best_by_key.get(dedup_key)
        if previous is None or score > previous[0]:
            best_by_key[dedup_key] = (score, params)
    ranked = sorted(
        best_by_key.values(),
        key=lambda item: item[0],
        reverse=True,
    )
    return ranked[:top_n]


def _sample_weighted_index(scores: list[float], rng: random.Random) -> int:
    """Sample one index with score-proportional positive weights."""
    min_score = min(scores)
    shifted = [score - min_score + 1e-9 for score in scores]
    total = sum(shifted)
    if total <= 0.0:
        return rng.randrange(len(scores))
    target = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(shifted):
        cumulative += weight
        if target <= cumulative:
            return index
    return len(scores) - 1


def sample_trial_params_history_guided(
    *,
    search_space: dict[str, dict[str, object]],
    seed: int,
    history_trials: list[tuple[float, dict[str, Scalar]]],
    random_fraction: float,
    mutation_rate: float,
) -> dict[str, Scalar]:
    """Sample one trial params set using historical top trials as anchors."""
    rng = random.Random(seed)
    if not history_trials or rng.random() < random_fraction:
        return _sample_trial_params_with_rng(search_space, rng)

    scores = [score for score, _ in history_trials]
    anchor_index = _sample_weighted_index(scores, rng)
    sampled = dict(history_trials[anchor_index][1])

    for key in sorted(search_space):
        if rng.random() < mutation_rate:
            sampled[key] = _sample_value(search_space[key], rng)
    return sampled


def detect_gpu_ids(setting: object) -> list[str]:
    """Resolve GPU ids from config, CUDA_VISIBLE_DEVICES, or nvidia-smi."""
    if isinstance(setting, list):
        parsed = [str(item).strip() for item in setting if str(item).strip()]
        return parsed
    if isinstance(setting, str):
        text = setting.strip()
        if text and text != "auto":
            return [part.strip() for part in text.split(",") if part.strip()]
    elif setting is not None:
        raise ValueError("gpu_ids must be auto, string list, or array.")

    env_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_visible:
        return [part.strip() for part in env_visible.split(",") if part.strip()]

    cmd = [
        "nvidia-smi",
        "--query-gpu=index",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines


def resolve_max_parallel(setting: object, gpu_count: int) -> int:
    """Resolve max concurrent trials from config and available GPUs."""
    parsed: int
    if isinstance(setting, str):
        text = setting.strip()
        if text.lower() == "auto":
            if gpu_count > 0:
                return gpu_count
            return 1
        try:
            parsed = int(text)
        except ValueError as exc:
            raise ValueError("max_parallel_trials must be auto or integer.") from exc
    elif isinstance(setting, int):
        parsed = setting
    else:
        raise ValueError("max_parallel_trials must be auto or integer.")
    if parsed <= 0:
        raise ValueError("max_parallel_trials must be > 0.")
    if gpu_count > 0:
        return min(parsed, gpu_count)
    return parsed


def _set_active_max_parallel_trials(value: int) -> int:
    """Set global active parallel-trial count and return previous value."""
    global _ACTIVE_MAX_PARALLEL_TRIALS
    previous_value = _ACTIVE_MAX_PARALLEL_TRIALS
    _ACTIVE_MAX_PARALLEL_TRIALS = max(1, int(value))
    return previous_value


def _is_auto_num_workers(value: ArgValue) -> bool:
    """Return whether one ``num_workers`` value requests auto resolution."""
    if not isinstance(value, str):
        return False
    return value.strip().lower() == "auto"


def _resolve_hparam_auto_num_workers(max_parallel_trials: int) -> int:
    """Resolve conservative per-trial workers for parallel HPO workloads.

    The hparam sweep launches many trials concurrently, and each model trial may
    create multiple DataLoaders. A direct ``cpu_count // 2`` per trial often
    oversubscribes CPU resources heavily. This resolver keeps the existing
    conservative default, then additionally caps it by ``cpu_count // parallel``
    so per-trial workers never exceed the CPU budget implied by GPU-run
    parallelism.
    """
    cpu_count = os.cpu_count() or 4
    parallel = max(1, int(max_parallel_trials))
    per_trial_cpu_budget = max(1, cpu_count // parallel)
    workers = max(1, per_trial_cpu_budget // 4)
    if cpu_count >= 64 and parallel >= 4:
        workers = max(workers, 4)
    current_default = min(8, workers)
    return min(current_default, per_trial_cpu_budget)


def _resolve_trial_num_workers(num_workers_value: ArgValue) -> Optional[int]:
    """Resolve effective trial ``num_workers`` override when value is ``auto``."""
    if not _is_auto_num_workers(num_workers_value):
        return None
    return _resolve_hparam_auto_num_workers(_ACTIVE_MAX_PARALLEL_TRIALS)


def _iter_cuda_header_candidates() -> Iterator[Path]:
    """Yield plausible filesystem candidates for ``cuda.h``."""
    seen: set[Path] = set()

    def _add(path: Path) -> Iterator[Path]:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        yield resolved

    env_roots = (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        os.environ.get("CONDA_PREFIX"),
    )
    for root in env_roots:
        if root is None or not root.strip():
            continue
        root_path = Path(root.strip())
        yield from _add(root_path / "include" / "cuda.h")
        targets_root = root_path / "targets"
        yield from _add(targets_root / "x86_64-linux" / "include" / "cuda.h")
        if targets_root.exists():
            for include_dir in targets_root.glob("*/include"):
                yield from _add(include_dir / "cuda.h")

    tool_names = ("ptxas", "nvcc")
    for tool_name in tool_names:
        tool_path_text = shutil.which(tool_name)
        if tool_path_text is None:
            continue
        tool_path = Path(tool_path_text).resolve()
        if tool_path.parent.name == "bin":
            yield from _add(tool_path.parent.parent / "include" / "cuda.h")
            targets_root = tool_path.parent.parent / "targets"
            yield from _add(targets_root / "x86_64-linux" / "include" / "cuda.h")
            if targets_root.exists():
                for include_dir in targets_root.glob("*/include"):
                    yield from _add(include_dir / "cuda.h")

    explicit_tool_env = (
        os.environ.get("TRITON_PTXAS_PATH"),
        os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
    )
    for path_text in explicit_tool_env:
        if path_text is None or not path_text.strip():
            continue
        tool_path = Path(path_text.strip())
        if tool_path.parent.name == "bin":
            yield from _add(tool_path.parent.parent / "include" / "cuda.h")
            targets_root = tool_path.parent.parent / "targets"
            yield from _add(targets_root / "x86_64-linux" / "include" / "cuda.h")
            if targets_root.exists():
                for include_dir in targets_root.glob("*/include"):
                    yield from _add(include_dir / "cuda.h")

    common_candidates = (
        Path("/usr/local/cuda/include/cuda.h"),
        Path("/usr/include/cuda.h"),
        Path("/opt/cuda/include/cuda.h"),
    )
    for candidate in common_candidates:
        yield from _add(candidate)


def _find_cuda_header() -> Optional[Path]:
    """Return first discovered ``cuda.h`` candidate path that exists."""
    for candidate in _iter_cuda_header_candidates():
        if candidate.exists():
            return candidate
    return None


def _is_oom_text(text: str) -> bool:
    """Return whether text includes a CUDA OOM pattern."""
    lowered = text.lower()
    keywords = (
        "out of memory",
        "cuda error: out of memory",
        "cudnn_status_alloc_failed",
    )
    return any(keyword in lowered for keyword in keywords)


def _is_non_retryable_oom_text(text: str) -> bool:
    """Return whether text marks OOM as non-retryable by caller contract."""
    lowered = text.lower()
    markers = (
        "non_retryable_oom",
        "non-retryable oom",
    )
    return any(marker in lowered for marker in markers)


def _has_internal_oom_backoff(text: str) -> bool:
    """Return whether training already attempted internal batch OOM backoff."""
    lowered = text.lower()
    marker = "retry with smaller batch size"
    return marker in lowered


def _build_run_model_command(
    project_root: Path,
    args: dict[str, ArgValue],
) -> list[str]:
    """Build command list for ``src/run_model.py`` execution."""
    helper_keys = {
        "conv_depth",
        "channel_candidates",
        "kernel_candidates",
        "donor_conv_depth",
        "acceptor_conv_depth",
        "donor_channel_candidates",
        "acceptor_channel_candidates",
        "donor_kernel_candidates",
        "acceptor_kernel_candidates",
    }
    cmd = [sys.executable, "-u", str(project_root / "src" / "run_model.py")]
    for key in sorted(args):
        if key in helper_keys:
            continue
        value = args[key]
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])
    return cmd


def _extract_task_metric(
    summary: dict[str, object],
    task_name: str,
    metric_name: str,
) -> Optional[float]:
    """Extract one task-level best metric from a train summary JSON payload."""
    raw_task = summary.get(task_name)
    if not isinstance(raw_task, dict):
        return None
    best_metric_key = f"best_{metric_name}"
    best_metric_value = raw_task.get(best_metric_key)
    if isinstance(best_metric_value, (int, float)):
        return float(best_metric_value)
    best_metric = raw_task.get("best_metric")
    best_score = raw_task.get("best_score")
    if best_metric == metric_name and isinstance(best_score, (int, float)):
        return float(best_score)
    return None


def _extract_pr_auc(summary: dict[str, object], task_name: str) -> Optional[float]:
    """Extract PR-AUC for one task from train summary JSON."""
    return _extract_task_metric(summary, task_name, "pr_auc")


def _extract_roc_auc(summary: dict[str, object], task_name: str) -> Optional[float]:
    """Extract ROC-AUC for one task from train summary JSON."""
    return _extract_task_metric(summary, task_name, "roc_auc")


def _extract_max_f1(summary: dict[str, object], task_name: str) -> Optional[float]:
    """Extract max-F1 for one task from train summary JSON."""
    return _extract_task_metric(summary, task_name, "max_f1")


def _extract_best_epoch(summary: dict[str, object], task_name: str) -> Optional[int]:
    """Extract best epoch index for one task from train summary JSON."""
    raw_task = summary.get(task_name)
    if not isinstance(raw_task, dict):
        return None
    best_epoch = raw_task.get("best_epoch")
    if isinstance(best_epoch, int) and best_epoch > 0:
        return best_epoch
    return None


def _extract_metric_best_epoch_from_history(
    history: object,
    metric_name: str,
) -> Optional[int]:
    """Extract metric-best epoch from one epoch-history style payload."""
    if not isinstance(history, list):
        return None
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    for row in history:
        if not isinstance(row, dict):
            continue
        metric_value = row.get(metric_name)
        if not isinstance(metric_value, (int, float)):
            continue
        metric_float = float(metric_value)
        if not math.isfinite(metric_float):
            continue
        epoch_value = _to_positive_int(row.get("epoch"))
        if epoch_value is None:
            continue
        if best_value is None or metric_float > best_value:
            best_value = metric_float
            best_epoch = epoch_value
    return best_epoch


def _extract_best_epoch_for_metric(
    summary: dict[str, object],
    task_name: str,
    metric_name: str,
) -> Optional[int]:
    """Extract metric-aligned best epoch for one task."""
    raw_task = summary.get(task_name)
    if not isinstance(raw_task, dict):
        return None
    history_best_epoch = _extract_metric_best_epoch_from_history(
        raw_task.get("epoch_history"),
        metric_name,
    )
    if history_best_epoch is not None:
        return history_best_epoch
    legacy_history_best_epoch = _extract_metric_best_epoch_from_history(
        raw_task.get("epochs"),
        metric_name,
    )
    if legacy_history_best_epoch is not None:
        return legacy_history_best_epoch
    best_metric = raw_task.get("best_metric")
    best_epoch = _to_positive_int(raw_task.get("best_epoch"))
    if best_metric == metric_name and best_epoch is not None:
        return best_epoch
    if (
        metric_name == "pr_auc"
        and best_epoch is not None
        and isinstance(raw_task.get("best_pr_auc"), (int, float))
    ):
        return best_epoch
    return None


def _extract_checkpoint_paths_from_metrics(
    metrics_json_path: str,
) -> dict[str, str]:
    """Extract donor/acceptor checkpoint paths from one metrics JSON file."""
    metrics_path = Path(metrics_json_path).resolve()
    raw = read_json_object(metrics_path)
    if raw is None:
        return {}
    extracted = extract_checkpoint_paths(
        raw,
        base_dir=metrics_path.parent,
        existing_only=False,
    )
    resolved: dict[str, str] = {
        f"{task_name}_checkpoint_path": str(path)
        for task_name, path in extracted.items()
    }
    return resolved


def _resolve_model_root(project_root: Path) -> Path:
    """Resolve model root directory from environment or project default."""
    raw = os.environ.get("INTRONMODEL_MODEL_ROOT", "").strip()
    if raw == "":
        return (project_root / "model").resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path.resolve()


def _is_path_within(path: Path, root: Path) -> bool:
    """Return whether one path is located under ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _collect_checkpoint_paths_from_row(row: TrialResult) -> set[Path]:
    """Collect checkpoint file paths referenced by one trial result."""
    payload = _extract_checkpoint_paths_from_metrics(row.metrics_json)
    out: set[Path] = set()
    for raw_path in payload.values():
        resolved = Path(raw_path).resolve()
        if resolved.exists():
            out.add(resolved)
    return out


def _prune_non_best_trial_checkpoints(
    *,
    project_root: Path,
    trial_rows: list[TrialResult],
    best_row: Optional[TrialResult],
    min_mtime_epoch: float,
) -> int:
    """Delete non-best trial checkpoints while keeping pre-existing files."""
    if best_row is None:
        return 0

    all_paths: set[Path] = set()
    for row in trial_rows:
        if row.status != "success":
            continue
        all_paths.update(_collect_checkpoint_paths_from_row(row))
    if not all_paths:
        return 0

    keep_paths = _collect_checkpoint_paths_from_row(best_row)
    model_root = _resolve_model_root(project_root)
    deleted_count = 0
    for path in sorted(all_paths):
        if path in keep_paths:
            continue
        if not path.exists() or not path.is_file():
            continue
        if not _is_path_within(path, model_root):
            continue
        try:
            modified = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if modified < min_mtime_epoch:
            continue
        path.unlink()
        deleted_count += 1
    return deleted_count


def _serialize_top_trials(
    rows: list[TrialResult],
    top_k: int,
) -> list[dict[str, object]]:
    """Serialize ranked top-k trials for JSON export."""
    serialized: list[dict[str, object]] = []
    for rank, row in enumerate(rows[:top_k], start=1):
        serialized.append(
            {
                "rank": rank,
                "phase": row.phase,
                "trial_id": row.trial_id,
                "objective_metric": row.objective_metric,
                "objective_score": row.objective_score,
                "donor_pr_auc": row.donor_pr_auc,
                "acceptor_pr_auc": row.acceptor_pr_auc,
                "mean_pr_auc": row.mean_pr_auc,
                "sampled_params": row.sampled_params,
                "metrics_json": row.metrics_json,
                "log_file": row.log_file,
            }
        )
    return serialized


def _write_tuning_leaderboard(
    *,
    config: SearchConfig,
    ranked_rows: list[TrialResult],
    best_row: Optional[TrialResult],
) -> None:
    """Write tuning leaderboard JSON under run and model tuning directories."""
    model_obj = config.base_args.get("model")
    model_name = str(model_obj) if isinstance(model_obj, str) else "unknown"
    target = str(config.base_args.get("train_target", "both"))
    top_entries = _serialize_top_trials(ranked_rows, config.top_k)
    best_checkpoint_paths = (
        _extract_checkpoint_paths_from_metrics(best_row.metrics_json)
        if best_row is not None
        else {}
    )
    payload: dict[str, object] = {
        "species": config.species,
        "model": model_name,
        "target": target,
        "objective_metric": config.objective_metric,
        "top_k": config.top_k,
        "entries": top_entries,
        "best_checkpoint_paths": best_checkpoint_paths,
    }

    run_level_path = config.output_dir / f"leaderboard_top{config.top_k}.json"
    run_level_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if config.output_dir.parent.name in {"donor", "acceptor", "both"}:
        model_tuning_dir = config.output_dir.parent.parent
    else:
        model_tuning_dir = config.output_dir.parent
    model_level_path = model_tuning_dir / f"leaderboard_top{config.top_k}.json"
    existing = read_json_object(model_level_path)
    merged: dict[str, object]
    if (
        existing is not None
        and existing.get("species") == config.species
        and existing.get("model") == model_name
        and existing.get("top_k") == config.top_k
        and isinstance(existing.get("targets"), dict)
    ):
        targets_obj = dict(existing["targets"])
        targets_obj[target] = payload
        merged = {
            "species": config.species,
            "model": model_name,
            "top_k": config.top_k,
            "targets": targets_obj,
        }
    else:
        merged = {
            "species": config.species,
            "model": model_name,
            "top_k": config.top_k,
            "targets": {target: payload},
        }
    model_level_path.parent.mkdir(parents=True, exist_ok=True)
    model_level_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


def _read_objective_best_epoch_from_metrics(
    *,
    metrics_json_path: str,
    objective_metric: str,
) -> Optional[int]:
    """Read objective-aligned best epoch from one trial metrics JSON."""
    try:
        raw = json.loads(Path(metrics_json_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None

    scope, metric_name = _parse_objective_metric_name(objective_metric)
    if scope == "donor":
        return _extract_best_epoch_for_metric(raw, "donor", metric_name)
    if scope == "acceptor":
        return _extract_best_epoch_for_metric(raw, "acceptor", metric_name)
    if scope == "pair":
        return _extract_best_epoch_for_metric(raw, "pair", metric_name)
    if scope != "mean":
        return None
    donor_epoch = _extract_best_epoch_for_metric(raw, "donor", metric_name)
    acceptor_epoch = _extract_best_epoch_for_metric(raw, "acceptor", metric_name)
    if donor_epoch is None or acceptor_epoch is None:
        return None
    return max(donor_epoch, acceptor_epoch)


def _iter_stream_lines(stream: object) -> Iterator[str]:
    """Yield decoded text lines from a subprocess stream."""
    if stream is None:
        return
    for raw_line in stream:
        if isinstance(raw_line, str):
            yield raw_line


def _set_active_trial_stream_mode(mode: str) -> str:
    """Set global trial stream mode and return previous mode."""
    global _ACTIVE_TRIAL_STREAM_MODE
    previous_mode = _ACTIVE_TRIAL_STREAM_MODE
    _ACTIVE_TRIAL_STREAM_MODE = mode
    return previous_mode


def _register_active_trial_process(process: subprocess.Popen[str]) -> None:
    """Register one running trial subprocess for interrupt cleanup."""
    with _ACTIVE_TRIAL_PROCESSES_LOCK:
        _ACTIVE_TRIAL_PROCESSES.add(process)


def _deregister_active_trial_process(process: subprocess.Popen[str]) -> None:
    """Deregister one trial subprocess after it exits."""
    with _ACTIVE_TRIAL_PROCESSES_LOCK:
        _ACTIVE_TRIAL_PROCESSES.discard(process)


def _interrupt_active_trial_processes(wait_timeout_sec: float = 3.0) -> None:
    """Terminate all tracked trial subprocesses after a user interrupt."""
    with _ACTIVE_TRIAL_PROCESSES_LOCK:
        active_processes = [proc for proc in _ACTIVE_TRIAL_PROCESSES]

    for process in active_processes:
        if process.poll() is not None:
            continue
        try:
            process.terminate()
        except OSError:
            continue

    for process in active_processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=wait_timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                continue


def _resolve_trial_stream_mode(setting: str, max_parallel_trials: int) -> str:
    """Resolve effective trial stream mode from setting and runtime parallelism."""
    parsed = _validate_trial_stream_mode(setting, "trial_stream_mode")
    if parsed != "auto":
        return parsed
    if max_parallel_trials > 1:
        return "errors"
    return "full"


def _resolve_phase_execution_mode(
    *,
    process_mode: str,
    phase: str,
) -> str:
    """Resolve per-phase execution backend from process-mode policy."""
    if process_mode == "subprocess":
        return _DEFAULT_PHASE_EXECUTION_MODE
    if process_mode == "persistent_all":
        return _PERSISTENT_PHASE_EXECUTION_MODE
    if process_mode == "persistent_quick":
        if phase == "quick":
            return _PERSISTENT_PHASE_EXECUTION_MODE
        return _DEFAULT_PHASE_EXECUTION_MODE
    raise ValueError(
        "trial_process_mode must be one of: "
        f"{', '.join(sorted(TRIAL_PROCESS_MODE_CHOICES))}."
    )


def _resolve_workload_execution_mode(
    *,
    phase_execution_mode: str,
    trial_count: int,
    max_parallel_trials: int,
) -> str:
    """Resolve effective execution mode from phase policy and workload size.

    Persistent workers are beneficial when one slot processes multiple trials.
    If the workload fits in a single wave (``trial_count <= max_parallel_trials``),
    this resolver falls back to subprocess mode to avoid persistent queue/process
    overhead.
    """
    if phase_execution_mode == _DEFAULT_PHASE_EXECUTION_MODE:
        return _DEFAULT_PHASE_EXECUTION_MODE
    if phase_execution_mode != _PERSISTENT_PHASE_EXECUTION_MODE:
        raise ValueError(
            "phase_execution_mode must be one of: "
            f"{_DEFAULT_PHASE_EXECUTION_MODE}, "
            f"{_PERSISTENT_PHASE_EXECUTION_MODE}."
        )
    if trial_count <= max(1, int(max_parallel_trials)):
        return _DEFAULT_PHASE_EXECUTION_MODE
    return _PERSISTENT_PHASE_EXECUTION_MODE


def _should_stream_trial_line(line: str, mode: str) -> bool:
    """Return whether one trial subprocess output line should be mirrored."""
    if mode == "full":
        return True
    if mode == "silent":
        return False
    lowered = line.lower()
    keywords = (
        "traceback",
        "error",
        "failed",
        "exception",
        "oom",
        "nan",
        "inf",
    )
    return any(keyword in lowered for keyword in keywords)


def _emit_trial_output_lines(
    *,
    output_text: str,
    phase: str,
    trial_id: int,
    stream_mode: str,
) -> None:
    """Mirror collected trial output lines to stdout based on stream mode."""
    prefix = f"[hparam_search][{phase} {trial_id:04d}] "
    for raw_line in output_text.splitlines():
        if raw_line == "":
            continue
        if _should_stream_trial_line(raw_line, stream_mode):
            print(f"{prefix}{raw_line}", flush=True)


def _extract_run_model_argv(cmd: list[str]) -> list[str]:
    """Extract ``run_model.py`` argv from one built command list."""
    for index, token in enumerate(cmd):
        if token.endswith("run_model.py"):
            return cmd[index + 1 :]
    raise ValueError("run_model.py entrypoint not found in command.")


@contextmanager
def _temporary_cwd(cwd: Path) -> Iterator[None]:
    """Temporarily switch process working directory for in-process execution."""
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _temporary_env(env: dict[str, str]) -> Iterator[None]:
    """Temporarily replace process environment for in-process execution."""
    previous = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _run_command_with_streaming(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    phase: str,
    trial_id: int,
) -> tuple[int, str]:
    """Run a command and stream merged output while collecting it."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    _register_active_trial_process(proc)

    try:
        collected: list[str] = []
        stream_mode = _ACTIVE_TRIAL_STREAM_MODE
        prefix = f"[hparam_search][{phase} {trial_id:04d}] "
        for line in _iter_stream_lines(proc.stdout):
            collected.append(line)
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            if _should_stream_trial_line(stripped, stream_mode):
                print(f"{prefix}{stripped}", flush=True)

        return_code = int(proc.wait())
        return return_code, "".join(collected)
    finally:
        _deregister_active_trial_process(proc)


def _run_command_inprocess(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    phase: str,
    trial_id: int,
) -> tuple[int, str]:
    """Run ``run_model.py`` in-process while capturing merged output."""
    from run_model import main as run_model_main

    run_model_argv = _extract_run_model_argv(cmd)
    captured = io.StringIO()
    return_code = 0
    with (
        _temporary_cwd(cwd),
        _temporary_env(env),
        redirect_stdout(captured),
        redirect_stderr(captured),
    ):
        try:
            run_model_main(run_model_argv)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return_code = int(exc.code)
            else:
                return_code = 1
        except Exception:
            traceback.print_exc()
            return_code = 1

    combined_output = captured.getvalue()
    _emit_trial_output_lines(
        output_text=combined_output,
        phase=phase,
        trial_id=trial_id,
        stream_mode=_ACTIVE_TRIAL_STREAM_MODE,
    )
    return return_code, combined_output


TrialCommandRunner = Callable[..., tuple[int, str]]


def _build_failed_trial_result(
    *,
    config: SearchConfig,
    phase: str,
    trial_id: int,
    assigned_gpu_id: Optional[str],
    sampled_params: dict[str, Scalar],
    effective_batch_size: int,
    oom_retries: int,
    error_message: str,
    return_code: int,
    duration_sec: float,
    metrics_json: Path,
    log_file: Path,
    donor_pr_auc: Optional[float] = None,
    acceptor_pr_auc: Optional[float] = None,
    mean_pr_auc: Optional[float] = None,
) -> TrialResult:
    """Build one standardized failed trial result payload."""
    return TrialResult(
        phase=phase,
        trial_id=trial_id,
        status="failed",
        gpu_id=assigned_gpu_id,
        sampled_params=sampled_params,
        effective_batch_size=effective_batch_size,
        oom_retries=oom_retries,
        donor_pr_auc=donor_pr_auc,
        acceptor_pr_auc=acceptor_pr_auc,
        mean_pr_auc=mean_pr_auc,
        objective_metric=config.objective_metric,
        objective_score=None,
        error_message=error_message,
        return_code=return_code,
        duration_sec=duration_sec,
        metrics_json=str(metrics_json),
        log_file=str(log_file),
    )


def _run_trial_with_command_runner(
    *,
    config: SearchConfig,
    phase: str,
    trial_id: int,
    sampled_params: dict[str, Scalar],
    overrides: dict[str, ArgValue],
    assigned_gpu_id: Optional[str],
    metrics_json: Path,
    log_file: Path,
    command_runner: TrialCommandRunner,
) -> TrialResult:
    """Run one trial with the provided command runner backend."""
    base_model_name_obj = config.base_args.get("model", "")
    base_model_name = str(base_model_name_obj)
    sampled_params = _materialize_dnabert_readout_params(
        model_name=base_model_name,
        sampled_params=sampled_params,
        base_args=config.base_args,
    )
    merged_args: dict[str, ArgValue] = dict(config.base_args)
    for key, value in sampled_params.items():
        merged_args[key] = value
    for key, value in overrides.items():
        merged_args[key] = value
    model_name = merged_args.get("model")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("base_args.model must be a non-empty string.")
    if model_name.strip().lower() == "cnn":
        merged_args.setdefault("report_train_metrics", 0)
    merged_args["species"] = config.species
    trial_artifact_base = metrics_json.parent / metrics_json.stem
    is_test_max_f1_objective = config.objective_metric == "test_max_f1"
    is_test_pr_auc_objective = config.objective_metric == "test_pr_auc"
    eval_output_path: Optional[Path] = None
    if is_test_max_f1_objective:
        merged_args["train_only"] = False
        merged_args["site_output_tsv"] = str(trial_artifact_base) + ".site.tsv"
        merged_args["intron_output_tsv"] = str(trial_artifact_base) + ".intron.tsv"
        merged_args["transcript_output_tsv"] = (
            str(trial_artifact_base) + ".transcript.tsv"
        )
        eval_output_path = Path(str(trial_artifact_base) + ".eval.txt")
        merged_args["eval_output_txt"] = str(eval_output_path)
    else:
        merged_args["train_only"] = True
    merged_args["metrics_json"] = str(metrics_json)
    resolved_num_workers = _resolve_trial_num_workers(merged_args.get("num_workers"))
    if resolved_num_workers is not None:
        merged_args["num_workers"] = resolved_num_workers

    base_batch = merged_args.get("batch_size")
    if not isinstance(base_batch, int):
        raise ValueError("batch_size must resolve to an integer per trial.")
    current_batch = base_batch
    oom_retries = 0
    started_at = time.time()
    return_code = -1

    while True:
        merged_args["batch_size"] = current_batch
        cmd = _build_run_model_command(config.project_root, merged_args)
        env = os.environ.copy()
        if assigned_gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu_id)

        attempt_index = oom_retries + 1
        attempt_header = (
            f"phase={phase} trial_id={trial_id} attempt={attempt_index} "
            f"batch_size={current_batch} gpu_id={assigned_gpu_id}\n"
            f"command={shlex.join(cmd)}\n"
        )
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(attempt_header)

        return_code, combined_output = command_runner(
            cmd=cmd,
            cwd=config.project_root,
            env=env,
            phase=phase,
            trial_id=trial_id,
        )
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"return_code={return_code}\n")
            handle.write("\n[combined]\n")
            handle.write(combined_output)
            handle.write("\n")

        if return_code == 0:
            break

        is_oom = _is_oom_text(combined_output)
        is_non_retryable_oom = _is_non_retryable_oom_text(combined_output)
        has_internal_backoff = _has_internal_oom_backoff(combined_output)
        can_retry = (
            is_oom
            and not is_non_retryable_oom
            and not has_internal_backoff
            and oom_retries < config.max_oom_retries
            and current_batch > config.min_batch_size
        )
        if not can_retry:
            if is_non_retryable_oom:
                error_message = (
                    f"Training failed with non-retryable OOM (exit={return_code}). "
                    "Reduce model complexity or search-space bounds."
                )
            elif is_oom and has_internal_backoff:
                error_message = (
                    "Training failed after internal OOM backoff "
                    f"(exit={return_code}). See trial log for details."
                )
            else:
                error_message = (
                    f"Training command failed (exit={return_code}). "
                    "See trial log for details."
                )
            duration_sec = time.time() - started_at
            return _build_failed_trial_result(
                config=config,
                phase=phase,
                trial_id=trial_id,
                assigned_gpu_id=assigned_gpu_id,
                sampled_params=sampled_params,
                effective_batch_size=current_batch,
                oom_retries=oom_retries,
                error_message=error_message,
                return_code=return_code,
                duration_sec=duration_sec,
                metrics_json=metrics_json,
                log_file=log_file,
            )

        next_batch = max(config.min_batch_size, current_batch // 2)
        if next_batch >= current_batch:
            duration_sec = time.time() - started_at
            return _build_failed_trial_result(
                config=config,
                phase=phase,
                trial_id=trial_id,
                assigned_gpu_id=assigned_gpu_id,
                sampled_params=sampled_params,
                effective_batch_size=current_batch,
                oom_retries=oom_retries,
                error_message=(
                    "CUDA OOM encountered, but cannot reduce batch further."
                ),
                return_code=return_code,
                duration_sec=duration_sec,
                metrics_json=metrics_json,
                log_file=log_file,
            )
        oom_retries += 1
        current_batch = next_batch

    duration_sec = time.time() - started_at
    try:
        summary = json.loads(metrics_json.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return _build_failed_trial_result(
            config=config,
            phase=phase,
            trial_id=trial_id,
            assigned_gpu_id=assigned_gpu_id,
            sampled_params=sampled_params,
            effective_batch_size=current_batch,
            oom_retries=oom_retries,
            error_message="Metrics JSON missing or invalid.",
            return_code=return_code,
            duration_sec=duration_sec,
            metrics_json=metrics_json,
            log_file=log_file,
        )
    if not isinstance(summary, dict):
        return _build_failed_trial_result(
            config=config,
            phase=phase,
            trial_id=trial_id,
            assigned_gpu_id=assigned_gpu_id,
            sampled_params=sampled_params,
            effective_batch_size=current_batch,
            oom_retries=oom_retries,
            error_message="Metrics JSON top-level value must be an object.",
            return_code=return_code,
            duration_sec=duration_sec,
            metrics_json=metrics_json,
            log_file=log_file,
        )

    donor_pr_auc = _extract_pr_auc(summary, "donor")
    acceptor_pr_auc = _extract_pr_auc(summary, "acceptor")
    pair_pr_auc = _extract_pr_auc(summary, "pair")
    donor_roc_auc = _extract_roc_auc(summary, "donor")
    acceptor_roc_auc = _extract_roc_auc(summary, "acceptor")
    pair_roc_auc = _extract_roc_auc(summary, "pair")
    donor_max_f1 = _extract_max_f1(summary, "donor")
    acceptor_max_f1 = _extract_max_f1(summary, "acceptor")
    pair_max_f1 = _extract_max_f1(summary, "pair")
    validation_protocol = _extract_validation_protocol(summary)
    validation_signature = _extract_validation_signature(summary)
    mean_pr_auc = _compute_mean_metric(
        donor_metric=donor_pr_auc,
        acceptor_metric=acceptor_pr_auc,
        pair_metric=pair_pr_auc,
        objective_metric=config.objective_metric,
        pair_objective_metric="pair_pr_auc",
    )
    mean_roc_auc = _compute_mean_metric(
        donor_metric=donor_roc_auc,
        acceptor_metric=acceptor_roc_auc,
        pair_metric=pair_roc_auc,
        objective_metric=config.objective_metric,
        pair_objective_metric="pair_roc_auc",
    )
    mean_max_f1 = _compute_mean_metric(
        donor_metric=donor_max_f1,
        acceptor_metric=acceptor_max_f1,
        pair_metric=pair_max_f1,
        objective_metric=config.objective_metric,
        pair_objective_metric="pair_max_f1",
    )
    test_max_f1: Optional[float] = None
    if is_test_max_f1_objective:
        if eval_output_path is None:
            return _build_failed_trial_result(
                config=config,
                phase=phase,
                trial_id=trial_id,
                assigned_gpu_id=assigned_gpu_id,
                sampled_params=sampled_params,
                effective_batch_size=current_batch,
                oom_retries=oom_retries,
                error_message="Internal error: missing test objective output path.",
                return_code=return_code,
                duration_sec=duration_sec,
                metrics_json=metrics_json,
                log_file=log_file,
                donor_pr_auc=donor_pr_auc,
                acceptor_pr_auc=acceptor_pr_auc,
                mean_pr_auc=mean_pr_auc,
            )
        test_max_f1 = _extract_test_objective_score(
            objective_metric=config.objective_metric,
            eval_output_path=eval_output_path,
        )
    test_pr_auc: Optional[float] = None
    if is_test_pr_auc_objective:
        try:
            test_pr_auc = _compute_test_pr_auc_objective(
                config=config,
                merged_args=merged_args,
                metrics_json=metrics_json,
                trial_artifact_base=trial_artifact_base,
            )
        except (ValueError, FileNotFoundError) as exc:
            return _build_failed_trial_result(
                config=config,
                phase=phase,
                trial_id=trial_id,
                assigned_gpu_id=assigned_gpu_id,
                sampled_params=sampled_params,
                effective_batch_size=current_batch,
                oom_retries=oom_retries,
                error_message=f"test_pr_auc evaluation failed: {exc}",
                return_code=return_code,
                duration_sec=duration_sec,
                metrics_json=metrics_json,
                log_file=log_file,
                donor_pr_auc=donor_pr_auc,
                acceptor_pr_auc=acceptor_pr_auc,
                mean_pr_auc=mean_pr_auc,
            )
    objective_values: dict[str, Optional[float]] = {
        "mean_pr_auc": mean_pr_auc,
        "donor_pr_auc": donor_pr_auc,
        "acceptor_pr_auc": acceptor_pr_auc,
        "pair_pr_auc": pair_pr_auc,
        "mean_roc_auc": mean_roc_auc,
        "donor_roc_auc": donor_roc_auc,
        "acceptor_roc_auc": acceptor_roc_auc,
        "pair_roc_auc": pair_roc_auc,
        "mean_max_f1": mean_max_f1,
        "donor_max_f1": donor_max_f1,
        "acceptor_max_f1": acceptor_max_f1,
        "pair_max_f1": pair_max_f1,
        "test_pr_auc": test_pr_auc,
        "test_max_f1": test_max_f1,
    }
    objective_score = _select_objective_score(config.objective_metric, objective_values)
    if objective_score is None:
        return _build_failed_trial_result(
            config=config,
            phase=phase,
            trial_id=trial_id,
            assigned_gpu_id=assigned_gpu_id,
            sampled_params=sampled_params,
            effective_batch_size=current_batch,
            oom_retries=oom_retries,
            error_message=(
                "Missing objective metric in training summary: "
                f"{config.objective_metric}."
            ),
            return_code=return_code,
            duration_sec=duration_sec,
            metrics_json=metrics_json,
            log_file=log_file,
            donor_pr_auc=donor_pr_auc,
            acceptor_pr_auc=acceptor_pr_auc,
            mean_pr_auc=mean_pr_auc,
        )

    return TrialResult(
        phase=phase,
        trial_id=trial_id,
        status="success",
        gpu_id=assigned_gpu_id,
        sampled_params=sampled_params,
        effective_batch_size=current_batch,
        oom_retries=oom_retries,
        donor_pr_auc=donor_pr_auc,
        acceptor_pr_auc=acceptor_pr_auc,
        mean_pr_auc=mean_pr_auc,
        objective_metric=config.objective_metric,
        objective_score=objective_score,
        error_message=None,
        return_code=return_code,
        duration_sec=duration_sec,
        metrics_json=str(metrics_json),
        log_file=str(log_file),
        validation_signature=validation_signature,
        validation_protocol=validation_protocol,
        selection_score=objective_score,
    )


def run_trial(
    *,
    config: SearchConfig,
    phase: str,
    trial_id: int,
    sampled_params: dict[str, Scalar],
    overrides: dict[str, ArgValue],
    assigned_gpu_id: Optional[str],
    metrics_json: Path,
    log_file: Path,
) -> TrialResult:
    """Run one trial using subprocess command execution."""
    return _run_trial_with_command_runner(
        config=config,
        phase=phase,
        trial_id=trial_id,
        sampled_params=sampled_params,
        overrides=overrides,
        assigned_gpu_id=assigned_gpu_id,
        metrics_json=metrics_json,
        log_file=log_file,
        command_runner=_run_command_with_streaming,
    )


def run_trial_inprocess(
    *,
    config: SearchConfig,
    phase: str,
    trial_id: int,
    sampled_params: dict[str, Scalar],
    overrides: dict[str, ArgValue],
    assigned_gpu_id: Optional[str],
    metrics_json: Path,
    log_file: Path,
) -> TrialResult:
    """Run one trial by calling ``run_model.main`` in the same process."""
    return _run_trial_with_command_runner(
        config=config,
        phase=phase,
        trial_id=trial_id,
        sampled_params=sampled_params,
        overrides=overrides,
        assigned_gpu_id=assigned_gpu_id,
        metrics_json=metrics_json,
        log_file=log_file,
        command_runner=_run_command_inprocess,
    )


def rank_successful_trials(results: list[TrialResult]) -> list[TrialResult]:
    """Rank successful trials by objective score descending."""
    successful = [
        row
        for row in results
        if row.status == "success" and row.objective_score is not None
    ]
    successful.sort(
        key=lambda row: (
            float(row.objective_score) if row.objective_score is not None else -1.0,
            -row.effective_batch_size,
        ),
        reverse=True,
    )
    return successful


def _format_float(value: Optional[float]) -> str:
    """Format float values for TSV output."""
    if value is None:
        return ""
    return f"{value:.8f}"


def _to_positive_int(value: object) -> Optional[int]:
    """Convert one scalar-like value to a positive integer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = int(text)
        except ValueError:
            return None
    else:
        return None
    if parsed <= 0:
        return None
    return parsed


def _to_bool(value: object) -> bool:
    """Convert common CLI-style scalar values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return False


def _normalize_cnn_pair_fusion_mode(raw_mode: object) -> str:
    """Normalize cnn_pair fusion mode with backward-compatible aliases."""
    mode = str(raw_mode).strip().lower()
    if mode == "early_channel":
        return "early"
    return mode


def _normalize_cnn_v2_pair_mode(raw_mode: object) -> str:
    """Normalize cnn_v2 pair mode aliases for architecture validation."""
    mode = str(raw_mode).strip().lower()
    if mode in {"off", "false", "0", "independent"}:
        return "independent"
    if mode in {"on", "true", "1", "pair"}:
        return "pair"
    return mode


def _resolve_cnn_architecture_validation_model_name(
    *,
    model_name: str,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> str:
    """Resolve effective CNN-family model for shape validation.

    Returns
    -------
    str
        One of ``"cnn"``, ``"cnn_pair"``, or ``""`` when no shape check applies.
    """
    normalized_model = model_name.strip().lower()
    if normalized_model in {"cnn", "cnn_pair"}:
        return normalized_model
    if normalized_model == "cnn_v2":
        pair_mode_raw = sampled_params.get("pair_mode")
        if pair_mode_raw is None:
            pair_mode_raw = base_args.get("pair_mode", "pair")
        pair_mode = _normalize_cnn_v2_pair_mode(pair_mode_raw)
        if pair_mode == "independent":
            return "cnn"
        input_mode_raw = sampled_params.get("input_mode")
        if input_mode_raw is None:
            input_mode_raw = base_args.get("input_mode", "onehot")
        if str(input_mode_raw).strip().lower() == "onehot":
            return "cnn_pair"
        return ""
    if normalized_model == "cnn_v2_pair":
        input_mode_raw = sampled_params.get("input_mode")
        if input_mode_raw is None:
            input_mode_raw = base_args.get("input_mode", "onehot")
        if str(input_mode_raw).strip().lower() == "onehot":
            return "cnn_pair"
        return ""
    return ""


def _derive_validation_protocol_from_args(
    *,
    merged_args: dict[str, ArgValue],
    objective_metric: str,
) -> dict[str, object]:
    """Build validation protocol payload for one run-argument mapping."""
    val_frac_raw = merged_args.get("val_frac")
    val_frac: Optional[float]
    if isinstance(val_frac_raw, bool) or val_frac_raw is None:
        val_frac = None
    elif isinstance(val_frac_raw, (int, float)):
        val_frac = float(val_frac_raw)
    elif isinstance(val_frac_raw, str):
        stripped = val_frac_raw.strip()
        val_frac = float(stripped) if stripped else None
    else:
        val_frac = None

    seed_raw = merged_args.get("seed")
    seed: Optional[int]
    if isinstance(seed_raw, bool) or seed_raw is None:
        seed = None
    elif isinstance(seed_raw, int):
        seed = int(seed_raw)
    elif isinstance(seed_raw, float):
        seed = int(seed_raw)
    elif isinstance(seed_raw, str):
        stripped = seed_raw.strip()
        seed = int(stripped) if stripped else None
    else:
        seed = None

    train_pos_path = merged_args.get("train_pos_path")
    train_neg_path = merged_args.get("train_neg_path")
    model_name = str(merged_args.get("model", "")).strip().lower()
    train_target = str(merged_args.get("train_target", "")).strip().lower()
    pair_mode = str(merged_args.get("pair_mode", "")).strip().lower()
    include_pair_mixed_negatives = False
    if train_target == "pair":
        include_pair_mixed_negatives = True
    elif model_name == "cnn_v2" and pair_mode in {"pair", "on", "true", "1"}:
        include_pair_mixed_negatives = True
    elif model_name == "cnn_v2_pair":
        include_pair_mixed_negatives = True
    elif model_name in {"cnn_pair", "bilstm_pair", "cnn_v3"}:
        include_pair_mixed_negatives = True
    return build_validation_protocol(
        val_frac=val_frac,
        seed=seed,
        train_pos_path=(
            str(train_pos_path) if isinstance(train_pos_path, (str, Path)) else None
        ),
        train_neg_path=(
            str(train_neg_path) if isinstance(train_neg_path, (str, Path)) else None
        ),
        metric_primary=objective_metric,
        split_type=(
            "test_transcript_eval"
            if _is_test_objective_metric(objective_metric)
            else "stratified_site"
        ),
        include_pair_mixed_negatives=include_pair_mixed_negatives,
    )


def _parse_conv_channels(value: object) -> Optional[list[int]]:
    """Parse ``conv_channels`` value from string/list to positive ints."""
    if value is None:
        return None
    parts: list[object]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return None

    channels: list[int] = []
    for part in parts:
        parsed = _to_positive_int(part)
        if parsed is None:
            return None
        channels.append(parsed)
    if not channels:
        return None
    return channels


def _resolve_kernel_sizes_for_depth(
    *,
    kernel_raw: object,
    fallback_scalar_raw: object,
    depth: int,
) -> Optional[list[int]]:
    """Resolve one kernel-size list aligned to the given depth."""
    parsed = _parse_conv_channels(kernel_raw)
    if parsed is None:
        fallback = _to_positive_int(kernel_raw)
        if fallback is None:
            fallback = _to_positive_int(fallback_scalar_raw)
        if fallback is None:
            return None
        parsed = [fallback]

    if len(parsed) == 1:
        return parsed * depth
    if len(parsed) < depth:
        return parsed + ([parsed[-1]] * (depth - len(parsed)))
    if len(parsed) > depth:
        return parsed[:depth]
    return parsed


def _coerce_positive_int_list(value: object) -> Optional[list[int]]:
    """Coerce one value into a positive integer list."""
    parsed = _parse_conv_channels(value)
    if parsed is not None:
        return parsed
    scalar = _to_positive_int(value)
    if scalar is None:
        return None
    return [scalar]


def _default_conv_depth(
    *,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
    conv_key: str,
) -> int:
    """Infer fallback CNN depth from conv_channels or use default depth."""
    conv_raw = sampled_params.get(conv_key)
    if conv_raw is None:
        conv_raw = base_args.get(conv_key)
    parsed = _parse_conv_channels(conv_raw)
    if parsed is not None and parsed:
        return len(parsed)
    return 3


def _sample_list_by_depth(
    *,
    candidates: list[int],
    depth: int,
    rng: random.Random,
) -> list[int]:
    """Sample one depth-aligned list from candidates independently."""
    sampled: list[int] = []
    for _ in range(depth):
        sampled.append(candidates[rng.randrange(len(candidates))])
    return sampled


def _resolve_candidate_pool(
    *,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
    key: str,
) -> Optional[list[int]]:
    """Resolve one positive-integer candidate pool from sampled/base args."""
    raw = sampled_params.get(key)
    if raw is None:
        raw = base_args.get(key)
    return _coerce_positive_int_list(raw)


def _stringify_int_list(values: list[int]) -> str:
    """Serialize one integer list to comma-separated string."""
    return ",".join(str(value) for value in values)


def _resolve_max_pool_size(
    *,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> int:
    """Resolve CNN max-pooling width with backward-compatible fallback."""
    raw = sampled_params.get("max_pool_size")
    if raw is None:
        raw = base_args.get("max_pool_size")
    if raw is not None:
        resolved = _to_positive_int(raw)
        if resolved is not None:
            return resolved

    legacy_raw = sampled_params.get("use_max_pool")
    if legacy_raw is None:
        legacy_raw = base_args.get("use_max_pool")
    if legacy_raw is not None:
        return 2 if _to_bool(legacy_raw) else 1
    return 2


def _resolve_conv_stride(
    *,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> int:
    """Resolve shared CNN convolution stride."""
    raw = sampled_params.get("conv_stride")
    if raw is None:
        raw = base_args.get("conv_stride")
    resolved = _to_positive_int(raw)
    if resolved is not None:
        return resolved
    return 1


def _resolve_kernel_argument(
    *,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
    kernel_key: str,
    scalar_key: str,
    depth: int,
) -> Optional[list[int]]:
    """Resolve one depth-aligned kernel-size list."""
    kernel_raw = sampled_params.get(kernel_key)
    if kernel_raw is None:
        kernel_raw = base_args.get(kernel_key)
    scalar_raw = sampled_params.get(scalar_key)
    if scalar_raw is None:
        scalar_raw = base_args.get(scalar_key)
    return _resolve_kernel_sizes_for_depth(
        kernel_raw=kernel_raw,
        fallback_scalar_raw=scalar_raw,
        depth=depth,
    )


def _conv_output_length(length: int, kernel_size: int, stride: int) -> int:
    """Compute one Conv1d output length with same-style padding."""
    if length <= 0 or kernel_size <= 0 or stride <= 0:
        return 0
    padding = kernel_size // 2
    numerator = length + (2 * padding) - (kernel_size - 1) - 1
    if numerator < 0:
        return 0
    return (numerator // stride) + 1


def _apply_cnn_length_schedule(
    length: int,
    kernel_sizes: Sequence[int],
    conv_stride: int,
    max_pool_size: int,
) -> int:
    """Apply one CNN stack schedule and return the resulting length."""
    current = length
    if not kernel_sizes:
        return current
    for kernel_size in kernel_sizes:
        current = _conv_output_length(current, kernel_size, conv_stride)
        if current <= 0:
            return 0
        if max_pool_size > 1:
            current = current // max_pool_size
            if current <= 0:
                return 0
    return current


def _is_valid_cnn_architecture(
    *,
    model_name: str,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> bool:
    """Return whether sampled CNN-family architecture is shape-valid."""
    validation_model_name = _resolve_cnn_architecture_validation_model_name(
        model_name=model_name,
        sampled_params=sampled_params,
        base_args=base_args,
    )
    if validation_model_name == "":
        return True

    max_pool_size = _resolve_max_pool_size(
        sampled_params=sampled_params,
        base_args=base_args,
    )
    if max_pool_size <= 0:
        return False
    conv_stride = _resolve_conv_stride(
        sampled_params=sampled_params,
        base_args=base_args,
    )
    if conv_stride <= 0:
        return False

    donor_len_raw = sampled_params.get("donor_len")
    if donor_len_raw is None:
        donor_len_raw = base_args.get("donor_len")
    acceptor_len_raw = sampled_params.get("acceptor_len")
    if acceptor_len_raw is None:
        acceptor_len_raw = base_args.get("acceptor_len")

    donor_len = _to_positive_int(donor_len_raw)
    acceptor_len = _to_positive_int(acceptor_len_raw)

    if validation_model_name == "cnn":
        train_target_raw = sampled_params.get("train_target")
        if train_target_raw is None:
            train_target_raw = base_args.get("train_target", "both")
        train_target = str(train_target_raw).strip().lower()
        if str(model_name).strip().lower() == "cnn_v2":
            pair_mode_raw = sampled_params.get("pair_mode")
            if pair_mode_raw is None:
                pair_mode_raw = base_args.get("pair_mode", "pair")
            if _normalize_cnn_v2_pair_mode(pair_mode_raw) == "independent":
                # cnn_v2 independent path delegates to cnn with forced both targets.
                train_target = "both"

        conv_channels_raw = sampled_params.get("conv_channels")
        if conv_channels_raw is None:
            conv_channels_raw = base_args.get("conv_channels")
        conv_channels = _parse_conv_channels(conv_channels_raw)
        if conv_channels is None:
            lightweight = _to_bool(base_args.get("lightweight"))
            conv_channels = [64, 128] if lightweight else [64, 128, 256]
        depth = len(conv_channels)
        kernel_sizes = _resolve_kernel_argument(
            sampled_params=sampled_params,
            base_args=base_args,
            kernel_key="kernel_sizes",
            scalar_key="kernel_size",
            depth=depth,
        )
        if kernel_sizes is None:
            kernel_sizes = [7] * depth

        if train_target in {"both", "donor"} and donor_len is not None:
            if (
                _apply_cnn_length_schedule(
                    donor_len,
                    kernel_sizes,
                    conv_stride,
                    max_pool_size,
                )
                <= 0
            ):
                return False
        if train_target in {"both", "acceptor"} and acceptor_len is not None:
            if (
                _apply_cnn_length_schedule(
                    acceptor_len,
                    kernel_sizes,
                    conv_stride,
                    max_pool_size,
                )
                <= 0
            ):
                return False
        return True

    fusion_mode_raw = sampled_params.get("fusion_mode")
    if fusion_mode_raw is None:
        fusion_mode_raw = base_args.get("fusion_mode", "late")
    fusion_mode = _normalize_cnn_pair_fusion_mode(fusion_mode_raw)

    donor_conv_channels_raw = sampled_params.get("donor_conv_channels")
    if donor_conv_channels_raw is None:
        donor_conv_channels_raw = base_args.get("donor_conv_channels")
    acceptor_conv_channels_raw = sampled_params.get("acceptor_conv_channels")
    if acceptor_conv_channels_raw is None:
        acceptor_conv_channels_raw = base_args.get("acceptor_conv_channels")
    shared_conv_channels_raw = sampled_params.get("conv_channels")
    if shared_conv_channels_raw is None:
        shared_conv_channels_raw = base_args.get("conv_channels")

    shared_conv_channels = _parse_conv_channels(shared_conv_channels_raw)
    donor_conv_channels = _parse_conv_channels(donor_conv_channels_raw)
    if donor_conv_channels is None:
        donor_conv_channels = shared_conv_channels
    acceptor_conv_channels = _parse_conv_channels(acceptor_conv_channels_raw)
    if acceptor_conv_channels is None:
        acceptor_conv_channels = shared_conv_channels
    if donor_conv_channels is None:
        lightweight = _to_bool(base_args.get("lightweight"))
        donor_conv_channels = [64, 128] if lightweight else [64, 128, 256]
    if acceptor_conv_channels is None:
        lightweight = _to_bool(base_args.get("lightweight"))
        acceptor_conv_channels = [64, 128] if lightweight else [64, 128, 256]

    donor_depth = len(donor_conv_channels)
    acceptor_depth = len(acceptor_conv_channels)
    donor_kernel_sizes = _resolve_kernel_argument(
        sampled_params=sampled_params,
        base_args=base_args,
        kernel_key="donor_kernel_sizes",
        scalar_key="kernel_size",
        depth=donor_depth,
    )
    if donor_kernel_sizes is None:
        donor_kernel_sizes = _resolve_kernel_argument(
            sampled_params=sampled_params,
            base_args=base_args,
            kernel_key="kernel_sizes",
            scalar_key="kernel_size",
            depth=donor_depth,
        )
    acceptor_kernel_sizes = _resolve_kernel_argument(
        sampled_params=sampled_params,
        base_args=base_args,
        kernel_key="acceptor_kernel_sizes",
        scalar_key="kernel_size",
        depth=acceptor_depth,
    )
    if acceptor_kernel_sizes is None:
        acceptor_kernel_sizes = _resolve_kernel_argument(
            sampled_params=sampled_params,
            base_args=base_args,
            kernel_key="kernel_sizes",
            scalar_key="kernel_size",
            depth=acceptor_depth,
        )
    if donor_kernel_sizes is None:
        donor_kernel_sizes = [7] * donor_depth
    if acceptor_kernel_sizes is None:
        acceptor_kernel_sizes = [7] * acceptor_depth

    if fusion_mode in {"early", "mid"}:
        if (
            donor_len is not None
            and acceptor_len is not None
            and donor_len != acceptor_len
        ):
            return False
        if donor_conv_channels != acceptor_conv_channels:
            return False
        if donor_kernel_sizes != acceptor_kernel_sizes:
            return False
        shared_len = donor_len if donor_len is not None else acceptor_len
        if shared_len is None:
            return True
        if fusion_mode == "early":
            return (
                _apply_cnn_length_schedule(
                    shared_len,
                    donor_kernel_sizes,
                    conv_stride,
                    max_pool_size,
                )
                > 0
            )

        split_index = max(1, donor_depth // 2)
        prefix_length = _apply_cnn_length_schedule(
            shared_len,
            donor_kernel_sizes[:split_index],
            conv_stride,
            max_pool_size,
        )
        if prefix_length <= 0:
            return False
        return (
            _apply_cnn_length_schedule(
                prefix_length,
                donor_kernel_sizes[split_index:],
                conv_stride,
                max_pool_size,
            )
            > 0
        )

    if donor_len is not None:
        if (
            _apply_cnn_length_schedule(
                donor_len,
                donor_kernel_sizes,
                conv_stride,
                max_pool_size,
            )
            <= 0
        ):
            return False
    if acceptor_len is not None:
        if (
            _apply_cnn_length_schedule(
                acceptor_len,
                acceptor_kernel_sizes,
                conv_stride,
                max_pool_size,
            )
            <= 0
        ):
            return False
    return True


def _materialize_cnn_architecture_params(
    *,
    model_name: str,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
    rng: random.Random,
) -> dict[str, Scalar]:
    """Derive CNN architecture arguments from independent search keys."""
    out = dict(sampled_params)

    def _materialize_branch(
        *,
        depth_key: str,
        channel_pool_key: str,
        kernel_pool_key: str,
        conv_out_key: str,
        kernel_out_key: str,
        shared_depth_key: Optional[str] = None,
        shared_channel_key: Optional[str] = None,
        shared_kernel_key: Optional[str] = None,
    ) -> None:
        explicit_conv_raw = out.get(conv_out_key)
        if explicit_conv_raw is None:
            explicit_conv_raw = base_args.get(conv_out_key)
        explicit_conv = _parse_conv_channels(explicit_conv_raw)

        explicit_kernel_raw = out.get(kernel_out_key)
        if explicit_kernel_raw is None:
            explicit_kernel_raw = base_args.get(kernel_out_key)
        explicit_kernel = _parse_conv_channels(explicit_kernel_raw)

        depth_raw = out.get(depth_key)
        if depth_raw is None:
            depth_raw = base_args.get(depth_key)
        if depth_raw is None and shared_depth_key is not None:
            depth_raw = out.get(shared_depth_key)
            if depth_raw is None:
                depth_raw = base_args.get(shared_depth_key)
        depth = _to_positive_int(depth_raw)
        if explicit_conv is not None:
            depth = len(explicit_conv)
        elif explicit_kernel is not None:
            depth = len(explicit_kernel)
        if depth is None:
            depth = _default_conv_depth(
                sampled_params=out,
                base_args=base_args,
                conv_key=conv_out_key,
            )

        channel_candidates = _resolve_candidate_pool(
            sampled_params=out,
            base_args=base_args,
            key=channel_pool_key,
        )
        if channel_candidates is None and shared_channel_key is not None:
            channel_candidates = _resolve_candidate_pool(
                sampled_params=out,
                base_args=base_args,
                key=shared_channel_key,
            )
        if channel_candidates is None:
            channel_candidates = [64, 96, 128, 192, 256, 384, 512]

        kernel_candidates = _resolve_candidate_pool(
            sampled_params=out,
            base_args=base_args,
            key=kernel_pool_key,
        )
        if kernel_candidates is None and shared_kernel_key is not None:
            kernel_candidates = _resolve_candidate_pool(
                sampled_params=out,
                base_args=base_args,
                key=shared_kernel_key,
            )
        if kernel_candidates is None:
            kernel_candidates = [3, 5, 7, 9, 11, 13, 15]
        kernel_candidates = [value for value in kernel_candidates if value % 2 == 1]
        if not kernel_candidates:
            raise ValueError("kernel_candidates must include at least one odd value.")

        if explicit_conv is None:
            out[conv_out_key] = _stringify_int_list(
                _sample_list_by_depth(
                    candidates=channel_candidates,
                    depth=depth,
                    rng=rng,
                )
            )
        if explicit_kernel is None:
            out[kernel_out_key] = _stringify_int_list(
                _sample_list_by_depth(
                    candidates=kernel_candidates,
                    depth=depth,
                    rng=rng,
                )
            )

    if model_name in {"cnn", "cnn_resdil"}:
        has_arch_keys = any(
            key in out or key in base_args
            for key in ("conv_depth", "channel_candidates", "kernel_candidates")
        )
        if has_arch_keys:
            _materialize_branch(
                depth_key="conv_depth",
                channel_pool_key="channel_candidates",
                kernel_pool_key="kernel_candidates",
                conv_out_key="conv_channels",
                kernel_out_key="kernel_sizes",
            )
    elif model_name == "cnn_pair":
        fusion_mode_raw = out.get("fusion_mode")
        if fusion_mode_raw is None:
            fusion_mode_raw = base_args.get("fusion_mode", "late")
        fusion_mode = _normalize_cnn_pair_fusion_mode(fusion_mode_raw)
        out["fusion_mode"] = fusion_mode
        has_arch_keys = any(
            key in out or key in base_args
            for key in (
                "conv_depth",
                "channel_candidates",
                "kernel_candidates",
                "donor_conv_depth",
                "acceptor_conv_depth",
                "donor_channel_candidates",
                "acceptor_channel_candidates",
                "donor_kernel_candidates",
                "acceptor_kernel_candidates",
            )
        )
        if has_arch_keys:
            donor_has_arch_keys = any(
                key in out or key in base_args
                for key in (
                    "donor_conv_depth",
                    "donor_channel_candidates",
                    "donor_kernel_candidates",
                )
            )
            acceptor_has_arch_keys = any(
                key in out or key in base_args
                for key in (
                    "acceptor_conv_depth",
                    "acceptor_channel_candidates",
                    "acceptor_kernel_candidates",
                )
            )
            if fusion_mode in {"early", "mid"}:
                if donor_has_arch_keys:
                    _materialize_branch(
                        depth_key="donor_conv_depth",
                        channel_pool_key="donor_channel_candidates",
                        kernel_pool_key="donor_kernel_candidates",
                        conv_out_key="donor_conv_channels",
                        kernel_out_key="donor_kernel_sizes",
                        shared_depth_key="conv_depth",
                        shared_channel_key="channel_candidates",
                        shared_kernel_key="kernel_candidates",
                    )
                elif acceptor_has_arch_keys:
                    _materialize_branch(
                        depth_key="acceptor_conv_depth",
                        channel_pool_key="acceptor_channel_candidates",
                        kernel_pool_key="acceptor_kernel_candidates",
                        conv_out_key="donor_conv_channels",
                        kernel_out_key="donor_kernel_sizes",
                        shared_depth_key="conv_depth",
                        shared_channel_key="channel_candidates",
                        shared_kernel_key="kernel_candidates",
                    )
                else:
                    _materialize_branch(
                        depth_key="conv_depth",
                        channel_pool_key="channel_candidates",
                        kernel_pool_key="kernel_candidates",
                        conv_out_key="donor_conv_channels",
                        kernel_out_key="donor_kernel_sizes",
                    )
                out["acceptor_conv_channels"] = out["donor_conv_channels"]
                out["acceptor_kernel_sizes"] = out["donor_kernel_sizes"]
            else:
                _materialize_branch(
                    depth_key="donor_conv_depth",
                    channel_pool_key="donor_channel_candidates",
                    kernel_pool_key="donor_kernel_candidates",
                    conv_out_key="donor_conv_channels",
                    kernel_out_key="donor_kernel_sizes",
                    shared_depth_key="conv_depth",
                    shared_channel_key="channel_candidates",
                    shared_kernel_key="kernel_candidates",
                )
                _materialize_branch(
                    depth_key="acceptor_conv_depth",
                    channel_pool_key="acceptor_channel_candidates",
                    kernel_pool_key="acceptor_kernel_candidates",
                    conv_out_key="acceptor_conv_channels",
                    kernel_out_key="acceptor_kernel_sizes",
                    shared_depth_key="conv_depth",
                    shared_channel_key="channel_candidates",
                    shared_kernel_key="kernel_candidates",
                )

    helper_keys = {
        "conv_depth",
        "channel_candidates",
        "kernel_candidates",
        "donor_conv_depth",
        "acceptor_conv_depth",
        "donor_channel_candidates",
        "acceptor_channel_candidates",
        "donor_kernel_candidates",
        "acceptor_kernel_candidates",
    }
    for key in helper_keys:
        out.pop(key, None)
    return out


def estimate_cnn_param_complexity(
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> Optional[int]:
    """Estimate CNN trainable-parameter complexity for one trial.

    The estimate follows the current ``BasicSpliceCNN`` architecture:
    convolution blocks (Conv1d + BatchNorm affine parameters) and two
    fully connected layers.
    """
    conv_channels_raw = sampled_params.get("conv_channels")
    if conv_channels_raw is None:
        conv_channels_raw = base_args.get("conv_channels")
    conv_channels = _parse_conv_channels(conv_channels_raw)
    if conv_channels is None:
        lightweight = _to_bool(base_args.get("lightweight"))
        conv_channels = [64, 128] if lightweight else [64, 128, 256]

    kernel_raw = sampled_params.get("kernel_sizes")
    if kernel_raw is None:
        kernel_raw = base_args.get("kernel_sizes")
    scalar_kernel_raw = sampled_params.get("kernel_size")
    if scalar_kernel_raw is None:
        scalar_kernel_raw = base_args.get("kernel_size", 7)
    kernel_sizes = _resolve_kernel_sizes_for_depth(
        kernel_raw=kernel_raw,
        fallback_scalar_raw=scalar_kernel_raw,
        depth=len(conv_channels),
    )
    if kernel_sizes is None:
        return None

    fc_hidden_raw = sampled_params.get("fc_hidden")
    if fc_hidden_raw is None:
        fc_hidden_raw = base_args.get("fc_hidden", 128)
    fc_hidden = _to_positive_int(fc_hidden_raw)
    if fc_hidden is None:
        return None

    total_params = _estimate_cnn_encoder_params_layerwise(
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
    )
    prev_channels = conv_channels[-1]
    total_params += (prev_channels * fc_hidden) + fc_hidden
    total_params += fc_hidden + 1
    return total_params


def estimate_cnn_resdil_param_complexity(
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> Optional[int]:
    """Estimate trainable parameters for ``cnn_resdil``.

    The estimate mirrors ``ResDilSpliceCNN``:
    stem conv + batch norm, one residual block per ``conv_channels`` entry
    (two Conv1d layers with batch norms and optional projection), then
    a two-layer fully connected head.
    """
    conv_channels_raw = sampled_params.get("conv_channels")
    if conv_channels_raw is None:
        conv_channels_raw = base_args.get("conv_channels")
    conv_channels = _parse_conv_channels(conv_channels_raw)
    if conv_channels is None:
        lightweight = _to_bool(base_args.get("lightweight"))
        conv_channels = [64, 128] if lightweight else [64, 128, 256]

    kernel_raw = sampled_params.get("kernel_sizes")
    if kernel_raw is None:
        kernel_raw = base_args.get("kernel_sizes")
    scalar_kernel_raw = sampled_params.get("kernel_size")
    if scalar_kernel_raw is None:
        scalar_kernel_raw = base_args.get("kernel_size", 7)
    kernel_sizes = _resolve_kernel_sizes_for_depth(
        kernel_raw=kernel_raw,
        fallback_scalar_raw=scalar_kernel_raw,
        depth=len(conv_channels),
    )
    if kernel_sizes is None:
        return None

    fc_hidden_raw = sampled_params.get("fc_hidden")
    if fc_hidden_raw is None:
        fc_hidden_raw = base_args.get("fc_hidden", 128)
    fc_hidden = _to_positive_int(fc_hidden_raw)
    if fc_hidden is None:
        return None

    stem_channels = conv_channels[0]
    total_params = 0

    # Stem Conv1d(4 -> stem) + stem BatchNorm affine params.
    stem_kernel_size = kernel_sizes[0]
    total_params += (4 * stem_channels * stem_kernel_size) + stem_channels
    total_params += 2 * stem_channels

    prev_channels = stem_channels
    for channel, kernel_size in zip(conv_channels, kernel_sizes):
        # Residual block conv1 + bn1.
        total_params += (prev_channels * channel * kernel_size) + channel
        total_params += 2 * channel
        # Residual block conv2 + bn2.
        total_params += (channel * channel * kernel_size) + channel
        total_params += 2 * channel
        # Projection path when channel width changes.
        if prev_channels != channel:
            total_params += (prev_channels * channel) + channel
            total_params += 2 * channel
        prev_channels = channel

    # Fully connected head.
    total_params += (prev_channels * fc_hidden) + fc_hidden
    total_params += fc_hidden + 1
    return total_params


def estimate_tcn_param_complexity(
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> Optional[int]:
    """Estimate trainable parameters for ``tcn``.

    The estimate mirrors ``TCNSpliceCNN``:
    stem conv + batch norm, repeated dilated residual blocks across
    ``conv_channels`` (with optional projection per block), then a
    two-layer fully connected head.
    """
    conv_channels_raw = sampled_params.get("conv_channels")
    if conv_channels_raw is None:
        conv_channels_raw = base_args.get("conv_channels")
    conv_channels = _parse_conv_channels(conv_channels_raw)
    if conv_channels is None:
        lightweight = _to_bool(base_args.get("lightweight"))
        conv_channels = [64, 128] if lightweight else [64, 128, 256]

    kernel_raw = sampled_params.get("kernel_size")
    if kernel_raw is None:
        kernel_raw = base_args.get("kernel_size", 7)
    kernel_size = _to_positive_int(kernel_raw)
    if kernel_size is None:
        return None

    repeats_raw = sampled_params.get("tcn_block_repeats")
    if repeats_raw is None:
        repeats_raw = base_args.get("tcn_block_repeats", 2)
    block_repeats = _to_positive_int(repeats_raw)
    if block_repeats is None:
        return None

    fc_hidden_raw = sampled_params.get("fc_hidden")
    if fc_hidden_raw is None:
        fc_hidden_raw = base_args.get("fc_hidden", 128)
    fc_hidden = _to_positive_int(fc_hidden_raw)
    if fc_hidden is None:
        return None

    stem_channels = conv_channels[0]
    total_params = 0

    # Stem Conv1d(4 -> stem) + stem BatchNorm affine params.
    total_params += (4 * stem_channels * kernel_size) + stem_channels
    total_params += 2 * stem_channels

    prev_channels = stem_channels
    for _ in range(block_repeats):
        for channel in conv_channels:
            # Residual block conv1 + bn1.
            total_params += (prev_channels * channel * kernel_size) + channel
            total_params += 2 * channel
            # Residual block conv2 + bn2.
            total_params += (channel * channel * kernel_size) + channel
            total_params += 2 * channel
            # Projection path when channel width changes.
            if prev_channels != channel:
                total_params += (prev_channels * channel) + channel
                total_params += 2 * channel
            prev_channels = channel

    # Fully connected head.
    total_params += (prev_channels * fc_hidden) + fc_hidden
    total_params += fc_hidden + 1
    return total_params


def _estimate_cnn_encoder_params(
    *,
    conv_channels: list[int],
    kernel_size: int,
) -> int:
    """Estimate one CNN encoder parameter count (conv+batch norm blocks)."""
    total_params = 0
    prev_channels = 4
    for channel in conv_channels:
        conv_params = (prev_channels * channel * kernel_size) + channel
        batch_norm_params = 2 * channel
        total_params += conv_params + batch_norm_params
        prev_channels = channel
    return total_params


def _estimate_cnn_encoder_params_layerwise(
    *,
    conv_channels: list[int],
    kernel_sizes: list[int],
    in_channels: int = 4,
) -> int:
    """Estimate one CNN encoder parameter count with layer-wise kernels."""
    if len(conv_channels) != len(kernel_sizes):
        raise ValueError("conv_channels and kernel_sizes lengths must match.")
    if in_channels <= 0:
        raise ValueError("in_channels must be positive.")
    total_params = 0
    prev_channels = in_channels
    for channel, kernel_size in zip(conv_channels, kernel_sizes):
        conv_params = (prev_channels * channel * kernel_size) + channel
        batch_norm_params = 2 * channel
        total_params += conv_params + batch_norm_params
        prev_channels = channel
    return total_params


def _resolve_branch_kernel_sizes(
    *,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
    branch_key: str,
    depth: int,
    shared_kernel_raw: object,
    scalar_kernel_raw: object,
) -> Optional[list[int]]:
    """Resolve one branch kernel-size list from branch/shared/scalar settings."""
    branch_raw: object = sampled_params.get(branch_key)
    if branch_raw is None:
        branch_raw = base_args.get(branch_key)
    if branch_raw is None:
        branch_raw = shared_kernel_raw
    if branch_raw is None:
        branch_raw = scalar_kernel_raw

    parsed_list = _parse_conv_channels(branch_raw)
    if parsed_list is None:
        scalar_kernel = _to_positive_int(branch_raw)
        if scalar_kernel is None:
            return None
        parsed_list = [scalar_kernel]

    if len(parsed_list) == 1:
        return parsed_list * depth
    if len(parsed_list) < depth:
        return parsed_list + ([parsed_list[-1]] * (depth - len(parsed_list)))
    if len(parsed_list) > depth:
        return parsed_list[:depth]
    return parsed_list


def estimate_cnn_pair_param_complexity(
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> Optional[int]:
    """Estimate trainable parameters for ``cnn_pair``.

    Pair architecture supports:
    - late fusion: independent donor/acceptor encoders then MLP head
    - mid fusion: branch-specific prefix then shared fused tail
    - early fusion: one shared encoder over concatenated channels
    """
    fusion_mode_raw: object = sampled_params.get("fusion_mode")
    if fusion_mode_raw is None:
        fusion_mode_raw = base_args.get("fusion_mode", "late")
    fusion_mode = _normalize_cnn_pair_fusion_mode(fusion_mode_raw)
    if fusion_mode not in {"late", "mid", "early"}:
        return None

    shared_conv_channels_raw = sampled_params.get("conv_channels")
    if shared_conv_channels_raw is None:
        shared_conv_channels_raw = base_args.get("conv_channels")
    shared_conv_channels = _parse_conv_channels(shared_conv_channels_raw)

    donor_conv_channels_raw = sampled_params.get("donor_conv_channels")
    if donor_conv_channels_raw is None:
        donor_conv_channels_raw = base_args.get("donor_conv_channels")
    donor_conv_channels = _parse_conv_channels(donor_conv_channels_raw)
    if donor_conv_channels is None:
        donor_conv_channels = shared_conv_channels

    acceptor_conv_channels_raw = sampled_params.get("acceptor_conv_channels")
    if acceptor_conv_channels_raw is None:
        acceptor_conv_channels_raw = base_args.get("acceptor_conv_channels")
    acceptor_conv_channels = _parse_conv_channels(acceptor_conv_channels_raw)
    if acceptor_conv_channels is None:
        acceptor_conv_channels = shared_conv_channels

    if donor_conv_channels is None:
        lightweight = _to_bool(base_args.get("lightweight"))
        donor_conv_channels = [64, 128] if lightweight else [64, 128, 256]
    if acceptor_conv_channels is None:
        lightweight = _to_bool(base_args.get("lightweight"))
        acceptor_conv_channels = [64, 128] if lightweight else [64, 128, 256]

    shared_kernel_raw = sampled_params.get("kernel_sizes")
    if shared_kernel_raw is None:
        shared_kernel_raw = base_args.get("kernel_sizes")
    scalar_kernel_raw = sampled_params.get("kernel_size")
    if scalar_kernel_raw is None:
        scalar_kernel_raw = base_args.get("kernel_size", 7)

    donor_kernel_sizes = _resolve_branch_kernel_sizes(
        sampled_params=sampled_params,
        base_args=base_args,
        branch_key="donor_kernel_sizes",
        depth=len(donor_conv_channels),
        shared_kernel_raw=shared_kernel_raw,
        scalar_kernel_raw=scalar_kernel_raw,
    )
    if donor_kernel_sizes is None:
        return None
    acceptor_kernel_sizes = _resolve_branch_kernel_sizes(
        sampled_params=sampled_params,
        base_args=base_args,
        branch_key="acceptor_kernel_sizes",
        depth=len(acceptor_conv_channels),
        shared_kernel_raw=shared_kernel_raw,
        scalar_kernel_raw=scalar_kernel_raw,
    )
    if acceptor_kernel_sizes is None:
        return None

    fc_hidden_raw = sampled_params.get("fc_hidden")
    if fc_hidden_raw is None:
        fc_hidden_raw = base_args.get("fc_hidden", 128)
    fc_hidden = _to_positive_int(fc_hidden_raw)
    if fc_hidden is None:
        return None

    if fusion_mode == "early":
        if donor_conv_channels != acceptor_conv_channels:
            return None
        if donor_kernel_sizes != acceptor_kernel_sizes:
            return None
        encoder_params = _estimate_cnn_encoder_params_layerwise(
            conv_channels=donor_conv_channels,
            kernel_sizes=donor_kernel_sizes,
            in_channels=8,
        )
        pair_dim = donor_conv_channels[-1]
        head_params = (pair_dim * fc_hidden) + fc_hidden + fc_hidden + 1
        return encoder_params + head_params

    if fusion_mode == "mid":
        if donor_conv_channels != acceptor_conv_channels:
            return None
        if donor_kernel_sizes != acceptor_kernel_sizes:
            return None
        split_index = max(1, len(donor_conv_channels) // 2)
        prefix_channels = donor_conv_channels[:split_index]
        prefix_kernels = donor_kernel_sizes[:split_index]
        suffix_channels = donor_conv_channels[split_index:]
        suffix_kernels = donor_kernel_sizes[split_index:]

        donor_prefix_params = _estimate_cnn_encoder_params_layerwise(
            conv_channels=prefix_channels,
            kernel_sizes=prefix_kernels,
            in_channels=4,
        )
        acceptor_prefix_params = _estimate_cnn_encoder_params_layerwise(
            conv_channels=prefix_channels,
            kernel_sizes=prefix_kernels,
            in_channels=4,
        )
        tail_params = 0
        pair_dim = 2 * prefix_channels[-1]
        if suffix_channels:
            tail_params = _estimate_cnn_encoder_params_layerwise(
                conv_channels=suffix_channels,
                kernel_sizes=suffix_kernels,
                in_channels=2 * prefix_channels[-1],
            )
            pair_dim = suffix_channels[-1]
        head_params = (pair_dim * fc_hidden) + fc_hidden + fc_hidden + 1
        return donor_prefix_params + acceptor_prefix_params + tail_params + head_params

    donor_params = _estimate_cnn_encoder_params_layerwise(
        conv_channels=donor_conv_channels,
        kernel_sizes=donor_kernel_sizes,
    )
    acceptor_params = _estimate_cnn_encoder_params_layerwise(
        conv_channels=acceptor_conv_channels,
        kernel_sizes=acceptor_kernel_sizes,
    )
    pair_dim = donor_conv_channels[-1] + acceptor_conv_channels[-1]
    head_params = (pair_dim * fc_hidden) + fc_hidden + fc_hidden + 1
    return donor_params + acceptor_params + head_params


def estimate_model_param_complexity(
    *,
    model_name: str,
    sampled_params: dict[str, Scalar],
    base_args: dict[str, ArgValue],
) -> Optional[int]:
    """Estimate model complexity for supported model families."""
    if model_name == "cnn":
        return estimate_cnn_param_complexity(
            sampled_params=sampled_params,
            base_args=base_args,
        )
    if model_name == "cnn_resdil":
        return estimate_cnn_resdil_param_complexity(
            sampled_params=sampled_params,
            base_args=base_args,
        )
    if model_name == "tcn":
        return estimate_tcn_param_complexity(
            sampled_params=sampled_params,
            base_args=base_args,
        )
    if model_name == "cnn_pair":
        return estimate_cnn_pair_param_complexity(
            sampled_params=sampled_params,
            base_args=base_args,
        )
    return None


def _rolling_mean_curve(
    points: list[tuple[int, float]],
    window_size: int,
) -> list[tuple[float, float]]:
    """Compute a centered rolling-mean curve on sorted points."""
    if not points:
        return []
    if window_size <= 1:
        return [(float(x), y) for x, y in points]
    half_window = window_size // 2
    curve: list[tuple[float, float]] = []
    for idx, (x_val, _) in enumerate(points):
        start = max(0, idx - half_window)
        end = min(len(points), idx + half_window + 1)
        segment = points[start:end]
        mean_y = sum(y for _, y in segment) / len(segment)
        curve.append((float(x_val), mean_y))
    return curve


def write_trials_tsv(path: Path, rows: list[TrialResult]) -> None:
    """Write trial table as TSV."""
    all_param_names: list[str] = sorted(
        {key for row in rows for key in row.sampled_params}
    )
    headers = [
        "phase",
        "trial_id",
        "status",
        "gpu_id",
        "effective_batch_size",
        "oom_retries",
        "donor_pr_auc",
        "acceptor_pr_auc",
        "mean_pr_auc",
        "objective_metric",
        "objective_score",
        "return_code",
        "duration_sec",
        "metrics_json",
        "log_file",
        "error_message",
    ] + all_param_names
    lines = ["\t".join(headers)]
    for row in rows:
        fixed_values = [
            row.phase,
            str(row.trial_id),
            row.status,
            "" if row.gpu_id is None else row.gpu_id,
            str(row.effective_batch_size),
            str(row.oom_retries),
            _format_float(row.donor_pr_auc),
            _format_float(row.acceptor_pr_auc),
            _format_float(row.mean_pr_auc),
            row.objective_metric,
            _format_float(row.objective_score),
            str(row.return_code),
            f"{row.duration_sec:.4f}",
            row.metrics_json,
            row.log_file,
            "" if row.error_message is None else row.error_message,
        ]
        param_values = [
            str(row.sampled_params.get(name, "")) for name in all_param_names
        ]
        lines.append("\t".join(fixed_values + param_values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(
    path: Path,
    *,
    config: SearchConfig,
    gpu_ids: list[str],
    quick_rows: list[TrialResult],
    full_rows: list[TrialResult],
    best_row: Optional[TrialResult],
    previous_global_best_score: Optional[float],
) -> None:
    """Write human-readable run summary."""
    quick_success = len([row for row in quick_rows if row.status == "success"])
    full_success = len([row for row in full_rows if row.status == "success"])
    lines = [
        "# Hyperparameter Search Summary",
        "",
        f"- Species: `{config.species}`",
        f"- Objective: `{config.objective_metric}`",
        f"- GPU slots: `{', '.join(gpu_ids) if gpu_ids else 'cpu-fallback'}`",
        (f"- Quick phase: {config.quick_trials} trials, {quick_success} successful"),
        f"- Full phase: {len(full_rows)} trials, {full_success} successful",
        "",
    ]
    if previous_global_best_score is not None:
        lines.extend(
            [
                "## Reference Best",
                "",
                (
                    "- Previous global best "
                    f"({config.objective_metric}): "
                    f"`{previous_global_best_score:.6f}`"
                ),
                "",
            ]
        )
    if best_row is None:
        lines.extend(
            [
                "## Best Trial",
                "",
                "No successful trial was produced.",
            ]
        )
    else:
        metric_lines: list[str] = []
        if best_row.objective_metric.startswith("pair_"):
            metric_lines.append(
                f"- Pair objective score: `{best_row.objective_score:.6f}`"
            )
        else:
            metric_lines.extend(
                [
                    (
                        "- Mean PR-AUC: "
                        f"`{_format_float(best_row.mean_pr_auc) or 'n/a'}`"
                    ),
                    (
                        "- Donor PR-AUC: "
                        f"`{_format_float(best_row.donor_pr_auc) or 'n/a'}`"
                    ),
                    (
                        "- Acceptor PR-AUC: "
                        f"`{_format_float(best_row.acceptor_pr_auc) or 'n/a'}`"
                    ),
                ]
            )
        lines.extend(
            [
                "## Best Trial",
                "",
                f"- Phase: `{best_row.phase}`",
                f"- Trial ID: `{best_row.trial_id}`",
                (
                    f"- Objective ({best_row.objective_metric}): "
                    f"`{best_row.objective_score:.6f}`"
                ),
                f"- Effective batch size: `{best_row.effective_batch_size}`",
                "",
            ]
        )
        lines.extend(metric_lines)
        lines.extend(
            [
                "",
                "### Parameters",
                "",
            ]
        )
        for key in sorted(best_row.sampled_params):
            lines.append(f"- `{key}`: `{best_row.sampled_params[key]}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_trial_start(
    *,
    phase: str,
    trial_id: int,
    assigned_gpu: Optional[str],
) -> None:
    """Print one standardized trial start line."""
    slot_label = "cpu" if assigned_gpu is None else f"gpu:{assigned_gpu}"
    print(
        f"[hparam_search] {phase} trial {trial_id:04d} started on {slot_label}.",
        flush=True,
    )


def _print_trial_result(
    *,
    phase: str,
    trial_count: int,
    completed_count: int,
    result: TrialResult,
) -> None:
    """Print one standardized trial completion line."""
    metric_text = (
        "-" if result.objective_score is None else f"{result.objective_score:.6f}"
    )
    print(
        f"[hparam_search] {phase} trial {result.trial_id:04d} "
        f"{result.status} ({completed_count}/{trial_count}) "
        f"{result.objective_metric}={metric_text}.",
        flush=True,
    )


def _run_phase_subprocess(
    *,
    phase: str,
    config: SearchConfig,
    trial_count: int,
    trial_params: list[dict[str, Scalar]],
    overrides: dict[str, ArgValue],
    slots: list[Optional[str]],
    out_dir: Path,
) -> list[TrialResult]:
    """Run one phase using subprocess-backed trials."""
    pending_indices = list(range(trial_count))
    running: dict[Future[TrialResult], Optional[str]] = {}
    collected: list[TrialResult] = []
    executor = ThreadPoolExecutor(max_workers=max(1, len(slots)))
    try:
        while pending_indices or running:
            while pending_indices and slots:
                trial_id = pending_indices.pop(0)
                assigned_gpu = slots.pop(0)
                metrics_json = out_dir / f"{phase}_trial_{trial_id:04d}.metrics.json"
                log_file = out_dir / f"{phase}_trial_{trial_id:04d}.log.txt"
                future = executor.submit(
                    run_trial,
                    config=config,
                    phase=phase,
                    trial_id=trial_id,
                    sampled_params=trial_params[trial_id],
                    overrides=overrides,
                    assigned_gpu_id=assigned_gpu,
                    metrics_json=metrics_json,
                    log_file=log_file,
                )
                running[future] = assigned_gpu
                _print_trial_start(
                    phase=phase,
                    trial_id=trial_id,
                    assigned_gpu=assigned_gpu,
                )

            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                assigned_gpu = running.pop(future)
                slots.append(assigned_gpu)
                result = future.result()
                collected.append(result)
                _print_trial_result(
                    phase=phase,
                    trial_count=trial_count,
                    completed_count=len(collected),
                    result=result,
                )
    except KeyboardInterrupt:
        print(
            "[hparam_search] Interrupt received. Terminating active trials...",
            flush=True,
        )
        for future in running:
            future.cancel()
        _interrupt_active_trial_processes()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return collected


def _build_worker_failure_result(
    *,
    config: SearchConfig,
    phase: str,
    task: PersistentTrialTask,
    assigned_gpu_id: Optional[str],
    error_message: str,
) -> TrialResult:
    """Build one failed result when a persistent worker hits an internal error."""
    return TrialResult(
        phase=phase,
        trial_id=task.trial_id,
        status="failed",
        gpu_id=assigned_gpu_id,
        sampled_params=task.sampled_params,
        effective_batch_size=0,
        oom_retries=0,
        donor_pr_auc=None,
        acceptor_pr_auc=None,
        mean_pr_auc=None,
        objective_metric=config.objective_metric,
        objective_score=None,
        error_message=error_message,
        return_code=1,
        duration_sec=0.0,
        metrics_json=task.metrics_json,
        log_file=task.log_file,
    )


def _prewarm_persistent_trial_worker(
    *,
    config: SearchConfig,
    assigned_gpu_id: Optional[str],
) -> None:
    """Prewarm one persistent worker if the selected model exposes a hook."""
    from models.registry import load_model_module

    model_name_obj = config.base_args.get("model")
    if not isinstance(model_name_obj, str) or not model_name_obj.strip():
        return
    model_module = load_model_module(model_name_obj)
    prewarm_fn = getattr(model_module, "prewarm_persistent_worker", None)
    if not callable(prewarm_fn):
        return
    prewarm_fn(dict(config.base_args), assigned_gpu_id)


def _persistent_trial_worker_main(
    *,
    slot_index: int,
    assigned_gpu_id: Optional[str],
    config: SearchConfig,
    phase: str,
    overrides: dict[str, ArgValue],
    stream_mode: str,
    max_parallel_trials: int,
    task_queue: object,
    result_queue: object,
) -> None:
    """Persistent worker loop that executes multiple trials in one process."""
    previous_stream_mode = _set_active_trial_stream_mode(stream_mode)
    previous_parallel = _set_active_max_parallel_trials(max_parallel_trials)
    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assigned_gpu_id is not None:
        # CUDA_VISIBLE_DEVICES must be fixed before prewarm/training so CUDA
        # runtime initialization cannot bind this worker to an unintended GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu_id)
    try:
        _prewarm_persistent_trial_worker(
            config=config,
            assigned_gpu_id=assigned_gpu_id,
        )
        while True:
            task = task_queue.get()
            if task is None:
                return
            if not isinstance(task, PersistentTrialTask):
                continue
            try:
                result = run_trial_inprocess(
                    config=config,
                    phase=phase,
                    trial_id=task.trial_id,
                    sampled_params=task.sampled_params,
                    overrides=overrides,
                    assigned_gpu_id=assigned_gpu_id,
                    metrics_json=Path(task.metrics_json),
                    log_file=Path(task.log_file),
                )
            except Exception as exc:
                result = _build_worker_failure_result(
                    config=config,
                    phase=phase,
                    task=task,
                    assigned_gpu_id=assigned_gpu_id,
                    error_message=(
                        "Persistent trial worker failed with "
                        f"{exc.__class__.__name__}: {exc}"
                    ),
                )
            result_queue.put(
                PersistentTrialOutcome(
                    slot_index=slot_index,
                    result=result,
                )
            )
    finally:
        if assigned_gpu_id is not None:
            if previous_cuda_visible is None:
                _ = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible
        _set_active_max_parallel_trials(previous_parallel)
        _set_active_trial_stream_mode(previous_stream_mode)


def _run_phase_persistent(
    *,
    phase: str,
    config: SearchConfig,
    trial_count: int,
    trial_params: list[dict[str, Scalar]],
    overrides: dict[str, ArgValue],
    slots: list[Optional[str]],
    out_dir: Path,
    max_parallel_trials: int,
    stream_mode: str,
) -> list[TrialResult]:
    """Run one phase using persistent in-process trial workers."""
    context = mp.get_context("spawn")
    task_queues = [context.Queue() for _ in slots]
    result_queue = context.Queue()
    workers: list[mp.Process] = []
    slot_busy = [False for _ in slots]
    slot_active_trial: list[Optional[int]] = [None for _ in slots]
    collected: list[TrialResult] = []
    pending_indices = list(range(trial_count))
    for slot_index, assigned_gpu in enumerate(slots):
        worker = context.Process(
            target=_persistent_trial_worker_main,
            kwargs={
                "slot_index": slot_index,
                "assigned_gpu_id": assigned_gpu,
                "config": config,
                "phase": phase,
                "overrides": dict(overrides),
                "stream_mode": stream_mode,
                "max_parallel_trials": max_parallel_trials,
                "task_queue": task_queues[slot_index],
                "result_queue": result_queue,
            },
        )
        worker.start()
        workers.append(worker)

    try:
        while pending_indices or any(slot_busy):
            for slot_index, assigned_gpu in enumerate(slots):
                if not pending_indices or slot_busy[slot_index]:
                    continue
                trial_id = pending_indices.pop(0)
                metrics_json = out_dir / f"{phase}_trial_{trial_id:04d}.metrics.json"
                log_file = out_dir / f"{phase}_trial_{trial_id:04d}.log.txt"
                task = PersistentTrialTask(
                    trial_id=trial_id,
                    sampled_params=trial_params[trial_id],
                    metrics_json=str(metrics_json),
                    log_file=str(log_file),
                )
                task_queues[slot_index].put(task)
                slot_busy[slot_index] = True
                slot_active_trial[slot_index] = trial_id
                _print_trial_start(
                    phase=phase,
                    trial_id=trial_id,
                    assigned_gpu=assigned_gpu,
                )

            if not any(slot_busy):
                continue

            try:
                outcome_raw = result_queue.get(timeout=1.0)
            except Empty:
                for slot_index, worker in enumerate(workers):
                    if slot_busy[slot_index] and not worker.is_alive():
                        trial_id = slot_active_trial[slot_index]
                        raise RuntimeError(
                            "Persistent trial worker exited unexpectedly "
                            f"(slot={slot_index}, trial={trial_id})."
                        )
                continue

            if not isinstance(outcome_raw, PersistentTrialOutcome):
                continue
            outcome = outcome_raw
            slot_index = outcome.slot_index
            slot_busy[slot_index] = False
            slot_active_trial[slot_index] = None
            collected.append(outcome.result)
            _print_trial_result(
                phase=phase,
                trial_count=trial_count,
                completed_count=len(collected),
                result=outcome.result,
            )
    finally:
        for queue_obj in task_queues:
            queue_obj.put(None)
        for worker in workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5.0)
    return collected


def run_phase(
    *,
    phase: str,
    config: SearchConfig,
    trial_count: int,
    trial_params: list[dict[str, Scalar]],
    overrides: dict[str, ArgValue],
    gpu_ids: list[str],
    max_parallel_trials: int,
    out_dir: Path,
    execution_mode: str = "subprocess",
) -> list[TrialResult]:
    """Run one phase with slot-based scheduling and selectable execution mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_stream_mode = _resolve_trial_stream_mode(
        config.trial_stream_mode,
        max_parallel_trials,
    )
    previous_stream_mode = _set_active_trial_stream_mode(resolved_stream_mode)
    previous_max_parallel_trials = _set_active_max_parallel_trials(max_parallel_trials)
    if gpu_ids:
        slots: list[Optional[str]] = list(gpu_ids[:max_parallel_trials])
    else:
        slots = [None for _ in range(max_parallel_trials)]
    try:
        if execution_mode == _DEFAULT_PHASE_EXECUTION_MODE:
            return _run_phase_subprocess(
                phase=phase,
                config=config,
                trial_count=trial_count,
                trial_params=trial_params,
                overrides=overrides,
                slots=slots,
                out_dir=out_dir,
            )
        if execution_mode == _PERSISTENT_PHASE_EXECUTION_MODE:
            return _run_phase_persistent(
                phase=phase,
                config=config,
                trial_count=trial_count,
                trial_params=trial_params,
                overrides=overrides,
                slots=slots,
                out_dir=out_dir,
                max_parallel_trials=max_parallel_trials,
                stream_mode=resolved_stream_mode,
            )
        raise ValueError(
            "execution_mode must be one of: "
            f"{_DEFAULT_PHASE_EXECUTION_MODE}, "
            f"{_PERSISTENT_PHASE_EXECUTION_MODE}."
        )
    finally:
        _set_active_max_parallel_trials(previous_max_parallel_trials)
        _set_active_trial_stream_mode(previous_stream_mode)


def build_trial_params(
    *,
    config: SearchConfig,
    phase: str,
    count: int,
    seed_offset: int,
    seed_source: Optional[list[TrialResult]] = None,
    history_trials: Optional[list[tuple[float, dict[str, Scalar]]]] = None,
) -> list[dict[str, Scalar]]:
    """Build sampled parameter sets for a phase."""
    if seed_source is not None:
        return [dict(row.sampled_params) for row in seed_source[:count]]
    sampled: list[dict[str, Scalar]] = []
    max_resample_attempts = 64
    model_name = str(config.base_args.get("model", ""))
    for trial_id in range(count):
        trial_seed = config.base_seed + seed_offset + trial_id
        rng = random.Random(trial_seed)
        params: dict[str, Scalar]
        best_under_cap_params: Optional[dict[str, Scalar]] = None
        best_under_cap_complexity: Optional[int] = None
        last_invalid_reason = "Failed to sample a valid architecture."
        for _attempt in range(max_resample_attempts):
            if (
                phase == "quick"
                and config.search_algo == "history_guided"
                and history_trials
            ):
                params = sample_trial_params_history_guided(
                    search_space=config.search_space,
                    seed=rng.randrange(1 << 30),
                    history_trials=history_trials,
                    random_fraction=config.guided_random_fraction,
                    mutation_rate=config.guided_mutation_rate,
                )
            else:
                params = _sample_trial_params_with_rng(config.search_space, rng)
            params = _materialize_cnn_architecture_params(
                model_name=model_name,
                sampled_params=params,
                base_args=config.base_args,
                rng=rng,
            )
            params = _materialize_dnabert_readout_params(
                model_name=model_name,
                sampled_params=params,
                base_args=config.base_args,
            )
            if not _is_valid_cnn_architecture(
                model_name=model_name,
                sampled_params=params,
                base_args=config.base_args,
            ):
                last_invalid_reason = (
                    "Failed to sample a valid architecture after "
                    f"{max_resample_attempts} attempts."
                )
                continue
            if config.max_model_params is None:
                break
            complexity = estimate_model_param_complexity(
                model_name=model_name,
                sampled_params=params,
                base_args=config.base_args,
            )
            if complexity is None or complexity <= config.max_model_params:
                break
            if (
                best_under_cap_complexity is None
                or complexity < best_under_cap_complexity
            ):
                best_under_cap_params = dict(params)
                best_under_cap_complexity = complexity
        else:
            if best_under_cap_params is not None:
                params = best_under_cap_params
            else:
                raise ValueError(last_invalid_reason)
        sampled.append(params)
    return sampled


def write_best_config(
    path: Path,
    row: Optional[TrialResult],
    *,
    top_rows: Optional[list[TrialResult]] = None,
    top_k: Optional[int] = None,
    fallback_validation_protocol: Optional[dict[str, object]] = None,
    hparam_context: Optional[dict[str, object]] = None,
) -> None:
    """Write best-trial config JSON."""
    if row is None:
        payload: dict[str, object] = {"status": "no_successful_trial"}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    validation_protocol = (
        row.validation_protocol
        if row.validation_protocol is not None
        else fallback_validation_protocol
    )
    objective_best_epoch = _read_objective_best_epoch_from_metrics(
        metrics_json_path=row.metrics_json,
        objective_metric=row.objective_metric,
    )
    payload = {
        "status": "ok",
        "phase": row.phase,
        "trial_id": row.trial_id,
        "gpu_id": row.gpu_id,
        "donor_pr_auc": row.donor_pr_auc,
        "acceptor_pr_auc": row.acceptor_pr_auc,
        "mean_pr_auc": row.mean_pr_auc,
        "objective_metric": row.objective_metric,
        "objective_score": row.objective_score,
        "effective_batch_size": row.effective_batch_size,
        "oom_retries": row.oom_retries,
        "sampled_params": row.sampled_params,
        "selection_score": row.selection_score,
        "objective_best_epoch": objective_best_epoch,
        "validation_protocol": validation_protocol,
        "hparam_context": hparam_context,
        "metrics_json": row.metrics_json,
    }
    if top_rows is not None and top_k is not None and top_k > 0:
        payload["top_trials"] = _serialize_top_trials(top_rows, top_k)
    payload.update(_extract_checkpoint_paths_from_metrics(row.metrics_json))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def maybe_update_global_best(
    *,
    global_best_path: Optional[Path],
    best_row: Optional[TrialResult],
    fallback_validation_protocol: Optional[dict[str, object]] = None,
    hparam_context: Optional[dict[str, object]] = None,
) -> None:
    """Update global best config if current run improves the best score."""
    if global_best_path is None or best_row is None or best_row.objective_score is None:
        return
    previous_score = _read_best_objective_score(
        global_best_path,
        best_row.objective_metric,
        expected_hparam_context=hparam_context,
    )
    if previous_score is not None and previous_score > best_row.objective_score:
        print(
            "[hparam_search] Keep previous global best "
            f"{previous_score:.6f} > {best_row.objective_score:.6f}: "
            f"{global_best_path}",
            flush=True,
        )
        return
    global_best_path.parent.mkdir(parents=True, exist_ok=True)
    write_best_config(
        global_best_path,
        best_row,
        fallback_validation_protocol=fallback_validation_protocol,
        hparam_context=hparam_context,
    )
    print(
        "[hparam_search] Updated global best config: "
        f"{global_best_path} "
        f"({best_row.objective_metric}={best_row.objective_score:.6f})",
        flush=True,
    )


def write_visualization(
    path: Path,
    *,
    model_name: str,
    species: str,
    objective_metric: str,
    quick_rows: list[TrialResult],
    full_rows: list[TrialResult],
    base_args: dict[str, ArgValue],
) -> Optional[str]:
    """Write a tuning visualization image if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib is not available; skipped plot generation."

    def _extract_points(rows: list[TrialResult]) -> list[tuple[int, float]]:
        points: list[tuple[int, float]] = []
        for row in rows:
            if row.status != "success" or row.objective_score is None:
                continue
            complexity = estimate_model_param_complexity(
                model_name=model_name,
                sampled_params=row.sampled_params,
                base_args=base_args,
            )
            if complexity is None:
                continue
            points.append((complexity, float(row.objective_score)))
        return points

    quick_points = _extract_points(quick_rows)
    full_points = _extract_points(full_rows)
    all_points = quick_points + full_points
    if not all_points:
        return "No successful trials with valid complexity estimate."

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=140)

    if quick_points:
        ax.scatter(
            [point[0] for point in quick_points],
            [point[1] for point in quick_points],
            s=28,
            alpha=0.55,
            label="quick",
            marker="o",
        )
    if full_points:
        ax.scatter(
            [point[0] for point in full_points],
            [point[1] for point in full_points],
            s=52,
            alpha=0.9,
            label="full",
            marker="^",
        )

    best_complexity, best_score = max(all_points, key=lambda item: item[1])
    ax.scatter(
        [best_complexity],
        [best_score],
        s=140,
        marker="*",
        color="gold",
        edgecolors="black",
        linewidths=0.8,
        label="best",
        zorder=5,
    )

    sorted_points = sorted(all_points, key=lambda item: item[0])
    if len(sorted_points) >= 5:
        window_size = max(5, len(sorted_points) // 8)
        if window_size % 2 == 0:
            window_size += 1
        trend_curve = _rolling_mean_curve(sorted_points, window_size)
        ax.plot(
            [point[0] for point in trend_curve],
            [point[1] for point in trend_curve],
            linewidth=1.8,
            color="black",
            label=f"trend (rolling mean, w={window_size})",
        )

    ax.set_xscale("log")
    ax.set_title(f"{species} Tuning: {objective_metric} vs model complexity")
    ax.set_xlabel("Estimated model complexity (trainable parameters, log scale)")
    ax.set_ylabel(objective_metric)
    ax.grid(alpha=0.3, linestyle="--")
    if ax.get_legend_handles_labels()[1]:
        ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return None


def run_search(config: SearchConfig) -> int:
    """Execute the full two-phase search run."""
    run_started_epoch = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_validation_protocol = _derive_validation_protocol_from_args(
        merged_args=dict(config.base_args),
        objective_metric=config.objective_metric,
    )
    full_overrides = dict(config.full_overrides)
    full_overrides.setdefault("epochs", config.full_epochs)
    full_overrides.setdefault("compile_mode", "auto")
    full_epochs_value = _to_positive_int(full_overrides.get("epochs"))
    if full_epochs_value is None:
        full_epochs_value = config.full_epochs
    fixed_run_args = _build_fixed_run_args_context(
        base_args=dict(config.base_args),
        full_overrides=full_overrides,
        search_space=config.search_space,
    )
    current_hparam_context = _build_hparam_context(
        objective_metric=config.objective_metric,
        full_epochs=full_epochs_value,
        validation_protocol=baseline_validation_protocol,
        fixed_run_args=fixed_run_args,
    )
    gpu_ids = detect_gpu_ids(config.gpu_ids_setting)
    max_parallel_trials = resolve_max_parallel(
        setting=config.max_parallel_trials_setting,
        gpu_count=len(gpu_ids),
    )
    resolved_trial_stream_mode = _resolve_trial_stream_mode(
        config.trial_stream_mode,
        max_parallel_trials,
    )
    quick_mode_policy = _resolve_phase_execution_mode(
        process_mode=config.trial_process_mode,
        phase="quick",
    )
    full_mode_policy = _resolve_phase_execution_mode(
        process_mode=config.trial_process_mode,
        phase="full",
    )
    quick_execution_mode = _resolve_workload_execution_mode(
        phase_execution_mode=quick_mode_policy,
        trial_count=config.quick_trials,
        max_parallel_trials=max_parallel_trials,
    )
    if gpu_ids:
        gpu_summary = ",".join(gpu_ids[:max_parallel_trials])
        print(
            f"[hparam_search] Using GPU slots: {gpu_summary} "
            f"(parallel={max_parallel_trials}).",
            flush=True,
        )
    else:
        print(
            f"[hparam_search] No GPU detected; using CPU "
            f"(parallel={max_parallel_trials}).",
            flush=True,
        )
    print(
        "[hparam_search] Trial stdout stream mode: "
        f"{resolved_trial_stream_mode} (logs are still saved per trial).",
        flush=True,
    )
    print(
        "[hparam_search] Trial process mode: "
        f"{config.trial_process_mode} (quick_policy={quick_mode_policy}, "
        f"full_policy={full_mode_policy}, quick={quick_execution_mode}).",
        flush=True,
    )
    uses_auto_num_workers = any(
        _is_auto_num_workers(value)
        for value in (
            config.base_args.get("num_workers"),
            config.quick_overrides.get("num_workers"),
            config.full_overrides.get("num_workers"),
        )
    )
    if uses_auto_num_workers:
        resolved_auto_workers = _resolve_hparam_auto_num_workers(max_parallel_trials)
        cpu_count = os.cpu_count() or 4
        print(
            "[hparam_search] num_workers auto resolved to "
            f"{resolved_auto_workers} per trial "
            f"(cpu_count={cpu_count}, parallel={max_parallel_trials}).",
            flush=True,
        )
    previous_global_best_score = _read_best_objective_score(
        config.global_best_config_path,
        config.objective_metric,
        expected_hparam_context=current_hparam_context,
    )
    if previous_global_best_score is not None:
        print(
            "[hparam_search] Reference global best "
            f"{config.objective_metric}={previous_global_best_score:.6f}.",
            flush=True,
        )

    history_trials: list[tuple[float, dict[str, Scalar]]] = []
    if config.search_algo == "history_guided":
        history_trials = load_historical_trials(
            output_dir=config.output_dir,
            search_space=config.search_space,
            objective_metric=config.objective_metric,
            top_n=config.history_top_n,
            base_args=config.base_args,
        )
        print(
            "[hparam_search] Search algorithm: history_guided "
            f"(history_candidates={len(history_trials)}, "
            f"top_n={config.history_top_n}, "
            f"random_fraction={config.guided_random_fraction:.2f}, "
            f"mutation_rate={config.guided_mutation_rate:.2f}).",
            flush=True,
        )
    else:
        print("[hparam_search] Search algorithm: random.", flush=True)

    seed_best_config: Optional[SeedBestConfig] = None
    seed_best_params: Optional[dict[str, Scalar]] = None
    seed_best_context_mismatch = False
    if config.seed_best_config_path is not None:
        try:
            seed_best_config = load_seed_best_config(
                path=config.seed_best_config_path,
                search_space=config.search_space,
                base_args=config.base_args,
                default_objective_metric=config.objective_metric,
            )
        except ValueError as exc:
            print(
                f"[hparam_search] Seed best config ignored due to parse error: {exc}",
                flush=True,
            )
        else:
            if seed_best_config is not None:
                seed_best_params = dict(seed_best_config.sampled_params)
                seed_best_context_mismatch = not _contexts_match(
                    seed_best_config.hparam_context,
                    current_hparam_context,
                )
                print(
                    "[hparam_search] Loaded seed best sampled params from "
                    f"{config.seed_best_config_path}.",
                    flush=True,
                )
                if seed_best_context_mismatch:
                    score_text = (
                        "-"
                        if seed_best_config.objective_score is None
                        else f"{seed_best_config.objective_score:.6f}"
                    )
                    epoch_text = (
                        "-"
                        if seed_best_config.objective_best_epoch is None
                        else str(seed_best_config.objective_best_epoch)
                    )
                    print(
                        "[hparam_search] Seed best context changed. "
                        "Schedule one full-phase recheck with stored seed: "
                        f"prev_score={score_text}, prev_best_epoch={epoch_text}.",
                        flush=True,
                    )

    quick_params = build_trial_params(
        config=config,
        phase="quick",
        count=config.quick_trials,
        seed_offset=0,
        history_trials=history_trials,
    )
    if seed_best_params is not None and quick_params:
        quick_params[0] = dict(seed_best_params)
        print(
            "[hparam_search] Quick trial 0000 replaced with seed best sampled params.",
            flush=True,
        )
    quick_overrides = dict(config.quick_overrides)
    quick_overrides.setdefault("epochs", config.quick_epochs)
    quick_overrides.setdefault("compile_mode", "off")
    print(
        f"[hparam_search] Quick phase: {config.quick_trials} trials, "
        f"epochs={quick_overrides.get('epochs')}.",
        flush=True,
    )
    quick_rows = run_phase(
        phase="quick",
        config=config,
        trial_count=config.quick_trials,
        trial_params=quick_params,
        overrides=quick_overrides,
        gpu_ids=gpu_ids,
        max_parallel_trials=max_parallel_trials,
        out_dir=config.output_dir,
        execution_mode=quick_execution_mode,
    )
    write_trials_tsv(config.output_dir / "quick_trials.tsv", quick_rows)
    ranked_quick = rank_successful_trials(quick_rows)

    full_rows: list[TrialResult]
    if config.skip_full_phase:
        full_rows = []
        print(
            "[hparam_search] Full phase skipped by config (skip_full_phase=true).",
            flush=True,
        )
    else:
        selected_for_full = ranked_quick[: config.top_k]
        full_compile_mode = (
            str(full_overrides.get("compile_mode", "auto")).strip().lower()
        )
        if full_compile_mode == "auto" and gpu_ids:
            cuda_header_path = _find_cuda_header()
            if cuda_header_path is None:
                full_overrides["compile_mode"] = "off"
                print(
                    "[hparam_search] Full phase compile_mode auto -> off: "
                    "cuda.h not found. Install CUDA toolkit headers (or set "
                    "CUDA_HOME/CUDA_PATH) to enable torch.compile.",
                    flush=True,
                )
        filtered_for_full: list[TrialResult] = []
        skipped_same_best_epoch = 0
        skipped_seed_context_match = 0
        seed_best_key: Optional[str] = None
        if seed_best_params is not None:
            seed_best_key = json.dumps(
                seed_best_params,
                sort_keys=True,
                separators=(",", ":"),
            )
        for row in selected_for_full:
            row_key = json.dumps(
                row.sampled_params,
                sort_keys=True,
                separators=(",", ":"),
            )
            if (
                seed_best_key is not None
                and row_key == seed_best_key
                and not seed_best_context_mismatch
            ):
                skipped_seed_context_match += 1
                continue
            quick_best_epoch = _read_objective_best_epoch_from_metrics(
                metrics_json_path=row.metrics_json,
                objective_metric=config.objective_metric,
            )
            if (
                full_epochs_value is not None
                and quick_best_epoch is not None
                and quick_best_epoch == full_epochs_value
            ):
                skipped_same_best_epoch += 1
                continue
            filtered_for_full.append(row)
        full_params = [dict(row.sampled_params) for row in filtered_for_full]
        injected_seed_full_recheck = False
        if seed_best_params is not None and seed_best_context_mismatch:
            existing_param_keys = {
                json.dumps(params, sort_keys=True, separators=(",", ":"))
                for params in full_params
            }
            if seed_best_key not in existing_param_keys:
                full_params.append(dict(seed_best_params))
                injected_seed_full_recheck = True
        full_count = len(full_params)
        full_execution_mode = _resolve_workload_execution_mode(
            phase_execution_mode=full_mode_policy,
            trial_count=full_count,
            max_parallel_trials=max_parallel_trials,
        )
        print(
            f"[hparam_search] Full phase: top_k={config.top_k}, "
            f"selected={full_count}, skipped_same_best_epoch={skipped_same_best_epoch}, "
            f"skipped_seed_context_match={skipped_seed_context_match}, "
            f"injected_seed_full_recheck={injected_seed_full_recheck}, "
            f"epochs={full_overrides.get('epochs')}, objective={config.objective_metric}, "
            f"execution_mode={full_execution_mode}.",
            flush=True,
        )
        if full_count > 0:
            full_rows = run_phase(
                phase="full",
                config=config,
                trial_count=full_count,
                trial_params=full_params,
                overrides=full_overrides,
                gpu_ids=gpu_ids,
                max_parallel_trials=max_parallel_trials,
                out_dir=config.output_dir,
                execution_mode=full_execution_mode,
            )
        else:
            full_rows = []
    write_trials_tsv(config.output_dir / "full_trials.tsv", full_rows)

    ranked_full = rank_successful_trials(full_rows)
    if ranked_full:
        best_row = ranked_full[0]
        ranked_for_export = ranked_full
    elif ranked_quick:
        best_row = ranked_quick[0]
        ranked_for_export = ranked_quick
    else:
        best_row = None
        ranked_for_export = []

    write_best_config(
        config.output_dir / "best_config.json",
        best_row,
        top_rows=ranked_for_export,
        top_k=config.top_k,
        fallback_validation_protocol=baseline_validation_protocol,
        hparam_context=current_hparam_context,
    )
    _write_tuning_leaderboard(
        config=config,
        ranked_rows=ranked_for_export,
        best_row=best_row,
    )
    pruned_count = _prune_non_best_trial_checkpoints(
        project_root=config.project_root,
        trial_rows=quick_rows + full_rows,
        best_row=best_row,
        min_mtime_epoch=run_started_epoch,
    )
    if pruned_count > 0:
        print(
            "[hparam_search] Pruned non-best trial checkpoints: "
            f"deleted={pruned_count}.",
            flush=True,
        )
    maybe_update_global_best(
        global_best_path=config.global_best_config_path,
        best_row=best_row,
        fallback_validation_protocol=baseline_validation_protocol,
        hparam_context=current_hparam_context,
    )
    if config.enable_visualization:
        viz_path = config.output_dir / f"{config.species}_snpr.png"
        viz_error = write_visualization(
            viz_path,
            model_name=str(config.base_args.get("model", "cnn")),
            species=config.species,
            objective_metric=config.objective_metric,
            quick_rows=quick_rows,
            full_rows=full_rows,
            base_args=config.base_args,
        )
        if viz_error is None:
            print(f"[hparam_search] Wrote tuning visualization: {viz_path}", flush=True)
        else:
            print(f"[hparam_search] Visualization skipped: {viz_error}", flush=True)
    else:
        print(
            "[hparam_search] Visualization disabled by config "
            "(enable_visualization=false).",
            flush=True,
        )
    write_summary_markdown(
        config.output_dir / "run_summary.md",
        config=config,
        gpu_ids=gpu_ids,
        quick_rows=quick_rows,
        full_rows=full_rows,
        best_row=best_row,
        previous_global_best_score=previous_global_best_score,
    )
    if best_row is None:
        print("[hparam_search] No successful trial found.")
        return 1
    if previous_global_best_score is not None and best_row.objective_score is not None:
        delta = best_row.objective_score - previous_global_best_score
        if delta >= 0.0:
            print(
                f"[hparam_search] Comparison to previous global best: +{delta:.6f}.",
                flush=True,
            )
        else:
            print(
                f"[hparam_search] Comparison to previous global best: {delta:.6f}.",
                flush=True,
            )
    print(
        f"[hparam_search] Best {best_row.objective_metric} "
        f"{best_row.objective_score:.6f} from {best_row.phase} "
        f"trial {best_row.trial_id}."
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Two-phase random HPO runner.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    actual_argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(actual_argv)
    config = load_config(Path(args.config))
    try:
        return run_search(config)
    except KeyboardInterrupt:
        _interrupt_active_trial_processes()
        print("[hparam_search] Interrupted by user.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
