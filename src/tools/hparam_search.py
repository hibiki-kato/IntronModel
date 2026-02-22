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
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from util.validation_protocol import (
    LEGACY_VALIDATION_SIGNATURE,
    build_validation_protocol,
    compute_validation_signature,
)

Scalar = int | float | str | bool
ArgValue = Scalar | None


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
                raise ValueError(
                    f"search_space['{name}'].scale must be linear or log."
                )
            if scale == "log" and float(min_value) <= 0.0:
                raise ValueError(
                    f"search_space['{name}'] log scale requires min > 0."
                )
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
        raise ValueError(
            f"search_space['{name}'].type must be categorical|float|int."
        )
    if not normalized:
        raise ValueError("search_space must define at least one parameter.")
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
    objective_metric = str(raw.get("objective_metric", "mean_pr_auc"))
    if objective_metric not in {
        "mean_pr_auc",
        "donor_pr_auc",
        "acceptor_pr_auc",
    }:
        raise ValueError(
            "objective_metric must be one of: "
            "mean_pr_auc, donor_pr_auc, acceptor_pr_auc."
        )
    normalized_space = _validate_search_space(search_space)
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

    normalized_base_args: dict[str, ArgValue] = {
        str(key): value for key, value in base_args.items()
    }
    normalized_base_args["model"] = model_name.strip()

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


def load_global_best_params(
    *,
    path: Optional[Path],
    search_space: dict[str, dict[str, object]],
    expected_validation_signature: Optional[str] = None,
) -> Optional[dict[str, Scalar]]:
    """Load and validate previous best sampled params for forced inclusion."""
    if path is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid global best config JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Global best config must be an object.")
    if raw.get("status") != "ok":
        return None
    if expected_validation_signature is not None:
        signature = raw.get("validation_signature")
        if isinstance(signature, str) and signature.strip():
            if signature.strip() != expected_validation_signature:
                print(
                    "[hparam_search] Skip global best due to validation "
                    "signature mismatch: "
                    f"{signature.strip()} != {expected_validation_signature}.",
                    flush=True,
                )
                return None

    sampled = raw.get("sampled_params")
    if not isinstance(sampled, dict):
        raise ValueError("Global best config missing sampled_params object.")

    normalized: dict[str, Scalar] = {}
    for key in sorted(search_space):
        if key not in sampled:
            raise ValueError(
                f"Global best config missing required key: sampled_params.{key}"
            )
        value = sampled[key]
        if not isinstance(value, (int, float, str, bool)):
            raise ValueError(
                f"Global best sampled_params.{key} must be a scalar value."
            )
        if not _value_matches_spec(value, search_space[key]):
            raise ValueError(
                f"Global best sampled_params.{key}={value} is not in current "
                "search space."
            )
        normalized[key] = value
    return normalized


def _select_objective_score(
    objective_metric: str,
    donor_pr_auc: Optional[float],
    acceptor_pr_auc: Optional[float],
    mean_pr_auc: Optional[float],
) -> Optional[float]:
    """Return the configured objective score from extracted metrics."""
    if objective_metric == "mean_pr_auc":
        return mean_pr_auc
    if objective_metric == "donor_pr_auc":
        return donor_pr_auc
    if objective_metric == "acceptor_pr_auc":
        return acceptor_pr_auc
    raise ValueError(f"Unsupported objective metric: {objective_metric}")


def _read_best_objective_score(
    path: Path,
    objective_metric: str,
    expected_validation_signature: Optional[str] = None,
) -> Optional[float]:
    """Read objective score from a best_config.json payload."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    if expected_validation_signature is not None:
        signature = raw.get("validation_signature")
        if isinstance(signature, str) and signature.strip():
            if signature.strip() != expected_validation_signature:
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


def load_historical_trials(
    *,
    output_dir: Path,
    search_space: dict[str, dict[str, object]],
    objective_metric: str,
    top_n: int,
) -> list[tuple[float, dict[str, Scalar]]]:
    """Load successful historical trials from sibling run directories."""
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
                            parsed = _parse_history_param_value(
                                raw_value=row.get(key, ""),
                                spec=search_space[key],
                            )
                            if parsed is None:
                                valid = False
                                break
                            params[key] = parsed
                        if valid:
                            collected.append((score, params))
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
        if text == "auto":
            return max(1, gpu_count if gpu_count > 0 else 1)
        try:
            parsed = int(text)
        except ValueError as exc:
            raise ValueError(
                "max_parallel_trials must be auto or integer."
            ) from exc
    elif isinstance(setting, int):
        parsed = setting
    else:
        raise ValueError("max_parallel_trials must be auto or integer.")
    if parsed <= 0:
        raise ValueError("max_parallel_trials must be > 0.")
    if gpu_count > 0:
        return min(parsed, gpu_count)
    return parsed


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
    cmd = [sys.executable, "-u", str(project_root / "src" / "run_model.py")]
    for key in sorted(args):
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


def _extract_pr_auc(summary: dict[str, object], task_name: str) -> Optional[float]:
    """Extract PR-AUC for one task from train summary JSON."""
    raw_task = summary.get(task_name)
    if not isinstance(raw_task, dict):
        return None
    best_pr_auc = raw_task.get("best_pr_auc")
    if isinstance(best_pr_auc, (int, float)):
        return float(best_pr_auc)
    best_metric = raw_task.get("best_metric")
    best_score = raw_task.get("best_score")
    if best_metric == "pr_auc" and isinstance(best_score, (int, float)):
        return float(best_score)
    return None


def _extract_best_epoch(summary: dict[str, object], task_name: str) -> Optional[int]:
    """Extract best epoch index for one task from train summary JSON."""
    raw_task = summary.get(task_name)
    if not isinstance(raw_task, dict):
        return None
    best_epoch = raw_task.get("best_epoch")
    if isinstance(best_epoch, int) and best_epoch > 0:
        return best_epoch
    return None


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

    if objective_metric == "donor_pr_auc":
        return _extract_best_epoch(raw, "donor")
    if objective_metric == "acceptor_pr_auc":
        return _extract_best_epoch(raw, "acceptor")
    if objective_metric == "mean_pr_auc":
        donor_epoch = _extract_best_epoch(raw, "donor")
        acceptor_epoch = _extract_best_epoch(raw, "acceptor")
        if donor_epoch is None or acceptor_epoch is None:
            return None
        return max(donor_epoch, acceptor_epoch)
    return None


def _iter_stream_lines(stream: object) -> Iterator[str]:
    """Yield decoded text lines from a subprocess stream."""
    if stream is None:
        return
    for raw_line in stream:
        if isinstance(raw_line, str):
            yield raw_line


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

    collected: list[str] = []
    prefix = f"[hparam_search][{phase} {trial_id:04d}] "
    for line in _iter_stream_lines(proc.stdout):
        collected.append(line)
        stripped = line.rstrip("\n")
        if stripped:
            print(f"{prefix}{stripped}", flush=True)
        else:
            print(prefix, flush=True)

    return_code = int(proc.wait())
    return return_code, "".join(collected)


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
    """Run one trial with OOM backoff and return structured result."""
    merged_args: dict[str, ArgValue] = dict(config.base_args)
    for key, value in sampled_params.items():
        merged_args[key] = value
    for key, value in overrides.items():
        merged_args[key] = value
    model_name = merged_args.get("model")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("base_args.model must be a non-empty string.")
    merged_args["species"] = config.species
    merged_args["train_only"] = True
    merged_args["metrics_json"] = str(metrics_json)

    base_batch = merged_args.get("batch_size")
    if not isinstance(base_batch, int):
        raise ValueError("batch_size must resolve to an integer per trial.")
    current_batch = base_batch
    oom_retries = 0
    started_at = time.time()
    error_message: Optional[str] = None
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

        return_code, combined_output = _run_command_with_streaming(
            cmd=cmd,
            cwd=config.project_root,
            env=env,
            phase=phase,
            trial_id=trial_id,
        )
        log_header = f"return_code={return_code}\n"
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(log_header)
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
                    f"Training failed after internal OOM backoff (exit={return_code}). "
                    "See trial log for details."
                )
            else:
                error_message = (
                    f"Training command failed (exit={return_code}). "
                    "See trial log for details."
                )
            duration_sec = time.time() - started_at
            return TrialResult(
                phase=phase,
                trial_id=trial_id,
                status="failed",
                gpu_id=assigned_gpu_id,
                sampled_params=sampled_params,
                effective_batch_size=current_batch,
                oom_retries=oom_retries,
                donor_pr_auc=None,
                acceptor_pr_auc=None,
                mean_pr_auc=None,
                objective_metric=config.objective_metric,
                objective_score=None,
                error_message=error_message,
                return_code=return_code,
                duration_sec=duration_sec,
                metrics_json=str(metrics_json),
                log_file=str(log_file),
            )

        next_batch = max(config.min_batch_size, current_batch // 2)
        if next_batch >= current_batch:
            error_message = "CUDA OOM encountered, but cannot reduce batch further."
            duration_sec = time.time() - started_at
            return TrialResult(
                phase=phase,
                trial_id=trial_id,
                status="failed",
                gpu_id=assigned_gpu_id,
                sampled_params=sampled_params,
                effective_batch_size=current_batch,
                oom_retries=oom_retries,
                donor_pr_auc=None,
                acceptor_pr_auc=None,
                mean_pr_auc=None,
                objective_metric=config.objective_metric,
                objective_score=None,
                error_message=error_message,
                return_code=return_code,
                duration_sec=duration_sec,
                metrics_json=str(metrics_json),
                log_file=str(log_file),
            )
        oom_retries += 1
        current_batch = next_batch

    duration_sec = time.time() - started_at
    try:
        summary = json.loads(metrics_json.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        error_message = "Metrics JSON missing or invalid."
        return TrialResult(
            phase=phase,
            trial_id=trial_id,
            status="failed",
            gpu_id=assigned_gpu_id,
            sampled_params=sampled_params,
            effective_batch_size=current_batch,
            oom_retries=oom_retries,
            donor_pr_auc=None,
            acceptor_pr_auc=None,
            mean_pr_auc=None,
            objective_metric=config.objective_metric,
            objective_score=None,
            error_message=error_message,
            return_code=return_code,
            duration_sec=duration_sec,
            metrics_json=str(metrics_json),
            log_file=str(log_file),
        )
    if not isinstance(summary, dict):
        error_message = "Metrics JSON top-level value must be an object."
        return TrialResult(
            phase=phase,
            trial_id=trial_id,
            status="failed",
            gpu_id=assigned_gpu_id,
            sampled_params=sampled_params,
            effective_batch_size=current_batch,
            oom_retries=oom_retries,
            donor_pr_auc=None,
            acceptor_pr_auc=None,
            mean_pr_auc=None,
            objective_metric=config.objective_metric,
            objective_score=None,
            error_message=error_message,
            return_code=return_code,
            duration_sec=duration_sec,
            metrics_json=str(metrics_json),
            log_file=str(log_file),
        )

    donor_pr_auc = _extract_pr_auc(summary, "donor")
    acceptor_pr_auc = _extract_pr_auc(summary, "acceptor")
    validation_protocol = _extract_validation_protocol(summary)
    validation_signature = _extract_validation_signature(summary)
    mean_pr_auc: Optional[float]
    if donor_pr_auc is None or acceptor_pr_auc is None:
        mean_pr_auc = None
    else:
        mean_pr_auc = (donor_pr_auc + acceptor_pr_auc) / 2.0

    objective_score = _select_objective_score(
        config.objective_metric,
        donor_pr_auc,
        acceptor_pr_auc,
        mean_pr_auc,
    )
    if objective_score is None:
        error_message = (
            "Missing objective metric in training summary: "
            f"{config.objective_metric}."
        )
        return TrialResult(
            phase=phase,
            trial_id=trial_id,
            status="failed",
            gpu_id=assigned_gpu_id,
            sampled_params=sampled_params,
            effective_batch_size=current_batch,
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
    return build_validation_protocol(
        val_frac=val_frac,
        seed=seed,
        train_pos_path=(
            str(train_pos_path)
            if isinstance(train_pos_path, (str, Path))
            else None
        ),
        train_neg_path=(
            str(train_neg_path)
            if isinstance(train_neg_path, (str, Path))
            else None
        ),
        metric_primary=objective_metric,
        split_type="stratified_site",
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

    kernel_raw = sampled_params.get("kernel_size")
    if kernel_raw is None:
        kernel_raw = base_args.get("kernel_size", 7)
    kernel_size = _to_positive_int(kernel_raw)
    if kernel_size is None:
        return None

    fc_hidden_raw = sampled_params.get("fc_hidden")
    if fc_hidden_raw is None:
        fc_hidden_raw = base_args.get("fc_hidden", 128)
    fc_hidden = _to_positive_int(fc_hidden_raw)
    if fc_hidden is None:
        return None

    total_params = 0
    prev_channels = 4
    for channel in conv_channels:
        conv_params = (prev_channels * channel * kernel_size) + channel
        batch_norm_params = 2 * channel
        total_params += conv_params + batch_norm_params
        prev_channels = channel

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

    kernel_raw = sampled_params.get("kernel_size")
    if kernel_raw is None:
        kernel_raw = base_args.get("kernel_size", 7)
    kernel_size = _to_positive_int(kernel_raw)
    if kernel_size is None:
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
        {
            key
            for row in rows
            for key in row.sampled_params
        }
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
            str(row.sampled_params.get(name, ""))
            for name in all_param_names
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
        (
            f"- Quick phase: {config.quick_trials} trials, "
            f"{quick_success} successful"
        ),
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
                f"- Effective batch size: `{best_row.effective_batch_size}`",
                "",
                "### Parameters",
                "",
            ]
        )
        for key in sorted(best_row.sampled_params):
            lines.append(f"- `{key}`: `{best_row.sampled_params[key]}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
) -> list[TrialResult]:
    """Run one phase with slot-based parallel scheduling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if gpu_ids:
        slots: list[Optional[str]] = list(gpu_ids[:max_parallel_trials])
    else:
        slots = [None for _ in range(max_parallel_trials)]

    pending_indices = list(range(trial_count))
    running: dict[Future[TrialResult], Optional[str]] = {}
    collected: list[TrialResult] = []

    with ThreadPoolExecutor(max_workers=max_parallel_trials) as executor:
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
                slot_label = (
                    "cpu" if assigned_gpu is None else f"gpu:{assigned_gpu}"
                )
                print(
                    f"[hparam_search] {phase} trial {trial_id:04d} started "
                    f"on {slot_label}.",
                    flush=True,
                )

            done, _ = wait(
                running.keys(),
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                assigned_gpu = running.pop(future)
                slots.append(assigned_gpu)
                result = future.result()
                collected.append(result)
                metric_text = (
                    "-"
                    if result.objective_score is None
                    else f"{result.objective_score:.6f}"
                )
                print(
                    f"[hparam_search] {phase} trial {result.trial_id:04d} "
                    f"{result.status} ({len(collected)}/{trial_count}) "
                    f"{result.objective_metric}={metric_text}.",
                    flush=True,
                )
    return collected


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
    for trial_id in range(count):
        trial_seed = config.base_seed + seed_offset + trial_id
        if (
            phase == "quick"
            and config.search_algo == "history_guided"
            and history_trials
        ):
            params = sample_trial_params_history_guided(
                search_space=config.search_space,
                seed=trial_seed,
                history_trials=history_trials,
                random_fraction=config.guided_random_fraction,
                mutation_rate=config.guided_mutation_rate,
            )
        else:
            params = sample_trial_params(
                search_space=config.search_space,
                seed=trial_seed,
            )
        sampled.append(params)
    return sampled


def write_best_config(
    path: Path,
    row: Optional[TrialResult],
    *,
    fallback_validation_protocol: Optional[dict[str, object]] = None,
    fallback_validation_signature: Optional[str] = None,
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
    validation_signature = row.validation_signature
    if (
        validation_signature == LEGACY_VALIDATION_SIGNATURE
        and fallback_validation_signature is not None
    ):
        validation_signature = fallback_validation_signature
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
        "validation_signature": validation_signature,
        "validation_protocol": validation_protocol,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def maybe_update_global_best(
    *,
    global_best_path: Optional[Path],
    best_row: Optional[TrialResult],
    fallback_validation_protocol: Optional[dict[str, object]] = None,
    fallback_validation_signature: Optional[str] = None,
) -> None:
    """Update global best config if current run improves the best score."""
    if (
        global_best_path is None
        or best_row is None
        or best_row.objective_score is None
    ):
        return
    previous_score = _read_best_objective_score(
        global_best_path,
        best_row.objective_metric,
        expected_validation_signature=(
            best_row.validation_signature
            if best_row.validation_signature != LEGACY_VALIDATION_SIGNATURE
            else fallback_validation_signature
        ),
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
        fallback_validation_signature=fallback_validation_signature,
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
    config.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_validation_protocol = _derive_validation_protocol_from_args(
        merged_args=dict(config.base_args),
        objective_metric=config.objective_metric,
    )
    baseline_validation_signature = compute_validation_signature(
        baseline_validation_protocol
    )
    gpu_ids = detect_gpu_ids(config.gpu_ids_setting)
    max_parallel_trials = resolve_max_parallel(
        setting=config.max_parallel_trials_setting,
        gpu_count=len(gpu_ids),
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
    previous_global_best_score = _read_best_objective_score(
        config.global_best_config_path,
        config.objective_metric,
        expected_validation_signature=baseline_validation_signature,
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

    seed_best_params: Optional[dict[str, Scalar]] = None
    if config.seed_best_config_path is not None:
        try:
            seed_best_params = load_global_best_params(
                path=config.seed_best_config_path,
                search_space=config.search_space,
                expected_validation_signature=baseline_validation_signature,
            )
        except ValueError as exc:
            print(
                "[hparam_search] Seed best config ignored due to validation "
                f"error: {exc}",
                flush=True,
            )
        else:
            if seed_best_params is not None:
                print(
                    "[hparam_search] Loaded seed best sampled params from "
                    f"{config.seed_best_config_path}.",
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
            "[hparam_search] Quick trial 0000 replaced with seed best "
            "sampled params.",
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
    )
    write_trials_tsv(config.output_dir / "quick_trials.tsv", quick_rows)
    ranked_quick = rank_successful_trials(quick_rows)

    selected_for_full = ranked_quick[: config.top_k]
    full_overrides = dict(config.full_overrides)
    full_overrides.setdefault("epochs", config.full_epochs)
    full_overrides.setdefault("compile_mode", "auto")
    full_epochs_value = _to_positive_int(full_overrides.get("epochs"))
    filtered_for_full: list[TrialResult] = []
    skipped_same_best_epoch = 0
    for row in selected_for_full:
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
    full_count = len(full_params)
    print(
        f"[hparam_search] Full phase: top_k={config.top_k}, "
        f"selected={full_count}, skipped_same_best_epoch={skipped_same_best_epoch}, "
        f"epochs={full_overrides.get('epochs')}, objective={config.objective_metric}.",
        flush=True,
    )
    full_rows: list[TrialResult]
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
        )
    else:
        full_rows = []
    write_trials_tsv(config.output_dir / "full_trials.tsv", full_rows)

    ranked_full = rank_successful_trials(full_rows)
    if ranked_full:
        best_row = ranked_full[0]
    elif ranked_quick:
        best_row = ranked_quick[0]
    else:
        best_row = None

    write_best_config(
        config.output_dir / "best_config.json",
        best_row,
        fallback_validation_protocol=baseline_validation_protocol,
        fallback_validation_signature=baseline_validation_signature,
    )
    maybe_update_global_best(
        global_best_path=config.global_best_config_path,
        best_row=best_row,
        fallback_validation_protocol=baseline_validation_protocol,
        fallback_validation_signature=baseline_validation_signature,
    )
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
    if (
        previous_global_best_score is not None
        and best_row.objective_score is not None
    ):
        delta = best_row.objective_score - previous_global_best_score
        if delta >= 0.0:
            print(
                "[hparam_search] Comparison to previous global best: "
                f"+{delta:.6f}.",
                flush=True,
            )
        else:
            print(
                "[hparam_search] Comparison to previous global best: "
                f"{delta:.6f}.",
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
    return run_search(config)


if __name__ == "__main__":
    raise SystemExit(main())
