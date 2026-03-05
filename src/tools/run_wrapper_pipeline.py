from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_model import (
    _build_checkpoint_paths,
    _build_checkpoint_stem_from_params,
    _infer_window_defaults,
    parse_args,
)
from util.checkpoint_io import (
    TaskName,
    extract_task_checkpoint_path,
    normalize_checkpoint_path,
    read_json_object,
)
from util.data_proc import build_output_stem, parse_name_fields
from util.model_task_paths import checkpoint_tasks_for_model


@dataclass(frozen=True)
class WrapperSpec:
    """Specification for one training/inference wrapper script."""

    script_name: str
    model_env_name: str
    supports_tuned_hparams: bool
    tuned_key_map: Mapping[str, str]
    stem_param_builder: str
    required_arg_keys: tuple[str, ...]
    per_task_override_keys: tuple[str, ...]


def _project_root() -> Path:
    """Return repository root inferred from this script path."""

    return PROJECT_ROOT


def _require_env(env: Mapping[str, str], key: str) -> str:
    """Return an environment value or raise an explicit error."""

    value = env.get(key)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def _as_int(raw: str, key: str) -> int:
    """Parse int with a key-specific validation error."""

    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer: {raw}") from exc


def _as_float(raw: str, key: str) -> float:
    """Parse float with a key-specific validation error."""

    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a float: {raw}") from exc


def _check_choice(value: str, allowed: tuple[str, ...], key: str) -> None:
    """Validate a string enum value."""

    if value not in allowed:
        joined = "|".join(allowed)
        raise ValueError(f"{key} must be {joined}.")


def _resolve_root(env_key: str, fallback: Path) -> str:
    """Resolve root path from env override or fallback path."""

    raw = os.environ.get(env_key)
    if raw is None or raw.strip() == "":
        return str(fallback)
    path = Path(raw)
    if not path.is_absolute():
        path = _project_root() / path
    return str(path)


def _resolve_species_case(raw_species: str, data_root: Path, label: str) -> str:
    """Resolve species directory case-insensitively under data root."""

    direct = data_root / raw_species
    if direct.is_dir():
        return raw_species

    if not data_root.exists():
        return raw_species

    matches: list[str] = [
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and path.name.lower() == raw_species.lower()
    ]
    if len(matches) == 1:
        print(
            f"[{label}] species case normalized: '{raw_species}' -> '{matches[0]}'",
            file=sys.stderr,
        )
        return matches[0]
    if len(matches) > 1:
        joined = " ".join(matches)
        raise ValueError(
            f"ambiguous species '{raw_species}'. case-insensitive matches: {joined}"
        )
    return raw_species


def _resolve_tuned_config_path(
    task: TaskName,
    explicit_path: str,
    species: str,
    data_root: Path,
    model_name: str,
    shared_path: str,
) -> Path | None:
    """Resolve tuned config path using wrapper compatibility order."""

    if explicit_path.strip() != "":
        return Path(explicit_path)

    task_path = data_root / species / "tuning" / model_name / task / "best_config.json"
    if task_path.is_file():
        return task_path

    if shared_path.strip() != "":
        shared = Path(shared_path)
        if shared.is_file():
            return shared

    legacy = data_root / species / "tuning" / model_name / "best_config.json"
    if legacy.is_file():
        return legacy

    return None


def _resolve_tuned_checkpoint_path(
    *,
    task: TaskName,
    tuned_config_path: Path,
) -> Path | None:
    """Resolve checkpoint path from one tuned best_config.json file."""
    payload = read_json_object(tuned_config_path)
    if payload is None:
        return None
    if str(payload.get("status", "")).strip().lower() != "ok":
        return None

    direct = extract_task_checkpoint_path(
        payload,
        task=task,
        base_dir=tuned_config_path.parent,
    )
    if direct is not None and direct.exists():
        return direct

    metrics_json_raw = payload.get("metrics_json")
    if isinstance(metrics_json_raw, str) and metrics_json_raw.strip():
        metrics_path = normalize_checkpoint_path(
            metrics_json_raw,
            base_dir=tuned_config_path.parent,
        )
        metrics_payload = read_json_object(metrics_path)
        if metrics_payload is not None:
            from_metrics_json = extract_task_checkpoint_path(
                metrics_payload,
                task=task,
                base_dir=metrics_path.parent,
            )
            if from_metrics_json is not None and from_metrics_json.exists():
                return from_metrics_json

    phase = payload.get("phase")
    trial_id = payload.get("trial_id")
    if not isinstance(phase, str) or not isinstance(trial_id, int):
        return None
    metrics_path = (
        tuned_config_path.parent / f"{phase}_trial_{trial_id:04d}.metrics.json"
    )
    metrics_payload = read_json_object(metrics_path)
    if metrics_payload is None:
        return None
    fallback = extract_task_checkpoint_path(
        metrics_payload,
        task=task,
        base_dir=metrics_path.parent,
    )
    if fallback is None or not fallback.exists():
        return None
    return fallback


def _extract_tuned_assignments(
    config_path: Path,
    task_prefix: str | None,
    key_map: Mapping[str, str],
) -> dict[str, str]:
    """Extract tuned key/value assignments from best_config.json payload."""

    payload_obj = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError("Config payload must be a JSON object.")

    status = str(payload_obj.get("status", "")).strip().lower()
    if status != "ok":
        show = status if status else "<missing>"
        raise ValueError(f"Expected status='ok', got: {show}")

    sampled_params_obj = payload_obj.get("sampled_params")
    if not isinstance(sampled_params_obj, dict):
        raise ValueError("sampled_params is missing or invalid.")

    assignments: dict[str, str] = {}
    for key, suffix in key_map.items():
        if key not in sampled_params_obj:
            continue
        value = sampled_params_obj[key]
        if value is None:
            continue

        if isinstance(value, bool):
            text = "1" if value else "0"
        elif isinstance(value, int):
            text = str(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite value for '{key}'.")
            text = format(value, ".15g")
        else:
            text = str(value)

        if task_prefix is None:
            assignments[suffix] = text
        else:
            assignments[f"{task_prefix}_{suffix}"] = text
    return assignments


def _resolve_tasks_for_target(
    *,
    train_target: str,
    model_tasks: tuple[TaskName, ...],
    train_only: bool,
) -> tuple[TaskName, ...]:
    """Resolve required tasks from one train target and model task set."""
    if len(model_tasks) == 1:
        expected = model_tasks[0]
        if train_target != expected:
            raise ValueError(f"TRAIN_TARGET must be {expected}.")
        return model_tasks

    if train_target not in ("both", *model_tasks):
        allowed = "|".join(("both", *model_tasks))
        raise ValueError(f"TRAIN_TARGET must be {allowed}.")
    if (not train_only) or train_target == "both":
        return model_tasks
    return (train_target,)


def _apply_tuned_overrides(
    spec: WrapperSpec,
    env: dict[str, str],
    data_root: Path,
) -> dict[TaskName, Path]:
    """Apply tuned hparam overrides to empty wrapper env fields."""

    resolved_configs: dict[TaskName, Path] = {}
    use_mode = env.get("USE_TUNED_HPARAMS", "off")
    if use_mode == "off":
        return resolved_configs

    species = env["SPECIES"]
    tuned_model_name = spec.model_env_name
    if spec.script_name == "dnabert.sh":
        resolved_model_name = env.get("MODEL", "").strip()
        if resolved_model_name != "":
            tuned_model_name = resolved_model_name

    if env.get("SHARED_TUNED_CONFIG_PATH", "").strip() == "":
        env["SHARED_TUNED_CONFIG_PATH"] = str(
            data_root / species / "tuning" / tuned_model_name / "best_config.json"
        )

    model_tasks = checkpoint_tasks_for_model(tuned_model_name)
    train_target = env["TRAIN_TARGET"]
    is_single_task_model = len(model_tasks) == 1
    if is_single_task_model:
        if train_target != model_tasks[0]:
            raise ValueError(f"TRAIN_TARGET must be {model_tasks[0]}.")
        tasks = model_tasks
    else:
        if train_target not in ("both", *model_tasks):
            allowed = "|".join(("both", *model_tasks))
            raise ValueError(f"TRAIN_TARGET must be {allowed}.")
        tasks = model_tasks if train_target == "both" else (train_target,)

    for task in tasks:
        explicit_key = f"{task.upper()}_TUNED_CONFIG_PATH"
        resolved = _resolve_tuned_config_path(
            task=task,
            explicit_path=env.get(explicit_key, ""),
            species=species,
            data_root=data_root,
            model_name=tuned_model_name,
            shared_path=env.get("SHARED_TUNED_CONFIG_PATH", ""),
        )
        if resolved is None:
            if use_mode == "required":
                raise ValueError(f"tuned {task} config is required but not found.")
            print(
                f"[{spec.script_name}] tuned {task} config not found; "
                "using CONFIG defaults.",
                file=sys.stderr,
            )
            continue

        if not resolved.is_file():
            if use_mode == "required":
                raise ValueError(f"tuned {task} config not found: {resolved}")
            print(
                f"[{spec.script_name}] tuned {task} config not found: {resolved}",
                file=sys.stderr,
            )
            continue

        task_prefix: str | None = None if is_single_task_model else task.upper()
        try:
            assignments = _extract_tuned_assignments(
                config_path=resolved,
                task_prefix=task_prefix,
                key_map=spec.tuned_key_map,
            )
        except ValueError as exc:
            if use_mode == "required":
                raise
            print(
                f"[{spec.script_name}] tuned {task} load failed: {exc}; "
                "using CONFIG defaults.",
                file=sys.stderr,
            )
            continue

        applied_count = 0
        kept_manual_count = 0
        for var_name, value in assignments.items():
            if env.get(var_name, "") != "":
                kept_manual_count += 1
                continue
            env[var_name] = value
            applied_count += 1

        print(
            f"[{spec.script_name}] tuned {task} loaded from {resolved} "
            f"(applied={applied_count}, kept_manual={kept_manual_count})"
        )
        resolved_configs[task] = resolved.resolve()
    return resolved_configs


def _resolve_expected_checkpoint_paths_for_run(
    run_args: list[str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Resolve strict checkpoint paths and required tasks for one run argument set."""
    args = parse_args(run_args)
    model_tasks = checkpoint_tasks_for_model(str(args.model))
    donor_len, acceptor_len, inferred_train_len = _infer_window_defaults(
        species=args.species,
        donor_len=args.donor_len,
        acceptor_len=args.acceptor_len,
    )
    checkpoint_stem = _build_checkpoint_stem_from_params(
        model_name=args.model,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        inferred_train_len=inferred_train_len,
        raw_params=dict(vars(args)),
    )
    checkpoint_paths = _build_checkpoint_paths(
        args.species,
        checkpoint_stem,
        tasks=model_tasks,
    )
    default_train_target = "both" if len(model_tasks) > 1 else model_tasks[0]
    train_target = str(
        getattr(args, "train_target", default_train_target)
    ).strip().lower()
    required_tasks = _resolve_tasks_for_target(
        train_target=train_target,
        model_tasks=model_tasks,
        train_only=bool(getattr(args, "train_only", False)),
    )
    return checkpoint_paths, required_tasks


def _materialize_checkpoint_alias(
    *,
    source_path: Path,
    target_path: Path,
) -> str:
    """Materialize one checkpoint alias via hardlink or copy."""
    if target_path.exists():
        return "existing"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)
        return "copy"
    return "hardlink"


def _ensure_tuned_checkpoint_aliases(
    *,
    spec: WrapperSpec,
    run_args: list[str],
    tuned_config_paths: Mapping[TaskName, Path],
) -> None:
    """Ensure strict checkpoint paths are backed by tuned best checkpoints."""
    if not tuned_config_paths:
        return
    checkpoint_paths, required_tasks = _resolve_expected_checkpoint_paths_for_run(
        run_args
    )
    for task in required_tasks:
        strict_path = Path(checkpoint_paths[task]).resolve()
        if strict_path.exists():
            continue
        tuned_config_path = tuned_config_paths.get(task)
        if tuned_config_path is None:
            continue
        tuned_checkpoint = _resolve_tuned_checkpoint_path(
            task=task,
            tuned_config_path=tuned_config_path,
        )
        if tuned_checkpoint is None:
            continue
        mode = _materialize_checkpoint_alias(
            source_path=tuned_checkpoint,
            target_path=strict_path,
        )
        print(
            f"[{spec.script_name}] tuned {task} checkpoint alias "
            f"{mode}: {tuned_checkpoint} -> {strict_path}"
        )


def _stem_params(builder: str, env: Mapping[str, str]) -> dict[str, object]:
    """Build output-stem parameter dictionary with wrapper-compatible casting."""

    base: dict[str, object] = {
        "donor_len": _as_int(_require_env(env, "DONOR_LEN"), "DONOR_LEN"),
        "acceptor_len": _as_int(_require_env(env, "ACCEPTOR_LEN"), "ACCEPTOR_LEN"),
        "epochs": _require_env(env, "EPOCHS"),
        "batch_size": _as_int(_require_env(env, "BATCH_SIZE"), "BATCH_SIZE"),
        "lr": _as_float(_require_env(env, "LR"), "LR"),
        "loss": _require_env(env, "LOSS"),
        "weight_decay": _as_float(
            _require_env(env, "WEIGHT_DECAY"), "WEIGHT_DECAY"
        ),
        "eta_min_ratio": _as_float(
            _require_env(env, "ETA_MIN_RATIO"), "ETA_MIN_RATIO"
        ),
        "grad_clip": _as_float(_require_env(env, "GRAD_CLIP"), "GRAD_CLIP"),
        "val_frac": _as_float(_require_env(env, "VAL_FRAC"), "VAL_FRAC"),
        "intron_score_op": _require_env(env, "INTRON_SCORE_OP"),
        "transcript_score_agg": _require_env(env, "TRANSCRIPT_SCORE_AGG"),
        "softmin_tau": _as_float(_require_env(env, "SOFTMIN_TAU"), "SOFTMIN_TAU"),
        "seed": _as_int(_require_env(env, "SEED"), "SEED"),
        "train_target": _require_env(env, "TRAIN_TARGET"),
    }
    tag_value = env.get("TAG", "").strip()
    if tag_value != "":
        base["tag"] = tag_value

    if builder in {"cnn", "cnn_resdil", "tcn"}:
        base["conv_channels"] = _require_env(env, "CONV_CHANNELS") or None
        if builder in {"cnn", "cnn_resdil"}:
            kernel_sizes_raw = _require_env(env, "KERNEL_SIZES").strip()
            if kernel_sizes_raw != "":
                base["kernel_sizes"] = kernel_sizes_raw
            else:
                base["kernel_sizes"] = _as_int(
                    _require_env(env, "KERNEL_SIZE"),
                    "KERNEL_SIZE",
                )
        else:
            base["kernel_size"] = _as_int(
                _require_env(env, "KERNEL_SIZE"),
                "KERNEL_SIZE",
            )
        base["dropout"] = _as_float(_require_env(env, "DROPOUT"), "DROPOUT")
        base["fc_hidden"] = _as_int(_require_env(env, "FC_HIDDEN"), "FC_HIDDEN")

    if builder == "tcn":
        base["tcn_block_repeats"] = _as_int(
            _require_env(env, "TCN_BLOCK_REPEATS"), "TCN_BLOCK_REPEATS"
        )
        base["tcn_causal"] = _as_int(_require_env(env, "TCN_CAUSAL"), "TCN_CAUSAL")

    if builder in {"bert", "dnabert"}:
        base["dropout"] = _as_float(_require_env(env, "DROPOUT"), "DROPOUT")

    return base


def _build_run_args(spec: WrapperSpec, env: Mapping[str, str]) -> list[str]:
    """Construct CLI args for src/run_model.py."""

    args: list[str] = []

    model_value = _require_env(env, "MODEL")
    args.extend(["--model", model_value])

    for key in spec.required_arg_keys:
        args.extend([f"--{key.lower()}", _require_env(env, key)])

    optional_global = (
        "FOCAL_ALPHA_POS",
        "F1_LAMBDA",
        "ASYM_ALPHA_POS",
        "ASYM_GAMMA_POS",
        "ASYM_GAMMA_NEG",
        "KERNEL_SIZES",
        "N_DIM",
        "TAG",
    )
    for key in optional_global:
        value = env.get(key, "")
        if value != "":
            args.extend([f"--{key.lower()}", value])

    optional_data_paths = (
        ("TRAIN_POS_PATH", "train_pos_path"),
        ("TRAIN_NEG_PATH", "train_neg_path"),
    )
    for env_key, cli_key in optional_data_paths:
        value = env.get(env_key, "")
        if value != "":
            args.extend([f"--{cli_key}", value])

    for prefix in ("DONOR", "ACCEPTOR"):
        for key in spec.per_task_override_keys:
            env_key = f"{prefix}_{key}"
            value = env.get(env_key, "")
            if value != "":
                args.extend([f"--{prefix.lower()}_{key.lower()}", value])

    if env.get("SKIP_TRAINING", "0") == "1":
        args.append("--skip_train")
    if env.get("CONTINUE_TRAINING", "0") == "1":
        args.append("--continue_train")
    if env.get("TRAIN_ONLY", "0") == "1":
        args.append("--train_only")

    site_score = env.get("PRECOMPUTED_SITE_SCORE_TSV", "")
    if site_score != "":
        args.extend(["--site_score_tsv", site_score])

    checkpoint_top_k = env.get("CHECKPOINT_TOP_K", "")
    if checkpoint_top_k != "":
        args.extend(["--checkpoint_top_k", checkpoint_top_k])
    checkpoint_prune_dry_run = env.get("CHECKPOINT_PRUNE_DRY_RUN", "")
    if checkpoint_prune_dry_run != "":
        args.extend(
            ["--checkpoint_prune_dry_run", checkpoint_prune_dry_run]
        )

    return args


def _validate_common(spec: WrapperSpec, env: Mapping[str, str]) -> None:
    """Validate shared wrapper configuration values."""

    _check_choice(
        _require_env(env, "COMPILE_MODE"),
        ("off", "on", "auto"),
        "COMPILE_MODE",
    )
    _check_choice(_require_env(env, "TRAIN_ONLY"), ("0", "1"), "TRAIN_ONLY")
    _check_choice(_require_env(env, "SKIP_TRAINING"), ("0", "1"), "SKIP_TRAINING")
    _check_choice(
        _require_env(env, "CONTINUE_TRAINING"),
        ("0", "1"),
        "CONTINUE_TRAINING",
    )
    model_name = _require_env(env, "MODEL")
    model_tasks = checkpoint_tasks_for_model(model_name)
    train_target = _require_env(env, "TRAIN_TARGET")
    allowed_targets = (
        ("both", *model_tasks) if len(model_tasks) > 1 else model_tasks
    )
    _check_choice(train_target, tuple(allowed_targets), "TRAIN_TARGET")

    if env["SKIP_TRAINING"] == "1" and env["CONTINUE_TRAINING"] == "1":
        raise ValueError("CONTINUE_TRAINING=1 cannot be used with SKIP_TRAINING=1.")
    if len(model_tasks) > 1 and train_target != "both" and env["TRAIN_ONLY"] != "1":
        raise ValueError("TRAIN_TARGET single-task mode requires TRAIN_ONLY=1.")

    if spec.supports_tuned_hparams:
        _check_choice(
            _require_env(env, "USE_TUNED_HPARAMS"),
            ("off", "auto", "required"),
            "USE_TUNED_HPARAMS",
        )

    mps_max_batch_size = _as_int(
        _require_env(env, "MPS_MAX_BATCH_SIZE"), "MPS_MAX_BATCH_SIZE"
    )
    if mps_max_batch_size <= 0:
        raise ValueError("MPS_MAX_BATCH_SIZE must be a positive integer.")


def _validate_dnabert_specific(env: Mapping[str, str]) -> None:
    """Validate DNABERT-specific wrapper controls."""

    variant = _require_env(env, "DNABERT_VARIANT")
    _check_choice(variant, ("2", "6"), "DNABERT_VARIANT")
    _check_choice(
        _require_env(env, "TRUST_REMOTE_CODE"),
        ("0", "1"),
        "TRUST_REMOTE_CODE",
    )


def _resolve_dnabert_model(env: dict[str, str], model_root: Path) -> None:
    """Populate MODEL and default pretrained path for DNABERT wrappers."""

    variant = _require_env(env, "DNABERT_VARIANT")
    env["MODEL"] = f"dnabert{variant}"

    if env.get("PRETRAINED_MODEL_NAME", "").strip() == "":
        relative = _require_env(env, "PRETRAINED_MODEL_RELATIVE_PATH")
        env["PRETRAINED_MODEL_NAME"] = str(model_root / relative)


def _check_dnabert_skip_training_preconditions(run_args: list[str]) -> None:
    """Replicate dnabert wrapper checkpoint existence checks."""

    args = parse_args(run_args)
    model_tasks = checkpoint_tasks_for_model(str(args.model))
    donor_len, acceptor_len, inferred_train_len = _infer_window_defaults(
        species=args.species,
        donor_len=args.donor_len,
        acceptor_len=args.acceptor_len,
    )
    checkpoint_stem = _build_checkpoint_stem_from_params(
        model_name=args.model,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        inferred_train_len=inferred_train_len,
        raw_params=dict(vars(args)),
    )
    checkpoint_paths = _build_checkpoint_paths(
        args.species,
        checkpoint_stem,
        tasks=model_tasks,
    )
    default_train_target = "both" if len(model_tasks) > 1 else model_tasks[0]
    train_target = str(
        getattr(args, "train_target", default_train_target)
    ).strip().lower()
    required_tasks = _resolve_tasks_for_target(
        train_target=train_target,
        model_tasks=model_tasks,
        train_only=bool(getattr(args, "train_only", False)),
    )

    missing_paths = [
        checkpoint_paths[task]
        for task in required_tasks
        if not Path(checkpoint_paths[task]).exists()
    ]
    if missing_paths:
        print(
            "[dnabert.sh] SKIP_TRAINING=1 requires existing checkpoints for "
            "the current config.",
            file=sys.stderr,
        )
        for missing in missing_paths:
            print(f"[dnabert.sh] missing checkpoint: {missing}", file=sys.stderr)
        print(
            "[dnabert.sh] Set SKIP_TRAINING=0 for a fresh run, or set "
            "PRECOMPUTED_SITE_SCORE_TSV to skip inference checkpoint loading.",
            file=sys.stderr,
        )
        raise ValueError("Missing checkpoint(s) for SKIP_TRAINING=1.")


def _parse_species_list(raw_species: str) -> list[str]:
    """Parse one-or-many species string into ordered unique entries."""
    tokens = re.split(r"[\s,]+", raw_species.strip())
    species_list: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        value = token.strip()
        if value == "":
            continue
        if value in seen:
            continue
        seen.add(value)
        species_list.append(value)
    if not species_list:
        raise ValueError("SPECIES must contain at least one species name.")
    return species_list


def _run_single_species(
    spec: WrapperSpec,
    *,
    project_root: Path,
    data_root: Path,
    env: dict[str, str],
    species: str,
) -> int:
    """Execute one wrapper pipeline run for one normalized species."""
    env["SPECIES"] = species
    os.environ["INTRONMODEL_MPS_MAX_BATCH_SIZE"] = _require_env(
        env,
        "MPS_MAX_BATCH_SIZE",
    )
    for key in ("TRAIN_POS_PATH", "TRAIN_NEG_PATH"):
        raw = env.get(key, "")
        if raw == "":
            continue
        env[key] = (
            raw.replace("{species}", species)
            .replace("{SPECIES}", species)
            .replace("${SPECIES}", species)
        )

    tuned_config_paths: dict[TaskName, Path] = {}
    if spec.supports_tuned_hparams:
        tuned_config_paths = _apply_tuned_overrides(spec, env, data_root)

    model_name = _require_env(env, "MODEL")
    donor_len = _as_int(_require_env(env, "DONOR_LEN"), "DONOR_LEN")
    acceptor_len = _as_int(_require_env(env, "ACCEPTOR_LEN"), "ACCEPTOR_LEN")
    name_fields = parse_name_fields(env.get("NAME_FIELDS", ""))

    params = _stem_params(spec.stem_param_builder, env)
    output_stem = build_output_stem(
        model_name=model_name,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        fallback_train_len=None,
        name_fields=name_fields,
        name_params=params,
    )

    test_tsv = data_root / species / "raw" / "transcripts.tsv"
    class_file = data_root / species / "raw" / "transcript_class.txt"
    output_site_score_tsv = data_root / species / "site_score" / f"{output_stem}.tsv"
    output_trans_score_tsv = data_root / species / "trans_score" / f"{output_stem}.tsv"
    output_eval_score_txt = data_root / species / "eval_score" / f"{output_stem}.txt"
    metrics_json_path = data_root / species / "site_score" / f"{output_stem}.train.json"
    learning_curve_png = (
        data_root / species / "site_score" / f"{output_stem}_learning_curve.png"
    )

    env["TEST_TSV"] = str(test_tsv)
    env["CLASS_FILE"] = str(class_file)
    env["OUTPUT_SITE_SCORE_TSV"] = str(output_site_score_tsv)
    env["OUTPUT_TRANS_SCORE_TSV"] = str(output_trans_score_tsv)
    env["OUTPUT_EVAL_SCORE_TXT"] = str(output_eval_score_txt)

    run_args = _build_run_args(spec, env)
    if (
        env.get("SKIP_TRAINING", "0") == "1"
        or env.get("CONTINUE_TRAINING", "0") == "1"
    ):
        _ensure_tuned_checkpoint_aliases(
            spec=spec,
            run_args=run_args,
            tuned_config_paths=tuned_config_paths,
        )
    run_args.extend(
        [
            "--test_tsv",
            env["TEST_TSV"],
            "--class_file",
            env["CLASS_FILE"],
            "--site_output_tsv",
            env["OUTPUT_SITE_SCORE_TSV"],
            "--transcript_output_tsv",
            env["OUTPUT_TRANS_SCORE_TSV"],
            "--eval_output_txt",
            env["OUTPUT_EVAL_SCORE_TXT"],
            "--metrics_json",
            str(metrics_json_path),
        ]
    )

    if spec.script_name == "dnabert.sh":
        is_skip = env.get("SKIP_TRAINING", "0") == "1"
        has_precomputed = env.get("PRECOMPUTED_SITE_SCORE_TSV", "") != ""
        if is_skip and not has_precomputed:
            _check_dnabert_skip_training_preconditions(run_args)

    print(f"[{spec.script_name}] Start unified pipeline")
    if "PERF_MODE" in env:
        print(
            f"[{spec.script_name}] species={species} "
            f"perf_mode={env['PERF_MODE']} train_only={env['TRAIN_ONLY']}"
        )
    else:
        print(
            f"[{spec.script_name}] species={species} "
            f"train_only={env['TRAIN_ONLY']}"
        )

    process = subprocess.run(
        [sys.executable, str(project_root / "src" / "run_model.py"), *run_args],
        check=False,
        env=os.environ.copy(),
    )
    if process.returncode != 0:
        return process.returncode

    learning_curve_mode = env.get("PLOT_LEARNING_CURVE", "1").strip().lower()
    if learning_curve_mode not in {"0", "1"}:
        raise ValueError("PLOT_LEARNING_CURVE must be 0 or 1.")
    if learning_curve_mode == "1":
        if metrics_json_path.exists():
            curve_proc = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "src" / "tools" / "plot_learning_curve.py"),
                    "--metrics-json",
                    str(metrics_json_path),
                    "--output",
                    str(learning_curve_png),
                ],
                check=False,
                env=os.environ.copy(),
            )
            if curve_proc.returncode != 0:
                print(
                    f"[{spec.script_name}] learning-curve plot failed "
                    f"(metrics={metrics_json_path}).",
                    file=sys.stderr,
                )
            else:
                print(f"[{spec.script_name}] learning_curve={learning_curve_png}")
        else:
            print(
                f"[{spec.script_name}] metrics JSON not found for learning curve: "
                f"{metrics_json_path}",
                file=sys.stderr,
            )

    print(f"[{spec.script_name}] Done")
    print(f"[{spec.script_name}] site_score={output_site_score_tsv}")
    print(f"[{spec.script_name}] transcript_score={output_trans_score_tsv}")
    print(f"[{spec.script_name}] eval_score={output_eval_score_txt}")
    return 0


def _run(spec: WrapperSpec) -> int:
    """Execute a wrapper pipeline from environment-backed CONFIG values."""

    project_root = _project_root()
    data_root = Path(_resolve_root("INTRONMODEL_DATA_ROOT", project_root / "data"))
    model_root = Path(_resolve_root("INTRONMODEL_MODEL_ROOT", project_root / "model"))

    os.environ["INTRONMODEL_DATA_ROOT"] = str(data_root)
    os.environ["INTRONMODEL_MODEL_ROOT"] = str(model_root)

    env = dict(os.environ)

    if spec.script_name == "dnabert.sh":
        _validate_dnabert_specific(env)
        _resolve_dnabert_model(env, model_root)

    _validate_common(spec, env)
    raw_species_value = _require_env(env, "SPECIES")
    raw_species_list = _parse_species_list(raw_species_value)
    normalized_species = [
        _resolve_species_case(
            raw_species=raw_species,
            data_root=data_root,
            label=spec.script_name,
        )
        for raw_species in raw_species_list
    ]

    if len(normalized_species) > 1:
        joined = ",".join(normalized_species)
        print(f"[{spec.script_name}] multi-species run: {joined}")

    base_env = dict(env)
    for index, species in enumerate(normalized_species, start=1):
        if len(normalized_species) > 1:
            print(
                f"[{spec.script_name}] species batch progress: "
                f"{index}/{len(normalized_species)} ({species})"
            )
        species_env = dict(base_env)
        code = _run_single_species(
            spec,
            project_root=project_root,
            data_root=data_root,
            env=species_env,
            species=species,
        )
        if code != 0:
            return code
    return 0


SPECS: dict[str, WrapperSpec] = {
    "cnn.sh": WrapperSpec(
        script_name="cnn.sh",
        model_env_name="cnn",
        supports_tuned_hparams=True,
        tuned_key_map={
            "batch_size": "BATCH_SIZE",
            "lr": "LR",
            "loss": "LOSS",
            "conv_channels": "CONV_CHANNELS",
            "kernel_sizes": "KERNEL_SIZES",
            "donor_kernel_sizes": "DONOR_KERNEL_SIZES",
            "acceptor_kernel_sizes": "ACCEPTOR_KERNEL_SIZES",
            "dropout": "DROPOUT",
            "fc_hidden": "FC_HIDDEN",
            "weight_decay": "WEIGHT_DECAY",
            "eta_min_ratio": "ETA_MIN_RATIO",
            "val_frac": "VAL_FRAC",
            "grad_clip": "GRAD_CLIP",
            "pos_weight_cap": "POS_WEIGHT_CAP",
            "focal_gamma": "FOCAL_GAMMA",
            "focal_alpha_pos": "FOCAL_ALPHA_POS",
            "f1_lambda": "F1_LAMBDA",
            "asym_gamma_pos": "ASYM_GAMMA_POS",
            "asym_gamma_neg": "ASYM_GAMMA_NEG",
            "asym_alpha_pos": "ASYM_ALPHA_POS",
        },
        stem_param_builder="cnn",
        required_arg_keys=(
            "SPECIES",
            "DONOR_LEN",
            "ACCEPTOR_LEN",
            "EPOCHS",
            "MAX_EPOCHS",
            "EARLY_STOP_PATIENCE",
            "EARLY_STOP_MIN_DELTA",
            "TRAIN_TARGET",
            "SEQUENCE_TRANSFORM",
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "CONV_CHANNELS",
            "KERNEL_SIZES",
            "DROPOUT",
            "FC_HIDDEN",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "GRAD_CLIP",
            "VAL_FRAC",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "F1_LAMBDA",
            "NAME_FIELDS",
            "INTRON_SCORE_OP",
            "TRANSCRIPT_SCORE_AGG",
            "SOFTMIN_TAU",
            "SEED",
            "DEVICE",
            "VISUALIZE",
            "USE_AMP",
            "AMP_DTYPE",
            "COMPILE_MODE",
            "ALLOW_TF32",
            "CUDNN_BENCHMARK",
            "DETERMINISTIC",
            "NUM_WORKERS",
            "PREFETCH_FACTOR",
            "PERSISTENT_WORKERS",
            "PIN_MEMORY",
            "MIN_BATCH_SIZE",
            "MAX_OOM_RETRIES",
        ),
        per_task_override_keys=(
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "CONV_CHANNELS",
            "KERNEL_SIZES",
            "DROPOUT",
            "FC_HIDDEN",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "VAL_FRAC",
            "GRAD_CLIP",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "FOCAL_ALPHA_POS",
            "F1_LAMBDA",
            "ASYM_GAMMA_POS",
            "ASYM_GAMMA_NEG",
            "ASYM_ALPHA_POS",
        ),
    ),
    "cnn_pair.sh": WrapperSpec(
        script_name="cnn_pair.sh",
        model_env_name="cnn_pair",
        supports_tuned_hparams=True,
        tuned_key_map={
            "batch_size": "BATCH_SIZE",
            "lr": "LR",
            "loss": "LOSS",
            "conv_channels": "CONV_CHANNELS",
            "donor_conv_channels": "DONOR_CONV_CHANNELS",
            "acceptor_conv_channels": "ACCEPTOR_CONV_CHANNELS",
            "kernel_sizes": "KERNEL_SIZES",
            "donor_kernel_sizes": "DONOR_KERNEL_SIZES",
            "acceptor_kernel_sizes": "ACCEPTOR_KERNEL_SIZES",
            "dropout": "DROPOUT",
            "fc_hidden": "FC_HIDDEN",
            "weight_decay": "WEIGHT_DECAY",
            "eta_min_ratio": "ETA_MIN_RATIO",
            "val_frac": "VAL_FRAC",
            "grad_clip": "GRAD_CLIP",
            "pos_weight_cap": "POS_WEIGHT_CAP",
            "focal_gamma": "FOCAL_GAMMA",
            "focal_alpha_pos": "FOCAL_ALPHA_POS",
            "f1_lambda": "F1_LAMBDA",
            "asym_gamma_pos": "ASYM_GAMMA_POS",
            "asym_gamma_neg": "ASYM_GAMMA_NEG",
            "asym_alpha_pos": "ASYM_ALPHA_POS",
        },
        stem_param_builder="cnn",
        required_arg_keys=(
            "SPECIES",
            "DONOR_LEN",
            "ACCEPTOR_LEN",
            "EPOCHS",
            "MAX_EPOCHS",
            "EARLY_STOP_PATIENCE",
            "EARLY_STOP_MIN_DELTA",
            "TRAIN_TARGET",
            "SEQUENCE_TRANSFORM",
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "CONV_CHANNELS",
            "KERNEL_SIZES",
            "DROPOUT",
            "FC_HIDDEN",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "GRAD_CLIP",
            "VAL_FRAC",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "F1_LAMBDA",
            "NAME_FIELDS",
            "INTRON_SCORE_OP",
            "TRANSCRIPT_SCORE_AGG",
            "SOFTMIN_TAU",
            "SEED",
            "DEVICE",
            "VISUALIZE",
            "USE_AMP",
            "AMP_DTYPE",
            "COMPILE_MODE",
            "ALLOW_TF32",
            "CUDNN_BENCHMARK",
            "DETERMINISTIC",
            "NUM_WORKERS",
            "PREFETCH_FACTOR",
            "PERSISTENT_WORKERS",
            "PIN_MEMORY",
            "MIN_BATCH_SIZE",
            "MAX_OOM_RETRIES",
        ),
        per_task_override_keys=("CONV_CHANNELS", "KERNEL_SIZES"),
    ),
    "cnn_resdil.sh": WrapperSpec(
        script_name="cnn_resdil.sh",
        model_env_name="cnn_resdil",
        supports_tuned_hparams=True,
        tuned_key_map={
            "batch_size": "BATCH_SIZE",
            "lr": "LR",
            "loss": "LOSS",
            "conv_channels": "CONV_CHANNELS",
            "kernel_sizes": "KERNEL_SIZES",
            "donor_kernel_sizes": "DONOR_KERNEL_SIZES",
            "acceptor_kernel_sizes": "ACCEPTOR_KERNEL_SIZES",
            "dropout": "DROPOUT",
            "fc_hidden": "FC_HIDDEN",
            "weight_decay": "WEIGHT_DECAY",
            "eta_min_ratio": "ETA_MIN_RATIO",
            "val_frac": "VAL_FRAC",
            "grad_clip": "GRAD_CLIP",
            "pos_weight_cap": "POS_WEIGHT_CAP",
            "focal_gamma": "FOCAL_GAMMA",
            "focal_alpha_pos": "FOCAL_ALPHA_POS",
            "asym_gamma_pos": "ASYM_GAMMA_POS",
            "asym_gamma_neg": "ASYM_GAMMA_NEG",
            "asym_alpha_pos": "ASYM_ALPHA_POS",
        },
        stem_param_builder="cnn_resdil",
        required_arg_keys=(
            "SPECIES",
            "DONOR_LEN",
            "ACCEPTOR_LEN",
            "EPOCHS",
            "MAX_EPOCHS",
            "EARLY_STOP_PATIENCE",
            "EARLY_STOP_MIN_DELTA",
            "TRAIN_TARGET",
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "CONV_CHANNELS",
            "KERNEL_SIZES",
            "DROPOUT",
            "FC_HIDDEN",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "GRAD_CLIP",
            "VAL_FRAC",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "NAME_FIELDS",
            "INTRON_SCORE_OP",
            "TRANSCRIPT_SCORE_AGG",
            "SOFTMIN_TAU",
            "SEED",
            "DEVICE",
            "VISUALIZE",
            "USE_AMP",
            "AMP_DTYPE",
            "COMPILE_MODE",
            "ALLOW_TF32",
            "CUDNN_BENCHMARK",
            "DETERMINISTIC",
            "NUM_WORKERS",
            "PREFETCH_FACTOR",
            "PERSISTENT_WORKERS",
            "PIN_MEMORY",
            "MIN_BATCH_SIZE",
            "MAX_OOM_RETRIES",
        ),
        per_task_override_keys=(
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "CONV_CHANNELS",
            "KERNEL_SIZES",
            "DROPOUT",
            "FC_HIDDEN",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "VAL_FRAC",
            "GRAD_CLIP",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "FOCAL_ALPHA_POS",
            "ASYM_GAMMA_POS",
            "ASYM_GAMMA_NEG",
            "ASYM_ALPHA_POS",
        ),
    ),
    "tcn.sh": WrapperSpec(
        script_name="tcn.sh",
        model_env_name="tcn",
        supports_tuned_hparams=True,
        tuned_key_map={
            "batch_size": "BATCH_SIZE",
            "lr": "LR",
            "loss": "LOSS",
            "conv_channels": "CONV_CHANNELS",
            "kernel_size": "KERNEL_SIZE",
            "tcn_block_repeats": "TCN_BLOCK_REPEATS",
            "tcn_causal": "TCN_CAUSAL",
            "dropout": "DROPOUT",
            "fc_hidden": "FC_HIDDEN",
            "weight_decay": "WEIGHT_DECAY",
            "eta_min_ratio": "ETA_MIN_RATIO",
            "val_frac": "VAL_FRAC",
            "grad_clip": "GRAD_CLIP",
            "pos_weight_cap": "POS_WEIGHT_CAP",
            "focal_gamma": "FOCAL_GAMMA",
            "focal_alpha_pos": "FOCAL_ALPHA_POS",
            "asym_gamma_pos": "ASYM_GAMMA_POS",
            "asym_gamma_neg": "ASYM_GAMMA_NEG",
            "asym_alpha_pos": "ASYM_ALPHA_POS",
        },
        stem_param_builder="tcn",
        required_arg_keys=(
            "SPECIES",
            "DONOR_LEN",
            "ACCEPTOR_LEN",
            "EPOCHS",
            "MAX_EPOCHS",
            "EARLY_STOP_PATIENCE",
            "EARLY_STOP_MIN_DELTA",
            "TRAIN_TARGET",
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "CONV_CHANNELS",
            "KERNEL_SIZE",
            "TCN_BLOCK_REPEATS",
            "TCN_CAUSAL",
            "DROPOUT",
            "FC_HIDDEN",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "GRAD_CLIP",
            "VAL_FRAC",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "NAME_FIELDS",
            "INTRON_SCORE_OP",
            "TRANSCRIPT_SCORE_AGG",
            "SOFTMIN_TAU",
            "SEED",
            "DEVICE",
            "VISUALIZE",
            "USE_AMP",
            "AMP_DTYPE",
            "COMPILE_MODE",
            "ALLOW_TF32",
            "CUDNN_BENCHMARK",
            "DETERMINISTIC",
            "NUM_WORKERS",
            "PREFETCH_FACTOR",
            "PERSISTENT_WORKERS",
            "PIN_MEMORY",
            "MIN_BATCH_SIZE",
            "MAX_OOM_RETRIES",
        ),
        per_task_override_keys=(
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "CONV_CHANNELS",
            "KERNEL_SIZE",
            "TCN_BLOCK_REPEATS",
            "TCN_CAUSAL",
            "DROPOUT",
            "FC_HIDDEN",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "VAL_FRAC",
            "GRAD_CLIP",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "FOCAL_ALPHA_POS",
            "ASYM_GAMMA_POS",
            "ASYM_GAMMA_NEG",
            "ASYM_ALPHA_POS",
        ),
    ),
    "bert.sh": WrapperSpec(
        script_name="bert.sh",
        model_env_name="bert",
        supports_tuned_hparams=True,
        tuned_key_map={
            "batch_size": "BATCH_SIZE",
            "lr": "LR",
            "loss": "LOSS",
            "kmer_k": "KMER_K",
            "max_tokens": "MAX_TOKENS",
            "d_model": "D_MODEL",
            "n_heads": "N_HEADS",
            "n_layers": "N_LAYERS",
            "ff_mult": "FF_MULT",
            "dropout": "DROPOUT",
            "weight_decay": "WEIGHT_DECAY",
            "eta_min_ratio": "ETA_MIN_RATIO",
            "val_frac": "VAL_FRAC",
            "grad_clip": "GRAD_CLIP",
            "pos_weight_cap": "POS_WEIGHT_CAP",
            "focal_gamma": "FOCAL_GAMMA",
            "focal_alpha_pos": "FOCAL_ALPHA_POS",
            "asym_gamma_pos": "ASYM_GAMMA_POS",
            "asym_gamma_neg": "ASYM_GAMMA_NEG",
            "asym_alpha_pos": "ASYM_ALPHA_POS",
        },
        stem_param_builder="bert",
        required_arg_keys=(
            "SPECIES",
            "DONOR_LEN",
            "ACCEPTOR_LEN",
            "EPOCHS",
            "MAX_EPOCHS",
            "EARLY_STOP_PATIENCE",
            "EARLY_STOP_MIN_DELTA",
            "TRAIN_TARGET",
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "KMER_K",
            "MAX_TOKENS",
            "D_MODEL",
            "N_HEADS",
            "N_LAYERS",
            "FF_MULT",
            "DROPOUT",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "GRAD_CLIP",
            "VAL_FRAC",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "NAME_FIELDS",
            "INTRON_SCORE_OP",
            "TRANSCRIPT_SCORE_AGG",
            "SOFTMIN_TAU",
            "SEED",
            "DEVICE",
            "VISUALIZE",
            "USE_AMP",
            "AMP_DTYPE",
            "COMPILE_MODE",
            "ALLOW_TF32",
            "CUDNN_BENCHMARK",
            "DETERMINISTIC",
            "NUM_WORKERS",
            "PREFETCH_FACTOR",
            "PERSISTENT_WORKERS",
            "PIN_MEMORY",
            "MIN_BATCH_SIZE",
            "MAX_OOM_RETRIES",
        ),
        per_task_override_keys=(
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "KMER_K",
            "MAX_TOKENS",
            "D_MODEL",
            "N_HEADS",
            "N_LAYERS",
            "FF_MULT",
            "DROPOUT",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "VAL_FRAC",
            "GRAD_CLIP",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "FOCAL_ALPHA_POS",
            "ASYM_GAMMA_POS",
            "ASYM_GAMMA_NEG",
            "ASYM_ALPHA_POS",
        ),
    ),
    "reservoir.sh": WrapperSpec(
        script_name="reservoir.sh",
        model_env_name="reservoir",
        supports_tuned_hparams=True,
        tuned_key_map={
            "batch_size": "BATCH_SIZE",
            "lr": "LR",
            "loss": "LOSS",
            "input_mode": "INPUT_MODE",
            "kmer_k": "KMER_K",
            "max_tokens": "MAX_TOKENS",
            "input_dim": "INPUT_DIM",
            "reservoir_size": "RESERVOIR_SIZE",
            "spectral_radius": "SPECTRAL_RADIUS",
            "leak": "LEAK",
            "sparsity": "SPARSITY",
            "input_scale": "INPUT_SCALE",
            "pooling": "POOLING",
            "mts_rep": "MTS_REP",
            "dimred_method": "DIMRED_METHOD",
            "n_dim": "N_DIM",
            "readout_type": "READOUT_TYPE",
            "readout_hidden": "READOUT_HIDDEN",
            "readout_dropout": "READOUT_DROPOUT",
            "washout": "WASHOUT",
            "preroll_steps": "PREROLL_STEPS",
            "read_order": "READ_ORDER",
            "weight_decay": "WEIGHT_DECAY",
            "eta_min_ratio": "ETA_MIN_RATIO",
            "val_frac": "VAL_FRAC",
            "grad_clip": "GRAD_CLIP",
            "pos_weight_cap": "POS_WEIGHT_CAP",
            "focal_gamma": "FOCAL_GAMMA",
            "focal_alpha_pos": "FOCAL_ALPHA_POS",
            "asym_gamma_pos": "ASYM_GAMMA_POS",
            "asym_gamma_neg": "ASYM_GAMMA_NEG",
            "asym_alpha_pos": "ASYM_ALPHA_POS",
        },
        stem_param_builder="reservoir",
        required_arg_keys=(
            "SPECIES",
            "DONOR_LEN",
            "ACCEPTOR_LEN",
            "EPOCHS",
            "MAX_EPOCHS",
            "EARLY_STOP_PATIENCE",
            "EARLY_STOP_MIN_DELTA",
            "TRAIN_TARGET",
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "INPUT_MODE",
            "KMER_K",
            "MAX_TOKENS",
            "INPUT_DIM",
            "RESERVOIR_SIZE",
            "SPECTRAL_RADIUS",
            "LEAK",
            "SPARSITY",
            "INPUT_SCALE",
            "POOLING",
            "MTS_REP",
            "DIMRED_METHOD",
            "READOUT_TYPE",
            "READOUT_HIDDEN",
            "READOUT_DROPOUT",
            "WASHOUT",
            "PREROLL_STEPS",
            "READ_ORDER",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "GRAD_CLIP",
            "VAL_FRAC",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "NAME_FIELDS",
            "INTRON_SCORE_OP",
            "TRANSCRIPT_SCORE_AGG",
            "SOFTMIN_TAU",
            "SEED",
            "DEVICE",
            "VISUALIZE",
            "USE_AMP",
            "AMP_DTYPE",
            "COMPILE_MODE",
            "ALLOW_TF32",
            "CUDNN_BENCHMARK",
            "DETERMINISTIC",
            "NUM_WORKERS",
            "PREFETCH_FACTOR",
            "PERSISTENT_WORKERS",
            "PIN_MEMORY",
            "MIN_BATCH_SIZE",
            "MAX_OOM_RETRIES",
        ),
        per_task_override_keys=(
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "INPUT_MODE",
            "KMER_K",
            "MAX_TOKENS",
            "INPUT_DIM",
            "RESERVOIR_SIZE",
            "SPECTRAL_RADIUS",
            "LEAK",
            "SPARSITY",
            "INPUT_SCALE",
            "POOLING",
            "MTS_REP",
            "DIMRED_METHOD",
            "N_DIM",
            "READOUT_TYPE",
            "READOUT_HIDDEN",
            "READOUT_DROPOUT",
            "WASHOUT",
            "PREROLL_STEPS",
            "READ_ORDER",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "VAL_FRAC",
            "GRAD_CLIP",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "FOCAL_ALPHA_POS",
            "ASYM_GAMMA_POS",
            "ASYM_GAMMA_NEG",
            "ASYM_ALPHA_POS",
        ),
    ),
    "dnabert.sh": WrapperSpec(
        script_name="dnabert.sh",
        model_env_name="dnabert",
        supports_tuned_hparams=True,
        tuned_key_map={
            "batch_size": "BATCH_SIZE",
            "lr": "LR",
            "loss": "LOSS",
            "max_tokens": "MAX_TOKENS",
            "dropout": "DROPOUT",
            "weight_decay": "WEIGHT_DECAY",
            "eta_min_ratio": "ETA_MIN_RATIO",
            "val_frac": "VAL_FRAC",
            "grad_clip": "GRAD_CLIP",
            "pos_weight_cap": "POS_WEIGHT_CAP",
            "focal_gamma": "FOCAL_GAMMA",
            "focal_alpha_pos": "FOCAL_ALPHA_POS",
            "asym_gamma_pos": "ASYM_GAMMA_POS",
            "asym_gamma_neg": "ASYM_GAMMA_NEG",
            "asym_alpha_pos": "ASYM_ALPHA_POS",
        },
        stem_param_builder="dnabert",
        required_arg_keys=(
            "SPECIES",
            "DONOR_LEN",
            "ACCEPTOR_LEN",
            "PRETRAINED_MODEL_NAME",
            "PRETRAINED_REVISION",
            "TRUST_REMOTE_CODE",
            "EPOCHS",
            "MAX_EPOCHS",
            "EARLY_STOP_PATIENCE",
            "EARLY_STOP_MIN_DELTA",
            "TRAIN_TARGET",
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "MAX_TOKENS",
            "DROPOUT",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "GRAD_CLIP",
            "VAL_FRAC",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "NAME_FIELDS",
            "INTRON_SCORE_OP",
            "TRANSCRIPT_SCORE_AGG",
            "SOFTMIN_TAU",
            "SEED",
            "DEVICE",
            "VISUALIZE",
            "USE_AMP",
            "AMP_DTYPE",
            "COMPILE_MODE",
            "ALLOW_TF32",
            "CUDNN_BENCHMARK",
            "DETERMINISTIC",
            "NUM_WORKERS",
            "PREFETCH_FACTOR",
            "PERSISTENT_WORKERS",
            "PIN_MEMORY",
            "MIN_BATCH_SIZE",
            "MAX_OOM_RETRIES",
        ),
        per_task_override_keys=(
            "BATCH_SIZE",
            "LR",
            "LOSS",
            "MAX_TOKENS",
            "DROPOUT",
            "WEIGHT_DECAY",
            "ETA_MIN_RATIO",
            "VAL_FRAC",
            "GRAD_CLIP",
            "POS_WEIGHT_CAP",
            "FOCAL_GAMMA",
            "FOCAL_ALPHA_POS",
            "ASYM_GAMMA_POS",
            "ASYM_GAMMA_NEG",
            "ASYM_ALPHA_POS",
        ),
    ),
}


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and run the requested wrapper."""

    parser = argparse.ArgumentParser(description="Run wrapper pipeline backend")
    parser.add_argument(
        "--script-name",
        required=True,
        choices=tuple(SPECS.keys()),
        help="Wrapper shell script name (e.g., cnn.sh).",
    )
    args = parser.parse_args(argv)

    spec = SPECS[args.script_name]
    try:
        return _run(spec)
    except ValueError as exc:
        print(f"[{spec.script_name}] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
