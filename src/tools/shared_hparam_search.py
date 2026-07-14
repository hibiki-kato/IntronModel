"""Cross-species shared hyperparameter search for cnn_v4.

A candidate is trained independently for every requested species.  Its rank is
the arithmetic mean of the same task metric across those species; no weights,
checkpoints, or best-config files are shared between species.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dev.IntronModel.src.tools import hparam_search


@dataclass(frozen=True)
class SharedSearchConfig:
    """Validated input for one task-specific cross-species search."""

    project_root: Path
    data_root: Path
    output_dir: Path
    species: tuple[str, ...]
    publish_best: bool
    task: str
    trials: int
    epochs: int
    seed: int
    objective_metric: str
    base_args: dict[str, hparam_search.ArgValue]
    search_space: dict[str, dict[str, object]]


def shared_best_config_path(data_root: Path, task: str) -> Path:
    """Return the sole best-config path for one shared cnn_v4 task."""
    normalized_task = task.strip().lower()
    if normalized_task not in {"donor", "acceptor"}:
        raise ValueError("task must be donor or acceptor.")
    return data_root / "tuning" / "cnn_v4_shared" / normalized_task / "best_config.json"


def aggregate_species_objectives(scores: Mapping[str, float]) -> float:
    """Return the unweighted mean, rejecting missing/non-finite measurements."""
    if not scores:
        raise ValueError("At least one species objective is required.")
    values = list(scores.values())
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Species objective scores must be finite.")
    return sum(values) / len(values)


def load_config(path: Path) -> SharedSearchConfig:
    """Load one shared-search JSON configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Shared search config must be a JSON object.")
    project_root = Path(str(raw.get("project_root", PROJECT_ROOT))).resolve()
    data_root = Path(str(raw["data_root"])).resolve()
    output_dir = Path(str(raw["output_dir"])).resolve()
    species_raw = raw.get("species")
    if not isinstance(species_raw, list) or not species_raw:
        raise ValueError("species must be a non-empty list.")
    species = tuple(str(value).strip() for value in species_raw)
    if any(not value for value in species) or len(set(species)) != len(species):
        raise ValueError("species entries must be non-empty and unique.")
    publish_best = raw.get("publish_best", True)
    if not isinstance(publish_best, bool):
        raise ValueError("publish_best must be a boolean.")
    task = str(raw.get("task", "")).strip().lower()
    if task not in {"donor", "acceptor"}:
        raise ValueError("task must be donor or acceptor.")
    trials = int(raw.get("trials", 0))
    epochs = int(raw.get("epochs", 0))
    if trials <= 0 or epochs <= 0:
        raise ValueError("trials and epochs must be positive integers.")
    base_args = raw.get("base_args")
    if not isinstance(base_args, dict):
        raise ValueError("base_args must be an object.")
    normalized_base_args = {str(key): value for key, value in base_args.items()}
    if normalized_base_args.get("model") != "cnn_v4":
        raise ValueError("base_args.model must be cnn_v4.")
    search_space = hparam_search._validate_search_space(raw.get("search_space"))
    objective_metric = str(raw.get("objective_metric", "pr_auc")).strip().lower()
    if objective_metric not in {"pr_auc", "roc_auc", "max_f1"}:
        raise ValueError("objective_metric must be pr_auc, roc_auc, or max_f1.")
    return SharedSearchConfig(
        project_root=project_root,
        data_root=data_root,
        output_dir=output_dir,
        species=species,
        publish_best=publish_best,
        task=task,
        trials=trials,
        epochs=epochs,
        seed=int(raw.get("seed", 1337)),
        objective_metric=objective_metric,
        base_args=normalized_base_args,
        search_space=search_space,
    )


def _run_species_trial(
    *,
    config: SharedSearchConfig,
    species: str,
    trial_id: int,
    sampled_params: dict[str, hparam_search.Scalar],
) -> dict[str, object]:
    """Run one species-local training job for a shared candidate."""
    trial_dir = config.output_dir / "trials" / f"trial_{trial_id:04d}" / species
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = trial_dir / "metrics.json"
    log_path = trial_dir / "train.log"
    args = dict(config.base_args)
    args.update(sampled_params)
    args.update(
        {
            "species": species,
            "train_target": config.task,
            "epochs": config.epochs,
            "max_epochs": config.epochs,
            "train_only": True,
            "metrics_json": str(metrics_path),
        }
    )
    command = hparam_search._build_run_model_command(config.project_root, args)
    completed = subprocess.run(
        command,
        cwd=config.project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        return {"species": species, "status": "failed", "return_code": completed.returncode,
                "log_file": str(log_path)}
    try:
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"species": species, "status": "failed", "return_code": completed.returncode,
                "log_file": str(log_path), "error": "Metrics JSON missing or invalid."}
    if not isinstance(summary, dict):
        return {"species": species, "status": "failed", "return_code": completed.returncode,
                "log_file": str(log_path), "error": "Metrics JSON must be an object."}
    score = hparam_search._extract_task_metric(summary, config.task, config.objective_metric)
    if score is None or not math.isfinite(score):
        return {"species": species, "status": "failed", "return_code": completed.returncode,
                "log_file": str(log_path), "error": "Requested task objective is missing."}
    return {
        "species": species,
        "status": "success",
        "return_code": completed.returncode,
        "objective_score": score,
        "metrics_json": str(metrics_path),
        "log_file": str(log_path),
        "checkpoint_paths": hparam_search._extract_checkpoint_paths_from_metrics(str(metrics_path)),
    }


def run_search(
    config: SharedSearchConfig,
    *,
    species_runner: Callable[..., dict[str, object]] = _run_species_trial,
) -> int:
    """Evaluate shared candidates and write exactly one task-level best config."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    best_row: dict[str, object] | None = None
    rows: list[dict[str, object]] = []
    for trial_id in range(config.trials):
        sampled_params = hparam_search._sample_trial_params_with_rng(config.search_space, rng)
        species_rows = [
            species_runner(config=config, species=species, trial_id=trial_id, sampled_params=sampled_params)
            for species in config.species
        ]
        scores = {
            str(row["species"]): float(row["objective_score"])
            for row in species_rows
            if row.get("status") == "success" and isinstance(row.get("objective_score"), (int, float))
        }
        row: dict[str, object] = {
            "trial_id": trial_id,
            "sampled_params": sampled_params,
            "species_trials": species_rows,
            "status": "success" if len(scores) == len(config.species) else "failed",
        }
        if row["status"] == "success":
            row["objective_score"] = aggregate_species_objectives(scores)
            if best_row is None or float(row["objective_score"]) > float(best_row["objective_score"]):
                best_row = row
        rows.append(row)
        print(f"[shared_hparam_search] trial={trial_id} status={row['status']} "
              f"objective={row.get('objective_score')}", flush=True)
    (config.output_dir / "trials.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    run_best_path = config.output_dir / "best_config.json"
    payload: dict[str, object] = {
        "status": "ok" if best_row is not None else "no_successful_trial",
        "model": "cnn_v4",
        "task": config.task,
        "objective_metric": config.objective_metric,
        "aggregation": "mean",
        "species": list(config.species),
        "provenance": {"search_output_dir": str(config.output_dir), "trials": config.trials,
                       "seed": config.seed, "published_to_shared_path": config.publish_best},
    }
    if best_row is not None:
        payload.update({"trial_id": best_row["trial_id"], "objective_score": best_row["objective_score"],
                        "sampled_params": best_row["sampled_params"], "species_trials": best_row["species_trials"],
                        "hparam_context": {"fixed_run_args": {"model": "cnn_v4", "train_target": config.task},
                                           "shared_across_species": list(config.species)}})
    run_best_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if config.publish_best:
        best_path = shared_best_config_path(config.data_root, config.task)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[shared_hparam_search] published_best_config={best_path}", flush=True)
    else:
        print(f"[shared_hparam_search] run_best_config={run_best_path} (not published)", flush=True)
    return 0 if best_row is not None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cnn_v4 cross-species shared HPO")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    return run_search(load_config(Path(args.config)))


if __name__ == "__main__":
    raise SystemExit(main())
