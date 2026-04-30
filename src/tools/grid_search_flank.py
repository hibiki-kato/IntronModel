"""Grid search over model upstream/downstream flank lengths.

Enumerates all 10×10 (upstream × downstream) combinations for donor and
acceptor targets, runs full training for each cell, and writes two figures:
  <output_dir>/grid_donor_<species>.png
  <output_dir>/grid_acceptor_<species>.png

Each figure shows validation max-F1 (left). When transcript test data exists,
two additional panels show held-out site-wise max-F1 and held-out
transcript-wise max-F1, where the inactive side is scored by the latest
published checkpoint for the same model family. Otherwise the second panel
shows validation PR-AUC.
annotated heatmaps.  Results are also saved to a JSON file so re-runs can
skip already-completed cells.

Usage
-----
  python src/tools/grid_search_flank.py \\
      --species Mmus \\
      --gpus 1,2,3,4 \\
      --epochs 15 \\
      --model cnn_v2 \\
      --output_dir data/Mmus/grid_search/cnn_v2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------


def _parse_grid_axis_values(env_name: str, default_values: list[int]) -> list[int]:
    raw_value = os.environ.get(env_name, "").strip()
    if not raw_value:
        return default_values
    parsed_values: list[int] = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed_values.append(int(item))
    if not parsed_values:
        raise ValueError(f"{env_name} must contain at least one integer")
    return parsed_values


UPSTREAM_VALS: list[int] = _parse_grid_axis_values(
    "INTRONMODEL_GRID_UPSTREAM_VALS",
    list(range(10, 101, 10)),
)
DOWNSTREAM_VALS: list[int] = _parse_grid_axis_values(
    "INTRONMODEL_GRID_DOWNSTREAM_VALS",
    list(range(10, 101, 10)),
)

TARGETS: list[str] = ["donor", "acceptor"]
CELLS_PER_TARGET: int = len(UPSTREAM_VALS) * len(DOWNSTREAM_VALS)
_TRIAL_METRICS_PATTERN = re.compile(r"full_trial_(\d{4})\.metrics\.json$")


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    upstream: int
    downstream: int
    target: str
    val_max_f1: Optional[float] = None
    val_pr_auc: Optional[float] = None
    test_site_max_f1: Optional[float] = None
    test_transcript_max_f1: Optional[float] = None
    test_max_f1: Optional[float] = None
    test_pr_auc: Optional[float] = None
    duration_sec: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: build per-cell SearchConfig and run one phase
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return project root (two levels above src/)."""
    return Path(__file__).resolve().parent.parent.parent


def _format_duration(seconds: float) -> str:
    """Return ``HH:MM:SS`` string for one elapsed duration."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _has_transcript_test_tsv(root: Path, species: str) -> bool:
    """Return whether canonical transcript test TSV exists for species."""
    return (root / "data" / species / "processed" / "transcripts.unique.tsv").exists()


def _cell_has_required_metrics(
    cell: CellResult,
    *,
    model_name: str,
    has_test: bool,
) -> bool:
    """Return whether cached cell carries all metrics needed for plotting."""
    del model_name
    if cell.val_max_f1 is None:
        return False
    if has_test:
        if cell.test_site_max_f1 is None:
            return False
        if cell.test_transcript_max_f1 is None:
            return False
        return True
    if cell.val_pr_auc is None:
        return False
    return True


def _resolve_grid_objective_metric(
    *,
    model_name: str,
    target: str,
    has_test: bool,
) -> str:
    """Resolve one grid objective metric for model/target/test availability."""
    del model_name, has_test
    return f"{target}_max_f1"


def _run_grid_target(
    *,
    species: str,
    target: str,
    gpu_ids: list[str],
    max_parallel: int,
    epochs: int,
    seed: int,
    batch_size: int,
    val_frac: float,
    output_dir: Path,
    results_path: Path,
    existing: dict[str, CellResult],
    model: str,
    compile_mode: str,
    infer_compile: int,
    infer_compile_mode: str,
    pretrained_model_name: str,
    pretrained_revision: str,
    trust_remote_code: int,
    max_tokens: str,
    head_layer_norm: int,
    on_trial_complete: Optional[Callable[[object], None]] = None,
    persist_prefix_cells: Optional[list[CellResult]] = None,
) -> list[CellResult]:
    """Train all grid cells for one target; skip cells already in *existing*."""
    sys.path.insert(0, str(_project_root() / "src"))
    from tools.hparam_search import (
        SearchConfig,
        TrialResult,
        detect_gpu_ids,
        resolve_max_parallel,
        run_phase,
    )

    root = _project_root()
    pos_path = f"data/{species}/processed/site_flank100.coding.err"
    neg_path = f"data/{species}/processed/site_flank100.neg.err"

    # The objective metric controls whether test PR-AUC is computed.
    # If the test TSV exists we use test_pr_auc (includes val metrics too);
    # otherwise fall back to val-only.
    has_test = _has_transcript_test_tsv(root, species)
    objective_metric = _resolve_grid_objective_metric(
        model_name=model,
        target=target,
        has_test=has_test,
    )

    # Minimal search space — only the two active dims.
    # The inactive dims (opposite site) are still listed so existing
    # infrastructure stays happy, but they will be nulled out by
    # _normalize_independent_site_sampled_params.
    search_space = {
        "donor_upstream": {"type": "int", "min": 10, "max": 100, "step": 10},
        "donor_downstream": {"type": "int", "min": 10, "max": 100, "step": 10},
        "acceptor_upstream": {"type": "int", "min": 10, "max": 100, "step": 10},
        "acceptor_downstream": {"type": "int", "min": 10, "max": 100, "step": 10},
    }

    # Fixed training arguments (not searched).
    base_args: dict = {
        "model": model,
        "species": species,
        "train_target": target,
        "seed": seed,
        "batch_size": batch_size,
        "val_frac": val_frac,
        "train_pos_path": pos_path,
        "train_neg_path": neg_path,
        "use_amp": 1,
        "amp_dtype": "auto",
        "allow_tf32": 1,
        "cudnn_benchmark": 1,
        "deterministic": 0,
        "num_workers": "auto",
        "prefetch_factor": 4,
        "persistent_workers": 1,
        "pin_memory": 1,
        "min_batch_size": 64,
        "max_oom_retries": 5,
        "visualize": "none",
        "name_fields": "none",
        "checkpoint_top_k": 1,
        "compile_mode": compile_mode,
        "infer_compile": infer_compile,
        "infer_compile_mode": infer_compile_mode,
    }
    if model == "cnn_v2":
        base_args.update(
            {
                "input_mode": "onehot",
                "pair_mode": "independent",
                "sequence_transform": "none",
                "embedding_dim": 32,
            }
        )
    if model == "dnabert2":
        base_args.update(
            {
                "pretrained_model_name": pretrained_model_name,
                "pretrained_revision": pretrained_revision,
                "trust_remote_code": trust_remote_code,
                "max_tokens": max_tokens,
                "head_layer_norm": head_layer_norm,
            }
        )

    phase_dir = output_dir / target
    phase_dir.mkdir(parents=True, exist_ok=True)

    config = SearchConfig(
        project_root=root,
        species=species,
        output_dir=phase_dir,
        quick_trials=0,
        quick_epochs=2,
        top_k=0,
        full_epochs=epochs,
        base_seed=seed,
        gpu_ids_setting=",".join(gpu_ids),
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=5,
        max_model_params=None,
        objective_metric=objective_metric,
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args=base_args,
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
        search_algo="random",
        trial_stream_mode="auto",
        trial_process_mode="subprocess",
        skip_full_phase=False,
        enable_visualization=False,
        enable_phase_overlap=False,
        gpu_release_events_path=None,
    )

    # Build grid param list; skip already-completed cells.
    all_params: list[dict] = []
    cell_keys: list[str] = []
    retained_cells: list[CellResult] = []
    for up in UPSTREAM_VALS:
        for dn in DOWNSTREAM_VALS:
            key = f"{target}_{up}_{dn}"
            existing_cell = existing.get(key)
            if existing_cell is not None and existing_cell.status == "done":
                if _cell_has_required_metrics(
                    existing_cell,
                    model_name=model,
                    has_test=has_test,
                ):
                    retained_cells.append(existing_cell)
                    continue
            params: dict
            if target == "donor":
                params = {
                    "donor_upstream": up,
                    "donor_downstream": dn,
                    "acceptor_upstream": 50,  # placeholder; will be nulled
                    "acceptor_downstream": 50,
                }
            else:
                params = {
                    "donor_upstream": 50,  # placeholder; will be nulled
                    "donor_downstream": 50,
                    "acceptor_upstream": up,
                    "acceptor_downstream": dn,
                }
            all_params.append(params)
            cell_keys.append(key)

    if not all_params:
        print(f"[grid] {target}: all cells already complete, skipping.")
        return retained_cells

    print(
        f"[grid] {target}: running {len(all_params)} cells "
        f"({len(UPSTREAM_VALS) * len(DOWNSTREAM_VALS) - len(all_params)} cached)"
    )

    resolved_gpus = detect_gpu_ids(config.gpu_ids_setting)
    resolved_max = resolve_max_parallel(
        config.max_parallel_trials_setting, len(resolved_gpus)
    )

    def _phase_trial_callback(
        result: TrialResult,
        completed_count: int,
        trial_count: int,
    ) -> None:
        del completed_count, trial_count
        if on_trial_complete is not None:
            on_trial_complete(result)

    trial_results: list[TrialResult] = run_phase(
        phase="full",
        config=config,
        trial_count=len(all_params),
        trial_params=all_params,
        overrides={"epochs": epochs},
        gpu_ids=resolved_gpus,
        max_parallel_trials=resolved_max,
        out_dir=phase_dir,
        execution_mode="subprocess",
        on_trial_complete=_phase_trial_callback,
    )

    # Collect results in submission order; run_phase returns completion order.
    ordered_trial_results = sorted(trial_results, key=lambda result: result.trial_id)
    cells: list[CellResult] = list(retained_cells)
    prefix_cells = list(persist_prefix_cells or [])

    for key, tr in zip(cell_keys, ordered_trial_results):
        up_str, dn_str = key.split("_")[1], key.split("_")[2]
        up, dn = int(up_str), int(dn_str)
        cell = CellResult(upstream=up, downstream=dn, target=target)
        cell.duration_sec = tr.duration_sec
        if tr.status == "success":
            cell.val_pr_auc = (
                tr.donor_pr_auc if target == "donor" else tr.acceptor_pr_auc
            )
            cell.val_max_f1 = _read_val_max_f1(tr.metrics_json, target)
            if cell.val_max_f1 is None:
                cell.val_max_f1 = tr.objective_score
            if has_test:
                eval_metrics = _compute_grid_test_metrics(
                    species=species,
                    model_name=model,
                    target=target,
                    metrics_json=tr.metrics_json,
                )
                cell.test_site_max_f1 = _to_float_or_none(
                    eval_metrics.get("test_site_max_f1")
                )
                cell.test_transcript_max_f1 = _to_float_or_none(
                    eval_metrics.get("test_transcript_max_f1")
                )
                cell.test_max_f1 = cell.test_transcript_max_f1
            cell.status = "done"
        else:
            cell.status = "failed"
            cell.error = tr.error_message
        cells.append(cell)

        # Persist after each cell so partial results survive interruption.
        _save_results(prefix_cells + cells, results_path)

    return cells


def _read_val_max_f1(metrics_json: str, target: str) -> Optional[float]:
    """Extract validation max-F1 from a training metrics JSON file."""
    try:
        with open(metrics_json, encoding="utf-8") as fh:
            data = json.load(fh)
        task_data = data.get(target, {})
        if isinstance(task_data, dict):
            best_max_f1 = task_data.get("best_max_f1")
            if isinstance(best_max_f1, (int, float)):
                return float(best_max_f1)
            max_f1 = task_data.get("max_f1")
            if isinstance(max_f1, (int, float)):
                return float(max_f1)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


def _load_metrics_payload(metrics_json: str) -> dict[str, object]:
    """Read one metrics JSON payload or return an empty mapping."""
    try:
        payload = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _resolve_published_task_metrics_json(
    *,
    species: str,
    model_name: str,
    published_name: str,
    task: str,
    fallback_metrics_json: object,
) -> Optional[str]:
    """Resolve one published task's own metrics JSON from version snapshot."""
    from util.versioned_artifacts import resolve_versions_dir

    snapshot_name = published_name.strip()
    if snapshot_name != "":
        snapshot_path = (
            resolve_versions_dir(_project_root() / "data", species, model_name)
            / f"{snapshot_name}.json"
        )
        snapshot_payload = _load_metrics_payload(str(snapshot_path))
        best_configs = snapshot_payload.get("best_configs")
        if isinstance(best_configs, dict):
            task_payload = best_configs.get(task)
            if isinstance(task_payload, dict):
                metrics_json = str(task_payload.get("metrics_json", "")).strip()
                if metrics_json != "":
                    return metrics_json
    if isinstance(fallback_metrics_json, str):
        metrics_json = fallback_metrics_json.strip()
        if metrics_json != "":
            return metrics_json
    return None


def _trial_artifact_base(metrics_json: str) -> Path:
    """Return per-trial artifact base path derived from metrics JSON path."""
    metrics_path = Path(metrics_json)
    return metrics_path.parent / metrics_path.stem


def _grid_eval_metrics_path(metrics_json: str) -> Path:
    """Return sidecar JSON path used to persist held-out grid metrics."""
    return Path(str(_trial_artifact_base(metrics_json)) + ".grid_eval.json")


def _to_optional_int(value: object) -> Optional[int]:
    """Convert one scalar-like value to an optional integer."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _resolve_window_config(
    payload: dict[str, object],
) -> dict[str, Optional[int]]:
    """Extract site-window configuration from one metrics payload."""
    return {
        "donor_len": _to_optional_int(payload.get("donor_len")),
        "acceptor_len": _to_optional_int(payload.get("acceptor_len")),
        "donor_upstream": _to_optional_int(payload.get("donor_upstream")),
        "donor_downstream": _to_optional_int(payload.get("donor_downstream")),
        "acceptor_upstream": _to_optional_int(payload.get("acceptor_upstream")),
        "acceptor_downstream": _to_optional_int(payload.get("acceptor_downstream")),
    }


def _resolve_class_file(species: str) -> str:
    """Resolve transcript class file path using runtime defaults."""
    from util.data_proc import species_data_dirs

    dirs = species_data_dirs(species)
    processed_class_file = os.path.join(dirs["processed"], "transcript_class.txt")
    raw_class_file = os.path.join(dirs["raw"], "transcript_class.txt")
    return (
        processed_class_file if os.path.isfile(processed_class_file) else raw_class_file
    )


def _infer_site_rows_for_grid(
    *,
    species: str,
    model_name: str,
    donor_checkpoint_path: Path,
    acceptor_checkpoint_path: Path,
    window_config: dict[str, Optional[int]],
    batch_size: int,
    infer_compile: int,
    infer_compile_mode: str,
    sequence_transform: str,
    input_mode: str,
    pair_mode: str,
) -> list[dict[str, object]]:
    """Run one model's shared ``infer_site`` entrypoint for grid evaluation."""
    from models.registry import load_model_module

    model_module = load_model_module(model_name)
    common_args = argparse.Namespace(
        model=model_name,
        species=species,
        donor_len=window_config["donor_len"],
        acceptor_len=window_config["acceptor_len"],
        donor_upstream=window_config["donor_upstream"],
        donor_downstream=window_config["donor_downstream"],
        acceptor_upstream=window_config["acceptor_upstream"],
        acceptor_downstream=window_config["acceptor_downstream"],
        test_tsv=None,
        device="auto",
        donor_checkpoint_path=str(donor_checkpoint_path),
        acceptor_checkpoint_path=str(acceptor_checkpoint_path),
        pair_checkpoint_path="",
    )
    model_args = argparse.Namespace(
        batch_size=batch_size,
        infer_batch_size=batch_size,
        use_amp=1,
        infer_use_amp=1,
        amp_dtype="auto",
        infer_amp_dtype="auto",
        compile=0,
        compile_mode="off",
        infer_compile=infer_compile,
        infer_compile_mode=infer_compile_mode,
        sequence_transform=sequence_transform,
        input_mode=input_mode,
        pair_mode=pair_mode,
    )
    return model_module.infer_site(common_args, model_args)


def _extract_max_f1_from_eval_lines(lines: list[str]) -> Optional[float]:
    """Extract maximum F1 from ``evaluate_score_file`` output lines."""
    best_f1: Optional[float] = None
    for raw_line in lines:
        parts = raw_line.strip().split()
        if len(parts) < 6:
            continue
        try:
            f1_value = float(parts[5])
        except ValueError:
            continue
        if not np.isfinite(f1_value):
            continue
        if best_f1 is None or f1_value > best_f1:
            best_f1 = f1_value
    return best_f1


def _compute_site_max_f1_from_rows(
    *,
    rows: list[dict[str, object]],
    labels: dict[tuple[str, int], int],
) -> Optional[float]:
    """Compute site-wise held-out max-F1 from scored site rows."""
    from util.model_runtime import fallback_max_f1
    from util.transcript_eval import SCORE_SPACE_FIELD, coerce_score_to_probability

    label_values: list[int] = []
    prob_values: list[float] = []
    for row in rows:
        key = (str(row["transcript_id"]), int(row["intron_index"]))
        label = labels.get(key)
        if label is None:
            continue
        prob = coerce_score_to_probability(
            float(row["score"]),
            score_space=str(row.get(SCORE_SPACE_FIELD, "")),
        )
        label_values.append(int(label))
        prob_values.append(float(prob))
    if not label_values:
        return None
    return float(
        fallback_max_f1(
            np.asarray(label_values, dtype=np.int64),
            np.asarray(prob_values, dtype=np.float64),
        )
    )


def _compute_grid_test_metrics(
    *,
    species: str,
    model_name: str,
    target: str,
    metrics_json: str,
) -> dict[str, object]:
    """Compute held-out site/transcript metrics from one fresh grid trial."""
    from evaluate_scores import evaluate_score_file
    from run_model import (
        _expand_unique_site_rows,
        _load_optional_intron_labels,
        _load_required_unique_intron_map,
        _resolve_ref_gff_file,
    )
    from tools import hparam_search
    from util.transcript_eval import (
        aggregate_transcript_scores,
        write_transcript_scores,
    )
    from util.versioned_artifacts import resolve_published_run_assets

    payload = _load_metrics_payload(metrics_json)
    if not payload:
        return {}

    window_config = _resolve_window_config(payload)
    checkpoint_paths = hparam_search._extract_checkpoint_paths_from_metrics(
        metrics_json
    )
    checkpoint_key = f"{target}_checkpoint_path"
    checkpoint_raw = checkpoint_paths.get(checkpoint_key)
    if checkpoint_raw is None or checkpoint_raw.strip() == "":
        return {}

    transcript_max_f1: Optional[float] = None
    site_max_f1: Optional[float] = None
    partner_published_name: Optional[str] = None
    partner_task = "acceptor" if target == "donor" else "donor"
    project_root = _project_root()

    _common_infer_kwargs: dict = dict(
        species=species,
        model_name=model_name,
        batch_size=_to_optional_int(payload.get("batch_size")) or 512,
        infer_compile=_to_optional_int(payload.get("infer_compile")) or 0,
        infer_compile_mode=str(payload.get("infer_compile_mode", "off")) or "off",
        sequence_transform=str(payload.get("sequence_transform", "none")) or "none",
        input_mode=str(payload.get("input_mode", "onehot")) or "onehot",
        pair_mode=str(payload.get("pair_mode", "independent")) or "independent",
    )

    # --- Site-wise: only needs the current target's checkpoint ---
    # Mirror the current target's window dims to the partner slot so the model
    # can load (architecture is determined by upstream/downstream values).
    if Path(checkpoint_raw).is_file():
        site_window = dict(window_config)
        site_window[f"{partner_task}_len"] = window_config.get(f"{target}_len")
        site_window[f"{partner_task}_upstream"] = window_config.get(
            f"{target}_upstream"
        )
        site_window[f"{partner_task}_downstream"] = window_config.get(
            f"{target}_downstream"
        )
        current_checkpoint = Path(checkpoint_raw)
        site_rows = _infer_site_rows_for_grid(
            donor_checkpoint_path=current_checkpoint,
            acceptor_checkpoint_path=current_checkpoint,
            window_config=site_window,
            **_common_infer_kwargs,
        )
        site_max_f1 = _compute_site_max_f1_from_rows(
            rows=[
                row
                for row in site_rows
                if str(row.get("site_type", "")).strip().lower() == target
            ],
            labels=_load_optional_intron_labels(species),
        )

    # --- Transcript: partner site rows scored once from published checkpoint ---
    # Cache lives next to the target's trial directory so all 100 trials share it.
    partner_cache_path = (
        Path(metrics_json).parent.parent / f"partner_{partner_task}_site_rows.json"
    )
    partner_site_rows: Optional[list] = None

    if partner_cache_path.exists():
        try:
            partner_site_rows = json.loads(
                partner_cache_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            partner_site_rows = None

    if partner_site_rows is None:
        try:
            published_assets = resolve_published_run_assets(
                project_root=project_root,
                species=species,
                model_name=model_name,
                published_name=None,
                allow_missing_checkpoints=True,
            )
        except (FileNotFoundError, ValueError):
            published_assets = None
        if published_assets is not None:
            partner_published_name = published_assets.get("published_name")
            partner_checkpoint_raw = published_assets.get(
                f"{partner_task}_checkpoint_path"
            )
            partner_metrics_json_path = _resolve_published_task_metrics_json(
                species=species,
                model_name=model_name,
                published_name=(
                    partner_published_name
                    if isinstance(partner_published_name, str)
                    else ""
                ),
                task=partner_task,
                fallback_metrics_json=published_assets.get("metrics_json"),
            )
            if (
                isinstance(partner_checkpoint_raw, str)
                and partner_checkpoint_raw.strip() != ""
                and isinstance(partner_metrics_json_path, str)
                and partner_metrics_json_path.strip() != ""
                and Path(partner_checkpoint_raw).is_file()
            ):
                partner_payload = _load_metrics_payload(partner_metrics_json_path)
                partner_window = _resolve_window_config(partner_payload)
                # Mirror partner's own dims to both slots so the model can load.
                partner_slot_window = dict(partner_window)
                partner_slot_window[f"{target}_len"] = partner_window.get(
                    f"{partner_task}_len"
                )
                partner_slot_window[f"{target}_upstream"] = partner_window.get(
                    f"{partner_task}_upstream"
                )
                partner_slot_window[f"{target}_downstream"] = partner_window.get(
                    f"{partner_task}_downstream"
                )
                partner_checkpoint = Path(partner_checkpoint_raw)
                all_partner_rows = _infer_site_rows_for_grid(
                    donor_checkpoint_path=partner_checkpoint,
                    acceptor_checkpoint_path=partner_checkpoint,
                    window_config=partner_slot_window,
                    **_common_infer_kwargs,
                )
                partner_site_rows = [
                    row
                    for row in all_partner_rows
                    if str(row.get("site_type", "")).strip().lower() == partner_task
                ]
                try:
                    partner_cache_path.write_text(
                        json.dumps(partner_site_rows) + "\n", encoding="utf-8"
                    )
                except OSError:
                    pass

    if partner_site_rows:
        current_target_rows = [
            row
            for row in site_rows
            if str(row.get("site_type", "")).strip().lower() == target
        ]
        all_rows_for_transcript = current_target_rows + partner_site_rows
        unique_map = _load_required_unique_intron_map(species=species)
        mapped_site_rows = _expand_unique_site_rows(
            site_score_rows=all_rows_for_transcript,
            unique_map=unique_map,
        )
        transcript_rows = aggregate_transcript_scores(
            site_score_rows=mapped_site_rows,
            intron_score_op="+",
            transcript_score_agg="min",
            softmin_tau=1.0,
        )
        if transcript_rows:
            transcript_score_tsv = Path(
                str(_trial_artifact_base(metrics_json)) + ".test_transcript.tsv"
            )
            write_transcript_scores(str(transcript_score_tsv), transcript_rows)
            output_lines = evaluate_score_file(
                class_file=_resolve_class_file(species),
                score_file=transcript_score_tsv,
                ref_gff=_resolve_ref_gff_file(species, None),
            )
            eval_output_path = Path(
                str(_trial_artifact_base(metrics_json)) + ".test_transcript.eval.txt"
            )
            eval_output_path.write_text(
                ("\n".join(output_lines) + "\n") if output_lines else "",
                encoding="utf-8",
            )
            transcript_max_f1 = _extract_max_f1_from_eval_lines(output_lines)

    out: dict[str, object] = {
        "test_site_max_f1": site_max_f1,
        "test_transcript_max_f1": transcript_max_f1,
    }
    if partner_published_name is not None:
        out["partner_published_name"] = partner_published_name
    _grid_eval_metrics_path(metrics_json).write_text(
        json.dumps(out, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def _delete_trial_checkpoints(metrics_json: str) -> int:
    """Delete checkpoint files referenced by one trial metrics JSON."""
    sys.path.insert(0, str(_project_root() / "src"))
    from util.checkpoint_io import extract_checkpoint_paths, read_json_object

    metrics_path = Path(metrics_json)
    payload = read_json_object(metrics_path)
    if payload is None:
        return 0

    deleted_count = 0
    checkpoint_paths = extract_checkpoint_paths(
        payload,
        base_dir=metrics_path.parent,
        existing_only=True,
    )
    for checkpoint_path in checkpoint_paths.values():
        try:
            checkpoint_path.unlink()
        except FileNotFoundError:
            continue
        deleted_count += 1
    return deleted_count


def _cleanup_grid_target_checkpoints(target_dir: Path) -> int:
    """Delete all trial checkpoints referenced under one target directory."""
    if not target_dir.exists():
        return 0

    deleted_count = 0
    for metrics_path in sorted(target_dir.glob("full_trial_*.metrics.json")):
        if _TRIAL_METRICS_PATTERN.search(metrics_path.name) is None:
            continue
        deleted_count += _delete_trial_checkpoints(str(metrics_path))
    return deleted_count


def _cleanup_grid_checkpoints(output_dir: Path, targets: list[str]) -> int:
    """Delete cached trial checkpoints for selected targets in one grid run."""
    deleted_count = 0
    for target in targets:
        target_deleted = _cleanup_grid_target_checkpoints(output_dir / target)
        print(
            f"[grid] cleanup target={target} deleted_checkpoints={target_deleted}",
            flush=True,
        )
        deleted_count += target_deleted
    return deleted_count


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _results_to_json(cells: list[CellResult]) -> list[dict]:
    return [c.__dict__ for c in cells]


def _save_results(cells: list[CellResult], path: Path) -> None:
    path.write_text(
        json.dumps(_results_to_json(cells), indent=2) + "\n", encoding="utf-8"
    )


def _load_results(path: Path) -> dict[str, CellResult]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, CellResult] = {}
        for r in rows:
            c = CellResult(**r)
            key = f"{c.target}_{c.upstream}_{c.downstream}"
            out[key] = c
        return out
    except (json.JSONDecodeError, TypeError):
        return {}


def _to_float_or_none(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_cells_from_trial_metrics(
    *,
    output_dir: Path,
    target: str,
    has_test: bool,
    model_name: str,
) -> list[CellResult]:
    """Recover target cells from per-trial metrics when summary JSON is stale."""
    del model_name
    target_dir = output_dir / target
    if not target_dir.exists():
        return []

    recovered: dict[str, CellResult] = {}
    for metrics_path in sorted(target_dir.glob("full_trial_*.metrics.json")):
        if metrics_path.name.endswith(".metrics.test_pr_auc.json"):
            continue
        if _TRIAL_METRICS_PATTERN.search(metrics_path.name) is None:
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue

        upstream_raw = payload.get(f"{target}_upstream")
        downstream_raw = payload.get(f"{target}_downstream")
        if not isinstance(upstream_raw, int) or not isinstance(downstream_raw, int):
            continue

        task_payload = payload.get(target)
        if not isinstance(task_payload, dict):
            continue

        val_max_f1 = _to_float_or_none(task_payload.get("best_max_f1"))
        if val_max_f1 is None:
            val_max_f1 = _to_float_or_none(task_payload.get("max_f1"))
        val_pr_auc = _to_float_or_none(task_payload.get("best_pr_auc"))
        if val_pr_auc is None:
            val_pr_auc = _to_float_or_none(task_payload.get("pr_auc"))

        test_site_max_f1: Optional[float] = None
        test_transcript_max_f1: Optional[float] = None
        if has_test:
            eval_payload = _load_metrics_payload(
                str(_grid_eval_metrics_path(str(metrics_path)))
            )
            test_site_max_f1 = _to_float_or_none(eval_payload.get("test_site_max_f1"))
            test_transcript_max_f1 = _to_float_or_none(
                eval_payload.get("test_transcript_max_f1")
            )

        cell = CellResult(
            upstream=upstream_raw,
            downstream=downstream_raw,
            target=target,
            val_max_f1=val_max_f1,
            val_pr_auc=val_pr_auc,
            test_site_max_f1=test_site_max_f1,
            test_transcript_max_f1=test_transcript_max_f1,
            test_max_f1=test_transcript_max_f1,
            status="done",
        )
        recovered[f"{target}_{upstream_raw}_{downstream_raw}"] = cell

    return list(recovered.values())


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _build_grid(
    cells: list[CellResult],
    target: str,
    metric_attr: str,
) -> np.ndarray:
    """Return 10×10 grid array (rows=downstream, cols=upstream)."""
    grid = np.full((len(DOWNSTREAM_VALS), len(UPSTREAM_VALS)), np.nan)
    for cell in cells:
        if cell.target != target:
            continue
        try:
            row = DOWNSTREAM_VALS.index(cell.downstream)
            col = UPSTREAM_VALS.index(cell.upstream)
        except ValueError:
            continue
        val = getattr(cell, metric_attr)
        if val is not None:
            grid[row, col] = float(val)
    return grid


def _plot_heatmap(
    ax: plt.Axes,
    grid: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xlabel: str = "Upstream flank length",
    ylabel: str = "Downstream flank length",
) -> None:
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        ax.set_title(title + "\n(no data)")
        return

    lo = float(finite.min()) if vmin is None else vmin
    hi = float(finite.max()) if vmax is None else vmax
    if lo == hi:
        lo = max(0.0, lo - 0.01)
        hi = hi + 0.01

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )

    # Annotate each cell.
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = grid[r, c]
            if not np.isfinite(v):
                text = "—"
                color = "gray"
            else:
                text = f"{v:.3f}"
                # White text on dark background, black on light.
                norm_v = (v - lo) / (hi - lo) if hi > lo else 0.5
                color = "white" if norm_v < 0.6 else "black"
            ax.text(c, r, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(len(UPSTREAM_VALS)))
    ax.set_xticklabels(UPSTREAM_VALS, fontsize=8)
    ax.set_yticks(range(len(DOWNSTREAM_VALS)))
    ax.set_yticklabels(DOWNSTREAM_VALS, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=7)


def plot_grid(
    cells: list[CellResult],
    target: str,
    species: str,
    model_name: str,
    output_path: Path,
    has_test: bool,
) -> None:
    """Generate and save the heatmap figure for one target."""
    if has_test:
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"{model_name}  |  species={species}  |  target={target}",
        fontsize=12,
        fontweight="bold",
    )

    # Determine metric attr for val.
    val_attr = "val_max_f1"
    val_title = "Validation max-F1"

    val_grid = _build_grid(cells, target, val_attr)
    _plot_heatmap(axes[0], val_grid, val_title, cmap="plasma")

    if has_test:
        site_grid = _build_grid(cells, target, "test_site_max_f1")
        _plot_heatmap(axes[1], site_grid, "Test site-wise max-F1", cmap="viridis")
        transcript_grid = _build_grid(cells, target, "test_transcript_max_f1")
        _plot_heatmap(
            axes[2],
            transcript_grid,
            "Test transcript-wise max-F1",
            cmap="cividis",
        )
    else:
        val_pr_grid = _build_grid(cells, target, "val_pr_auc")
        _plot_heatmap(axes[1], val_pr_grid, "Validation PR-AUC", cmap="viridis")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] saved figure → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid search over model upstream/downstream window lengths."
    )
    p.add_argument("--species", required=True, help="Species name, e.g. Mmus")
    p.add_argument(
        "--model",
        default="cnn_v2",
        help="Model name passed to run_model.py (default: cnn_v2).",
    )
    p.add_argument(
        "--target",
        choices=["donor", "acceptor", "both"],
        default="both",
        help="Which target to grid-search (default: both).",
    )
    p.add_argument(
        "--gpus",
        default="auto",
        help="Comma-separated GPU IDs or 'auto' (default: auto).",
    )
    p.add_argument("--epochs", type=int, default=15, help="Training epochs per cell.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument(
        "--pretrained_model_name",
        default="",
        help="Optional DNABERT pretrained model id or local path.",
    )
    p.add_argument(
        "--pretrained_revision",
        default="",
        help="Optional DNABERT pretrained revision.",
    )
    p.add_argument(
        "--trust_remote_code",
        type=int,
        choices=[0, 1],
        default=1,
        help="Forward DNABERT trust_remote_code flag.",
    )
    p.add_argument(
        "--max_tokens",
        default="auto",
        help="Optional DNABERT max_tokens value.",
    )
    p.add_argument(
        "--head_layer_norm",
        type=int,
        choices=[0, 1],
        default=1,
        help="Optional DNABERT head LayerNorm flag.",
    )
    p.add_argument(
        "--compile_mode",
        choices=["off", "on", "auto", "quick", "full"],
        default="quick",
        help="Training compile mode (default: quick = reduce-overhead).",
    )
    p.add_argument(
        "--infer_compile",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable compiled inference/transcript scoring when set to 1.",
    )
    p.add_argument(
        "--infer_compile_mode",
        choices=["off", "on", "auto", "quick", "full"],
        default="quick",
        help="Inference compile mode (default: quick = reduce-overhead only).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: data/<species>/grid_search/<model>).",
    )
    p.add_argument(
        "--figures_only",
        action="store_true",
        help="Skip training; regenerate figures from cached results JSON only.",
    )
    p.add_argument(
        "--cleanup_only",
        action="store_true",
        help="Skip training/plotting; delete cached trial checkpoints only.",
    )
    p.add_argument("--global_trial_offset", type=int, default=0)
    p.add_argument("--global_trial_total", type=int, default=0)
    args = p.parse_args(argv)
    if args.figures_only and args.cleanup_only:
        p.error("--figures_only and --cleanup_only cannot be combined.")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    sys.path.insert(0, str(_project_root() / "src"))
    from util.process_title import (
        apply_process_title,
        apply_process_title_from_env,
        format_eta_process_title,
    )

    apply_process_title_from_env() or apply_process_title("ETA:--/-- --:--")

    # Subprocesses (training trials) inherit this env var, disabling their
    # per-trial ETA process title so only the global ETA is visible.
    os.environ.setdefault("INTRONMODEL_DISABLE_ETA_PROCESS_TITLE", "1")

    root = _project_root()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "data" / args.species / "grid_search" / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = TARGETS if args.target == "both" else [args.target]
    if args.cleanup_only:
        deleted_count = _cleanup_grid_checkpoints(output_dir, targets)
        print(
            f"[grid] cleanup species={args.species} deleted_checkpoints={deleted_count}",
            flush=True,
        )
        return 0

    has_test = _has_transcript_test_tsv(root, args.species)

    # GPU list
    gpu_list: list[str]
    if args.gpus == "auto":
        gpu_list = []  # detect_gpu_ids will handle it inside _run_grid_target
    else:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]

    results_path = output_dir / f"results_{args.species}.json"
    existing = _load_results(results_path)
    cached_done = sum(
        1
        for cell in existing.values()
        if cell.target in targets and cell.status == "done"
    )
    species_started_at = time.perf_counter()
    live_completed = 0
    global_trial_offset = max(0, int(args.global_trial_offset))
    local_trial_total = len(targets) * CELLS_PER_TARGET
    global_trial_total = int(args.global_trial_total)
    if global_trial_total <= 0:
        global_trial_total = global_trial_offset + local_trial_total

    def _emit_eta_progress(
        *,
        target_name: Optional[str] = None,
        result: Optional[object] = None,
    ) -> None:
        completed_total = min(
            global_trial_total,
            global_trial_offset + cached_done + live_completed,
        )
        remaining_total = max(0, global_trial_total - completed_total)
        elapsed_sec = time.perf_counter() - species_started_at
        eta_text = "unknown"
        if remaining_total == 0:
            eta_text = "00:00:00"
            apply_process_title(f"grid {args.species} done")
        elif live_completed > 0 and elapsed_sec > 0.0:
            remaining_secs = (elapsed_sec / live_completed) * remaining_total
            eta_text = _format_duration(remaining_secs)
            apply_process_title(
                f"grid {args.species} {format_eta_process_title(remaining_secs)}"
            )
        else:
            if not apply_process_title_from_env():
                apply_process_title(f"grid {args.species} ETA:--/-- --:--")

        detail = ""
        if target_name is not None and result is not None:
            sampled_params = getattr(result, "sampled_params", {})
            if isinstance(sampled_params, dict):
                prefix = target_name
                upstream = sampled_params.get(f"{prefix}_upstream")
                downstream = sampled_params.get(f"{prefix}_downstream")
                if upstream is not None and downstream is not None:
                    detail += f" target={target_name} window={upstream}/{downstream}"
            status = getattr(result, "status", None)
            if isinstance(status, str) and status:
                detail += f" status={status}"
            duration_sec = getattr(result, "duration_sec", None)
            if isinstance(duration_sec, (int, float)):
                detail += f" trial={_format_duration(float(duration_sec))}"
        print(
            f"[grid_eta] species={args.species} "
            f"global={completed_total}/{global_trial_total} "
            f"cached={cached_done} live={live_completed} "
            f"elapsed={_format_duration(elapsed_sec)} eta={eta_text}{detail}",
            flush=True,
        )

    all_cells: list[CellResult] = []

    def _merge_unique_cells(cells: list[CellResult]) -> list[CellResult]:
        merged: dict[str, CellResult] = {}
        for cell in cells:
            key = f"{cell.target}_{cell.upstream}_{cell.downstream}"
            merged[key] = cell
        return list(merged.values())

    if not args.figures_only:
        _emit_eta_progress()

    for target in targets:
        existing_target = {k: v for k, v in existing.items() if v.target == target}
        recovered_target_cells: list[CellResult] = []
        needs_recovery = len(existing_target) < CELLS_PER_TARGET or any(
            not _cell_has_required_metrics(
                cell,
                model_name=args.model,
                has_test=has_test,
            )
            for cell in existing_target.values()
        )
        if needs_recovery:
            recovered_target_cells = _load_cells_from_trial_metrics(
                output_dir=output_dir,
                target=target,
                has_test=has_test,
                model_name=args.model,
            )
            for cell in recovered_target_cells:
                key = f"{cell.target}_{cell.upstream}_{cell.downstream}"
                current = existing_target.get(key)
                if current is None or not _cell_has_required_metrics(
                    current,
                    model_name=args.model,
                    has_test=has_test,
                ):
                    if current is None or current.status != "done":
                        cached_done += 1
                    existing_target[key] = cell
                    existing[key] = cell
            if recovered_target_cells:
                print(
                    f"[grid] recovered {len(recovered_target_cells)} cells from "
                    f"trial metrics for target={target}.",
                    flush=True,
                )
        target_cells: list[CellResult] = list(existing_target.values())
        target_error: Optional[Exception] = None
        if args.figures_only:
            pass
        else:

            def _handle_trial_complete(
                result: object, target_name: str = target
            ) -> None:
                nonlocal live_completed
                live_completed += 1
                _emit_eta_progress(target_name=target_name, result=result)

            try:
                target_cells = _run_grid_target(
                    species=args.species,
                    target=target,
                    gpu_ids=gpu_list,
                    max_parallel=len(gpu_list) if gpu_list else 0,
                    epochs=args.epochs,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    val_frac=args.val_frac,
                    output_dir=output_dir,
                    results_path=results_path,
                    existing=existing_target,
                    model=args.model,
                    compile_mode=args.compile_mode,
                    infer_compile=args.infer_compile,
                    infer_compile_mode=args.infer_compile_mode,
                    pretrained_model_name=args.pretrained_model_name,
                    pretrained_revision=args.pretrained_revision,
                    trust_remote_code=args.trust_remote_code,
                    max_tokens=args.max_tokens,
                    head_layer_norm=args.head_layer_norm,
                    on_trial_complete=_handle_trial_complete,
                    persist_prefix_cells=_merge_unique_cells(
                        [
                            *all_cells,
                            *[v for v in existing.values() if v.target != target],
                        ]
                    ),
                )
            except Exception as exc:
                target_error = exc
                target_cells = _load_cells_from_trial_metrics(
                    output_dir=output_dir,
                    target=target,
                    has_test=has_test,
                    model_name=args.model,
                )
        all_cells.extend(target_cells)

        if target_cells:
            fig_path = output_dir / f"grid_{target}_{args.species}.png"
            plot_grid(
                cells=target_cells,
                target=target,
                species=args.species,
                model_name=args.model,
                output_path=fig_path,
                has_test=has_test,
            )
        if target_error is not None:
            raise target_error

    preserved_non_targets = [v for v in existing.values() if v.target not in targets]
    _save_results(
        _merge_unique_cells([*preserved_non_targets, *all_cells]), results_path
    )
    if not args.figures_only:
        _emit_eta_progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
