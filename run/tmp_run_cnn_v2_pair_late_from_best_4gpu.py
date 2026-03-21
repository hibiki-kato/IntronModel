#!/usr/bin/env python3
"""Run cnn_v2_pair late jobs with cnn_v2 best architectures on four GPUs.

This temporary launcher compares late-fusion pair models against the best
donor and acceptor architectures found by ``cnn_v2``. The script is fully
configured in-place so it can be executed directly on a 4-GPU server without
additional command-line arguments.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Iterator


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
PYTHON_BIN: Path = Path("/home/hibiki/miniforge3/envs/intronmodel/bin/python")
SPECIES: tuple[str, ...] = ("Athal", "Dmel", "Hsap", "Mmus")
GPUS: tuple[str, ...] = ("4", "5", "6", "7")
LOG_DIR: Path = PROJECT_ROOT / "run" / "tmp_cnn_v2_pair_late_logs"


def _load_json(path: Path) -> dict[str, object]:
    """Load one JSON object from disk.

    Parameters
    ----------
    path:
        Path to a JSON file.

    Returns
    -------
    dict[str, object]
        Parsed JSON object.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_text(value: object) -> str:
    """Convert one scalar value to shell-friendly text."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def _as_list_text(value: object) -> str:
    """Convert a scalar or sequence architecture value to comma text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ",".join(_as_text(item) for item in value)
    return _as_text(value)


def _require_sampled_params(path: Path) -> dict[str, object]:
    """Load and validate a best-config ``sampled_params`` mapping."""
    payload = _load_json(path)
    status = str(payload.get("status", "")).strip().lower()
    if status != "ok":
        raise ValueError(f"Expected status='ok' in {path}")
    sampled_params = payload.get("sampled_params")
    if not isinstance(sampled_params, dict):
        raise ValueError(f"Missing sampled_params in {path}")
    return sampled_params


def _build_run_args(project_root: Path, species: str) -> tuple[list[str], dict[str, str]]:
    """Build one ``run_model.py`` command for a single species."""
    tuning_root = project_root / "data" / species / "tuning"
    donor_path = tuning_root / "cnn_v2" / "donor" / "best_config.json"
    acceptor_path = tuning_root / "cnn_v2" / "acceptor" / "best_config.json"
    pair_path = tuning_root / "cnn_v2_pair" / "pair" / "best_config.json"

    donor_sampled = _require_sampled_params(donor_path)
    acceptor_sampled = _require_sampled_params(acceptor_path)
    pair_sampled = _require_sampled_params(pair_path)

    donor_conv_channels = _as_list_text(donor_sampled.get("conv_channels"))
    donor_kernel_sizes = _as_list_text(donor_sampled.get("kernel_sizes"))
    acceptor_conv_channels = _as_list_text(acceptor_sampled.get("conv_channels"))
    acceptor_kernel_sizes = _as_list_text(acceptor_sampled.get("kernel_sizes"))

    command: list[str] = [
        "--model",
        "cnn_v2_pair",
        "--species",
        species,
        "--donor_len",
        _as_text(pair_sampled.get("donor_len", 100)),
        "--acceptor_len",
        _as_text(pair_sampled.get("acceptor_len", 100)),
        "--device",
        "auto",
        "--seed",
        "1337",
        "--name_fields",
        "none",
        "--epochs",
        "10",
        "--max_epochs",
        "200",
        "--early_stop_patience",
        "12",
        "--early_stop_min_delta",
        "0.0",
        "--train_target",
        "pair",
        "--sequence_transform",
        _as_text(pair_sampled.get("sequence_transform", "none")),
        "--batch_size",
        _as_text(pair_sampled.get("batch_size", 512)),
        "--lr",
        _as_text(pair_sampled.get("lr", 5e-4)),
        "--loss",
        _as_text(pair_sampled.get("loss", "focal")),
        "--input_mode",
        _as_text(pair_sampled.get("input_mode", "onehot")),
        "--pair_mode",
        "pair",
        "--fusion_mode",
        "late",
        "--embedding_dim",
        _as_text(pair_sampled.get("embedding_dim", 32)),
        "--bpe_pretrained_model_name",
        "zhihan1996/DNABERT-2-117M",
        "--bpe_trust_remote_code",
        "0",
        "--conv_channels",
        "",
        "--kernel_sizes",
        "",
        "--donor_conv_channels",
        donor_conv_channels,
        "--acceptor_conv_channels",
        acceptor_conv_channels,
        "--donor_kernel_sizes",
        donor_kernel_sizes,
        "--acceptor_kernel_sizes",
        acceptor_kernel_sizes,
        "--max_pool_size",
        _as_text(pair_sampled.get("max_pool_size", 2)),
        "--conv_stride",
        _as_text(pair_sampled.get("conv_stride", 1)),
        "--head_type",
        _as_text(pair_sampled.get("head_type", "gap")),
        "--fc_hidden",
        _as_text(pair_sampled.get("fc_hidden", 128)),
        "--dropout",
        _as_text(pair_sampled.get("dropout", 0.3)),
        "--weight_decay",
        _as_text(pair_sampled.get("weight_decay", 0.01)),
        "--eta_min_ratio",
        "0.01",
        "--val_frac",
        "0.2",
        "--grad_clip",
        "5.0",
        "--pos_weight_cap",
        "20.0",
        "--focal_gamma",
        "2.0",
        "--f1_lambda",
        _as_text(pair_sampled.get("f1_lambda", 0.1)),
        "--use_amp",
        "1",
        "--amp_dtype",
        "auto",
        "--compile_mode",
        "off",
        "--allow_tf32",
        "1",
        "--cudnn_benchmark",
        "1",
        "--deterministic",
        "0",
        "--num_workers",
        "auto",
        "--prefetch_factor",
        "4",
        "--persistent_workers",
        "1",
        "--pin_memory",
        "1",
        "--min_batch_size",
        "64",
        "--max_oom_retries",
        "8",
        "--transcript_score_agg",
        "min",
        "--softmin_tau",
        "1.0",
        "--tag",
        "late_from_cnn_v2_best_4gpu",
    ]

    summary = {
        "species": species,
        "donor_path": str(donor_path),
        "acceptor_path": str(acceptor_path),
        "pair_path": str(pair_path),
        "donor_conv_channels": donor_conv_channels,
        "acceptor_conv_channels": acceptor_conv_channels,
        "donor_kernel_sizes": donor_kernel_sizes,
        "acceptor_kernel_sizes": acceptor_kernel_sizes,
    }
    return command, summary


def _launch_process(
    *,
    project_root: Path,
    gpu_id: str,
    command: list[str],
    log_path: Path,
) -> subprocess.Popen[str]:
    """Launch one species run on one GPU and stream output into a log file."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONPATH"] = str(project_root / "src")
    log_file = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                str(PYTHON_BIN),
                str(project_root / "src" / "run_model.py"),
                *command,
            ],
            cwd=project_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        log_file.close()
        raise
    log_file.close()
    return process


def _iter_pairs(items: list[str], gpus: list[str]) -> Iterator[tuple[str, str]]:
    """Pair species and GPU ids one-to-one."""
    if len(items) > len(gpus):
        raise ValueError("Need at least as many GPUs as species.")
    return iter(zip(items, gpus, strict=True))


def main() -> int:
    """Launch all comparison jobs and wait for completion."""
    species_list = list(SPECIES)
    gpu_list = list(GPUS)
    if len(species_list) != len(gpu_list):
        raise ValueError("Species and GPU lists must have the same length.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_log_dir = LOG_DIR / f"late_from_best_4gpu_{stamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    manifest_runs: dict[str, dict[str, str]] = {}
    processes: list[tuple[str, str, subprocess.Popen[str], Path]] = []

    print(
        "[late_from_best_4gpu] launching species jobs:",
        ", ".join(
            f"{species}@GPU{gpu}" for species, gpu in _iter_pairs(species_list, gpu_list)
        ),
    )

    for species, gpu_id in zip(species_list, gpu_list, strict=True):
        command, summary = _build_run_args(PROJECT_ROOT, species)
        log_path = run_log_dir / f"{species}.log"
        manifest_runs[species] = {
            "gpu_id": gpu_id,
            **summary,
            "log_path": str(log_path),
        }
        print(
            f"[late_from_best_4gpu] start species={species} gpu={gpu_id} "
            f"log={log_path}"
        )
        process = _launch_process(
            project_root=PROJECT_ROOT,
            gpu_id=gpu_id,
            command=command,
            log_path=log_path,
        )
        processes.append((species, gpu_id, process, log_path))

    manifest_path = run_log_dir / "manifest.json"
    manifest = {
        "project_root": str(PROJECT_ROOT),
        "python_bin": str(PYTHON_BIN),
        "species": species_list,
        "gpus": gpu_list,
        "runs": manifest_runs,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[late_from_best_4gpu] manifest={manifest_path}")

    failure: tuple[str, int] | None = None
    while processes:
        finished_index: int | None = None
        finished_species = ""
        finished_gpu = ""
        finished_log_path = Path()
        finished_rc = 0
        for index, (species, gpu_id, process, log_path) in enumerate(processes):
            return_code = process.poll()
            if return_code is None:
                continue
            finished_index = index
            finished_species = species
            finished_gpu = gpu_id
            finished_log_path = log_path
            finished_rc = int(return_code)
            break
        if finished_index is not None:
            processes.pop(finished_index)
            if finished_rc != 0:
                print(
                    f"[late_from_best_4gpu] failed species={finished_species} "
                    f"gpu={finished_gpu} rc={finished_rc} "
                    f"log={finished_log_path}"
                )
                failure = (finished_species, finished_rc)
            else:
                print(
                    f"[late_from_best_4gpu] done species={finished_species} "
                    f"gpu={finished_gpu} log={finished_log_path}"
                )
        if failure is not None:
            break
        if processes:
            time.sleep(30.0)

    if failure is not None:
        for species, gpu_id, process, _log_path in processes:
            if process.poll() is None:
                process.terminate()
                print(
                    f"[late_from_best_4gpu] terminating species={species} "
                    f"gpu={gpu_id}"
                )
        return 1

    print("[late_from_best_4gpu] all species completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
