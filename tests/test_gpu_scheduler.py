from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from tools import hparam_search


def _dummy_config(tmp_path: Path) -> hparam_search.SearchConfig:
    return hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=4,
        quick_epochs=1,
        top_k=2,
        full_epochs=1,
        base_seed=1337,
        gpu_ids_setting=["0", "1"],
        max_parallel_trials_setting=2,
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space={
            "batch_size": {"type": "categorical", "values": [512]},
        },
    )


def test_run_phase_uses_all_gpu_slots_without_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _dummy_config(tmp_path)
    lock = threading.Lock()
    active: set[str] = set()
    seen: list[str] = []

    def _fake_run_trial(
        *,
        config: hparam_search.SearchConfig,
        phase: str,
        trial_id: int,
        sampled_params: dict[str, hparam_search.Scalar],
        overrides: dict[str, hparam_search.ArgValue],
        assigned_gpu_id: str | None,
        metrics_json: Path,
        log_file: Path,
    ) -> hparam_search.TrialResult:
        del config, sampled_params, overrides, metrics_json, log_file
        assert assigned_gpu_id is not None
        with lock:
            assert assigned_gpu_id not in active
            active.add(assigned_gpu_id)
            seen.append(assigned_gpu_id)
        time.sleep(0.02)
        with lock:
            active.remove(assigned_gpu_id)
        return hparam_search.TrialResult(
            phase=phase,
            trial_id=trial_id,
            status="success",
            gpu_id=assigned_gpu_id,
            sampled_params={"batch_size": 512},
            effective_batch_size=512,
            oom_retries=0,
            donor_pr_auc=0.8,
            acceptor_pr_auc=0.8,
            mean_pr_auc=0.8,
            objective_metric="mean_pr_auc",
            objective_score=0.8,
            error_message=None,
            return_code=0,
            duration_sec=0.01,
            metrics_json="x.json",
            log_file="x.log",
        )

    monkeypatch.setattr(hparam_search, "run_trial", _fake_run_trial)

    rows = hparam_search.run_phase(
        phase="quick",
        config=config,
        trial_count=4,
        trial_params=[{"batch_size": 512} for _ in range(4)],
        overrides={"epochs": 1},
        gpu_ids=["0", "1"],
        max_parallel_trials=2,
        out_dir=tmp_path / "out",
    )

    assert len(rows) == 4
    assert set(seen) == {"0", "1"}


def test_run_phase_continues_when_some_trials_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _dummy_config(tmp_path)

    def _fake_run_trial(
        *,
        config: hparam_search.SearchConfig,
        phase: str,
        trial_id: int,
        sampled_params: dict[str, hparam_search.Scalar],
        overrides: dict[str, hparam_search.ArgValue],
        assigned_gpu_id: str | None,
        metrics_json: Path,
        log_file: Path,
    ) -> hparam_search.TrialResult:
        del config, sampled_params, overrides, assigned_gpu_id, metrics_json, log_file
        status = "failed" if trial_id % 2 == 1 else "success"
        return hparam_search.TrialResult(
            phase=phase,
            trial_id=trial_id,
            status=status,
            gpu_id="0",
            sampled_params={"batch_size": 512},
            effective_batch_size=512,
            oom_retries=0,
            donor_pr_auc=0.8 if status == "success" else None,
            acceptor_pr_auc=0.8 if status == "success" else None,
            mean_pr_auc=0.8 if status == "success" else None,
            objective_metric="mean_pr_auc",
            objective_score=0.8 if status == "success" else None,
            error_message="x" if status == "failed" else None,
            return_code=1 if status == "failed" else 0,
            duration_sec=0.01,
            metrics_json="x.json",
            log_file="x.log",
        )

    monkeypatch.setattr(hparam_search, "run_trial", _fake_run_trial)
    rows = hparam_search.run_phase(
        phase="quick",
        config=config,
        trial_count=5,
        trial_params=[{"batch_size": 512} for _ in range(5)],
        overrides={"epochs": 1},
        gpu_ids=["0", "1"],
        max_parallel_trials=2,
        out_dir=tmp_path / "out",
    )

    assert len(rows) == 5
    num_failed = len([row for row in rows if row.status == "failed"])
    assert num_failed == 2
