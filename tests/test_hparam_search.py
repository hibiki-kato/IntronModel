from __future__ import annotations

import math
import json
import os
import signal
import time
from pathlib import Path
from queue import Queue
from typing import Optional, cast

import pytest

import util.model_runtime as model_runtime
from tools import hparam_search


def _base_config_dict(tmp_path: Path) -> dict[str, object]:
    return {
        "project_root": str(tmp_path),
        "species": "Dmel",
        "output_dir": str(tmp_path / "out"),
        "quick_trials": 4,
        "quick_epochs": 2,
        "top_k": 2,
        "full_epochs": 4,
        "base_seed": 1337,
        "gpu_ids": "auto",
        "max_parallel_trials": "auto",
        "min_batch_size": 64,
        "max_oom_retries": 2,
        "objective_metric": "mean_pr_auc",
        "base_args": {
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 512,
            "epochs": 1,
        },
        "quick_overrides": {"compile_mode": "off"},
        "full_overrides": {"compile_mode": "auto"},
        "search_space": {
            "lr": {"type": "float", "min": 1e-5, "max": 1e-3, "scale": "log"},
            "batch_size": {"type": "categorical", "values": [512, 1024]},
            "kernel_size": {"type": "int", "min": 5, "max": 9, "step": 2},
        },
    }


def test_load_config_rejects_invalid_quick_trials(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    config["quick_trials"] = 0
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="quick_trials"):
        _ = hparam_search.load_config(config_path)


def test_load_config_requires_base_args_model(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    base_args = dict(config["base_args"])
    del base_args["model"]
    config["base_args"] = base_args
    config_path = tmp_path / "bad_model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="base_args.model"):
        _ = hparam_search.load_config(config_path)


def test_load_config_accepts_history_guided_settings(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    config["search_algo"] = "history_guided"
    config["history_top_n"] = 42
    config["guided_random_fraction"] = 0.2
    config["guided_mutation_rate"] = 0.4
    config_path = tmp_path / "guided.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    assert loaded.search_algo == "history_guided"
    assert loaded.history_top_n == 42
    assert loaded.guided_random_fraction == pytest.approx(0.2)
    assert loaded.guided_mutation_rate == pytest.approx(0.4)


def test_load_config_injects_site_window_len_defaults(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    base_args = dict(config["base_args"])
    _ = base_args.pop("donor_len", None)
    _ = base_args.pop("acceptor_len", None)
    config["base_args"] = base_args
    config_path = tmp_path / "site_window_defaults.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    assert loaded.base_args["donor_len"] == 100
    assert loaded.base_args["acceptor_len"] == 100
    assert loaded.search_space["donor_len"] == {
        "type": "int",
        "min": 40,
        "max": 100,
        "step": 10,
    }
    assert loaded.search_space["acceptor_len"] == {
        "type": "int",
        "min": 40,
        "max": 100,
        "step": 10,
    }


@pytest.mark.parametrize(
    ("train_target", "expected_keys"),
    [
        ("donor", {"donor_len"}),
        ("acceptor", {"acceptor_len"}),
    ],
)
def test_load_config_injects_only_target_window_len_for_single_target_tuning(
    tmp_path: Path,
    train_target: str,
    expected_keys: set[str],
) -> None:
    config = _base_config_dict(tmp_path)
    base_args = dict(config["base_args"])
    base_args["train_target"] = train_target
    config["base_args"] = base_args
    config_path = tmp_path / f"single_target_{train_target}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    actual_keys = {key for key in loaded.search_space if key.endswith("_len")}
    assert actual_keys == expected_keys


@pytest.mark.parametrize(
    "objective_metric",
    [
        "pair_pr_auc",
        "donor_roc_auc",
        "mean_max_f1",
        "test_pr_auc",
        "test_max_f1",
    ],
)
def test_load_config_accepts_supported_objective_metrics(
    tmp_path: Path,
    objective_metric: str,
) -> None:
    config = _base_config_dict(tmp_path)
    config["objective_metric"] = objective_metric
    config_path = tmp_path / "pair_metric.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    assert loaded.objective_metric == objective_metric


def test_load_config_rejects_invalid_search_algo(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    config["search_algo"] = "surrogate"
    config_path = tmp_path / "bad_algo.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="search_algo"):
        _ = hparam_search.load_config(config_path)


def test_derive_validation_protocol_uses_test_split_for_test_objective() -> None:
    protocol = hparam_search._derive_validation_protocol_from_args(
        merged_args={
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 512,
        },
        objective_metric="test_max_f1",
    )

    assert protocol["split_type"] == "test_transcript_eval"


def test_derive_validation_protocol_marks_cnn_v2_pair_as_pair_mode() -> None:
    protocol = hparam_search._derive_validation_protocol_from_args(
        merged_args={
            "model": "cnn_v2_pair",
            "species": "Dmel",
            "batch_size": 512,
            "train_target": "pair",
        },
        objective_metric="pair_pr_auc",
    )

    assert protocol["include_pair_mixed_negatives"] is True


def test_combine_donor_acceptor_rows_adds_log10_scores() -> None:
    rows = [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "donor",
            "score": math.log10(0.25),
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": math.log10(0.5),
        },
        {
            "transcript_id": "tx2",
            "intron_index": 1,
            "site_type": "donor",
            "score": math.log10(0.1),
        },
    ]

    combined = hparam_search._combine_donor_acceptor_rows(rows)

    assert len(combined) == 1
    assert combined[0]["transcript_id"] == "tx1"
    assert combined[0]["intron_index"] == 1
    assert combined[0]["site_type"] == "pair"
    assert combined[0]["score"] == pytest.approx(math.log10(0.125))


def test_load_config_accepts_trial_process_mode(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    config["trial_process_mode"] = "persistent_quick"
    config_path = tmp_path / "trial_process_mode.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    assert loaded.trial_process_mode == "persistent_quick"


def test_resolve_hparam_auto_num_workers_caps_at_eight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_runtime.os, "cpu_count", lambda: 64)

    assert hparam_search._resolve_hparam_auto_num_workers(1) == 8
    assert hparam_search._resolve_hparam_auto_num_workers(4) == 4


def test_load_config_rejects_invalid_trial_process_mode(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    config["trial_process_mode"] = "persistent"
    config_path = tmp_path / "bad_trial_process_mode.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="trial_process_mode"):
        _ = hparam_search.load_config(config_path)


def test_load_config_parses_skip_full_and_visualization_flags(
    tmp_path: Path,
) -> None:
    config = _base_config_dict(tmp_path)
    config["skip_full_phase"] = 1
    config["enable_visualization"] = "off"
    config_path = tmp_path / "flags.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    assert loaded.skip_full_phase is True
    assert loaded.enable_visualization is False


def test_load_config_parses_phase_overlap_flag(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    config["enable_phase_overlap"] = True
    config_path = tmp_path / "phase_overlap.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    assert loaded.enable_phase_overlap is True


def test_load_config_disables_visualization_when_base_visualize_is_none(
    tmp_path: Path,
) -> None:
    config = _base_config_dict(tmp_path)
    base_args = dict(config["base_args"])
    base_args["visualize"] = "none"
    config["base_args"] = base_args
    config_path = tmp_path / "visualize_none.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    assert loaded.enable_visualization is False


def test_load_config_explicit_visualization_flag_overrides_base_visualize(
    tmp_path: Path,
) -> None:
    config = _base_config_dict(tmp_path)
    base_args = dict(config["base_args"])
    base_args["visualize"] = "none"
    config["base_args"] = base_args
    config["enable_visualization"] = "on"
    config_path = tmp_path / "visualize_override.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    assert loaded.enable_visualization is True


def test_prewarm_persistent_trial_worker_calls_model_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=2,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "dnabert",
            "species": "Dmel",
            "batch_size": 16,
            "pretrained_model_name": "dummy-model",
        },
        quick_overrides={},
        full_overrides={},
        search_space={
            "batch_size": {"type": "categorical", "values": [16]},
        },
    )
    captured: dict[str, object] = {}

    class _FakeModule:
        def add_train_args(self, parser: object) -> None:
            del parser

        def add_infer_args(self, parser: object) -> None:
            del parser

        def train(self, common_args: object, model_args: object) -> dict[str, object]:
            del common_args, model_args
            return {}

        def infer_site(
            self,
            common_args: object,
            model_args: object,
        ) -> list[dict[str, object]]:
            del common_args, model_args
            return []

        def prewarm_persistent_worker(
            self,
            base_args: dict[str, object],
            assigned_gpu_id: str | None,
        ) -> None:
            captured["base_args"] = base_args
            captured["assigned_gpu_id"] = assigned_gpu_id

    import models.registry as registry

    monkeypatch.setattr(
        registry,
        "load_model_module",
        lambda model_name: _FakeModule(),
    )

    hparam_search._prewarm_persistent_trial_worker(
        config=config,
        assigned_gpu_id="1",
    )

    assert captured["assigned_gpu_id"] == "1"
    assert cast(dict[str, object], captured["base_args"])["model"] == "dnabert"


def test_persistent_worker_sets_visible_gpu_before_prewarm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "dnabert",
            "species": "Dmel",
            "batch_size": 16,
        },
        quick_overrides={},
        full_overrides={},
        search_space={
            "batch_size": {"type": "categorical", "values": [16]},
        },
    )
    captured_env: dict[str, str] = {}

    def _fake_prewarm(
        *,
        config: hparam_search.SearchConfig,
        assigned_gpu_id: str | None,
    ) -> None:
        del config, assigned_gpu_id
        captured_env["prewarm"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    def _fake_run_trial_inprocess(
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
        del overrides
        captured_env["trial"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return hparam_search.TrialResult(
            phase=phase,
            trial_id=trial_id,
            status="success",
            gpu_id=assigned_gpu_id,
            sampled_params=sampled_params,
            effective_batch_size=16,
            oom_retries=0,
            donor_pr_auc=0.6,
            acceptor_pr_auc=0.7,
            mean_pr_auc=0.65,
            objective_metric=config.objective_metric,
            objective_score=0.65,
            error_message=None,
            return_code=0,
            duration_sec=0.01,
            metrics_json=str(metrics_json),
            log_file=str(log_file),
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        hparam_search,
        "_prewarm_persistent_trial_worker",
        _fake_prewarm,
    )
    monkeypatch.setattr(
        hparam_search,
        "run_trial_inprocess",
        _fake_run_trial_inprocess,
    )

    task_queue: Queue[object] = Queue()
    result_queue: Queue[object] = Queue()
    task_queue.put(
        hparam_search.PersistentTrialTask(
            trial_id=0,
            sampled_params={"batch_size": 16},
            metrics_json=str(tmp_path / "metrics.json"),
            log_file=str(tmp_path / "trial.log"),
        )
    )
    task_queue.put(None)

    hparam_search._persistent_trial_worker_main(
        slot_index=0,
        assigned_gpu_id="6",
        config=config,
        phase="quick",
        overrides={},
        stream_mode="silent",
        max_parallel_trials=1,
        task_queue=task_queue,
        result_queue=result_queue,
    )

    outcome = result_queue.get_nowait()
    assert isinstance(outcome, hparam_search.PersistentTrialOutcome)
    assert captured_env["prewarm"] == "6"
    assert captured_env["trial"] == "6"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"


@pytest.mark.parametrize(
    ("mode", "phase", "expected"),
    [
        ("subprocess", "quick", "subprocess"),
        ("subprocess", "full", "subprocess"),
        ("persistent_quick", "quick", "persistent"),
        ("persistent_quick", "full", "subprocess"),
        ("persistent_all", "quick", "persistent"),
        ("persistent_all", "full", "persistent"),
    ],
)
def test_resolve_phase_execution_mode(
    mode: str,
    phase: str,
    expected: str,
) -> None:
    resolved = hparam_search._resolve_phase_execution_mode(
        process_mode=mode,
        phase=phase,
    )

    assert resolved == expected


@pytest.mark.parametrize(
    ("phase_execution_mode", "trial_count", "max_parallel_trials", "expected"),
    [
        ("subprocess", 1, 8, "subprocess"),
        ("persistent", 1, 8, "subprocess"),
        ("persistent", 8, 8, "subprocess"),
        ("persistent", 9, 8, "persistent"),
    ],
)
def test_resolve_workload_execution_mode(
    phase_execution_mode: str,
    trial_count: int,
    max_parallel_trials: int,
    expected: str,
) -> None:
    resolved = hparam_search._resolve_workload_execution_mode(
        phase_execution_mode=phase_execution_mode,
        trial_count=trial_count,
        max_parallel_trials=max_parallel_trials,
    )

    assert resolved == expected


def test_select_locked_quick_trials_returns_conservative_prefix() -> None:
    rows = [
        hparam_search.TrialResult(
            phase="quick",
            trial_id=0,
            status="success",
            gpu_id=None,
            sampled_params={"batch_size": 128},
            effective_batch_size=128,
            oom_retries=0,
            donor_pr_auc=0.6,
            acceptor_pr_auc=0.6,
            mean_pr_auc=0.6,
            objective_metric="mean_pr_auc",
            objective_score=0.82,
            error_message=None,
            return_code=0,
            duration_sec=0.01,
            metrics_json="",
            log_file="",
        ),
        hparam_search.TrialResult(
            phase="quick",
            trial_id=1,
            status="success",
            gpu_id=None,
            sampled_params={"batch_size": 256},
            effective_batch_size=256,
            oom_retries=0,
            donor_pr_auc=0.7,
            acceptor_pr_auc=0.7,
            mean_pr_auc=0.7,
            objective_metric="mean_pr_auc",
            objective_score=0.95,
            error_message=None,
            return_code=0,
            duration_sec=0.01,
            metrics_json="",
            log_file="",
        ),
        hparam_search.TrialResult(
            phase="quick",
            trial_id=2,
            status="success",
            gpu_id=None,
            sampled_params={"batch_size": 512},
            effective_batch_size=512,
            oom_retries=0,
            donor_pr_auc=0.65,
            acceptor_pr_auc=0.65,
            mean_pr_auc=0.65,
            objective_metric="mean_pr_auc",
            objective_score=0.88,
            error_message=None,
            return_code=0,
            duration_sec=0.01,
            metrics_json="",
            log_file="",
        ),
    ]

    selected = hparam_search._select_locked_quick_trials(
        completed_quick_rows=rows,
        unfinished_quick_count=1,
        top_k=3,
    )

    assert [row.trial_id for row in selected] == [1, 2]


def test_sample_trial_params_is_deterministic(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    space = hparam_search._validate_search_space(config["search_space"])
    first = hparam_search.sample_trial_params(space, seed=2026)
    second = hparam_search.sample_trial_params(space, seed=2026)
    assert first == second


def test_build_run_model_command_skips_architecture_helper_keys(
    tmp_path: Path,
) -> None:
    cmd = hparam_search._build_run_model_command(
        tmp_path,
        {
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 256,
            "conv_channels": "64,128,256",
            "kernel_sizes": "3,5,7",
            "conv_depth": 3,
            "channel_candidates": "64,128,256",
            "kernel_candidates": "3,5,7",
            "channel_order": "nondecreasing",
            "kernel_order": "nonincreasing",
            "conv_stride_candidates": "1,2",
            "max_pool_candidates": "1,2,3",
        },
    )

    assert "--conv_channels" in cmd
    assert "--kernel_sizes" in cmd
    assert "--conv_depth" not in cmd
    assert "--channel_candidates" not in cmd
    assert "--kernel_candidates" not in cmd
    assert "--channel_order" not in cmd
    assert "--kernel_order" not in cmd
    assert "--conv_stride_candidates" not in cmd
    assert "--max_pool_candidates" not in cmd


def test_build_run_model_command_skips_mask_helper_key(tmp_path: Path) -> None:
    cmd = hparam_search._build_run_model_command(
        tmp_path,
        {
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 256,
            "mask": "on",
            "sequence_transform": "none",
        },
    )

    assert "--mask" not in cmd
    assert cmd.count("--sequence_transform") == 1


def test_extract_sampled_params_from_best_config_converts_legacy_sequence_transform_value(
) -> None:
    params = hparam_search._extract_sampled_params_from_best_config(
        raw={
            "sampled_params": {
                "batch_size": 128,
                "sequence_transform": "truncate_outside_intron",
            }
        },
        search_space={
            "batch_size": {
                "type": "categorical",
                "values": [128],
            },
            "mask": {
                "type": "categorical",
                "values": ["off", "on"],
            },
        },
        base_args={},
    )

    assert params == {"batch_size": 128, "mask": "on"}


def test_extract_sampled_params_from_best_config_falls_back_to_fixed_run_args(
) -> None:
    params = hparam_search._extract_sampled_params_from_best_config(
        raw={
            "sampled_params": {
                "batch_size": 128,
                "lr": 2e-4,
            },
            "hparam_context": {
                "fixed_run_args": {
                    "input_mode": "onehot",
                }
            },
        },
        search_space={
            "batch_size": {
                "type": "categorical",
                "values": [128],
            },
            "input_mode": {
                "type": "categorical",
                "values": ["onehot", "kmer3", "bpe"],
            },
            "lr": {
                "type": "float",
                "min": 1e-5,
                "max": 1e-3,
                "scale": "log",
            },
        },
        base_args={
            "input_mode": "bpe",
        },
    )

    assert params == {
        "batch_size": 128,
        "input_mode": "onehot",
        "lr": pytest.approx(2e-4),
    }


def test_run_trial_translates_mask_to_sequence_transform(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_dict = _base_config_dict(tmp_path)
    config_dict["base_args"] = {
        "model": "cnn",
        "species": "Dmel",
        "batch_size": 128,
        "epochs": 1,
    }
    config_dict["search_space"] = {
        "batch_size": {
            "type": "categorical",
            "values": [128],
        },
        "mask": {
            "type": "categorical",
            "values": ["off", "on"],
        },
    }
    config_path.write_text(json.dumps(config_dict), encoding="utf-8")
    config = hparam_search.load_config(config_path)

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "donor": {
                    "best_metric": "pr_auc",
                    "best_score": 0.8,
                },
                "acceptor": {
                    "best_metric": "pr_auc",
                    "best_score": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "trial.log"
    captured_cmd: dict[str, list[str]] = {}

    def _fake_command_runner(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        captured_cmd["cmd"] = cmd
        return 0, ""

    result = hparam_search._run_trial_with_command_runner(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 128, "mask": "on"},
        overrides={},
        assigned_gpu_id=None,
        metrics_json=metrics_path,
        log_file=log_path,
        command_runner=_fake_command_runner,
    )

    assert result.status == "success"
    assert result.sampled_params["mask"] == "on"
    assert "--mask" not in captured_cmd["cmd"]
    assert "--sequence_transform" in captured_cmd["cmd"]
    assert "mask_outside_intron_n" in captured_cmd["cmd"]


def test_run_trial_drops_mask_for_independent_cnn_v2(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_dict = _base_config_dict(tmp_path)
    config_dict["base_args"] = {
        "model": "cnn_v2",
        "species": "Hsap",
        "pair_mode": "independent",
        "train_target": "both",
        "batch_size": 128,
        "epochs": 1,
    }
    config_dict["search_space"] = {
        "batch_size": {
            "type": "categorical",
            "values": [128],
        },
        "mask": {
            "type": "categorical",
            "values": ["off", "on"],
        },
    }
    config_path.write_text(json.dumps(config_dict), encoding="utf-8")
    config = hparam_search.load_config(config_path)

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "donor": {
                    "best_metric": "pr_auc",
                    "best_score": 0.8,
                },
                "acceptor": {
                    "best_metric": "pr_auc",
                    "best_score": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "trial.log"
    captured_cmd: dict[str, list[str]] = {}

    def _fake_command_runner(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        captured_cmd["cmd"] = cmd
        return 0, ""

    result = hparam_search._run_trial_with_command_runner(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 128, "mask": "on"},
        overrides={},
        assigned_gpu_id=None,
        metrics_json=metrics_path,
        log_file=log_path,
        command_runner=_fake_command_runner,
    )

    assert result.status == "success"
    assert "mask" not in result.sampled_params
    assert result.sampled_params["batch_size"] == 128
    assert "--mask" not in captured_cmd["cmd"]
    assert "--sequence_transform" in captured_cmd["cmd"]
    assert "none" in captured_cmd["cmd"]


def test_rank_successful_trials_prefers_high_mean_pr_auc() -> None:
    rows = [
        hparam_search.TrialResult(
            phase="quick",
            trial_id=0,
            status="success",
            gpu_id="0",
            sampled_params={"lr": 1e-4},
            effective_batch_size=512,
            oom_retries=0,
            donor_pr_auc=0.81,
            acceptor_pr_auc=0.83,
            mean_pr_auc=0.82,
            objective_metric="mean_pr_auc",
            objective_score=0.82,
            error_message=None,
            return_code=0,
            duration_sec=1.0,
            metrics_json="a.json",
            log_file="a.log",
        ),
        hparam_search.TrialResult(
            phase="quick",
            trial_id=1,
            status="success",
            gpu_id="1",
            sampled_params={"lr": 2e-4},
            effective_batch_size=1024,
            oom_retries=0,
            donor_pr_auc=0.85,
            acceptor_pr_auc=0.87,
            mean_pr_auc=0.86,
            objective_metric="mean_pr_auc",
            objective_score=0.86,
            error_message=None,
            return_code=0,
            duration_sec=1.0,
            metrics_json="b.json",
            log_file="b.log",
        ),
    ]
    ranked = hparam_search.rank_successful_trials(rows)
    assert ranked[0].trial_id == 1


def test_resolve_max_parallel_accepts_numeric_string() -> None:
    resolved = hparam_search.resolve_max_parallel("3", gpu_count=8)
    assert resolved == 3


def test_resolve_max_parallel_auto_uses_gpu_count() -> None:
    resolved = hparam_search.resolve_max_parallel("auto", gpu_count=4)
    assert resolved == 4


def test_resolve_max_parallel_auto_falls_back_to_one_without_gpus() -> None:
    resolved = hparam_search.resolve_max_parallel("auto", gpu_count=0)
    assert resolved == 1


def test_resolve_trial_num_workers_auto_is_parallel_aware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.hparam_search.os.cpu_count", lambda: 128)
    previous_parallel = hparam_search._set_active_max_parallel_trials(8)
    try:
        resolved = hparam_search._resolve_trial_num_workers("auto")
    finally:
        _ = hparam_search._set_active_max_parallel_trials(previous_parallel)
    assert resolved == 4


def test_resolve_trial_num_workers_auto_has_floor_on_large_multi_gpu_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.hparam_search.os.cpu_count", lambda: 64)
    previous_parallel = hparam_search._set_active_max_parallel_trials(8)
    try:
        resolved = hparam_search._resolve_trial_num_workers("auto")
    finally:
        _ = hparam_search._set_active_max_parallel_trials(previous_parallel)
    assert resolved == 4


def test_resolve_trial_num_workers_auto_is_capped_by_cpu_per_parallel_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.hparam_search.os.cpu_count", lambda: 64)
    previous_parallel = hparam_search._set_active_max_parallel_trials(32)
    try:
        resolved = hparam_search._resolve_trial_num_workers("auto")
    finally:
        _ = hparam_search._set_active_max_parallel_trials(previous_parallel)
    assert resolved == 2


def test_run_trial_rewrites_auto_num_workers_to_effective_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 512,
            "num_workers": "auto",
        },
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )

    captured_num_workers: list[str] = []
    captured_report_train_metrics: list[str] = []

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        for index, token in enumerate(cmd):
            if token == "--num_workers":
                captured_num_workers.append(cmd[index + 1])
                break
        for index, token in enumerate(cmd):
            if token == "--report_train_metrics":
                captured_report_train_metrics.append(cmd[index + 1])
                break
        metrics_path: Optional[Path] = None
        for index, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[index + 1])
                break
        assert metrics_path is not None
        payload = {
            "donor": {"best_pr_auc": 0.8},
            "acceptor": {"best_pr_auc": 0.7},
        }
        metrics_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )
    monkeypatch.setattr("tools.hparam_search.os.cpu_count", lambda: 128)

    previous_parallel = hparam_search._set_active_max_parallel_trials(8)
    try:
        _ = hparam_search.run_trial(
            config=config,
            phase="quick",
            trial_id=0,
            sampled_params={"batch_size": 512, "lr": 1e-4},
            overrides={"epochs": 1},
            assigned_gpu_id="0",
            metrics_json=tmp_path / "metrics.json",
            log_file=tmp_path / "trial.log",
        )
    finally:
        _ = hparam_search._set_active_max_parallel_trials(previous_parallel)

    assert captured_num_workers == ["4"]
    assert captured_report_train_metrics == ["0"]


def test_run_trial_preserves_explicit_report_train_metrics_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 512,
            "report_train_metrics": 1,
        },
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )

    captured_report_train_metrics: list[str] = []

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        for index, token in enumerate(cmd):
            if token == "--report_train_metrics":
                captured_report_train_metrics.append(cmd[index + 1])
                break
        metrics_path: Optional[Path] = None
        for index, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[index + 1])
                break
        assert metrics_path is not None
        payload = {
            "donor": {"best_pr_auc": 0.8},
            "acceptor": {"best_pr_auc": 0.7},
        }
        metrics_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )

    _ = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id="0",
        metrics_json=tmp_path / "metrics.json",
        log_file=tmp_path / "trial.log",
    )

    assert captured_report_train_metrics == ["1"]


def test_find_cuda_header_supports_conda_targets_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conda_prefix = tmp_path / "env"
    cuda_header = conda_prefix / "targets" / "x86_64-linux" / "include" / "cuda.h"
    cuda_header.parent.mkdir(parents=True, exist_ok=True)
    cuda_header.write_text("", encoding="utf-8")
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    monkeypatch.delenv("TRITON_PTXAS_BLACKWELL_PATH", raising=False)
    monkeypatch.setattr("tools.hparam_search.shutil.which", lambda _name: None)

    detected = hparam_search._find_cuda_header()

    assert detected == cuda_header.resolve()


def test_run_trial_oom_backoff_then_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=4,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )

    calls: list[int] = []

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        calls.append(1)
        metrics_path: Optional[Path] = None
        for idx, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[idx + 1])
                break
        assert metrics_path is not None
        if len(calls) == 1:
            return 1, "CUDA out of memory"
        metrics_path.write_text(
            json.dumps(
                {
                    "donor": {"best_pr_auc": 0.81},
                    "acceptor": {"best_pr_auc": 0.79},
                }
            ),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )

    result = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id="0",
        metrics_json=tmp_path / "metrics.json",
        log_file=tmp_path / "trial.log",
    )

    assert result.status == "success"
    assert result.oom_retries == 1
    assert result.effective_batch_size == 256
    assert result.mean_pr_auc == pytest.approx(0.80)


def test_run_trial_succeeds_with_single_task_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="donor_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        metrics_path: Optional[Path] = None
        for idx, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[idx + 1])
                break
        assert metrics_path is not None
        metrics_path.write_text(
            json.dumps({"donor": {"best_pr_auc": 0.82}}),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )

    result = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id=None,
        metrics_json=tmp_path / "metrics.json",
        log_file=tmp_path / "trial.log",
    )

    assert result.status == "success"
    assert result.objective_metric == "donor_pr_auc"
    assert result.objective_score == pytest.approx(0.82)
    assert result.mean_pr_auc is None


def test_run_trial_succeeds_with_pair_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="pair_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn_pair", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        metrics_path: Optional[Path] = None
        for idx, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[idx + 1])
                break
        assert metrics_path is not None
        metrics_path.write_text(
            json.dumps({"pair": {"best_pr_auc": 0.88}}),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )

    result = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id=None,
        metrics_json=tmp_path / "metrics_pair.json",
        log_file=tmp_path / "trial_pair.log",
    )

    assert result.status == "success"
    assert result.objective_metric == "pair_pr_auc"
    assert result.objective_score == pytest.approx(0.88)


def test_run_trial_succeeds_with_roc_auc_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="donor_roc_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        metrics_path: Optional[Path] = None
        for idx, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[idx + 1])
                break
        assert metrics_path is not None
        metrics_path.write_text(
            json.dumps({"donor": {"best_roc_auc": 0.91}}),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )

    result = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id=None,
        metrics_json=tmp_path / "metrics_roc.json",
        log_file=tmp_path / "trial_roc.log",
    )

    assert result.status == "success"
    assert result.objective_metric == "donor_roc_auc"
    assert result.objective_score == pytest.approx(0.91)


def test_run_trial_succeeds_with_max_f1_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="pair_max_f1",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn_pair", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        metrics_path: Optional[Path] = None
        for idx, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[idx + 1])
                break
        assert metrics_path is not None
        metrics_path.write_text(
            json.dumps({"pair": {"best_max_f1": 0.77}}),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )

    result = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id=None,
        metrics_json=tmp_path / "metrics_max_f1.json",
        log_file=tmp_path / "trial_max_f1.log",
    )

    assert result.status == "success"
    assert result.objective_metric == "pair_max_f1"
    assert result.objective_score == pytest.approx(0.77)


def test_run_trial_succeeds_with_test_max_f1_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="test_max_f1",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )
    captured_train_only: list[bool] = []
    captured_eval_output: list[str] = []

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        captured_train_only.append("--train_only" in cmd)
        metrics_path: Optional[Path] = None
        eval_output_path: Optional[Path] = None
        for idx, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[idx + 1])
            if token == "--eval_output_txt":
                eval_output_path = Path(cmd[idx + 1])
                captured_eval_output.append(cmd[idx + 1])
        assert metrics_path is not None
        assert eval_output_path is not None
        metrics_path.write_text(
            json.dumps(
                {
                    "donor": {"best_pr_auc": 0.66},
                    "acceptor": {"best_pr_auc": 0.61},
                }
            ),
            encoding="utf-8",
        )
        eval_output_path.write_text(
            "tx1 0.1 = 15.0 20.0 17.1\n"
            "tx2 0.2 = 18.0 22.0 19.8\n",
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )

    result = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id=None,
        metrics_json=tmp_path / "metrics_test_max_f1.json",
        log_file=tmp_path / "trial_test_max_f1.log",
    )

    assert result.status == "success"
    assert result.objective_metric == "test_max_f1"
    assert result.objective_score == pytest.approx(19.8)
    assert captured_train_only == [False]
    assert captured_eval_output


def test_run_trial_succeeds_with_test_pr_auc_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="test_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn",
            "species": "Dmel",
            "train_target": "donor",
            "batch_size": 512,
        },
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )
    captured_train_only: list[bool] = []

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        captured_train_only.append("--train_only" in cmd)
        metrics_path: Optional[Path] = None
        for idx, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[idx + 1])
                break
        assert metrics_path is not None
        metrics_path.write_text(
            json.dumps(
                {
                    "donor": {"best_pr_auc": 0.66},
                    "donor_checkpoint_path": str(tmp_path / "donor.pt"),
                    "acceptor_checkpoint_path": str(tmp_path / "acceptor.pt"),
                }
            ),
            encoding="utf-8",
        )
        return 0, "ok"

    def _fake_compute_test_pr_auc_objective(
        *,
        config: hparam_search.SearchConfig,
        merged_args: dict[str, hparam_search.ArgValue],
        metrics_json: Path,
        trial_artifact_base: Path,
    ) -> Optional[float]:
        del config, metrics_json, trial_artifact_base
        assert merged_args["train_only"] is True
        return 0.731

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )
    monkeypatch.setattr(
        hparam_search,
        "_compute_test_pr_auc_objective",
        _fake_compute_test_pr_auc_objective,
    )

    result = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id=None,
        metrics_json=tmp_path / "metrics_test_pr_auc.json",
        log_file=tmp_path / "trial_test_pr_auc.log",
    )

    assert result.status == "success"
    assert result.objective_metric == "test_pr_auc"
    assert result.objective_score == pytest.approx(0.731)
    assert captured_train_only == [True]


def test_run_trial_ignores_architecture_helper_keys_in_base_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="donor_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 512,
            "conv_depth": 3,
            "channel_candidates": "64,128,256",
            "kernel_candidates": "3,5,7",
            "conv_stride_candidates": "1,2",
            "max_pool_candidates": "1,2,3",
        },
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        assert "--conv_depth" not in cmd
        assert "--channel_candidates" not in cmd
        assert "--kernel_candidates" not in cmd
        assert "--conv_stride_candidates" not in cmd
        assert "--max_pool_candidates" not in cmd
        metrics_path: Optional[Path] = None
        for idx, token in enumerate(cmd):
            if token == "--metrics_json":
                metrics_path = Path(cmd[idx + 1])
                break
        assert metrics_path is not None
        metrics_path.write_text(
            json.dumps({"donor": {"best_pr_auc": 0.77}}),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )

    result = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id=None,
        metrics_json=tmp_path / "metrics_filtered.json",
        log_file=tmp_path / "trial_filtered.log",
    )

    assert result.status == "success"
    assert result.objective_score == pytest.approx(0.77)


def test_load_global_best_params_accepts_valid_file(tmp_path: Path) -> None:
    base_config = _base_config_dict(tmp_path)
    search_space = hparam_search._validate_search_space(base_config["search_space"])
    path = tmp_path / "best_config.json"
    payload = {
        "status": "ok",
        "mean_pr_auc": 0.8,
        "sampled_params": {
            "batch_size": 1024,
            "kernel_size": 7,
            "lr": 2e-4,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = hparam_search.load_global_best_params(
        path=path,
        search_space=search_space,
        base_args=base_config["base_args"],
    )

    assert loaded == payload["sampled_params"]


def test_load_global_best_params_rejects_out_of_range_value(
    tmp_path: Path,
) -> None:
    base_config = _base_config_dict(tmp_path)
    search_space = hparam_search._validate_search_space(base_config["search_space"])
    path = tmp_path / "best_config.json"
    payload = {
        "status": "ok",
        "sampled_params": {
            "batch_size": 2048,
            "kernel_size": 7,
            "lr": 2e-4,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="search space"):
        _ = hparam_search.load_global_best_params(
            path=path,
            search_space=search_space,
            base_args=base_config["base_args"],
        )


def test_load_global_best_params_ignores_validation_signature(
    tmp_path: Path,
) -> None:
    base_config = _base_config_dict(tmp_path)
    search_space = hparam_search._validate_search_space(base_config["search_space"])
    path = tmp_path / "best_config.json"
    payload = {
        "status": "ok",
        "validation_signature": "abcd1234ef56",
        "sampled_params": {
            "batch_size": 1024,
            "kernel_size": 7,
            "lr": 2e-4,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = hparam_search.load_global_best_params(
        path=path,
        search_space=search_space,
        base_args=base_config["base_args"],
    )
    assert loaded == payload["sampled_params"]


def test_load_global_best_params_keeps_sampled_params_for_missing_keys(
    tmp_path: Path,
) -> None:
    base_config = _base_config_dict(tmp_path)
    base_config["base_args"]["kernel_size"] = 7
    search_space = hparam_search._validate_search_space(base_config["search_space"])
    path = tmp_path / "best_config.json"
    payload = {
        "status": "ok",
        "sampled_params": {
            "batch_size": 1024,
            "lr": 2e-4,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = hparam_search.load_global_best_params(
        path=path,
        search_space=search_space,
        base_args=base_config["base_args"],
    )

    assert loaded is not None
    assert loaded["batch_size"] == 1024
    assert loaded["lr"] == pytest.approx(2e-4)
    assert "kernel_size" not in loaded


def test_write_best_config_includes_validation_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "best_config.json"
    donor_ckpt = tmp_path / "donor.pt"
    donor_ckpt.write_bytes(b"donor")
    acceptor_ckpt = tmp_path / "acceptor.pt"
    acceptor_ckpt.write_bytes(b"acceptor")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(donor_ckpt),
                "acceptor_checkpoint_path": str(acceptor_ckpt),
            }
        ),
        encoding="utf-8",
    )
    row = hparam_search.TrialResult(
        phase="full",
        trial_id=3,
        status="success",
        gpu_id="0",
        sampled_params={"batch_size": 512},
        effective_batch_size=512,
        oom_retries=0,
        donor_pr_auc=0.82,
        acceptor_pr_auc=0.81,
        mean_pr_auc=0.815,
        objective_metric="mean_pr_auc",
        objective_score=0.815,
        error_message=None,
        return_code=0,
        duration_sec=1.0,
        metrics_json=str(metrics_path),
        log_file="trial.log",
        validation_signature="feedbeefcafe",
        validation_protocol={
            "split_type": "stratified_site",
            "val_frac": 0.1,
            "seed": 1337,
            "train_source": {
                "train_pos_path": "data/Dmel/train/100bp.err",
                "train_neg_path": "data/Dmel/train/100bp.neg.err",
            },
            "metric_primary": "mean_pr_auc",
        },
        selection_score=0.815,
    )

    hparam_search.write_best_config(output_path, row)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["validation_protocol"]["split_type"] == "stratified_site"
    assert float(payload["selection_score"]) == pytest.approx(0.815)
    assert payload["hparam_context"] is None
    assert payload["objective_best_epoch"] is None
    assert payload["donor_checkpoint_path"] == str(donor_ckpt)
    assert payload["acceptor_checkpoint_path"] == str(acceptor_ckpt)


def test_write_best_config_includes_hparam_context_and_objective_best_epoch(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "best_config.json"
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "donor": {
                    "best_pr_auc": 0.81,
                    "best_epoch": 6,
                }
            }
        ),
        encoding="utf-8",
    )
    row = hparam_search.TrialResult(
        phase="full",
        trial_id=1,
        status="success",
        gpu_id="0",
        sampled_params={"batch_size": 512},
        effective_batch_size=512,
        oom_retries=0,
        donor_pr_auc=0.81,
        acceptor_pr_auc=None,
        mean_pr_auc=None,
        objective_metric="donor_pr_auc",
        objective_score=0.81,
        error_message=None,
        return_code=0,
        duration_sec=1.0,
        metrics_json=str(metrics_path),
        log_file="trial.log",
    )
    hparam_context = {
        "version": 2,
        "objective_metric": "donor_pr_auc",
        "full_epochs": 10,
        "validation_protocol": {"split_type": "stratified_site"},
    }

    hparam_search.write_best_config(
        output_path,
        row,
        hparam_context=hparam_context,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["hparam_context"] == hparam_context
    assert payload["objective_best_epoch"] == 6


def test_maybe_update_global_best_logs_score_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "best_config.json"
    row = hparam_search.TrialResult(
        phase="full",
        trial_id=1,
        status="success",
        gpu_id=None,
        sampled_params={"batch_size": 512},
        effective_batch_size=512,
        oom_retries=0,
        donor_pr_auc=0.95,
        acceptor_pr_auc=None,
        mean_pr_auc=0.95,
        objective_metric="mean_pr_auc",
        objective_score=0.95,
        error_message=None,
        return_code=0,
        duration_sec=1.0,
        metrics_json=str(tmp_path / "metrics.json"),
        log_file="trial.log",
    )

    monkeypatch.setattr(
        hparam_search,
        "_read_best_objective_score",
        lambda *args, **kwargs: 0.94,
    )
    monkeypatch.setattr(
        hparam_search,
        "write_best_config",
        lambda *args, **kwargs: None,
    )

    hparam_search.maybe_update_global_best(
        global_best_path=output_path,
        best_row=row,
    )

    captured = capsys.readouterr()
    assert "0.940000 -> 0.950000" in captured.out


def test_build_fixed_run_args_context_excludes_search_and_runtime_keys() -> None:
    fixed_run_args = hparam_search._build_fixed_run_args_context(
        base_args={
            "model": "cnn",
            "species": "Dmel",
            "donor_len": 100,
            "sequence_transform": "none",
            "input_mode": "onehot",
            "batch_size": 512,
            "visualize": "none",
            "num_workers": "auto",
        },
        full_overrides={
            "epochs": 8,
            "compile_mode": "off",
            "sequence_transform": "mask_sites",
        },
        search_space={
            "input_mode": {
                "type": "categorical",
                "values": ["onehot", "kmer3", "bpe"],
            },
            "batch_size": {
                "type": "categorical",
                "values": [512, 1024],
            }
        },
    )

    assert fixed_run_args == {
        "donor_len": 100,
        "model": "cnn",
        "sequence_transform": "mask_sites",
        "species": "Dmel",
    }


def test_read_objective_best_epoch_from_metrics_for_roc_auc_and_max_f1(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics_obj_epoch.json"
    metrics_path.write_text(
        json.dumps(
            {
                "donor": {
                    "best_metric": "pr_auc",
                    "best_epoch": 9,
                    "epoch_history": [
                        {"epoch": 1, "roc_auc": 0.70},
                        {"epoch": 2, "roc_auc": 0.85},
                        {"epoch": 3, "roc_auc": 0.82},
                    ],
                },
                "acceptor": {
                    "epoch_history": [
                        {"epoch": 1, "roc_auc": 0.78},
                        {"epoch": 2, "roc_auc": 0.80},
                    ]
                },
                "pair": {
                    "epoch_history": [
                        {"epoch": 1, "max_f1": 0.61},
                        {"epoch": 2, "max_f1": 0.74},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    donor_roc_epoch = hparam_search._read_objective_best_epoch_from_metrics(
        metrics_json_path=str(metrics_path),
        objective_metric="donor_roc_auc",
    )
    mean_roc_epoch = hparam_search._read_objective_best_epoch_from_metrics(
        metrics_json_path=str(metrics_path),
        objective_metric="mean_roc_auc",
    )
    pair_max_f1_epoch = hparam_search._read_objective_best_epoch_from_metrics(
        metrics_json_path=str(metrics_path),
        objective_metric="pair_max_f1",
    )

    assert donor_roc_epoch == 2
    assert mean_roc_epoch == 2
    assert pair_max_f1_epoch == 2


def test_run_search_ignores_global_best_in_quick_and_uses_species_plot_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    forced = {
        "batch_size": 1024,
        "kernel_size": 9,
        "lr": 1.5e-4,
    }
    global_best_path = tmp_path / "global_best.json"
    global_best_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "mean_pr_auc": 0.9,
                "sampled_params": forced,
            }
        ),
        encoding="utf-8",
    )

    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=2,
        top_k=2,
        full_epochs=4,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=global_best_path,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={"compile_mode": "off"},
        full_overrides={"compile_mode": "auto"},
        search_space=search_space,
    )

    captured: dict[str, object] = {}

    def _fake_detect_gpu_ids(setting: object) -> list[str]:
        del setting
        return []

    def _fake_run_phase(
        *,
        phase: str,
        config: hparam_search.SearchConfig,
        trial_count: int,
        trial_params: list[dict[str, hparam_search.Scalar]],
        overrides: dict[str, hparam_search.ArgValue],
        gpu_ids: list[str],
        max_parallel_trials: int,
        out_dir: Path,
        execution_mode: str,
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir
        assert execution_mode == "subprocess"
        if phase == "quick":
            captured["quick_params"] = [dict(params) for params in trial_params]
            captured["quick_first"] = dict(trial_params[0])
        if phase == "full":
            captured["full_params"] = [dict(params) for params in trial_params]
        rows: list[hparam_search.TrialResult] = []
        for trial_id in range(trial_count):
            score = 0.8 + (0.01 * trial_id)
            rows.append(
                hparam_search.TrialResult(
                    phase=phase,
                    trial_id=trial_id,
                    status="success",
                    gpu_id=None,
                    sampled_params=dict(trial_params[trial_id]),
                    effective_batch_size=512,
                    oom_retries=0,
                    donor_pr_auc=score,
                    acceptor_pr_auc=score,
                    mean_pr_auc=score,
                    objective_metric="mean_pr_auc",
                    objective_score=score,
                    error_message=None,
                    return_code=0,
                    duration_sec=0.1,
                    metrics_json=f"{phase}_{trial_id}.json",
                    log_file=f"{phase}_{trial_id}.log",
                )
            )
        return rows

    def _fake_write_visualization(
        path: Path,
        *,
        model_name: str,
        species: str,
        objective_metric: str,
        quick_rows: list[hparam_search.TrialResult],
        full_rows: list[hparam_search.TrialResult],
        base_args: dict[str, hparam_search.ArgValue],
    ) -> Optional[str]:
        del quick_rows, full_rows, base_args
        captured["viz_path"] = path
        captured["viz_model"] = model_name
        captured["viz_species"] = species
        captured["viz_metric"] = objective_metric
        return None

    monkeypatch.setattr(hparam_search, "detect_gpu_ids", _fake_detect_gpu_ids)
    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)
    monkeypatch.setattr(
        hparam_search,
        "write_visualization",
        _fake_write_visualization,
    )

    exit_code = hparam_search.run_search(config)

    assert exit_code == 0
    expected_quick_first = hparam_search.sample_trial_params(
        search_space=search_space,
        seed=config.base_seed,
    )
    assert captured["quick_first"] == expected_quick_first
    assert captured["quick_first"] != forced
    quick_params = cast(list[dict[str, object]], captured["quick_params"])
    full_params = cast(list[dict[str, object]], captured["full_params"])
    assert full_params == [quick_params[2], quick_params[1]]
    assert all(params != forced for params in full_params)
    assert captured["viz_model"] == "cnn"
    assert captured["viz_species"] == "Dmel"
    assert captured["viz_metric"] == "mean_pr_auc"
    assert isinstance(captured["viz_path"], Path)
    assert cast(Path, captured["viz_path"]).name == "Dmel_snpr.png"


def test_run_search_rechecks_global_best_when_context_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    forced = {
        "batch_size": 1024,
        "kernel_size": 9,
        "lr": 1.5e-4,
    }
    validation_protocol = hparam_search._derive_validation_protocol_from_args(
        merged_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        objective_metric="mean_pr_auc",
    )
    fixed_run_args = hparam_search._build_fixed_run_args_context(
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        full_overrides={"compile_mode": "auto", "epochs": 4},
        search_space=search_space,
    )
    global_best_path = tmp_path / "global_best_context_change.json"
    global_best_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "objective_metric": "mean_pr_auc",
                "objective_score": 0.93,
                "objective_best_epoch": 6,
                "sampled_params": forced,
                "hparam_context": {
                    "version": 2,
                    "objective_metric": "mean_pr_auc",
                    "full_epochs": 4,
                    "validation_protocol": validation_protocol,
                    "fixed_run_args": fixed_run_args,
                },
            }
        ),
        encoding="utf-8",
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=2,
        top_k=2,
        full_epochs=4,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_max_f1",
        global_best_config_path=global_best_path,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={"compile_mode": "off"},
        full_overrides={"compile_mode": "auto"},
        search_space=search_space,
    )
    captured: dict[str, object] = {}

    def _fake_detect_gpu_ids(setting: object) -> list[str]:
        del setting
        return []

    def _fake_run_phase(
        *,
        phase: str,
        config: hparam_search.SearchConfig,
        trial_count: int,
        trial_params: list[dict[str, hparam_search.Scalar]],
        overrides: dict[str, hparam_search.ArgValue],
        gpu_ids: list[str],
        max_parallel_trials: int,
        out_dir: Path,
        execution_mode: str,
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir
        assert execution_mode == "subprocess"
        if phase == "quick":
            captured["quick_params"] = [dict(params) for params in trial_params]
        if phase == "full":
            captured["full_params"] = [dict(params) for params in trial_params]
        rows: list[hparam_search.TrialResult] = []
        for trial_id in range(trial_count):
            score = 0.75 + (0.01 * trial_id)
            rows.append(
                hparam_search.TrialResult(
                    phase=phase,
                    trial_id=trial_id,
                    status="success",
                    gpu_id=None,
                    sampled_params=dict(trial_params[trial_id]),
                    effective_batch_size=512,
                    oom_retries=0,
                    donor_pr_auc=score,
                    acceptor_pr_auc=score,
                    mean_pr_auc=score,
                    objective_metric="mean_max_f1",
                    objective_score=score,
                    error_message=None,
                    return_code=0,
                    duration_sec=0.1,
                    metrics_json=str(tmp_path / f"{phase}_{trial_id}.metrics.json"),
                    log_file=f"{phase}_{trial_id}.log",
                )
            )
        return rows

    monkeypatch.setattr(hparam_search, "detect_gpu_ids", _fake_detect_gpu_ids)
    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)
    monkeypatch.setattr(
        hparam_search,
        "write_visualization",
        lambda *args, **kwargs: None,
    )

    exit_code = hparam_search.run_search(config)

    assert exit_code == 0
    quick_params = cast(list[dict[str, object]], captured["quick_params"])
    full_params = cast(list[dict[str, object]], captured["full_params"])
    assert quick_params[0] != forced
    assert full_params[0] == forced
    assert forced in full_params
    assert len(full_params) == 3


def test_run_search_adds_full_recheck_as_extra_trial_when_global_best_is_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    forced = {
        "batch_size": 1024,
        "kernel_size": 9,
        "lr": 1.5e-4,
    }
    validation_protocol = hparam_search._derive_validation_protocol_from_args(
        merged_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        objective_metric="mean_pr_auc",
    )
    fixed_run_args = hparam_search._build_fixed_run_args_context(
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        full_overrides={"compile_mode": "auto", "epochs": 4},
        search_space=search_space,
    )
    global_best_path = tmp_path / "global_best_selected.json"
    global_best_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "objective_metric": "mean_pr_auc",
                "objective_score": 0.93,
                "objective_best_epoch": 6,
                "sampled_params": forced,
                "hparam_context": {
                    "version": 2,
                    "objective_metric": "mean_pr_auc",
                    "full_epochs": 4,
                    "validation_protocol": validation_protocol,
                    "fixed_run_args": fixed_run_args,
                },
            }
        ),
        encoding="utf-8",
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=2,
        top_k=2,
        full_epochs=4,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_max_f1",
        global_best_config_path=global_best_path,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={"compile_mode": "off"},
        full_overrides={"compile_mode": "auto"},
        search_space=search_space,
    )
    captured: dict[str, object] = {}

    def _fake_detect_gpu_ids(setting: object) -> list[str]:
        del setting
        return []

    def _fake_build_trial_params(
        *,
        config: hparam_search.SearchConfig,
        phase: str,
        count: int,
        seed_offset: int,
        seed_source: list[hparam_search.TrialResult] | None = None,
        history_trials: list[tuple[float, dict[str, hparam_search.Scalar]]] | None = None,
    ) -> list[dict[str, hparam_search.Scalar]]:
        del config, seed_offset, seed_source, history_trials
        assert phase == "quick"
        assert count == 3
        return [
            dict(forced),
            {"batch_size": 512, "kernel_size": 7, "lr": 2.0e-4},
            {"batch_size": 256, "kernel_size": 5, "lr": 3.0e-4},
        ]

    def _fake_run_phase(
        *,
        phase: str,
        config: hparam_search.SearchConfig,
        trial_count: int,
        trial_params: list[dict[str, hparam_search.Scalar]],
        overrides: dict[str, hparam_search.ArgValue],
        gpu_ids: list[str],
        max_parallel_trials: int,
        out_dir: Path,
        execution_mode: str,
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir
        assert execution_mode == "subprocess"
        if phase == "quick":
            captured["quick_params"] = [dict(params) for params in trial_params]
        if phase == "full":
            captured["full_params"] = [dict(params) for params in trial_params]
        rows: list[hparam_search.TrialResult] = []
        for trial_id in range(trial_count):
            score = 0.99 - (0.05 * trial_id)
            rows.append(
                hparam_search.TrialResult(
                    phase=phase,
                    trial_id=trial_id,
                    status="success",
                    gpu_id=None,
                    sampled_params=dict(trial_params[trial_id]),
                    effective_batch_size=512,
                    oom_retries=0,
                    donor_pr_auc=score,
                    acceptor_pr_auc=score,
                    mean_pr_auc=score,
                    objective_metric="mean_max_f1",
                    objective_score=score,
                    error_message=None,
                    return_code=0,
                    duration_sec=0.1,
                    metrics_json=str(tmp_path / f"{phase}_{trial_id}.metrics.json"),
                    log_file=f"{phase}_{trial_id}.log",
                )
            )
        return rows

    monkeypatch.setattr(hparam_search, "detect_gpu_ids", _fake_detect_gpu_ids)
    monkeypatch.setattr(hparam_search, "build_trial_params", _fake_build_trial_params)
    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)
    monkeypatch.setattr(
        hparam_search,
        "write_visualization",
        lambda *args, **kwargs: None,
    )

    exit_code = hparam_search.run_search(config)
    captured_out = capsys.readouterr().out

    assert exit_code == 0
    assert "Full recheck added!" in captured_out
    assert "base_full=2" in captured_out
    assert "recheck=+1" in captured_out
    assert "full=3" in captured_out
    full_params = cast(list[dict[str, object]], captured["full_params"])
    assert full_params.count(forced) == 2
    assert len(full_params) == 3


def test_run_search_uses_trial_process_mode_per_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=2,
        quick_epochs=2,
        top_k=1,
        full_epochs=4,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={"compile_mode": "off"},
        full_overrides={"compile_mode": "off"},
        search_space=search_space,
        trial_process_mode="persistent_quick",
    )
    captured_modes: dict[str, str] = {}

    def _fake_detect_gpu_ids(setting: object) -> list[str]:
        del setting
        return []

    def _fake_run_phase(
        *,
        phase: str,
        config: hparam_search.SearchConfig,
        trial_count: int,
        trial_params: list[dict[str, hparam_search.Scalar]],
        overrides: dict[str, hparam_search.ArgValue],
        gpu_ids: list[str],
        max_parallel_trials: int,
        out_dir: Path,
        execution_mode: str,
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir
        captured_modes[phase] = execution_mode
        rows: list[hparam_search.TrialResult] = []
        for trial_id in range(trial_count):
            score = 0.7 + (0.1 * trial_id)
            rows.append(
                hparam_search.TrialResult(
                    phase=phase,
                    trial_id=trial_id,
                    status="success",
                    gpu_id=None,
                    sampled_params=dict(trial_params[trial_id]),
                    effective_batch_size=512,
                    oom_retries=0,
                    donor_pr_auc=score,
                    acceptor_pr_auc=score,
                    mean_pr_auc=score,
                    objective_metric="mean_pr_auc",
                    objective_score=score,
                    error_message=None,
                    return_code=0,
                    duration_sec=0.1,
                    metrics_json=str(tmp_path / f"{phase}_{trial_id}.metrics.json"),
                    log_file=f"{phase}_{trial_id}.log",
                )
            )
        return rows

    monkeypatch.setattr(hparam_search, "detect_gpu_ids", _fake_detect_gpu_ids)
    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)
    monkeypatch.setattr(
        hparam_search,
        "write_visualization",
        lambda *args, **kwargs: None,
    )

    exit_code = hparam_search.run_search(config)

    assert exit_code == 0
    assert captured_modes == {"quick": "persistent", "full": "subprocess"}


def test_run_search_skips_full_phase_and_visualization_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=1,
        top_k=2,
        full_epochs=1,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={"compile_mode": "off"},
        full_overrides={"compile_mode": "off"},
        search_space=search_space,
        skip_full_phase=True,
        enable_visualization=False,
    )

    phase_calls: list[str] = []
    visualization_called = False

    def _fake_detect_gpu_ids(setting: object) -> list[str]:
        del setting
        return []

    def _fake_run_phase(
        *,
        phase: str,
        config: hparam_search.SearchConfig,
        trial_count: int,
        trial_params: list[dict[str, hparam_search.Scalar]],
        overrides: dict[str, hparam_search.ArgValue],
        gpu_ids: list[str],
        max_parallel_trials: int,
        out_dir: Path,
        execution_mode: str,
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir, execution_mode
        phase_calls.append(phase)
        rows: list[hparam_search.TrialResult] = []
        for trial_id in range(trial_count):
            score = 0.7 + (0.01 * trial_id)
            rows.append(
                hparam_search.TrialResult(
                    phase=phase,
                    trial_id=trial_id,
                    status="success",
                    gpu_id=None,
                    sampled_params=dict(trial_params[trial_id]),
                    effective_batch_size=512,
                    oom_retries=0,
                    donor_pr_auc=score,
                    acceptor_pr_auc=score,
                    mean_pr_auc=score,
                    objective_metric="mean_pr_auc",
                    objective_score=score,
                    error_message=None,
                    return_code=0,
                    duration_sec=0.1,
                    metrics_json=f"{phase}_{trial_id}.json",
                    log_file=f"{phase}_{trial_id}.log",
                )
            )
        return rows

    def _fake_write_visualization(
        path: Path,
        *,
        model_name: str,
        species: str,
        objective_metric: str,
        quick_rows: list[hparam_search.TrialResult],
        full_rows: list[hparam_search.TrialResult],
        base_args: dict[str, hparam_search.ArgValue],
    ) -> Optional[str]:
        nonlocal visualization_called
        del path, model_name, species, objective_metric, quick_rows, full_rows, base_args
        visualization_called = True
        return None

    monkeypatch.setattr(hparam_search, "detect_gpu_ids", _fake_detect_gpu_ids)
    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)
    monkeypatch.setattr(
        hparam_search,
        "write_visualization",
        _fake_write_visualization,
    )

    exit_code = hparam_search.run_search(config)

    assert exit_code == 0
    assert phase_calls == ["quick"]
    assert visualization_called is False


def test_run_search_downgrades_persistent_when_trials_fit_one_wave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=2,
        quick_epochs=2,
        top_k=1,
        full_epochs=4,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={"compile_mode": "off"},
        full_overrides={"compile_mode": "off"},
        search_space=search_space,
        trial_process_mode="persistent_all",
    )
    captured_modes: dict[str, str] = {}

    def _fake_detect_gpu_ids(setting: object) -> list[str]:
        del setting
        return ["0", "1"]

    def _fake_run_phase(
        *,
        phase: str,
        config: hparam_search.SearchConfig,
        trial_count: int,
        trial_params: list[dict[str, hparam_search.Scalar]],
        overrides: dict[str, hparam_search.ArgValue],
        gpu_ids: list[str],
        max_parallel_trials: int,
        out_dir: Path,
        execution_mode: str,
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir
        captured_modes[phase] = execution_mode
        rows: list[hparam_search.TrialResult] = []
        for trial_id in range(trial_count):
            score = 0.7 + (0.1 * trial_id)
            rows.append(
                hparam_search.TrialResult(
                    phase=phase,
                    trial_id=trial_id,
                    status="success",
                    gpu_id=None,
                    sampled_params=dict(trial_params[trial_id]),
                    effective_batch_size=512,
                    oom_retries=0,
                    donor_pr_auc=score,
                    acceptor_pr_auc=score,
                    mean_pr_auc=score,
                    objective_metric="mean_pr_auc",
                    objective_score=score,
                    error_message=None,
                    return_code=0,
                    duration_sec=0.1,
                    metrics_json=str(tmp_path / f"{phase}_{trial_id}.metrics.json"),
                    log_file=f"{phase}_{trial_id}.log",
                )
            )
        return rows

    monkeypatch.setattr(hparam_search, "detect_gpu_ids", _fake_detect_gpu_ids)
    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)
    monkeypatch.setattr(
        hparam_search,
        "write_visualization",
        lambda *args, **kwargs: None,
    )

    exit_code = hparam_search.run_search(config)

    assert exit_code == 0
    assert captured_modes == {"quick": "subprocess", "full": "subprocess"}


def test_run_search_injects_seed_into_full_when_context_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    seed_params: dict[str, hparam_search.Scalar] = {
        "batch_size": 1024,
        "kernel_size": 9,
        "lr": 1.5e-4,
    }
    seed_best_path = tmp_path / "seed_best.json"
    seed_best_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "objective_metric": "mean_pr_auc",
                "objective_score": 0.91,
                "objective_best_epoch": 9,
                "sampled_params": seed_params,
                "hparam_context": {
                    "version": 2,
                    "objective_metric": "mean_pr_auc",
                    "full_epochs": 8,
                    "validation_protocol": {"split_type": "stratified_site"},
                },
            }
        ),
        encoding="utf-8",
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=2,
        top_k=2,
        full_epochs=4,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=seed_best_path,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={"compile_mode": "off"},
        full_overrides={"compile_mode": "auto"},
        search_space=search_space,
    )
    captured: dict[str, object] = {}

    def _fake_detect_gpu_ids(setting: object) -> list[str]:
        del setting
        return []

    def _fake_run_phase(
        *,
        phase: str,
        config: hparam_search.SearchConfig,
        trial_count: int,
        trial_params: list[dict[str, hparam_search.Scalar]],
        overrides: dict[str, hparam_search.ArgValue],
        gpu_ids: list[str],
        max_parallel_trials: int,
        out_dir: Path,
        execution_mode: str,
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir
        assert execution_mode == "subprocess"
        if phase == "quick":
            captured["quick_params"] = [dict(params) for params in trial_params]
        if phase == "full":
            captured["full_params"] = [dict(params) for params in trial_params]
        rows: list[hparam_search.TrialResult] = []
        for trial_id in range(trial_count):
            if phase == "quick":
                quick_scores = [0.70, 0.85, 0.84]
                score = quick_scores[trial_id]
            else:
                score = 0.86 + (0.01 * trial_id)
            rows.append(
                hparam_search.TrialResult(
                    phase=phase,
                    trial_id=trial_id,
                    status="success",
                    gpu_id=None,
                    sampled_params=dict(trial_params[trial_id]),
                    effective_batch_size=512,
                    oom_retries=0,
                    donor_pr_auc=score,
                    acceptor_pr_auc=score,
                    mean_pr_auc=score,
                    objective_metric="mean_pr_auc",
                    objective_score=score,
                    error_message=None,
                    return_code=0,
                    duration_sec=0.1,
                    metrics_json=str(tmp_path / f"{phase}_{trial_id}.metrics.json"),
                    log_file=f"{phase}_{trial_id}.log",
                )
            )
        return rows

    monkeypatch.setattr(hparam_search, "detect_gpu_ids", _fake_detect_gpu_ids)
    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)
    monkeypatch.setattr(
        hparam_search,
        "write_visualization",
        lambda *args, **kwargs: None,
    )

    exit_code = hparam_search.run_search(config)

    assert exit_code == 0
    quick_params = cast(list[dict[str, object]], captured["quick_params"])
    full_params = cast(list[dict[str, object]], captured["full_params"])
    assert quick_params[0] == seed_params
    assert full_params[0] == seed_params
    assert seed_params in full_params
    assert len(full_params) == 3


def test_run_search_skips_seed_full_recheck_when_context_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    seed_params: dict[str, hparam_search.Scalar] = {
        "batch_size": 1024,
        "kernel_size": 9,
        "lr": 1.5e-4,
    }
    validation_protocol = hparam_search._derive_validation_protocol_from_args(
        merged_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        objective_metric="mean_pr_auc",
    )
    fixed_run_args = hparam_search._build_fixed_run_args_context(
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        full_overrides={"compile_mode": "auto", "epochs": 4},
        search_space=search_space,
    )
    hparam_context = hparam_search._build_hparam_context(
        objective_metric="mean_pr_auc",
        full_epochs=4,
        validation_protocol=validation_protocol,
        fixed_run_args=fixed_run_args,
    )
    seed_best_path = tmp_path / "seed_best_context_match.json"
    seed_best_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "objective_metric": "mean_pr_auc",
                "objective_score": 0.92,
                "objective_best_epoch": 7,
                "sampled_params": seed_params,
                "hparam_context": hparam_context,
            }
        ),
        encoding="utf-8",
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=2,
        top_k=2,
        full_epochs=4,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=seed_best_path,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={"compile_mode": "off"},
        full_overrides={"compile_mode": "auto"},
        search_space=search_space,
    )
    captured: dict[str, object] = {}

    def _fake_detect_gpu_ids(setting: object) -> list[str]:
        del setting
        return []

    def _fake_run_phase(
        *,
        phase: str,
        config: hparam_search.SearchConfig,
        trial_count: int,
        trial_params: list[dict[str, hparam_search.Scalar]],
        overrides: dict[str, hparam_search.ArgValue],
        gpu_ids: list[str],
        max_parallel_trials: int,
        out_dir: Path,
        execution_mode: str,
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir
        assert execution_mode == "subprocess"
        if phase == "quick":
            captured["quick_params"] = [dict(params) for params in trial_params]
        if phase == "full":
            captured["full_params"] = [dict(params) for params in trial_params]
        rows: list[hparam_search.TrialResult] = []
        for trial_id in range(trial_count):
            if phase == "quick":
                quick_scores = [0.99, 0.80, 0.79]
                score = quick_scores[trial_id]
            else:
                score = 0.81 + (0.01 * trial_id)
            rows.append(
                hparam_search.TrialResult(
                    phase=phase,
                    trial_id=trial_id,
                    status="success",
                    gpu_id=None,
                    sampled_params=dict(trial_params[trial_id]),
                    effective_batch_size=512,
                    oom_retries=0,
                    donor_pr_auc=score,
                    acceptor_pr_auc=score,
                    mean_pr_auc=score,
                    objective_metric="mean_pr_auc",
                    objective_score=score,
                    error_message=None,
                    return_code=0,
                    duration_sec=0.1,
                    metrics_json=str(tmp_path / f"{phase}_{trial_id}.metrics.json"),
                    log_file=f"{phase}_{trial_id}.log",
                )
            )
        return rows

    monkeypatch.setattr(hparam_search, "detect_gpu_ids", _fake_detect_gpu_ids)
    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)
    monkeypatch.setattr(
        hparam_search,
        "write_visualization",
        lambda *args, **kwargs: None,
    )

    exit_code = hparam_search.run_search(config)

    assert exit_code == 0
    full_params = cast(list[dict[str, object]], captured["full_params"])
    assert seed_params not in full_params
    assert len(full_params) == 2


def test_run_search_injects_seed_into_full_when_fixed_run_args_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    seed_params: dict[str, hparam_search.Scalar] = {
        "batch_size": 1024,
        "kernel_size": 9,
        "lr": 1.5e-4,
    }
    validation_protocol = hparam_search._derive_validation_protocol_from_args(
        merged_args={
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 512,
            "donor_len": 100,
        },
        objective_metric="mean_pr_auc",
    )
    fixed_run_args = hparam_search._build_fixed_run_args_context(
        base_args={
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 512,
            "donor_len": 100,
        },
        full_overrides={"compile_mode": "auto", "epochs": 4},
        search_space=search_space,
    )
    seed_best_path = tmp_path / "seed_best_fixed_arg_change.json"
    seed_best_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "objective_metric": "mean_pr_auc",
                "objective_score": 0.92,
                "objective_best_epoch": 7,
                "sampled_params": seed_params,
                "hparam_context": hparam_search._build_hparam_context(
                    objective_metric="mean_pr_auc",
                    full_epochs=4,
                    validation_protocol=validation_protocol,
                    fixed_run_args=fixed_run_args,
                ),
            }
        ),
        encoding="utf-8",
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=2,
        top_k=2,
        full_epochs=4,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=2,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=seed_best_path,
        base_args={
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 512,
            "donor_len": 120,
        },
        quick_overrides={"compile_mode": "off"},
        full_overrides={"compile_mode": "auto"},
        search_space=search_space,
    )
    captured: dict[str, object] = {}

    def _fake_detect_gpu_ids(setting: object) -> list[str]:
        del setting
        return []

    def _fake_run_phase(
        *,
        phase: str,
        config: hparam_search.SearchConfig,
        trial_count: int,
        trial_params: list[dict[str, hparam_search.Scalar]],
        overrides: dict[str, hparam_search.ArgValue],
        gpu_ids: list[str],
        max_parallel_trials: int,
        out_dir: Path,
        execution_mode: str,
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir
        assert execution_mode == "subprocess"
        if phase == "quick":
            captured["quick_params"] = [dict(params) for params in trial_params]
        if phase == "full":
            captured["full_params"] = [dict(params) for params in trial_params]
        rows: list[hparam_search.TrialResult] = []
        for trial_id in range(trial_count):
            if phase == "quick":
                quick_scores = [0.70, 0.80, 0.79]
                score = quick_scores[trial_id]
            else:
                score = 0.81 + (0.01 * trial_id)
            rows.append(
                hparam_search.TrialResult(
                    phase=phase,
                    trial_id=trial_id,
                    status="success",
                    gpu_id=None,
                    sampled_params=dict(trial_params[trial_id]),
                    effective_batch_size=512,
                    oom_retries=0,
                    donor_pr_auc=score,
                    acceptor_pr_auc=score,
                    mean_pr_auc=score,
                    objective_metric="mean_pr_auc",
                    objective_score=score,
                    error_message=None,
                    return_code=0,
                    duration_sec=0.1,
                    metrics_json=str(tmp_path / f"{phase}_{trial_id}.metrics.json"),
                    log_file=f"{phase}_{trial_id}.log",
                )
            )
        return rows

    monkeypatch.setattr(hparam_search, "detect_gpu_ids", _fake_detect_gpu_ids)
    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)
    monkeypatch.setattr(
        hparam_search,
        "write_visualization",
        lambda *args, **kwargs: None,
    )

    exit_code = hparam_search.run_search(config)

    assert exit_code == 0
    quick_params = cast(list[dict[str, object]], captured["quick_params"])
    full_params = cast(list[dict[str, object]], captured["full_params"])
    assert quick_params[0] == seed_params
    assert full_params[0] == seed_params
    assert seed_params in full_params
    assert len(full_params) == 3


def test_run_trial_uses_model_from_base_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="donor_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn_resdil", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space=hparam_search._validate_search_space(config_dict["search_space"]),
    )

    observed_model: list[str] = []

    def _fake_run_command_with_streaming(
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        phase: str,
        trial_id: int,
    ) -> tuple[int, str]:
        del cwd, env, phase, trial_id
        metrics_path: Optional[Path] = None
        for idx, token in enumerate(cmd):
            if token == "--model":
                observed_model.append(cmd[idx + 1])
            if token == "--metrics_json":
                metrics_path = Path(cmd[idx + 1])
        assert metrics_path is not None
        metrics_path.write_text(
            json.dumps({"donor": {"best_pr_auc": 0.8}}),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(
        hparam_search,
        "_run_command_with_streaming",
        _fake_run_command_with_streaming,
    )

    result = hparam_search.run_trial(
        config=config,
        phase="quick",
        trial_id=0,
        sampled_params={"batch_size": 512, "lr": 1e-4},
        overrides={"epochs": 1},
        assigned_gpu_id=None,
        metrics_json=tmp_path / "metrics.json",
        log_file=tmp_path / "trial.log",
    )

    assert result.status == "success"
    assert observed_model == ["cnn_resdil"]


def test_estimate_cnn_param_complexity_known_configuration() -> None:
    sampled_params: dict[str, hparam_search.Scalar] = {
        "conv_channels": "64,128",
        "kernel_size": 7,
        "fc_hidden": 128,
    }
    complexity = hparam_search.estimate_cnn_param_complexity(
        sampled_params=sampled_params,
        base_args={},
    )
    assert complexity == 76353


def test_estimate_cnn_param_complexity_falls_back_to_base_defaults() -> None:
    complexity = hparam_search.estimate_cnn_param_complexity(
        sampled_params={},
        base_args={"lightweight": False},
    )
    assert complexity is not None
    assert complexity > 0


def test_estimate_tcn_param_complexity_known_configuration() -> None:
    sampled_params: dict[str, hparam_search.Scalar] = {
        "conv_channels": "64,128",
        "kernel_size": 7,
        "tcn_block_repeats": 2,
        "fc_hidden": 128,
    }
    complexity = hparam_search.estimate_tcn_param_complexity(
        sampled_params=sampled_params,
        base_args={},
    )
    assert complexity == 533889


def test_estimate_cnn_pair_param_complexity_known_configuration() -> None:
    sampled_params: dict[str, hparam_search.Scalar] = {
        "donor_conv_channels": "64,128",
        "acceptor_conv_channels": "128,256",
        "kernel_size": 7,
        "fc_hidden": 128,
    }
    complexity = hparam_search.estimate_cnn_pair_param_complexity(
        sampled_params=sampled_params,
        base_args={},
    )
    assert complexity == 343233


def test_estimate_cnn_pair_param_complexity_early_configuration() -> None:
    sampled_params: dict[str, hparam_search.Scalar] = {
        "fusion_mode": "early",
        "donor_conv_channels": "64,128",
        "acceptor_conv_channels": "64,128",
        "donor_kernel_sizes": "7,7",
        "acceptor_kernel_sizes": "7,7",
        "fc_hidden": 128,
    }
    complexity = hparam_search.estimate_cnn_pair_param_complexity(
        sampled_params=sampled_params,
        base_args={},
    )
    assert complexity == 78145


def test_estimate_cnn_pair_param_complexity_mid_configuration() -> None:
    sampled_params: dict[str, hparam_search.Scalar] = {
        "fusion_mode": "mid",
        "donor_conv_channels": "64,128",
        "acceptor_conv_channels": "64,128",
        "donor_kernel_sizes": "7,7",
        "acceptor_kernel_sizes": "7,7",
        "fc_hidden": 128,
    }
    complexity = hparam_search.estimate_cnn_pair_param_complexity(
        sampled_params=sampled_params,
        base_args={},
    )
    assert complexity == 135681


def test_load_historical_trials_reads_and_ranks_sibling_runs(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        _base_config_dict(tmp_path)["search_space"]
    )
    tuning_root = tmp_path / "tuning" / "cnn" / "donor"
    current_output = tuning_root / "20260220_000000"
    current_output.mkdir(parents=True, exist_ok=True)

    run_one = tuning_root / "20260219_010101"
    run_one.mkdir(parents=True, exist_ok=True)
    (run_one / "quick_trials.tsv").write_text(
        "\n".join(
            [
                "\t".join(
                    [
                        "status",
                        "objective_score",
                        "batch_size",
                        "kernel_size",
                        "lr",
                    ]
                ),
                "\t".join(["success", "0.82", "512", "7", "0.0002"]),
                "\t".join(["success", "0.81", "512", "7", "0.0002"]),
                "\t".join(["failed", "", "1024", "9", "0.0004"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_two = tuning_root / "20260219_020202"
    run_two.mkdir(parents=True, exist_ok=True)
    (run_two / "full_trials.tsv").write_text(
        "\n".join(
            [
                "\t".join(
                    [
                        "status",
                        "objective_score",
                        "batch_size",
                        "kernel_size",
                        "lr",
                    ]
                ),
                "\t".join(["success", "0.84", "1024", "9", "0.0004"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = hparam_search.load_historical_trials(
        output_dir=current_output,
        search_space=search_space,
        objective_metric="mean_pr_auc",
        top_n=5,
    )

    assert len(loaded) == 2
    assert loaded[0][0] == pytest.approx(0.84)
    assert loaded[0][1]["batch_size"] == 1024
    assert loaded[1][0] == pytest.approx(0.82)
    assert loaded[1][1]["kernel_size"] == 7


def test_load_historical_trials_defaults_missing_window_lengths_to_100(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [512]},
            "donor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
            "acceptor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
        }
    )
    tuning_root = tmp_path / "tuning" / "cnn" / "donor"
    current_output = tuning_root / "20260220_000000"
    current_output.mkdir(parents=True, exist_ok=True)

    run_one = tuning_root / "20260219_010101"
    run_one.mkdir(parents=True, exist_ok=True)
    (run_one / "quick_trials.tsv").write_text(
        "\n".join(
            [
                "\t".join(["status", "objective_score", "batch_size"]),
                "\t".join(["success", "0.82", "512"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = hparam_search.load_historical_trials(
        output_dir=current_output,
        search_space=search_space,
        objective_metric="mean_pr_auc",
        top_n=5,
        base_args={},
    )

    assert len(loaded) == 1
    assert loaded[0][1]["donor_len"] == 100
    assert loaded[0][1]["acceptor_len"] == 100


def test_build_trial_params_history_guided_is_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config_dict(tmp_path)
    search_space = hparam_search._validate_search_space(config_dict["search_space"])
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1337,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
        search_algo="history_guided",
        history_top_n=8,
        guided_random_fraction=0.0,
        guided_mutation_rate=0.0,
    )
    history_trials = [
        (0.83, {"batch_size": 512, "kernel_size": 7, "lr": 2e-4}),
        (0.81, {"batch_size": 1024, "kernel_size": 9, "lr": 1.5e-4}),
        (0.79, {"batch_size": 512, "kernel_size": 5, "lr": 1e-3}),
    ]

    guided_rows = [
        {"batch_size": 512, "kernel_size": 7, "lr": 2e-4},
        {"batch_size": 1024, "kernel_size": 9, "lr": 1.5e-4},
        {"batch_size": 512, "kernel_size": 5, "lr": 1e-3},
    ]
    call_count = {"value": 0}

    def _fake_history_guided(
        *,
        search_space: dict[str, hparam_search.SearchDimension],
        seed: int,
        history_trials: list[tuple[float, dict[str, hparam_search.Scalar]]],
        random_fraction: float,
        mutation_rate: float,
    ) -> dict[str, hparam_search.Scalar]:
        del search_space, seed, history_trials, random_fraction, mutation_rate
        index = call_count["value"] % len(guided_rows)
        call_count["value"] += 1
        return dict(guided_rows[index])

    monkeypatch.setattr(
        hparam_search,
        "sample_trial_params_history_guided",
        _fake_history_guided,
    )

    first = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=3,
        seed_offset=0,
        history_trials=history_trials,
    )
    second = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=3,
        seed_offset=0,
        history_trials=history_trials,
    )

    assert first == second
    anchors = [row[1] for row in history_trials]
    for params in first:
        assert params in anchors


def test_build_trial_params_skips_duplicate_quick_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "lr": {"type": "categorical", "values": [1e-4, 2e-4]},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=2,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=7,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 256},
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )
    sampled_rows = iter(
        [
            {"batch_size": 256, "lr": 1e-4},
            {"batch_size": 256, "lr": 1e-4},
            {"batch_size": 256, "lr": 2e-4},
        ]
    )
    call_count = {"value": 0}

    def _fake_sample(
        _search_space: dict[str, hparam_search.SearchDimension],
        _rng: object,
    ) -> dict[str, hparam_search.Scalar]:
        call_count["value"] += 1
        return next(sampled_rows)

    def _fake_materialize(
        *,
        model_name: str,
        sampled_params: dict[str, hparam_search.Scalar],
        base_args: dict[str, hparam_search.ArgValue],
        rng: object,
    ) -> dict[str, hparam_search.Scalar]:
        del model_name, base_args, rng
        return dict(sampled_params)

    monkeypatch.setattr(hparam_search, "_sample_trial_params_with_rng", _fake_sample)
    monkeypatch.setattr(
        hparam_search,
        "_materialize_cnn_architecture_params",
        _fake_materialize,
    )
    monkeypatch.setattr(
        hparam_search,
        "_is_valid_cnn_architecture",
        lambda **kwargs: True,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=2,
        seed_offset=0,
    )

    assert call_count["value"] == 3
    assert params == [
        {"batch_size": 256, "lr": 1e-4},
        {"batch_size": 256, "lr": 2e-4},
    ]


def test_build_trial_params_dnabert_linear_drops_inactive_readout_keys(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "readout_type": {"type": "categorical", "values": ["linear"]},
            "readout_cnn_kernel_size": {"type": "categorical", "values": [3, 5, 7]},
            "readout_mlp_hidden_dim": {
                "type": "categorical",
                "values": [128, 256, 512],
            },
            "readout_mlp_layers": {"type": "int", "min": 1, "max": 3, "step": 1},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=13,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=4,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="pair_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "dnabert_pair",
            "species": "Dmel",
            "batch_size": 16,
            "readout_type": "linear",
            "readout_cnn_kernel_size": 3,
            "readout_mlp_hidden_dim": 256,
            "readout_mlp_layers": 1,
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=1,
        seed_offset=0,
    )[0]

    assert params["readout_type"] == "linear"
    assert "readout_cnn_kernel_size" not in params
    assert "readout_mlp_hidden_dim" not in params
    assert "readout_mlp_layers" not in params


def test_build_trial_params_history_guided_normalizes_dnabert_readout_keys(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "readout_type": {"type": "categorical", "values": ["linear", "mlp"]},
            "readout_cnn_kernel_size": {"type": "categorical", "values": [3, 5, 7]},
            "readout_mlp_hidden_dim": {
                "type": "categorical",
                "values": [128, 256, 512],
            },
            "readout_mlp_layers": {"type": "int", "min": 1, "max": 3, "step": 1},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=17,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=4,
        max_oom_retries=0,
        max_model_params=None,
        objective_metric="pair_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "dnabert_pair",
            "species": "Dmel",
            "batch_size": 16,
            "readout_type": "linear",
            "readout_cnn_kernel_size": 3,
            "readout_mlp_hidden_dim": 256,
            "readout_mlp_layers": 1,
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
        search_algo="history_guided",
        guided_random_fraction=0.0,
        guided_mutation_rate=0.0,
    )
    history_trials = [
        (
            0.8,
            {
                "readout_type": "linear",
                "readout_cnn_kernel_size": 7,
                "readout_mlp_hidden_dim": 512,
                "readout_mlp_layers": 3,
            },
        ),
    ]

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=1,
        seed_offset=0,
        history_trials=history_trials,
    )[0]

    assert params["readout_type"] == "linear"
    assert "readout_cnn_kernel_size" not in params
    assert "readout_mlp_hidden_dim" not in params
    assert "readout_mlp_layers" not in params


def test_build_trial_params_materializes_independent_cnn_architecture(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "conv_depth": {"type": "categorical", "values": [3]},
            "channel_candidates": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "kernel_candidates": {
                "type": "categorical",
                "values": ["3,5,7"],
            },
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=2,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=7,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 256},
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=2,
        seed_offset=0,
    )

    for row in params:
        assert "conv_channels" in row
        assert "kernel_sizes" in row
        assert "conv_depth" not in row
        assert "channel_candidates" not in row
        assert "kernel_candidates" not in row
        channels = [int(value) for value in str(row["conv_channels"]).split(",")]
        kernels = [int(value) for value in str(row["kernel_sizes"]).split(",")]
        assert len(channels) == 3
        assert len(kernels) == 3
        assert all(value in {64, 128, 256} for value in channels)
        assert all(value in {3, 5, 7} for value in kernels)


def test_build_trial_params_materializes_cnn_v2_stride_pool_with_constraints(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "conv_depth": {"type": "categorical", "values": [3]},
            "channel_candidates": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "kernel_candidates": {
                "type": "categorical",
                "values": ["3,5,7"],
            },
            "channel_order": {
                "type": "categorical",
                "values": ["nondecreasing"],
            },
            "kernel_order": {
                "type": "categorical",
                "values": ["nonincreasing"],
            },
            "conv_stride_candidates": {
                "type": "categorical",
                "values": ["1,2"],
            },
            "max_pool_candidates": {
                "type": "categorical",
                "values": ["2,4"],
            },
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=17,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="donor_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn_v2",
            "species": "Dmel",
            "batch_size": 256,
            "donor_len": 8,
            "pair_mode": "independent",
            "train_target": "donor",
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=3,
        seed_offset=0,
    )

    for row in params:
        assert row["conv_stride"] == 1
        assert row["max_pool_size"] == 2
        assert "conv_stride_candidates" not in row
        assert "max_pool_candidates" not in row
        assert "channel_order" not in row
        assert "kernel_order" not in row

        channels = [int(value) for value in str(row["conv_channels"]).split(",")]
        kernels = [int(value) for value in str(row["kernel_sizes"]).split(",")]
        assert channels == sorted(channels)
        assert kernels == sorted(kernels, reverse=True)


def test_build_trial_params_preserves_explicit_cnn_layer_layout(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "conv_channels": {
                "type": "categorical",
                "values": ["64,96,128"],
            },
            "kernel_sizes": {
                "type": "categorical",
                "values": ["11,7,5"],
            },
            "max_pool_size": {"type": "categorical", "values": [1]},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=7,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 256,
            "conv_depth": 4,
            "channel_candidates": "64,128,256,512",
            "kernel_candidates": "3,5,7,9",
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=1,
        seed_offset=0,
    )

    assert params == [
        {
            "batch_size": 256,
            "conv_channels": "64,96,128",
            "kernel_sizes": "11,7,5",
            "max_pool_size": 1,
        }
    ]


def test_build_trial_params_materializes_stride_pool_without_arch_helper_keys(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "conv_channels": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "kernel_sizes": {
                "type": "categorical",
                "values": ["7,7,7"],
            },
            "conv_stride_candidates": {
                "type": "categorical",
                "values": ["1,2"],
            },
            "max_pool_candidates": {
                "type": "categorical",
                "values": ["2,4"],
            },
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=19,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="donor_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn_v2",
            "species": "Dmel",
            "batch_size": 256,
            "donor_len": 8,
            "pair_mode": "independent",
            "train_target": "donor",
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=1,
        seed_offset=0,
    )

    assert params == [
        {
            "batch_size": 256,
            "conv_channels": "64,128,256",
            "kernel_sizes": "7,7,7",
            "conv_stride": 1,
            "max_pool_size": 2,
        }
    ]


def test_build_trial_params_rejects_invalid_cnn_resdil_pool_schedule(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "conv_channels": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "kernel_sizes": {
                "type": "categorical",
                "values": ["7,7,7"],
            },
            "max_pool_size": {"type": "categorical", "values": [2]},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=23,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn_resdil",
            "species": "Dmel",
            "batch_size": 256,
            "donor_len": 3,
            "acceptor_len": 3,
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    with pytest.raises(ValueError, match="valid architecture"):
        _ = hparam_search.build_trial_params(
            config=config,
            phase="quick",
            count=1,
            seed_offset=0,
        )


@pytest.mark.parametrize("fusion_mode", ["early", "mid"])
def test_build_trial_params_materializes_shared_pair_architecture_for_fusion_modes(
    tmp_path: Path,
    fusion_mode: str,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "fusion_mode": {"type": "categorical", "values": [fusion_mode]},
            "donor_conv_depth": {"type": "categorical", "values": [3]},
            "donor_channel_candidates": {
                "type": "categorical",
                "values": ["64,96,128"],
            },
            "donor_kernel_candidates": {
                "type": "categorical",
                "values": ["3,5,7"],
            },
            "acceptor_conv_depth": {"type": "categorical", "values": [5]},
            "acceptor_channel_candidates": {
                "type": "categorical",
                "values": ["512,768,1024"],
            },
            "acceptor_kernel_candidates": {
                "type": "categorical",
                "values": ["11,13,15"],
            },
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=2,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=7,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="pair_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn_pair", "species": "Dmel", "batch_size": 256},
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=2,
        seed_offset=0,
    )

    for row in params:
        assert row["fusion_mode"] == fusion_mode
        assert row["donor_conv_channels"] == row["acceptor_conv_channels"]
        assert row["donor_kernel_sizes"] == row["acceptor_kernel_sizes"]
        assert "donor_conv_depth" not in row
        assert "acceptor_conv_depth" not in row
        assert "donor_channel_candidates" not in row
        assert "acceptor_channel_candidates" not in row
        assert "donor_kernel_candidates" not in row
        assert "acceptor_kernel_candidates" not in row


def test_build_trial_params_preserves_explicit_pair_layer_layout(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "fusion_mode": {"type": "categorical", "values": ["late"]},
            "donor_conv_channels": {
                "type": "categorical",
                "values": ["64,96,128"],
            },
            "acceptor_conv_channels": {
                "type": "categorical",
                "values": ["96,128,128"],
            },
            "donor_kernel_sizes": {
                "type": "categorical",
                "values": ["11,7,5"],
            },
            "acceptor_kernel_sizes": {
                "type": "categorical",
                "values": ["9,7,5"],
            },
            "max_pool_size": {"type": "categorical", "values": [1]},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=7,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn_pair",
            "species": "Dmel",
            "batch_size": 256,
            "donor_conv_depth": 5,
            "acceptor_conv_depth": 4,
            "donor_channel_candidates": "64,128,256,512",
            "acceptor_channel_candidates": "64,128,256,512",
            "donor_kernel_candidates": "3,5,7,9",
            "acceptor_kernel_candidates": "3,5,7,9",
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=1,
        seed_offset=0,
    )

    assert params == [
        {
            "batch_size": 256,
            "fusion_mode": "late",
            "donor_conv_channels": "64,96,128",
            "acceptor_conv_channels": "96,128,128",
            "donor_kernel_sizes": "11,7,5",
            "acceptor_kernel_sizes": "9,7,5",
            "max_pool_size": 1,
        }
    ]


def test_build_trial_params_materializes_cnn_v2_pair_stride_pool_candidates(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "input_mode": {"type": "categorical", "values": ["onehot"]},
            "fusion_mode": {"type": "categorical", "values": ["late"]},
            "conv_depth": {"type": "categorical", "values": [3]},
            "channel_candidates": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "kernel_candidates": {
                "type": "categorical",
                "values": ["7"],
            },
            "conv_stride_candidates": {
                "type": "categorical",
                "values": ["1,2"],
            },
            "max_pool_candidates": {
                "type": "categorical",
                "values": ["2,4"],
            },
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=2,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=37,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="pair_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn_v2_pair",
            "species": "Dmel",
            "batch_size": 256,
            "input_mode": "onehot",
            "pair_mode": "pair",
            "donor_len": 8,
            "acceptor_len": 8,
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=2,
        seed_offset=0,
    )

    for row in params:
        assert row["input_mode"] == "onehot"
        assert row["fusion_mode"] == "late"
        assert row["conv_stride"] == 1
        assert row["max_pool_size"] == 2
        assert "donor_conv_channels" in row
        assert "acceptor_conv_channels" in row
        assert "donor_kernel_sizes" in row
        assert "acceptor_kernel_sizes" in row
        assert "conv_depth" not in row
        assert "channel_candidates" not in row
        assert "kernel_candidates" not in row
        assert "conv_stride_candidates" not in row
        assert "max_pool_candidates" not in row


def test_build_trial_params_resamples_invalid_cnn_pool_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "conv_channels": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "kernel_sizes": {
                "type": "categorical",
                "values": ["7,7,7"],
            },
            "max_pool_size": {"type": "categorical", "values": [2, 4]},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=11,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn",
            "species": "Dmel",
            "batch_size": 256,
            "donor_len": 8,
            "acceptor_len": 8,
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    sampled_rows = iter(
        [
            {
                "batch_size": 256,
                "conv_channels": "64,128,256",
                "kernel_sizes": "7,7,7",
                "max_pool_size": 4,
            },
            {
                "batch_size": 256,
                "conv_channels": "64,128,256",
                "kernel_sizes": "7,7,7",
                "max_pool_size": 2,
            },
        ]
    )
    call_count = {"value": 0}

    def _fake_sample(
        _search_space: dict[str, hparam_search.SearchDimension],
        _rng: object,
    ) -> dict[str, hparam_search.Scalar]:
        call_count["value"] += 1
        return next(sampled_rows)

    monkeypatch.setattr(hparam_search, "_sample_trial_params_with_rng", _fake_sample)

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=1,
        seed_offset=0,
    )

    assert call_count["value"] == 2
    assert params == [
        {
            "batch_size": 256,
            "conv_channels": "64,128,256",
            "kernel_sizes": "7,7,7",
            "max_pool_size": 2,
        }
    ]


def test_build_trial_params_resamples_invalid_cnn_v2_pool_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "conv_channels": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "kernel_sizes": {
                "type": "categorical",
                "values": ["7,7,7"],
            },
            "max_pool_size": {"type": "categorical", "values": [2, 4]},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=11,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn_v2",
            "species": "Dmel",
            "batch_size": 256,
            "donor_len": 8,
            "acceptor_len": 8,
            "pair_mode": "independent",
            "input_mode": "onehot",
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    sampled_rows = iter(
        [
            {
                "batch_size": 256,
                "conv_channels": "64,128,256",
                "kernel_sizes": "7,7,7",
                "max_pool_size": 4,
            },
            {
                "batch_size": 256,
                "conv_channels": "64,128,256",
                "kernel_sizes": "7,7,7",
                "max_pool_size": 2,
            },
        ]
    )
    call_count = {"value": 0}

    def _fake_sample(
        _search_space: dict[str, hparam_search.SearchDimension],
        _rng: object,
    ) -> dict[str, hparam_search.Scalar]:
        call_count["value"] += 1
        return next(sampled_rows)

    monkeypatch.setattr(hparam_search, "_sample_trial_params_with_rng", _fake_sample)

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=1,
        seed_offset=0,
    )

    assert call_count["value"] == 2
    assert params == [
        {
            "batch_size": 256,
            "conv_channels": "64,128,256",
            "kernel_sizes": "7,7,7",
            "max_pool_size": 2,
        }
    ]


def test_build_trial_params_resamples_invalid_cnn_v2_pair_onehot_pool_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "conv_channels": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "kernel_sizes": {
                "type": "categorical",
                "values": ["7,7,7"],
            },
            "max_pool_size": {"type": "categorical", "values": [2, 4]},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=11,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="pair_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn_v2_pair",
            "species": "Dmel",
            "batch_size": 256,
            "donor_len": 8,
            "acceptor_len": 8,
            "input_mode": "onehot",
            "pair_mode": "pair",
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    sampled_rows = iter(
        [
            {
                "batch_size": 256,
                "conv_channels": "64,128,256",
                "kernel_sizes": "7,7,7",
                "max_pool_size": 4,
            },
            {
                "batch_size": 256,
                "conv_channels": "64,128,256",
                "kernel_sizes": "7,7,7",
                "max_pool_size": 2,
            },
        ]
    )
    call_count = {"value": 0}

    def _fake_sample(
        _search_space: dict[str, hparam_search.SearchDimension],
        _rng: object,
    ) -> dict[str, hparam_search.Scalar]:
        call_count["value"] += 1
        return next(sampled_rows)

    monkeypatch.setattr(hparam_search, "_sample_trial_params_with_rng", _fake_sample)

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=1,
        seed_offset=0,
    )

    assert call_count["value"] == 2
    assert params == [
        {
            "batch_size": 256,
            "conv_channels": "64,128,256",
            "kernel_sizes": "7,7,7",
            "max_pool_size": 2,
        }
    ]


def test_build_trial_params_resamples_invalid_pair_fusion_lengths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [256]},
            "fusion_mode": {"type": "categorical", "values": ["early", "late"]},
            "donor_conv_channels": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "acceptor_conv_channels": {
                "type": "categorical",
                "values": ["64,128,256"],
            },
            "donor_kernel_sizes": {
                "type": "categorical",
                "values": ["7,7,7"],
            },
            "acceptor_kernel_sizes": {
                "type": "categorical",
                "values": ["7,7,7"],
            },
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=23,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="pair_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={
            "model": "cnn_pair",
            "species": "Dmel",
            "batch_size": 256,
            "donor_len": 100,
            "acceptor_len": 80,
        },
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    sampled_rows = iter(
        [
            {
                "batch_size": 256,
                "fusion_mode": "early",
                "donor_conv_channels": "64,128,256",
                "acceptor_conv_channels": "64,128,256",
                "donor_kernel_sizes": "7,7,7",
                "acceptor_kernel_sizes": "7,7,7",
            },
            {
                "batch_size": 256,
                "fusion_mode": "late",
                "donor_conv_channels": "64,128,256",
                "acceptor_conv_channels": "64,128,256",
                "donor_kernel_sizes": "7,7,7",
                "acceptor_kernel_sizes": "7,7,7",
            },
        ]
    )
    call_count = {"value": 0}

    def _fake_sample(
        _search_space: dict[str, hparam_search.SearchDimension],
        _rng: object,
    ) -> dict[str, hparam_search.Scalar]:
        call_count["value"] += 1
        return next(sampled_rows)

    monkeypatch.setattr(hparam_search, "_sample_trial_params_with_rng", _fake_sample)

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=1,
        seed_offset=0,
    )

    assert call_count["value"] == 2
    assert params == [
        {
            "batch_size": 256,
            "fusion_mode": "late",
            "donor_conv_channels": "64,128,256",
            "acceptor_conv_channels": "64,128,256",
            "donor_kernel_sizes": "7,7,7",
            "acceptor_kernel_sizes": "7,7,7",
        }
    ]


def test_build_trial_params_respects_max_model_params(
    tmp_path: Path,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [128]},
            "fc_hidden": {"type": "categorical", "values": [128]},
            "conv_depth": {"type": "categorical", "values": [5]},
            "channel_candidates": {
                "type": "categorical",
                "values": ["2048,3072", "32,64"],
            },
            "kernel_candidates": {
                "type": "categorical",
                "values": ["15", "3,5"],
            },
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=3,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=13,
        gpu_ids_setting="auto",
        max_parallel_trials_setting="auto",
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=1_000_000,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 128},
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )

    params = hparam_search.build_trial_params(
        config=config,
        phase="quick",
        count=3,
        seed_offset=0,
    )
    for row in params:
        complexity = hparam_search.estimate_model_param_complexity(
            model_name="cnn",
            sampled_params=row,
            base_args=config.base_args,
        )
        assert complexity is not None
        assert complexity <= 1_000_000


def test_run_phase_subprocess_interrupt_triggers_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [512]},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=1,
        quick_epochs=1,
        top_k=1,
        full_epochs=1,
        base_seed=1337,
        gpu_ids_setting=["0"],
        max_parallel_trials_setting=1,
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 512},
        quick_overrides={},
        full_overrides={},
        search_space=search_space,
    )
    cleanup_calls: dict[str, int] = {"count": 0}

    def _fake_interrupt_active_trial_processes(
        wait_timeout_sec: float = 3.0,
    ) -> None:
        del wait_timeout_sec
        cleanup_calls["count"] += 1

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
        del (
            config,
            phase,
            trial_id,
            sampled_params,
            overrides,
            assigned_gpu_id,
            metrics_json,
            log_file,
        )
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        hparam_search,
        "_interrupt_active_trial_processes",
        _fake_interrupt_active_trial_processes,
    )
    monkeypatch.setattr(hparam_search, "run_trial", _fake_run_trial)

    with pytest.raises(KeyboardInterrupt):
        _ = hparam_search.run_phase(
            phase="quick",
            config=config,
            trial_count=1,
            trial_params=[{"batch_size": 512}],
            overrides={"epochs": 1},
            gpu_ids=["0"],
            max_parallel_trials=1,
            out_dir=tmp_path / "out",
        )
    assert cleanup_calls["count"] == 1


def test_run_quick_full_overlap_subprocess_promotes_full_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_space = hparam_search._validate_search_space(
        {
            "batch_size": {"type": "categorical", "values": [128, 256]},
        }
    )
    config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=tmp_path / "out",
        quick_trials=2,
        quick_epochs=1,
        top_k=2,
        full_epochs=4,
        base_seed=1337,
        gpu_ids_setting=["0", "1"],
        max_parallel_trials_setting=2,
        min_batch_size=64,
        max_oom_retries=1,
        max_model_params=None,
        objective_metric="mean_pr_auc",
        global_best_config_path=None,
        seed_best_config_path=None,
        base_args={"model": "cnn", "species": "Dmel", "batch_size": 128},
        quick_overrides={"epochs": 1},
        full_overrides={"epochs": 4},
        search_space=search_space,
        enable_phase_overlap=True,
    )
    events: list[tuple[str, int, str, float]] = []

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
        del overrides
        start_time = time.monotonic()
        events.append((phase, trial_id, "start", start_time))
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if phase == "quick" and trial_id == 0:
            time.sleep(0.25)
            objective_score = 0.81
        elif phase == "quick":
            time.sleep(0.05)
            objective_score = 0.93
        else:
            time.sleep(0.05)
            objective_score = 0.97
        metrics_json.write_text(
            json.dumps(
                {
                    "donor": {
                        "best_metric": "pr_auc",
                        "best_score": objective_score,
                        "best_epoch": 1,
                    },
                    "acceptor": {
                        "best_metric": "pr_auc",
                        "best_score": objective_score,
                        "best_epoch": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        log_file.write_text("ok", encoding="utf-8")
        end_time = time.monotonic()
        events.append((phase, trial_id, "end", end_time))
        return hparam_search.TrialResult(
            phase=phase,
            trial_id=trial_id,
            status="success",
            gpu_id=assigned_gpu_id,
            sampled_params=sampled_params,
            effective_batch_size=int(sampled_params["batch_size"]),
            oom_retries=0,
            donor_pr_auc=objective_score,
            acceptor_pr_auc=objective_score,
            mean_pr_auc=objective_score,
            objective_metric="mean_pr_auc",
            objective_score=objective_score,
            error_message=None,
            return_code=0,
            duration_sec=end_time - start_time,
            metrics_json=str(metrics_json),
            log_file=str(log_file),
        )

    monkeypatch.setattr(hparam_search, "run_trial", _fake_run_trial)

    quick_rows, full_rows = hparam_search._run_quick_full_overlap_subprocess(
        config=config,
        quick_params=[
            {"batch_size": 128},
            {"batch_size": 256},
        ],
        quick_overrides={"epochs": 1},
        full_overrides={"epochs": 4},
        gpu_ids=["0", "1"],
        max_parallel_trials=2,
        out_dir=config.output_dir,
        seed_best_params=None,
        seed_best_context_mismatch=False,
        global_best_recheck_params=None,
        global_best_recheck_context_mismatch=False,
        full_epochs_value=4,
    )

    quick0_end = next(
        timestamp
        for phase, trial_id, kind, timestamp in events
        if phase == "quick" and trial_id == 0 and kind == "end"
    )
    first_full_start = min(
        timestamp
        for phase, _trial_id, kind, timestamp in events
        if phase == "full" and kind == "start"
    )

    assert len(quick_rows) == 2
    assert len(full_rows) >= 1
    assert first_full_start < quick0_end


def test_main_returns_130_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config_dict(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    cleanup_calls: dict[str, int] = {"count": 0}

    def _fake_run_search(_config: hparam_search.SearchConfig) -> int:
        raise KeyboardInterrupt()

    def _fake_interrupt_active_trial_processes(
        wait_timeout_sec: float = 3.0,
    ) -> None:
        del wait_timeout_sec
        cleanup_calls["count"] += 1

    monkeypatch.setattr(hparam_search, "run_search", _fake_run_search)
    monkeypatch.setattr(
        hparam_search,
        "_interrupt_active_trial_processes",
        _fake_interrupt_active_trial_processes,
    )

    exit_code = hparam_search.main(["--config", str(config_path)])

    assert exit_code == 130
    assert cleanup_calls["count"] == 1


def test_main_installs_sigterm_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config_dict(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls: list[int] = []

    def _fake_signal(signum: int, handler: object) -> object:
        del handler
        calls.append(signum)
        return object()

    monkeypatch.setattr(hparam_search.signal, "signal", _fake_signal)
    monkeypatch.setattr(hparam_search, "run_search", lambda _config: 0)

    exit_code = hparam_search.main(["--config", str(config_path)])

    assert exit_code == 0
    assert signal.SIGINT in calls
    assert signal.SIGTERM in calls
