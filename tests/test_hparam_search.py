from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, cast

import pytest

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


def test_load_config_accepts_pair_objective_metric(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    config["objective_metric"] = "pair_pr_auc"
    config_path = tmp_path / "pair_metric.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hparam_search.load_config(config_path)

    assert loaded.objective_metric == "pair_pr_auc"


def test_load_config_rejects_invalid_search_algo(tmp_path: Path) -> None:
    config = _base_config_dict(tmp_path)
    config["search_algo"] = "surrogate"
    config_path = tmp_path / "bad_algo.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="search_algo"):
        _ = hparam_search.load_config(config_path)


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
        },
    )

    assert "--conv_channels" in cmd
    assert "--kernel_sizes" in cmd
    assert "--conv_depth" not in cmd
    assert "--channel_candidates" not in cmd
    assert "--kernel_candidates" not in cmd


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


def test_load_global_best_params_skips_signature_mismatch(tmp_path: Path) -> None:
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
        expected_validation_signature="deadbeefcafe",
    )
    assert loaded is None


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
    assert payload["validation_signature"] == "feedbeefcafe"
    assert payload["validation_protocol"]["split_type"] == "stratified_site"
    assert float(payload["selection_score"]) == pytest.approx(0.815)
    assert payload["donor_checkpoint_path"] == str(donor_ckpt)
    assert payload["acceptor_checkpoint_path"] == str(acceptor_ckpt)


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
    ) -> list[hparam_search.TrialResult]:
        del config, overrides, gpu_ids, max_parallel_trials, out_dir
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


def test_build_trial_params_history_guided_is_reproducible(
    tmp_path: Path,
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
    ]

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
