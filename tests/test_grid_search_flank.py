from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import grid_search_flank as grid_search_window
from util import versioned_artifacts


def test_delete_trial_checkpoints_removes_referenced_files(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "donor_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    metrics_path = tmp_path / "trial.metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "donor": {
                    "checkpoint": str(checkpoint_path),
                }
            }
        ),
        encoding="utf-8",
    )

    deleted_count = grid_search_window._delete_trial_checkpoints(str(metrics_path))

    assert deleted_count == 1
    assert not checkpoint_path.exists()


def test_run_grid_target_passes_epochs_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import hparam_search

    metrics_path = tmp_path / "full_trial_0000.metrics.json"
    metrics_path.write_text(
        json.dumps({"donor": {"best_max_f1": 0.75}}),
        encoding="utf-8",
    )
    captured_overrides: list[dict[str, object]] = []
    captured_objective_metrics: list[str] = []

    monkeypatch.setattr(grid_search_window, "UPSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "DOWNSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "CELLS_PER_TARGET", 1)
    monkeypatch.setattr(
        grid_search_window, "_has_transcript_test_tsv", lambda *_: False
    )
    monkeypatch.setattr(
        grid_search_window,
        "_compute_grid_test_metrics",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        hparam_search,
        "detect_gpu_ids",
        lambda _setting: ["0"],
    )
    monkeypatch.setattr(
        hparam_search,
        "resolve_max_parallel",
        lambda _setting, _gpu_count: 1,
    )

    def _fake_run_phase(**kwargs: object) -> list[hparam_search.TrialResult]:
        captured_overrides.append(dict(kwargs["overrides"]))
        captured_objective_metrics.append(kwargs["config"].objective_metric)
        return [
            hparam_search.TrialResult(
                phase="full",
                trial_id=0,
                status="success",
                gpu_id="0",
                sampled_params={"donor_upstream": 10, "donor_downstream": 10},
                effective_batch_size=512,
                oom_retries=0,
                donor_pr_auc=0.8,
                acceptor_pr_auc=None,
                mean_pr_auc=0.8,
                objective_metric="donor_max_f1",
                objective_score=0.81,
                error_message=None,
                return_code=0,
                duration_sec=1.0,
                metrics_json=str(metrics_path),
                log_file=str(tmp_path / "trial.log"),
            )
        ]

    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)

    cells = grid_search_window._run_grid_target(
        species="Athal",
        target="donor",
        gpu_ids=["0"],
        max_parallel=1,
        epochs=15,
        seed=1337,
        batch_size=512,
        val_frac=0.2,
        output_dir=tmp_path,
        results_path=tmp_path / "results.json",
        existing={},
        model="cnn_v2",
        compile_mode="off",
        infer_compile=0,
        infer_compile_mode="off",
        pretrained_model_name="",
        pretrained_revision="",
        trust_remote_code=1,
        max_tokens="auto",
        head_layer_norm=1,
    )

    assert captured_overrides == [{"epochs": 15}]
    assert captured_objective_metrics == ["donor_max_f1"]
    assert len(cells) == 1
    assert cells[0].val_max_f1 == pytest.approx(0.75)
    assert cells[0].test_max_f1 is None


def test_run_grid_target_passes_dnabert2_base_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import hparam_search

    metrics_path = tmp_path / "full_trial_0000.metrics.json"
    metrics_path.write_text(
        json.dumps({"donor": {"best_max_f1": 0.75}}),
        encoding="utf-8",
    )
    captured_base_args: list[dict[str, object]] = []

    monkeypatch.setattr(grid_search_window, "UPSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "DOWNSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "CELLS_PER_TARGET", 1)
    monkeypatch.setattr(
        grid_search_window, "_has_transcript_test_tsv", lambda *_: False
    )
    monkeypatch.setattr(
        grid_search_window,
        "_compute_grid_test_metrics",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(hparam_search, "detect_gpu_ids", lambda _setting: ["0"])
    monkeypatch.setattr(
        hparam_search,
        "resolve_max_parallel",
        lambda _setting, _gpu_count: 1,
    )

    def _fake_run_phase(**kwargs: object) -> list[hparam_search.TrialResult]:
        config = kwargs["config"]
        captured_base_args.append(dict(config.base_args))
        return [
            hparam_search.TrialResult(
                phase="full",
                trial_id=0,
                status="success",
                gpu_id="0",
                sampled_params={"donor_upstream": 10, "donor_downstream": 10},
                effective_batch_size=64,
                oom_retries=0,
                donor_pr_auc=0.8,
                acceptor_pr_auc=None,
                mean_pr_auc=0.8,
                objective_metric="donor_max_f1",
                objective_score=0.81,
                error_message=None,
                return_code=0,
                duration_sec=1.0,
                metrics_json=str(metrics_path),
                log_file=str(tmp_path / "trial.log"),
            )
        ]

    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)

    grid_search_window._run_grid_target(
        species="Dmel",
        target="donor",
        gpu_ids=["0"],
        max_parallel=1,
        epochs=6,
        seed=1337,
        batch_size=64,
        val_frac=0.2,
        output_dir=tmp_path,
        results_path=tmp_path / "results.json",
        existing={},
        model="dnabert2",
        compile_mode="off",
        infer_compile=0,
        infer_compile_mode="off",
        pretrained_model_name="/tmp/dnabert2",
        pretrained_revision="",
        trust_remote_code=1,
        max_tokens="auto",
        head_layer_norm=1,
    )

    assert captured_base_args == [
        {
            "model": "dnabert2",
            "species": "Dmel",
            "train_target": "donor",
            "donor_len": 100,
            "acceptor_len": 100,
            "seed": 1337,
            "batch_size": 64,
            "val_frac": 0.2,
            "train_pos_path": "data/Dmel/processed/site_flank100.coding.err",
            "train_neg_path": "data/Dmel/processed/site_flank100.neg.err",
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
            "compile_mode": "off",
            "infer_compile": 0,
            "infer_compile_mode": "off",
            "pretrained_model_name": "/tmp/dnabert2",
            "pretrained_revision": "",
            "trust_remote_code": 1,
            "max_tokens": "auto",
            "head_layer_norm": 1,
        }
    ]


def test_run_grid_target_computes_test_metrics_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import hparam_search

    metrics_path = tmp_path / "full_trial_0000.metrics.json"
    metrics_path.write_text(
        json.dumps({"donor": {"best_max_f1": 0.74}}),
        encoding="utf-8",
    )
    captured_objective_metrics: list[str] = []

    monkeypatch.setattr(grid_search_window, "UPSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "DOWNSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "CELLS_PER_TARGET", 1)
    monkeypatch.setattr(grid_search_window, "_has_transcript_test_tsv", lambda *_: True)
    monkeypatch.setattr(
        grid_search_window,
        "_compute_grid_test_metrics",
        lambda **_kwargs: {
            "test_site_max_f1": 0.89,
            "test_transcript_max_f1": 0.91,
        },
    )
    monkeypatch.setattr(hparam_search, "detect_gpu_ids", lambda _setting: ["0"])
    monkeypatch.setattr(
        hparam_search,
        "resolve_max_parallel",
        lambda _setting, _gpu_count: 1,
    )

    def _fake_run_phase(**kwargs: object) -> list[hparam_search.TrialResult]:
        captured_objective_metrics.append(kwargs["config"].objective_metric)
        return [
            hparam_search.TrialResult(
                phase="full",
                trial_id=0,
                status="success",
                gpu_id="0",
                sampled_params={"donor_upstream": 10, "donor_downstream": 10},
                effective_batch_size=512,
                oom_retries=0,
                donor_pr_auc=0.82,
                acceptor_pr_auc=None,
                mean_pr_auc=0.82,
                objective_metric="donor_max_f1",
                objective_score=0.74,
                error_message=None,
                return_code=0,
                duration_sec=1.0,
                metrics_json=str(metrics_path),
                log_file=str(tmp_path / "trial.log"),
            )
        ]

    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)

    cells = grid_search_window._run_grid_target(
        species="Athal",
        target="donor",
        gpu_ids=["0"],
        max_parallel=1,
        epochs=15,
        seed=1337,
        batch_size=512,
        val_frac=0.2,
        output_dir=tmp_path,
        results_path=tmp_path / "results.json",
        existing={},
        model="cnn_v2",
        compile_mode="off",
        infer_compile=0,
        infer_compile_mode="off",
        pretrained_model_name="",
        pretrained_revision="",
        trust_remote_code=1,
        max_tokens="auto",
        head_layer_norm=1,
    )

    assert captured_objective_metrics == ["donor_max_f1"]
    assert len(cells) == 1
    assert cells[0].val_pr_auc == pytest.approx(0.82)
    assert cells[0].val_max_f1 == pytest.approx(0.74)
    assert cells[0].test_site_max_f1 == pytest.approx(0.89)
    assert cells[0].test_transcript_max_f1 == pytest.approx(0.91)
    assert cells[0].test_max_f1 == pytest.approx(0.91)


def test_run_grid_target_uses_validation_max_f1_for_dnabert2_when_test_tsv_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import hparam_search

    metrics_path = tmp_path / "full_trial_0000.metrics.json"
    metrics_path.write_text(
        json.dumps({"donor": {"best_max_f1": 0.77}}),
        encoding="utf-8",
    )
    captured_objective_metrics: list[str] = []

    monkeypatch.setattr(grid_search_window, "UPSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "DOWNSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "CELLS_PER_TARGET", 1)
    monkeypatch.setattr(grid_search_window, "_has_transcript_test_tsv", lambda *_: True)
    monkeypatch.setattr(
        grid_search_window,
        "_compute_grid_test_metrics",
        lambda **_kwargs: {
            "test_site_max_f1": 0.84,
            "test_transcript_max_f1": None,
        },
    )
    monkeypatch.setattr(hparam_search, "detect_gpu_ids", lambda _setting: ["0"])
    monkeypatch.setattr(
        hparam_search,
        "resolve_max_parallel",
        lambda _setting, _gpu_count: 1,
    )

    def _fake_run_phase(**kwargs: object) -> list[hparam_search.TrialResult]:
        captured_objective_metrics.append(kwargs["config"].objective_metric)
        return [
            hparam_search.TrialResult(
                phase="full",
                trial_id=0,
                status="success",
                gpu_id="0",
                sampled_params={"donor_upstream": 10, "donor_downstream": 10},
                effective_batch_size=64,
                oom_retries=0,
                donor_pr_auc=0.83,
                acceptor_pr_auc=None,
                mean_pr_auc=0.83,
                objective_metric="donor_max_f1",
                objective_score=0.77,
                error_message=None,
                return_code=0,
                duration_sec=1.0,
                metrics_json=str(metrics_path),
                log_file=str(tmp_path / "trial.log"),
            )
        ]

    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)

    cells = grid_search_window._run_grid_target(
        species="Athal",
        target="donor",
        gpu_ids=["0"],
        max_parallel=1,
        epochs=6,
        seed=1337,
        batch_size=64,
        val_frac=0.2,
        output_dir=tmp_path,
        results_path=tmp_path / "results.json",
        existing={},
        model="dnabert2",
        compile_mode="off",
        infer_compile=0,
        infer_compile_mode="off",
        pretrained_model_name="/tmp/dnabert2",
        pretrained_revision="",
        trust_remote_code=1,
        max_tokens="auto",
        head_layer_norm=1,
    )

    assert captured_objective_metrics == ["donor_max_f1"]
    assert len(cells) == 1
    assert cells[0].val_max_f1 == pytest.approx(0.77)
    assert cells[0].test_site_max_f1 == pytest.approx(0.84)
    assert cells[0].test_transcript_max_f1 is None
    assert cells[0].test_max_f1 is None


def test_run_grid_target_reruns_cached_cells_missing_test_site_max_f1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import hparam_search

    metrics_path = tmp_path / "full_trial_0000.metrics.json"
    metrics_path.write_text(
        json.dumps({"donor": {"best_max_f1": 0.73}}),
        encoding="utf-8",
    )
    run_phase_calls = 0

    monkeypatch.setattr(grid_search_window, "UPSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "DOWNSTREAM_VALS", [10])
    monkeypatch.setattr(grid_search_window, "CELLS_PER_TARGET", 1)
    monkeypatch.setattr(grid_search_window, "_has_transcript_test_tsv", lambda *_: True)
    monkeypatch.setattr(
        grid_search_window,
        "_compute_grid_test_metrics",
        lambda **_kwargs: {
            "test_site_max_f1": 0.87,
            "test_transcript_max_f1": 0.9,
        },
    )
    monkeypatch.setattr(hparam_search, "detect_gpu_ids", lambda _setting: ["0"])
    monkeypatch.setattr(
        hparam_search,
        "resolve_max_parallel",
        lambda _setting, _gpu_count: 1,
    )

    def _fake_run_phase(**kwargs: object) -> list[hparam_search.TrialResult]:
        nonlocal run_phase_calls
        run_phase_calls += 1
        return [
            hparam_search.TrialResult(
                phase="full",
                trial_id=0,
                status="success",
                gpu_id="0",
                sampled_params={"donor_upstream": 10, "donor_downstream": 10},
                effective_batch_size=512,
                oom_retries=0,
                donor_pr_auc=0.81,
                acceptor_pr_auc=None,
                mean_pr_auc=0.81,
                objective_metric="donor_max_f1",
                objective_score=0.73,
                error_message=None,
                return_code=0,
                duration_sec=1.0,
                metrics_json=str(metrics_path),
                log_file=str(tmp_path / "trial.log"),
            )
        ]

    monkeypatch.setattr(hparam_search, "run_phase", _fake_run_phase)

    existing = {
        "donor_10_10": grid_search_window.CellResult(
            upstream=10,
            downstream=10,
            target="donor",
            val_max_f1=0.72,
            val_pr_auc=0.8,
            test_site_max_f1=None,
            test_transcript_max_f1=0.88,
            status="done",
        )
    }
    cells = grid_search_window._run_grid_target(
        species="Athal",
        target="donor",
        gpu_ids=["0"],
        max_parallel=1,
        epochs=15,
        seed=1337,
        batch_size=512,
        val_frac=0.2,
        output_dir=tmp_path,
        results_path=tmp_path / "results.json",
        existing=existing,
        model="cnn_v2",
        compile_mode="off",
        infer_compile=0,
        infer_compile_mode="off",
        pretrained_model_name="",
        pretrained_revision="",
        trust_remote_code=1,
        max_tokens="auto",
        head_layer_norm=1,
    )

    assert run_phase_calls == 1
    assert len(cells) == 1
    assert cells[0].val_max_f1 == pytest.approx(0.73)
    assert cells[0].test_site_max_f1 == pytest.approx(0.87)
    assert cells[0].test_transcript_max_f1 == pytest.approx(0.9)
    assert cells[0].test_max_f1 == pytest.approx(0.9)


def test_load_cells_from_trial_metrics_recovers_target_cells(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "acceptor"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "full_trial_0000.metrics.json").write_text(
        json.dumps(
            {
                "acceptor_upstream": 30,
                "acceptor_downstream": 40,
                "acceptor": {
                    "best_max_f1": 0.82,
                    "best_pr_auc": 0.91,
                    "best_score": 0.88,
                },
            }
        ),
        encoding="utf-8",
    )
    (target_dir / "full_trial_0000.metrics.grid_eval.json").write_text(
        json.dumps(
            {
                "test_site_max_f1": 0.86,
                "test_transcript_max_f1": 0.88,
            }
        ),
        encoding="utf-8",
    )

    cells = grid_search_window._load_cells_from_trial_metrics(
        output_dir=tmp_path,
        target="acceptor",
        has_test=True,
        model_name="cnn_v2",
    )

    assert len(cells) == 1
    assert cells[0].target == "acceptor"
    assert cells[0].upstream == 30
    assert cells[0].downstream == 40
    assert cells[0].val_max_f1 == pytest.approx(0.82)
    assert cells[0].val_pr_auc == pytest.approx(0.91)
    assert cells[0].test_site_max_f1 == pytest.approx(0.86)
    assert cells[0].test_transcript_max_f1 == pytest.approx(0.88)
    assert cells[0].test_max_f1 == pytest.approx(0.88)


def test_main_eta_counts_recovered_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(grid_search_window, "TARGETS", ["donor"])
    monkeypatch.setattr(grid_search_window, "CELLS_PER_TARGET", 2)
    monkeypatch.setattr(grid_search_window, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        grid_search_window, "_has_transcript_test_tsv", lambda *_args: False
    )
    monkeypatch.setattr(grid_search_window, "_load_results", lambda _path: {})
    monkeypatch.setattr(
        grid_search_window,
        "_load_cells_from_trial_metrics",
        lambda **_kwargs: [
            grid_search_window.CellResult(
                upstream=10,
                downstream=10,
                target="donor",
                val_max_f1=0.8,
                status="done",
            )
        ],
    )
    monkeypatch.setattr(grid_search_window, "plot_grid", lambda **_kwargs: None)

    def _fake_run_grid_target(**kwargs: object) -> list[grid_search_window.CellResult]:
        callback = kwargs["on_trial_complete"]
        callback(
            SimpleNamespace(
                sampled_params={"donor_upstream": 20, "donor_downstream": 20},
                status="success",
                duration_sec=1.0,
            )
        )
        return [
            grid_search_window.CellResult(
                upstream=10,
                downstream=10,
                target="donor",
                val_max_f1=0.8,
                status="done",
            ),
            grid_search_window.CellResult(
                upstream=20,
                downstream=20,
                target="donor",
                val_max_f1=0.81,
                status="done",
            ),
        ]

    monkeypatch.setattr(grid_search_window, "_run_grid_target", _fake_run_grid_target)

    exit_code = grid_search_window.main(
        [
            "--species",
            "Athal",
            "--target",
            "donor",
            "--gpus",
            "0",
            "--epochs",
            "1",
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "cached=1 live=1" in out
    assert "global=2/2" in out


def test_compute_grid_test_metrics_tolerates_missing_latest_published_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import hparam_search

    metrics_path = tmp_path / "full_trial_0000.metrics.json"
    metrics_path.write_text(json.dumps({"batch_size": 512}), encoding="utf-8")

    monkeypatch.setattr(
        hparam_search,
        "_extract_checkpoint_paths_from_metrics",
        lambda _path: {"donor_checkpoint_path": str(tmp_path / "trial.pt")},
    )
    monkeypatch.setattr(
        versioned_artifacts,
        "resolve_latest_published_run_assets",
        lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(
        versioned_artifacts,
        "resolve_latest_published_name",
        lambda *_args, **_kwargs: None,
    )

    result = grid_search_window._compute_grid_test_metrics(
        species="Athal",
        model_name="cnn_v2",
        target="donor",
        metrics_json=str(metrics_path),
    )

    assert result == {
        "test_site_max_f1": None,
        "test_transcript_max_f1": None,
    }
    saved = json.loads(
        (tmp_path / "full_trial_0000.metrics.grid_eval.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == result
