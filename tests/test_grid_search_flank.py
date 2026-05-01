from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import grid_search_flank as grid_search_window
from util import data_proc
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


def test_resolve_published_task_metrics_json_ignores_dot_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions_dir = tmp_path / "SpX" / "tuning" / "dnabert2" / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    (versions_dir / "dnabert2.01.json").write_text(
        json.dumps(
            {
                "best_configs": {
                    "donor": {"metrics_json": "."},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        versioned_artifacts,
        "resolve_versions_dir",
        lambda _data_root, _species, _model_name: versions_dir,
    )

    resolved = grid_search_window._resolve_published_task_metrics_json(
        species="SpX",
        model_name="dnabert2",
        published_name="dnabert2.01",
        task="donor",
        fallback_metrics_json=".",
    )

    assert resolved is None


def test_validate_grid_test_prerequisites_requires_partner_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "transcripts.unique.tsv").write_text("", encoding="utf-8")
    (processed_dir / "transcripts.unique.map.tsv").write_text("", encoding="utf-8")
    (processed_dir / "intron_eval_flank10.unique.tsv").write_text(
        "",
        encoding="utf-8",
    )
    class_file = processed_dir / "transcript_class.txt"
    class_file.write_text("", encoding="utf-8")
    ref_gff = raw_dir / "reference.gff3"
    ref_gff.write_text("##gff-version 3\n", encoding="utf-8")
    metrics_json = raw_dir / "published.train.json"
    metrics_json.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        data_proc,
        "species_data_dirs",
        lambda _species: {
            "base": str(tmp_path),
            "processed": str(processed_dir),
            "raw": str(raw_dir),
        },
    )
    monkeypatch.setattr(grid_search_window, "_has_transcript_test_tsv", lambda *_: True)
    monkeypatch.setattr(grid_search_window, "_resolve_class_file", lambda _species: str(class_file))
    monkeypatch.setattr(
        versioned_artifacts,
        "resolve_published_run_assets",
        lambda **_kwargs: {
            "published_name": "cnn_v2.99",
            "metrics_json": str(metrics_json),
            "donor_checkpoint_path": str(raw_dir / "missing_donor.pt"),
            "acceptor_checkpoint_path": str(raw_dir / "acceptor.pt"),
        },
    )

    with pytest.raises(FileNotFoundError, match="missing published donor checkpoint"):
        grid_search_window._validate_grid_test_prerequisites(
            root=tmp_path,
            species="SpX",
            model_name="cnn_v2",
            targets=["acceptor"],
        )


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


def test_main_figures_only_recovers_stale_right_panel_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_cell = grid_search_window.CellResult(
        upstream=10,
        downstream=10,
        target="acceptor",
        val_max_f1=0.8,
        test_site_max_f1=0.87,
        test_transcript_max_f1=None,
        status="done",
    )
    recovered_cell = grid_search_window.CellResult(
        upstream=10,
        downstream=10,
        target="acceptor",
        val_max_f1=0.8,
        test_site_max_f1=0.87,
        test_transcript_max_f1=0.9,
        status="done",
    )
    plotted_cells: list[grid_search_window.CellResult] = []

    monkeypatch.setattr(grid_search_window, "TARGETS", ["acceptor"])
    monkeypatch.setattr(grid_search_window, "CELLS_PER_TARGET", 1)
    monkeypatch.setattr(grid_search_window, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        grid_search_window, "_has_transcript_test_tsv", lambda *_args: True
    )
    monkeypatch.setattr(
        grid_search_window,
        "_load_results",
        lambda _path: {"acceptor_10_10": stale_cell},
    )
    monkeypatch.setattr(
        grid_search_window,
        "_load_cells_from_trial_metrics",
        lambda **_kwargs: [recovered_cell],
    )
    monkeypatch.setattr(
        grid_search_window,
        "_save_results",
        lambda _cells, _path: None,
    )
    monkeypatch.setattr(
        grid_search_window,
        "plot_grid",
        lambda **kwargs: plotted_cells.extend(kwargs["cells"]),
    )

    exit_code = grid_search_window.main(
        [
            "--species",
            "Athal",
            "--target",
            "acceptor",
            "--gpus",
            "0",
            "--epochs",
            "1",
            "--figures_only",
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert len(plotted_cells) == 1
    assert plotted_cells[0].test_transcript_max_f1 == pytest.approx(0.9)


def test_main_figures_only_skips_grid_test_prereq_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plotted_cells: list[grid_search_window.CellResult] = []

    monkeypatch.setattr(grid_search_window, "TARGETS", ["acceptor"])
    monkeypatch.setattr(grid_search_window, "CELLS_PER_TARGET", 1)
    monkeypatch.setattr(grid_search_window, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        grid_search_window, "_has_transcript_test_tsv", lambda *_args: True
    )
    monkeypatch.setattr(
        grid_search_window,
        "_validate_grid_test_prerequisites",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    monkeypatch.setattr(
        grid_search_window,
        "_load_results",
        lambda _path: {
            "acceptor_10_10": grid_search_window.CellResult(
                upstream=10,
                downstream=10,
                target="acceptor",
                val_max_f1=0.8,
                test_site_max_f1=0.87,
                test_transcript_max_f1=0.9,
                status="done",
            )
        },
    )
    monkeypatch.setattr(
        grid_search_window,
        "_save_results",
        lambda _cells, _path: None,
    )
    monkeypatch.setattr(
        grid_search_window,
        "plot_grid",
        lambda **kwargs: plotted_cells.extend(kwargs["cells"]),
    )

    exit_code = grid_search_window.main(
        [
            "--species",
            "Athal",
            "--target",
            "acceptor",
            "--gpus",
            "0",
            "--epochs",
            "1",
            "--figures_only",
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert len(plotted_cells) == 1


@pytest.mark.parametrize(
    ("target", "published_metrics_side", "expected_upstream", "expected_downstream"),
    [
        ("donor", "donor", 23, 24),
        ("acceptor", "acceptor", 11, 12),
    ],
)
def test_compute_grid_test_metrics_uses_partner_side_snapshot_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    published_metrics_side: str,
    expected_upstream: int,
    expected_downstream: int,
) -> None:
    import evaluate_scores
    import run_model
    from tools import hparam_search
    from util import transcript_eval

    trial_dir = tmp_path / target
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = trial_dir / "full_trial_0000.metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "batch_size": 512,
                "donor_upstream": 31,
                "donor_downstream": 32,
                "acceptor_upstream": 41,
                "acceptor_downstream": 42,
            }
        ),
        encoding="utf-8",
    )
    trial_checkpoint = tmp_path / "trial.pt"
    trial_checkpoint.write_bytes(b"trial")
    partner_checkpoint = tmp_path / "partner.pt"
    partner_checkpoint.write_bytes(b"partner")
    donor_metrics_path = tmp_path / "donor.metrics.json"
    donor_metrics_path.write_text(
        json.dumps(
            {
                "donor_upstream": 11,
                "donor_downstream": 12,
                "acceptor_upstream": 13,
                "acceptor_downstream": 14,
            }
        ),
        encoding="utf-8",
    )
    acceptor_metrics_path = tmp_path / "acceptor.metrics.json"
    acceptor_metrics_path.write_text(
        json.dumps(
            {
                "donor_upstream": 21,
                "donor_downstream": 22,
                "acceptor_upstream": 23,
                "acceptor_downstream": 24,
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = (
        tmp_path / "data" / "Athal" / "tuning" / "cnn_v2" / "versions" / "cnn_v2.12.json"
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "best_configs": {
                    "donor": {"metrics_json": str(donor_metrics_path)},
                    "acceptor": {"metrics_json": str(acceptor_metrics_path)},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(grid_search_window, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        hparam_search,
        "_extract_checkpoint_paths_from_metrics",
        lambda _path: {f"{target}_checkpoint_path": str(trial_checkpoint)},
    )
    monkeypatch.setattr(
        versioned_artifacts,
        "resolve_published_run_assets",
        lambda **_kwargs: {
            "published_name": "cnn_v2.12",
            "donor_checkpoint_path": str(partner_checkpoint),
            "acceptor_checkpoint_path": str(partner_checkpoint),
            "metrics_json": str(
                donor_metrics_path
                if published_metrics_side == "donor"
                else acceptor_metrics_path
            ),
        },
    )
    monkeypatch.setattr(
        run_model,
        "_load_optional_intron_labels",
        lambda _species: {},
    )
    monkeypatch.setattr(
        run_model,
        "_load_required_unique_intron_map",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        run_model,
        "_expand_unique_site_rows",
        lambda site_score_rows, unique_map: site_score_rows,
    )
    monkeypatch.setattr(
        run_model,
        "_resolve_ref_gff_file",
        lambda _species, _version: str(tmp_path / "ref.gff"),
    )
    monkeypatch.setattr(
        transcript_eval,
        "aggregate_transcript_scores",
        lambda **_kwargs: [{"transcript_id": "tx1", "score": 0.5}],
    )
    monkeypatch.setattr(
        transcript_eval,
        "write_transcript_scores",
        lambda path, rows: Path(path).write_text("ok\n", encoding="utf-8"),
    )
    monkeypatch.setattr(
        evaluate_scores,
        "evaluate_score_file",
        lambda **_kwargs: ["max_f1=0.88"],
    )
    monkeypatch.setattr(
        grid_search_window,
        "_extract_max_f1_from_eval_lines",
        lambda _lines: 0.88,
    )
    monkeypatch.setattr(
        grid_search_window,
        "_compute_site_max_f1_from_rows",
        lambda **_kwargs: 0.91,
    )
    monkeypatch.setattr(
        grid_search_window,
        "_resolve_class_file",
        lambda _species: str(tmp_path / "class.txt"),
    )

    captured_windows: list[dict[str, int | None]] = []

    def _fake_infer_site_rows_for_grid(**kwargs: object) -> list[dict[str, object]]:
        captured_windows.append(dict(kwargs["window_config"]))
        return [
            {"site_type": "donor"},
            {"site_type": "acceptor"},
        ]

    monkeypatch.setattr(
        grid_search_window,
        "_infer_site_rows_for_grid",
        _fake_infer_site_rows_for_grid,
    )

    result = grid_search_window._compute_grid_test_metrics(
        species="Athal",
        model_name="cnn_v2",
        target=target,
        metrics_json=str(metrics_path),
    )

    assert result["test_site_max_f1"] == pytest.approx(0.91)
    assert result["test_transcript_max_f1"] == pytest.approx(0.88)
    assert len(captured_windows) == 2
    partner_window = captured_windows[1]
    assert partner_window[f"{target}_upstream"] == expected_upstream
    assert partner_window[f"{target}_downstream"] == expected_downstream


def test_compute_grid_test_metrics_tolerates_missing_latest_published_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import hparam_search

    trial_dir = tmp_path / "donor"
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = trial_dir / "full_trial_0000.metrics.json"
    metrics_path.write_text(json.dumps({"batch_size": 512}), encoding="utf-8")

    monkeypatch.setattr(
        hparam_search,
        "_extract_checkpoint_paths_from_metrics",
        lambda _path: {"donor_checkpoint_path": str(tmp_path / "trial.pt")},
    )
    monkeypatch.setattr(
        versioned_artifacts,
        "resolve_published_run_assets",
        lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
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
        (trial_dir / "full_trial_0000.metrics.grid_eval.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == result


def test_compute_grid_test_metrics_allows_missing_partner_metrics_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluate_scores
    import run_model
    from tools import hparam_search
    from util import transcript_eval

    trial_dir = tmp_path / "donor"
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = trial_dir / "full_trial_0000.metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "batch_size": 512,
                "donor_upstream": 31,
                "donor_downstream": 32,
            }
        ),
        encoding="utf-8",
    )
    trial_checkpoint = tmp_path / "trial.pt"
    partner_checkpoint = tmp_path / "partner.pt"
    trial_checkpoint.write_bytes(b"trial")
    partner_checkpoint.write_bytes(b"partner")

    monkeypatch.setattr(grid_search_window, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        hparam_search,
        "_extract_checkpoint_paths_from_metrics",
        lambda _path: {"donor_checkpoint_path": str(trial_checkpoint)},
    )
    monkeypatch.setattr(
        versioned_artifacts,
        "resolve_published_run_assets",
        lambda **_kwargs: {
            "published_name": "dnabert2.01",
            "donor_checkpoint_path": str(partner_checkpoint),
            "acceptor_checkpoint_path": str(partner_checkpoint),
            "metrics_json": ".",
        },
    )
    monkeypatch.setattr(run_model, "_load_optional_intron_labels", lambda _species: {})
    monkeypatch.setattr(
        run_model,
        "_load_required_unique_intron_map",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        run_model,
        "_expand_unique_site_rows",
        lambda site_score_rows, unique_map: site_score_rows,
    )
    monkeypatch.setattr(
        run_model,
        "_resolve_ref_gff_file",
        lambda _species, _version: str(tmp_path / "ref.gff"),
    )
    monkeypatch.setattr(
        transcript_eval,
        "aggregate_transcript_scores",
        lambda **_kwargs: [{"transcript_id": "tx1", "score": 0.5}],
    )
    monkeypatch.setattr(
        transcript_eval,
        "write_transcript_scores",
        lambda path, rows: Path(path).write_text("ok\n", encoding="utf-8"),
    )
    monkeypatch.setattr(
        evaluate_scores,
        "evaluate_score_file",
        lambda **_kwargs: ["max_f1=0.77"],
    )
    monkeypatch.setattr(
        grid_search_window,
        "_extract_max_f1_from_eval_lines",
        lambda _lines: 0.77,
    )
    monkeypatch.setattr(
        grid_search_window,
        "_compute_site_max_f1_from_rows",
        lambda **_kwargs: 0.91,
    )
    monkeypatch.setattr(
        grid_search_window,
        "_resolve_class_file",
        lambda _species: str(tmp_path / "class.txt"),
    )
    monkeypatch.setattr(
        grid_search_window,
        "_infer_site_rows_for_grid",
        lambda **_kwargs: [
            {"site_type": "donor"},
            {"site_type": "acceptor"},
        ],
    )

    result = grid_search_window._compute_grid_test_metrics(
        species="Athal",
        model_name="dnabert2",
        target="donor",
        metrics_json=str(metrics_path),
    )

    assert result["test_site_max_f1"] == pytest.approx(0.91)
    assert result["test_transcript_max_f1"] == pytest.approx(0.77)
