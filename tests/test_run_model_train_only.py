from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_model
from util.versioned_artifacts import VersionHistoryEntry
from util.versioned_artifacts import write_version_history


class _DummyModelModule:
    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def add_infer_args(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def train(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> dict[str, object]:
        del model_args
        return {
            "donor": {
                "checkpoint": str(common_args.donor_checkpoint_path),
                "best_metric": "pr_auc",
                "best_score": 0.91,
            },
            "acceptor": {
                "checkpoint": str(common_args.acceptor_checkpoint_path),
                "best_metric": "pr_auc",
                "best_score": 0.89,
            },
        }

    def infer_site(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> list[dict[str, object]]:
        del common_args, model_args
        raise AssertionError("infer_site must not run when --train_only is set.")


class _DummySingleTaskModelModule:
    def __init__(self) -> None:
        self.train_targets: list[str] = []

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def add_infer_args(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def train(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> dict[str, object]:
        assert common_args is model_args
        self.train_targets.append(str(model_args.train_target))
        return {
            "acceptor": {
                "checkpoint": str(common_args.acceptor_checkpoint_path),
                "best_metric": "pr_auc",
                "best_score": 0.87,
            },
        }

    def infer_site(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> list[dict[str, object]]:
        del common_args, model_args
        raise AssertionError("infer_site must not run when --train_only is set.")


class _DummyInitCaptureModelModule:
    def __init__(self, *, improved_score: float = 0.91) -> None:
        self.calls: list[dict[str, str]] = []
        self.improved_score = improved_score

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def add_infer_args(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def train(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> dict[str, object]:
        del model_args
        self.calls.append(
            {
                "donor_init": str(common_args.donor_init_checkpoint_path),
                "acceptor_init": str(common_args.acceptor_init_checkpoint_path),
                "donor_checkpoint": str(common_args.donor_checkpoint_path),
                "acceptor_checkpoint": str(common_args.acceptor_checkpoint_path),
            }
        )
        Path(common_args.donor_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(common_args.acceptor_checkpoint_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(common_args.donor_checkpoint_path).write_bytes(b"donor-trained")
        Path(common_args.acceptor_checkpoint_path).write_bytes(b"acceptor-trained")
        return {
            "donor": {
                "checkpoint": str(common_args.donor_checkpoint_path),
                "best_metric": "pr_auc",
                "best_score": self.improved_score,
            },
            "acceptor": {
                "checkpoint": str(common_args.acceptor_checkpoint_path),
                "best_metric": "pr_auc",
                "best_score": self.improved_score - 0.01,
            },
        }

    def infer_site(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> list[dict[str, object]]:
        del common_args, model_args
        raise AssertionError("infer_site must not run when --train_only is set.")


def test_build_parser_defaults_intron_score_op_to_plus() -> None:
    """Default pipeline intron-score operator should be log-space addition."""
    parser = run_model._build_parser(
        selected_model="cnn",
        skip_model_import_error=True,
    )

    args = parser.parse_args([])

    assert args.intron_score_op == "+"


def test_build_parser_accepts_cnn_v3_task_specific_pool_every() -> None:
    """cnn_v3 parser should accept task-specific residual pooling overrides."""
    parser = run_model._build_parser(
        selected_model="cnn_v3",
        skip_model_import_error=True,
    )

    args = parser.parse_args(
        [
            "--model",
            "cnn_v3",
            "--pool_every",
            "2",
            "--donor_pool_every",
            "1",
            "--acceptor_pool_every",
            "3",
        ]
    )

    assert args.pool_every == 2
    assert args.donor_pool_every == 1
    assert args.acceptor_pool_every == 3


def test_run_pipeline_train_only_skips_infer_and_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_json = tmp_path / "train_summary.json"
    parser = run_model._build_parser(
        selected_model="cnn",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn",
            "--species",
            "Dmel",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--epochs",
            "1",
            "--metrics_json",
            str(metrics_json),
            "--train_only",
        ]
    )

    def _load_dummy_model_module(model_name: str) -> _DummyModelModule:
        assert model_name == "cnn"
        return _DummyModelModule()

    def _fail_ref_gff(
        species: str,
        configured_path: str | None,
    ) -> str:
        del species, configured_path
        raise AssertionError(
            "ref_gff resolution must not run in train-only mode."
        )

    def _fail_aggregate(
        site_score_rows: list[dict[str, object]],
        intron_score_op: str,
        transcript_score_agg: str,
        softmin_tau: float,
    ) -> list[dict[str, object]]:
        del site_score_rows, intron_score_op, transcript_score_agg, softmin_tau
        raise AssertionError("transcript aggregation must not run in train-only mode.")

    monkeypatch.setattr(run_model, "load_model_module", _load_dummy_model_module)
    monkeypatch.setattr(run_model, "_resolve_ref_gff_file", _fail_ref_gff)
    monkeypatch.setattr(run_model, "aggregate_transcript_scores", _fail_aggregate)

    run_model.run_pipeline(args)

    assert metrics_json.exists()
    summary = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert summary["donor"]["best_metric"] == "pr_auc"
    assert float(summary["donor"]["best_score"]) == pytest.approx(0.91)
    assert summary["acceptor"]["best_metric"] == "pr_auc"
    assert float(summary["acceptor"]["best_score"]) == pytest.approx(0.89)
    assert summary["validation_protocol"]["split_type"] == "stratified_site"
    assert isinstance(summary["validation_signature"], str)
    assert len(summary["validation_signature"]) == 12
    train_source = summary["validation_protocol"]["train_source"]
    assert isinstance(train_source, dict)
    assert not Path(str(train_source["train_pos_path"])).is_absolute()
    assert not Path(str(train_source["train_neg_path"])).is_absolute()


def test_plot_range_defaults_are_none_for_species_specific_bounds() -> None:
    parser = run_model._build_parser(
        selected_model="cnn",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn",
            "--species",
            "Mmus",
        ]
    )
    assert args.x_min is None
    assert args.x_max is None
    assert args.y_min is None
    assert args.y_max is None


def test_run_pipeline_rejects_single_task_without_train_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = run_model._build_parser(
        selected_model="cnn",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn",
            "--species",
            "Dmel",
            "--train_target",
            "donor",
        ]
    )

    def _load_dummy_model_module(model_name: str) -> _DummyModelModule:
        assert model_name == "cnn"
        return _DummyModelModule()

    monkeypatch.setattr(run_model, "load_model_module", _load_dummy_model_module)

    with pytest.raises(ValueError, match="requires --train_only"):
        run_model.run_pipeline(args)


def test_run_pipeline_cnn_v2_train_only_keeps_single_task_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_json = tmp_path / "cnn_v2_acceptor_summary.json"
    parser = run_model._build_parser(
        selected_model="cnn_v2",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_v2",
            "--species",
            "Dmel",
            "--train_target",
            "acceptor",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--epochs",
            "1",
            "--train_only",
            "--metrics_json",
            str(metrics_json),
        ]
    )
    dummy_module = _DummySingleTaskModelModule()

    monkeypatch.setattr(
        run_model,
        "load_model_module",
        lambda model_name: dummy_module,
    )

    run_model.run_pipeline(args)

    assert dummy_module.train_targets == ["acceptor"]
    summary = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert "donor" not in summary
    assert summary["acceptor"]["best_metric"] == "pr_auc"
    assert float(summary["acceptor"]["best_score"]) == pytest.approx(0.87)


def test_run_pipeline_cnn_v2_train_only_allows_both_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_json = tmp_path / "cnn_v2_both_summary.json"
    parser = run_model._build_parser(
        selected_model="cnn_v2",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_v2",
            "--species",
            "Dmel",
            "--train_target",
            "both",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--epochs",
            "1",
            "--train_only",
            "--metrics_json",
            str(metrics_json),
        ]
    )
    dummy_module = _DummySingleTaskModelModule()

    monkeypatch.setattr(
        run_model,
        "load_model_module",
        lambda model_name: dummy_module,
    )

    run_model.run_pipeline(args)

    assert dummy_module.train_targets == ["both"]
    summary = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert summary["acceptor"]["best_metric"] == "pr_auc"
    assert float(summary["acceptor"]["best_score"]) == pytest.approx(0.87)


def _write_tuned_best_config(
    path: Path,
    *,
    task: str,
    published_name: str,
    objective_score: float,
    checkpoint_path: Path,
    batch_size: int = 512,
    sampled_params: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tuned_sampled_params: dict[str, object] = {
        "batch_size": batch_size,
    }
    if sampled_params is not None:
        tuned_sampled_params.update(sampled_params)
    payload = {
        "status": "ok",
        "objective_metric": "pr_auc",
        "objective_score": objective_score,
        "published_name": published_name,
        "published_at": "2026-04-06T00:00:00Z",
        "metrics_json": "",
        "sampled_params": tuned_sampled_params,
        "hparam_context": {
            "fixed_run_args": {
                "model": "cnn_v2",
                "species": "Dmel",
                "val_frac": 0.2,
            }
        },
        f"{task}_checkpoint_path": str(checkpoint_path),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_run_pipeline_cnn_v2_continue_uses_published_warm_start_for_tuned_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    donor_published = model_root / "Dmel" / "donor" / "cnn_v2.01.pt"
    acceptor_published = model_root / "Dmel" / "acceptor" / "cnn_v2.01.pt"
    donor_published.parent.mkdir(parents=True, exist_ok=True)
    acceptor_published.parent.mkdir(parents=True, exist_ok=True)
    donor_published.write_bytes(b"donor-published")
    acceptor_published.write_bytes(b"acceptor-published")
    write_version_history(
        data_root,
        "Dmel",
        "cnn_v2",
        [
            VersionHistoryEntry(
                version=1,
                published_name="cnn_v2.01",
                published_at="2026-04-06T00:00:00Z",
                source_best_config="data/Dmel/tuning/cnn_v2/donor/best_config.json",
                objective_metric="pr_auc",
                objective_score="0.8",
                updated_side="seed",
                carry_forward_side="",
                donor_checkpoint_path="model/Dmel/donor/cnn_v2.01.pt",
                acceptor_checkpoint_path="model/Dmel/acceptor/cnn_v2.01.pt",
                pair_checkpoint_path="",
                metrics_json="data/Dmel/learning_metric/cnn_v2.01.train.json",
                archive_status="live",
            )
        ],
    )
    donor_best = data_root / "Dmel" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    acceptor_best = (
        data_root / "Dmel" / "tuning" / "cnn_v2" / "acceptor" / "best_config.json"
    )
    _write_tuned_best_config(
        donor_best,
        task="donor",
        published_name="cnn_v2.01",
        objective_score=0.8,
        checkpoint_path=donor_published,
    )
    _write_tuned_best_config(
        acceptor_best,
        task="acceptor",
        published_name="cnn_v2.01",
        objective_score=0.79,
        checkpoint_path=acceptor_published,
    )

    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", str(data_root))
    monkeypatch.setenv("INTRONMODEL_MODEL_ROOT", str(model_root))
    monkeypatch.setattr(
        run_model,
        "prune_species_model_checkpoints",
        lambda **_: SimpleNamespace(
            total_candidates=0,
            kept_count=0,
            deleted_count=0,
            dry_run=False,
        ),
    )
    dummy_module = _DummyInitCaptureModelModule()
    monkeypatch.setattr(run_model, "load_model_module", lambda model_name: dummy_module)

    parser = run_model._build_parser(
        selected_model="cnn_v2",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_v2",
            "--species",
            "Dmel",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--continue_train",
            "--train_only",
            "--metrics_json",
            str(tmp_path / "summary.json"),
            "--donor_tuned_config_path",
            str(donor_best),
            "--acceptor_tuned_config_path",
            str(acceptor_best),
        ]
    )
    args.batch_size = 512
    args.epochs = "1"
    args.val_frac = 0.2
    args.train_target = "both"

    run_model.run_pipeline(args)

    assert dummy_module.calls[0]["donor_init"] == str(donor_published)
    assert dummy_module.calls[0]["acceptor_init"] == str(acceptor_published)
    assert dummy_module.calls[0]["donor_checkpoint"] == str(donor_published)
    assert dummy_module.calls[0]["acceptor_checkpoint"] == str(acceptor_published)


def test_run_pipeline_cnn_v2_continue_without_tuned_configs_uses_strict_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_model,
        "prune_species_model_checkpoints",
        lambda **_: SimpleNamespace(
            total_candidates=0,
            kept_count=0,
            deleted_count=0,
            dry_run=False,
        ),
    )
    dummy_module = _DummyInitCaptureModelModule()
    monkeypatch.setattr(run_model, "load_model_module", lambda model_name: dummy_module)

    parser = run_model._build_parser(
        selected_model="cnn_v2",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_v2",
            "--species",
            "Dmel",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--continue_train",
            "--train_only",
            "--metrics_json",
            str(tmp_path / "summary.json"),
        ]
    )
    args.batch_size = 512
    args.epochs = "1"
    args.val_frac = 0.2
    args.train_target = "both"
    args.pair_mode = "independent"
    checkpoint_stem = run_model._build_checkpoint_stem_from_params(
        model_name="cnn_v2",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=100,
        raw_params=dict(vars(args)),
    )
    checkpoint_paths = run_model._build_checkpoint_paths(
        "Dmel",
        checkpoint_stem,
        tasks=("donor", "acceptor"),
    )
    Path(checkpoint_paths["donor"]).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_paths["acceptor"]).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_paths["donor"]).write_bytes(b"donor-strict")
    Path(checkpoint_paths["acceptor"]).write_bytes(b"acceptor-strict")

    run_model.run_pipeline(args)

    assert dummy_module.calls[0]["donor_init"] == checkpoint_paths["donor"]
    assert dummy_module.calls[0]["acceptor_init"] == checkpoint_paths["acceptor"]


def test_run_pipeline_cnn_v2_without_continue_leaves_tuned_init_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    donor_best = data_root / "Dmel" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    acceptor_best = (
        data_root / "Dmel" / "tuning" / "cnn_v2" / "acceptor" / "best_config.json"
    )
    donor_checkpoint = model_root / "Dmel" / "donor" / "cnn_v2.01.pt"
    acceptor_checkpoint = model_root / "Dmel" / "acceptor" / "cnn_v2.01.pt"
    donor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    acceptor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    donor_checkpoint.write_bytes(b"donor")
    acceptor_checkpoint.write_bytes(b"acceptor")
    write_version_history(
        data_root,
        "Dmel",
        "cnn_v2",
        [
            VersionHistoryEntry(
                version=1,
                published_name="cnn_v2.01",
                published_at="2026-04-06T00:00:00Z",
                source_best_config="data/Dmel/tuning/cnn_v2/donor/best_config.json",
                objective_metric="pr_auc",
                objective_score="0.8",
                updated_side="seed",
                carry_forward_side="",
                donor_checkpoint_path="model/Dmel/donor/cnn_v2.01.pt",
                acceptor_checkpoint_path="model/Dmel/acceptor/cnn_v2.01.pt",
                pair_checkpoint_path="",
                metrics_json="",
                archive_status="live",
            )
        ],
    )
    _write_tuned_best_config(
        donor_best,
        task="donor",
        published_name="cnn_v2.01",
        objective_score=0.8,
        checkpoint_path=donor_checkpoint,
    )
    _write_tuned_best_config(
        acceptor_best,
        task="acceptor",
        published_name="cnn_v2.01",
        objective_score=0.79,
        checkpoint_path=acceptor_checkpoint,
    )
    monkeypatch.setattr(
        run_model,
        "prune_species_model_checkpoints",
        lambda **_: SimpleNamespace(
            total_candidates=0,
            kept_count=0,
            deleted_count=0,
            dry_run=False,
        ),
    )
    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", str(data_root))
    monkeypatch.setenv("INTRONMODEL_MODEL_ROOT", str(model_root))
    dummy_module = _DummyInitCaptureModelModule()
    monkeypatch.setattr(run_model, "load_model_module", lambda model_name: dummy_module)

    parser = run_model._build_parser(
        selected_model="cnn_v2",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_v2",
            "--species",
            "Dmel",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--train_only",
            "--metrics_json",
            str(tmp_path / "summary.json"),
            "--donor_tuned_config_path",
            str(donor_best),
            "--acceptor_tuned_config_path",
            str(acceptor_best),
        ]
    )
    args.batch_size = 512
    args.epochs = "1"
    args.val_frac = 0.2
    args.train_target = "both"

    run_model.run_pipeline(args)

    assert dummy_module.calls[0]["donor_init"] == ""
    assert dummy_module.calls[0]["acceptor_init"] == ""
    assert dummy_module.calls[0]["donor_checkpoint"] == str(donor_checkpoint)
    assert dummy_module.calls[0]["acceptor_checkpoint"] == str(acceptor_checkpoint)


def test_run_pipeline_cnn_v2_ignores_non_runtime_tuned_keys_for_version_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    donor_best = data_root / "Dmel" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    acceptor_best = (
        data_root / "Dmel" / "tuning" / "cnn_v2" / "acceptor" / "best_config.json"
    )
    donor_checkpoint = model_root / "Dmel" / "donor" / "cnn_v2.01.pt"
    acceptor_checkpoint = model_root / "Dmel" / "acceptor" / "cnn_v2.01.pt"
    write_version_history(
        data_root,
        "Dmel",
        "cnn_v2",
        [
            VersionHistoryEntry(
                version=1,
                published_name="cnn_v2.01",
                published_at="2026-04-06T00:00:00Z",
                source_best_config="data/Dmel/tuning/cnn_v2/donor/best_config.json",
                objective_metric="pr_auc",
                objective_score="0.8",
                updated_side="seed",
                carry_forward_side="",
                donor_checkpoint_path="model/Dmel/donor/cnn_v2.01.pt",
                acceptor_checkpoint_path="model/Dmel/acceptor/cnn_v2.01.pt",
                pair_checkpoint_path="",
                metrics_json="",
                archive_status="live",
            )
        ],
    )
    _write_tuned_best_config(
        donor_best,
        task="donor",
        published_name="cnn_v2.01",
        objective_score=0.8,
        checkpoint_path=donor_checkpoint,
        batch_size=512,
        sampled_params={
            "donor_len": 50,
            "acceptor_len": 70,
            "input_mode": "kmer3",
        },
    )
    _write_tuned_best_config(
        acceptor_best,
        task="acceptor",
        published_name="cnn_v2.01",
        objective_score=0.79,
        checkpoint_path=acceptor_checkpoint,
        batch_size=256,
        sampled_params={
            "donor_len": 90,
            "acceptor_len": 100,
            "input_mode": "onehot",
        },
    )

    monkeypatch.setattr(
        run_model,
        "prune_species_model_checkpoints",
        lambda **_: SimpleNamespace(
            total_candidates=0,
            kept_count=0,
            deleted_count=0,
            dry_run=False,
        ),
    )
    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", str(data_root))
    monkeypatch.setenv("INTRONMODEL_MODEL_ROOT", str(model_root))
    dummy_module = _DummyInitCaptureModelModule()
    monkeypatch.setattr(run_model, "load_model_module", lambda model_name: dummy_module)

    parser = run_model._build_parser(
        selected_model="cnn_v2",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_v2",
            "--species",
            "Dmel",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--train_only",
            "--metrics_json",
            str(tmp_path / "summary.json"),
            "--donor_tuned_config_path",
            str(donor_best),
            "--acceptor_tuned_config_path",
            str(acceptor_best),
        ]
    )
    args.batch_size = 64
    args.donor_batch_size = 512
    args.acceptor_batch_size = 256
    args.epochs = "1"
    args.val_frac = 0.2
    args.train_target = "both"

    run_model.run_pipeline(args)

    assert dummy_module.calls[0]["donor_checkpoint"] == str(donor_checkpoint)
    assert dummy_module.calls[0]["acceptor_checkpoint"] == str(acceptor_checkpoint)


def test_run_pipeline_cnn_v2_tuned_run_uses_versioned_default_metrics_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    donor_best = data_root / "Dmel" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    acceptor_best = (
        data_root / "Dmel" / "tuning" / "cnn_v2" / "acceptor" / "best_config.json"
    )
    donor_checkpoint = model_root / "Dmel" / "donor" / "cnn_v2.01.pt"
    acceptor_checkpoint = model_root / "Dmel" / "acceptor" / "cnn_v2.01.pt"
    write_version_history(
        data_root,
        "Dmel",
        "cnn_v2",
        [
            VersionHistoryEntry(
                version=1,
                published_name="cnn_v2.01",
                published_at="2026-04-06T00:00:00Z",
                source_best_config="data/Dmel/tuning/cnn_v2/donor/best_config.json",
                objective_metric="pr_auc",
                objective_score="0.8",
                updated_side="seed",
                carry_forward_side="",
                donor_checkpoint_path="model/Dmel/donor/cnn_v2.01.pt",
                acceptor_checkpoint_path="model/Dmel/acceptor/cnn_v2.01.pt",
                pair_checkpoint_path="",
                metrics_json="data/Dmel/learning_metric/cnn_v2.01.train.json",
                archive_status="live",
            )
        ],
    )
    _write_tuned_best_config(
        donor_best,
        task="donor",
        published_name="cnn_v2.01",
        objective_score=0.8,
        checkpoint_path=donor_checkpoint,
        batch_size=512,
        sampled_params={
            "donor_len": 50,
            "acceptor_len": 70,
            "input_mode": "kmer3",
        },
    )
    _write_tuned_best_config(
        acceptor_best,
        task="acceptor",
        published_name="cnn_v2.01",
        objective_score=0.79,
        checkpoint_path=acceptor_checkpoint,
        batch_size=256,
        sampled_params={
            "donor_len": 90,
            "acceptor_len": 100,
            "input_mode": "onehot",
        },
    )

    monkeypatch.setattr(
        run_model,
        "prune_species_model_checkpoints",
        lambda **_: SimpleNamespace(
            total_candidates=0,
            kept_count=0,
            deleted_count=0,
            dry_run=False,
        ),
    )
    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", str(data_root))
    monkeypatch.setenv("INTRONMODEL_MODEL_ROOT", str(model_root))
    dummy_module = _DummyInitCaptureModelModule()
    monkeypatch.setattr(run_model, "load_model_module", lambda model_name: dummy_module)

    parser = run_model._build_parser(
        selected_model="cnn_v2",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_v2",
            "--species",
            "Dmel",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--train_only",
            "--donor_tuned_config_path",
            str(donor_best),
            "--acceptor_tuned_config_path",
            str(acceptor_best),
        ]
    )
    args.batch_size = 64
    args.donor_batch_size = 512
    args.acceptor_batch_size = 256
    args.epochs = "1"
    args.val_frac = 0.2
    args.train_target = "both"

    run_model.run_pipeline(args)

    metrics_path = data_root / "Dmel" / "learning_metric" / "cnn_v2.01.train.json"
    assert metrics_path.exists()
    assert dummy_module.calls[0]["donor_checkpoint"] == str(donor_checkpoint)
    assert dummy_module.calls[0]["acceptor_checkpoint"] == str(acceptor_checkpoint)


def test_run_pipeline_refresh_skips_when_runtime_deviates_from_tuned_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    donor_published = model_root / "Dmel" / "donor" / "cnn_v2.01.pt"
    acceptor_published = model_root / "Dmel" / "acceptor" / "cnn_v2.01.pt"
    donor_published.parent.mkdir(parents=True, exist_ok=True)
    acceptor_published.parent.mkdir(parents=True, exist_ok=True)
    donor_published.write_bytes(b"donor-published")
    acceptor_published.write_bytes(b"acceptor-published")
    write_version_history(
        data_root,
        "Dmel",
        "cnn_v2",
        [
            VersionHistoryEntry(
                version=1,
                published_name="cnn_v2.01",
                published_at="2026-04-06T00:00:00Z",
                source_best_config="data/Dmel/tuning/cnn_v2/donor/best_config.json",
                objective_metric="pr_auc",
                objective_score="0.8",
                updated_side="seed",
                carry_forward_side="",
                donor_checkpoint_path="model/Dmel/donor/cnn_v2.01.pt",
                acceptor_checkpoint_path="model/Dmel/acceptor/cnn_v2.01.pt",
                pair_checkpoint_path="",
                metrics_json="",
                archive_status="live",
            )
        ],
    )
    donor_best = data_root / "Dmel" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    acceptor_best = (
        data_root / "Dmel" / "tuning" / "cnn_v2" / "acceptor" / "best_config.json"
    )
    _write_tuned_best_config(
        donor_best,
        task="donor",
        published_name="cnn_v2.01",
        objective_score=0.8,
        checkpoint_path=donor_published,
        batch_size=512,
    )
    _write_tuned_best_config(
        acceptor_best,
        task="acceptor",
        published_name="cnn_v2.01",
        objective_score=0.79,
        checkpoint_path=acceptor_published,
        batch_size=512,
    )

    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", str(data_root))
    monkeypatch.setenv("INTRONMODEL_MODEL_ROOT", str(model_root))
    monkeypatch.setattr(
        run_model,
        "prune_species_model_checkpoints",
        lambda **_: SimpleNamespace(
            total_candidates=0,
            kept_count=0,
            deleted_count=0,
            dry_run=False,
        ),
    )
    dummy_module = _DummyInitCaptureModelModule(improved_score=0.95)
    monkeypatch.setattr(run_model, "load_model_module", lambda model_name: dummy_module)

    parser = run_model._build_parser(
        selected_model="cnn_v2",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_v2",
            "--species",
            "Dmel",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--train_only",
            "--metrics_json",
            str(tmp_path / "summary.json"),
            "--donor_tuned_config_path",
            str(donor_best),
            "--acceptor_tuned_config_path",
            str(acceptor_best),
        ]
    )
    args.batch_size = 1024
    args.epochs = "1"
    args.val_frac = 0.2
    args.train_target = "both"

    run_model.run_pipeline(args)

    assert donor_published.read_bytes() == b"donor-published"
    assert acceptor_published.read_bytes() == b"acceptor-published"


def test_run_model_applies_process_title_from_env_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []

    monkeypatch.setattr(
        "util.process_title.apply_process_title_from_env",
        lambda: called.append(True) or True,
    )

    importlib.reload(run_model)

    assert called == [True]


def test_tune_time_scheduler_applies_process_title_from_env_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from tools import tune_time_scheduler

    called: list[bool] = []

    monkeypatch.setattr(
        "util.process_title.apply_process_title_from_env",
        lambda: called.append(True) or True,
    )

    importlib.reload(tune_time_scheduler)

    assert called == [True]
