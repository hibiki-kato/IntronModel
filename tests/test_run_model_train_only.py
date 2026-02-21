from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import run_model


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
