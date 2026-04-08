from __future__ import annotations

from pathlib import Path

import pytest

from tools import run_wrapper_pipeline
from util.versioned_artifacts import VersionHistoryEntry
from util.versioned_artifacts import write_version_history


def test_validate_dnabert_specific_accepts_variant_s() -> None:
    env = {
        "DNABERT_VARIANT": "s",
        "TRUST_REMOTE_CODE": "1",
    }
    run_wrapper_pipeline._validate_dnabert_specific(env)


def test_validate_dnabert_specific_rejects_unknown_variant() -> None:
    env = {
        "DNABERT_VARIANT": "x",
        "TRUST_REMOTE_CODE": "1",
    }
    with pytest.raises(ValueError):
        run_wrapper_pipeline._validate_dnabert_specific(env)


def test_resolve_dnabert_model_uses_dnaberts_for_variant_s(tmp_path: Path) -> None:
    env = {
        "DNABERT_VARIANT": "S",
        "PRETRAINED_MODEL_RELATIVE_PATH": "pretrained/dnabert-s",
        "PRETRAINED_MODEL_NAME": "",
    }
    run_wrapper_pipeline._resolve_dnabert_model(env, tmp_path)
    assert env["MODEL"] == "dnaberts"
    assert env["PRETRAINED_MODEL_NAME"] == str(tmp_path / "pretrained/dnabert-s")


def test_resolve_dnabert_model_uses_pair_suffix_for_pair_wrapper(
    tmp_path: Path,
) -> None:
    env = {
        "DNABERT_VARIANT": "2",
        "PRETRAINED_MODEL_RELATIVE_PATH": "pretrained/dnabert2",
        "PRETRAINED_MODEL_NAME": "",
    }
    run_wrapper_pipeline._resolve_dnabert_model(
        env,
        tmp_path,
        pair_mode=True,
    )
    assert env["MODEL"] == "dnabert2_pair"
    assert env["PRETRAINED_MODEL_NAME"] == str(tmp_path / "pretrained/dnabert2")


def test_resolve_dnabert_versioned_output_stem_uses_published_name(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    write_version_history(
        data_root,
        "SpX",
        "dnabert2",
        [
            VersionHistoryEntry(
                version=1,
                published_name="dnabert2.01",
                published_at="2026-01-01T00:00:00Z",
                source_best_config="data/SpX/tuning/dnabert2/donor/best_config.json",
                objective_metric="donor_pr_auc",
                objective_score="0.9",
                updated_side="donor",
                carry_forward_side="acceptor",
                donor_checkpoint_path="model/SpX/donor/dnabert2.01.pt",
                acceptor_checkpoint_path="model/SpX/acceptor/dnabert2.01.pt",
                pair_checkpoint_path="",
                metrics_json="data/SpX/learning_metric/dnabert2.01.train.json",
                archive_status="live",
            )
        ],
    )
    resolved = run_wrapper_pipeline._resolve_dnabert_versioned_output_stem(
        spec=run_wrapper_pipeline.SPECS["dnabert.sh"],
        env={
            "SKIP_TRAINING": "1",
            "TRAIN_ONLY": "0",
            "PRECOMPUTED_SITE_SCORE_TSV": "",
        },
        data_root=data_root,
        species="SpX",
        model_name="dnabert2",
        output_stem="dnabert2_100_100",
    )
    assert resolved == "dnabert2.01"


def test_resolve_dnabert_versioned_output_stem_keeps_default_when_not_skip() -> None:
    resolved = run_wrapper_pipeline._resolve_dnabert_versioned_output_stem(
        spec=run_wrapper_pipeline.SPECS["dnabert_pair.sh"],
        env={
            "SKIP_TRAINING": "0",
            "TRAIN_ONLY": "0",
            "PRECOMPUTED_SITE_SCORE_TSV": "",
        },
        data_root=Path("."),
        species="SpX",
        model_name="dnabert2_pair",
        output_stem="dnabert2_pair_100_100",
    )
    assert resolved == "dnabert2_pair_100_100"
