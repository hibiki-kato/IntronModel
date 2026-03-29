from __future__ import annotations

import json
import time
from pathlib import Path

from tools import hparam_search
from util.path_format import relativize_path_string


def _make_trial_result(
    *,
    trial_id: int,
    objective_score: float,
    metrics_json: Path,
) -> hparam_search.TrialResult:
    return hparam_search.TrialResult(
        phase="full",
        trial_id=trial_id,
        status="success",
        gpu_id="0",
        sampled_params={"batch_size": 512 + trial_id},
        effective_batch_size=512,
        oom_retries=0,
        donor_pr_auc=objective_score,
        acceptor_pr_auc=objective_score,
        mean_pr_auc=objective_score,
        objective_metric="mean_pr_auc",
        objective_score=objective_score,
        error_message=None,
        return_code=0,
        duration_sec=1.0,
        metrics_json=str(metrics_json),
        log_file=f"trial_{trial_id}.log",
        validation_signature="sig",
        validation_protocol=None,
        selection_score=objective_score,
    )


def test_write_best_config_includes_top_trials(tmp_path: Path) -> None:
    donor_ckpt = tmp_path / "donor.pt"
    donor_ckpt.write_bytes(b"donor")
    acceptor_ckpt = tmp_path / "acceptor.pt"
    acceptor_ckpt.write_bytes(b"acceptor")

    metrics_one = tmp_path / "one.metrics.json"
    metrics_one.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(donor_ckpt),
                "acceptor_checkpoint_path": str(acceptor_ckpt),
            }
        ),
        encoding="utf-8",
    )
    metrics_two = tmp_path / "two.metrics.json"
    metrics_two.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(donor_ckpt),
                "acceptor_checkpoint_path": str(acceptor_ckpt),
            }
        ),
        encoding="utf-8",
    )

    best_row = _make_trial_result(
        trial_id=0,
        objective_score=0.91,
        metrics_json=metrics_one,
    )
    second_row = _make_trial_result(
        trial_id=1,
        objective_score=0.90,
        metrics_json=metrics_two,
    )
    output_path = tmp_path / "best_config.json"

    hparam_search.write_best_config(
        output_path,
        best_row,
        top_rows=[best_row, second_row],
        top_k=2,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    top_trials = payload["top_trials"]
    assert isinstance(top_trials, list)
    assert len(top_trials) == 2
    assert top_trials[0]["rank"] == 1
    assert top_trials[0]["trial_id"] == 0
    assert top_trials[1]["rank"] == 2
    assert top_trials[1]["trial_id"] == 1
    assert payload["donor_checkpoint_path"] == relativize_path_string(str(donor_ckpt))
    assert payload["acceptor_checkpoint_path"] == relativize_path_string(
        str(acceptor_ckpt)
    )


def test_prune_non_best_trial_checkpoints_keeps_best_and_external(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    keep_ckpt = model_root / "Dmel" / "donor" / "keep.pt"
    keep_ckpt.parent.mkdir(parents=True, exist_ok=True)
    keep_ckpt.write_bytes(b"keep")
    drop_ckpt = model_root / "Dmel" / "donor" / "drop.pt"
    drop_ckpt.write_bytes(b"drop")
    acceptor_ckpt = model_root / "Dmel" / "acceptor" / "shared.pt"
    acceptor_ckpt.parent.mkdir(parents=True, exist_ok=True)
    acceptor_ckpt.write_bytes(b"acceptor")
    external_ckpt = tmp_path / "external.pt"
    external_ckpt.write_bytes(b"external")

    metrics_keep = tmp_path / "keep.metrics.json"
    metrics_keep.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(keep_ckpt),
                "acceptor_checkpoint_path": str(acceptor_ckpt),
            }
        ),
        encoding="utf-8",
    )
    metrics_drop = tmp_path / "drop.metrics.json"
    metrics_drop.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(drop_ckpt),
                "acceptor_checkpoint_path": str(external_ckpt),
            }
        ),
        encoding="utf-8",
    )

    best_row = _make_trial_result(
        trial_id=0,
        objective_score=0.91,
        metrics_json=metrics_keep,
    )
    drop_row = _make_trial_result(
        trial_id=1,
        objective_score=0.90,
        metrics_json=metrics_drop,
    )

    deleted_count = hparam_search._prune_non_best_trial_checkpoints(
        project_root=tmp_path,
        trial_rows=[best_row, drop_row],
        best_row=best_row,
        min_mtime_epoch=time.time() - 1.0,
    )

    assert deleted_count == 1
    assert keep_ckpt.exists()
    assert not drop_ckpt.exists()
    assert external_ckpt.exists()


def test_write_tuning_leaderboard_merges_targets(tmp_path: Path) -> None:
    donor_ckpt = tmp_path / "donor.pt"
    donor_ckpt.write_bytes(b"donor")
    acceptor_ckpt = tmp_path / "acceptor.pt"
    acceptor_ckpt.write_bytes(b"acceptor")

    donor_metrics = tmp_path / "donor.metrics.json"
    donor_metrics.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(donor_ckpt),
                "acceptor_checkpoint_path": str(acceptor_ckpt),
            }
        ),
        encoding="utf-8",
    )
    acceptor_metrics = tmp_path / "acceptor.metrics.json"
    acceptor_metrics.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(donor_ckpt),
                "acceptor_checkpoint_path": str(acceptor_ckpt),
            }
        ),
        encoding="utf-8",
    )
    donor_row = _make_trial_result(
        trial_id=0,
        objective_score=0.81,
        metrics_json=donor_metrics,
    )
    acceptor_row = _make_trial_result(
        trial_id=1,
        objective_score=0.79,
        metrics_json=acceptor_metrics,
    )

    donor_output_dir = tmp_path / "data" / "Dmel" / "tuning" / "cnn" / "donor" / "run01"
    acceptor_output_dir = (
        tmp_path / "data" / "Dmel" / "tuning" / "cnn" / "acceptor" / "run02"
    )
    donor_output_dir.mkdir(parents=True, exist_ok=True)
    acceptor_output_dir.mkdir(parents=True, exist_ok=True)

    donor_config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=donor_output_dir,
        quick_trials=1,
        quick_epochs=1,
        top_k=3,
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
        base_args={"model": "cnn", "species": "Dmel", "train_target": "donor"},
        quick_overrides={},
        full_overrides={},
        search_space={},
    )
    acceptor_config = hparam_search.SearchConfig(
        project_root=tmp_path,
        species="Dmel",
        output_dir=acceptor_output_dir,
        quick_trials=1,
        quick_epochs=1,
        top_k=3,
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
        base_args={"model": "cnn", "species": "Dmel", "train_target": "acceptor"},
        quick_overrides={},
        full_overrides={},
        search_space={},
    )

    hparam_search._write_tuning_leaderboard(
        config=donor_config,
        ranked_rows=[donor_row],
        best_row=donor_row,
    )
    hparam_search._write_tuning_leaderboard(
        config=acceptor_config,
        ranked_rows=[acceptor_row],
        best_row=acceptor_row,
    )

    model_level_path = (
        tmp_path / "data" / "Dmel" / "tuning" / "cnn" / "leaderboard_top3.json"
    )
    payload = json.loads(model_level_path.read_text(encoding="utf-8"))
    targets = payload["targets"]
    assert isinstance(targets, dict)
    assert "donor" in targets
    assert "acceptor" in targets
    assert len(targets["donor"]["entries"]) == 1
    assert len(targets["acceptor"]["entries"]) == 1
    assert (donor_output_dir / "leaderboard_top3.json").exists()
    assert (acceptor_output_dir / "leaderboard_top3.json").exists()
