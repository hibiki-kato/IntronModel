from __future__ import annotations

import json
from pathlib import Path

from util.tuning_rank_prune import prune_missing_rank_tuning_checkpoints


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


def _write_metrics(path: Path, donor_path: Path, acceptor_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "donor_checkpoint_path": str(donor_path),
        "acceptor_checkpoint_path": str(acceptor_path),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prune_missing_rank_deletes_only_unprotected_files(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"

    keep_donor = model_root / "Dmel" / "donor" / "keep.pt"
    keep_acceptor = model_root / "Dmel" / "acceptor" / "keep.pt"
    drop_donor = model_root / "Dmel" / "donor" / "drop.pt"
    drop_acceptor = model_root / "Dmel" / "acceptor" / "drop.pt"
    for path in (keep_donor, keep_acceptor, drop_donor, drop_acceptor):
        _write_checkpoint(path)

    tuning_dir = data_root / "Dmel" / "tuning" / "cnn" / "donor" / "run01"
    ranked_metrics = tuning_dir / "full_trial_0000.metrics.json"
    missing_rank_metrics = tuning_dir / "full_trial_0001.metrics.json"
    _write_metrics(ranked_metrics, keep_donor, keep_acceptor)
    _write_metrics(missing_rank_metrics, drop_donor, drop_acceptor)

    best_config_payload: dict[str, object] = {
        "status": "ok",
        "top_trials": [
            {
                "rank": 1,
                "phase": "full",
                "trial_id": 0,
                "metrics_json": str(ranked_metrics),
            },
            {
                "phase": "full",
                "trial_id": 1,
                "metrics_json": str(missing_rank_metrics),
            },
        ],
    }
    tuning_dir.mkdir(parents=True, exist_ok=True)
    (tuning_dir / "best_config.json").write_text(
        json.dumps(best_config_payload),
        encoding="utf-8",
    )

    report = prune_missing_rank_tuning_checkpoints(
        data_root=data_root,
        model_root=model_root,
        species="Dmel",
        model_name="cnn",
        dry_run=False,
    )

    assert report.scanned_best_configs == 1
    assert report.missing_rank_entries == 1
    assert report.deleted_count == 2
    assert keep_donor.exists()
    assert keep_acceptor.exists()
    assert not drop_donor.exists()
    assert not drop_acceptor.exists()


def test_prune_missing_rank_respects_dry_run(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"

    drop_donor = model_root / "Dmel" / "donor" / "drop.pt"
    drop_acceptor = model_root / "Dmel" / "acceptor" / "drop.pt"
    _write_checkpoint(drop_donor)
    _write_checkpoint(drop_acceptor)

    tuning_dir = data_root / "Dmel" / "tuning" / "cnn" / "donor" / "run01"
    missing_rank_metrics = tuning_dir / "full_trial_0001.metrics.json"
    _write_metrics(missing_rank_metrics, drop_donor, drop_acceptor)

    best_config_payload: dict[str, object] = {
        "status": "ok",
        "top_trials": [
            {
                "phase": "full",
                "trial_id": 1,
                "metrics_json": str(missing_rank_metrics),
            }
        ],
    }
    tuning_dir.mkdir(parents=True, exist_ok=True)
    (tuning_dir / "best_config.json").write_text(
        json.dumps(best_config_payload),
        encoding="utf-8",
    )

    report = prune_missing_rank_tuning_checkpoints(
        data_root=data_root,
        model_root=model_root,
        species="Dmel",
        model_name="cnn",
        dry_run=True,
    )

    assert report.deleted_count == 2
    assert drop_donor.exists()
    assert drop_acceptor.exists()
