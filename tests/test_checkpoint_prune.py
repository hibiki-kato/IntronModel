from __future__ import annotations

import json
from pathlib import Path

from util.checkpoint_prune import prune_species_model_checkpoints


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ckpt")


def _write_train_summary(
    *,
    path: Path,
    model_name: str,
    donor_checkpoint_path: Path,
    acceptor_checkpoint_path: Path,
    donor_best_score: float,
    acceptor_best_score: float,
    donor_pr_auc: float,
    acceptor_pr_auc: float,
    validation_signature: str | None,
) -> None:
    payload: dict[str, object] = {
        "model": model_name,
        "donor_checkpoint_path": str(donor_checkpoint_path),
        "acceptor_checkpoint_path": str(acceptor_checkpoint_path),
        "donor": {
            "checkpoint": str(donor_checkpoint_path),
            "best_metric": "pr_auc",
            "best_score": donor_best_score,
            "best_pr_auc": donor_pr_auc,
        },
        "acceptor": {
            "checkpoint": str(acceptor_checkpoint_path),
            "best_metric": "pr_auc",
            "best_score": acceptor_best_score,
            "best_pr_auc": acceptor_pr_auc,
        },
        "selection_score_by_task": {
            "donor": donor_best_score,
            "acceptor": acceptor_best_score,
        },
    }
    if validation_signature is not None:
        payload["validation_signature"] = validation_signature
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prune_keeps_top_k_per_validation_signature(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    species = "Dmel"
    model_name = "cnn"
    task = "donor"

    ckpt_a1 = model_root / species / task / "a1.pt"
    ckpt_a2 = model_root / species / task / "a2.pt"
    ckpt_b1 = model_root / species / task / "b1.pt"
    for path in (ckpt_a1, ckpt_a2, ckpt_b1):
        _write_checkpoint(path)
    acc = model_root / species / "acceptor" / "placeholder.pt"
    _write_checkpoint(acc)

    learning_metric = data_root / species / "learning_metric"
    _write_train_summary(
        path=learning_metric / "a1.train.json",
        model_name=model_name,
        donor_checkpoint_path=ckpt_a1,
        acceptor_checkpoint_path=acc,
        donor_best_score=0.70,
        acceptor_best_score=0.0,
        donor_pr_auc=0.70,
        acceptor_pr_auc=0.0,
        validation_signature="sig_a",
    )
    _write_train_summary(
        path=learning_metric / "a2.train.json",
        model_name=model_name,
        donor_checkpoint_path=ckpt_a2,
        acceptor_checkpoint_path=acc,
        donor_best_score=0.90,
        acceptor_best_score=0.0,
        donor_pr_auc=0.90,
        acceptor_pr_auc=0.0,
        validation_signature="sig_a",
    )
    _write_train_summary(
        path=learning_metric / "b1.train.json",
        model_name=model_name,
        donor_checkpoint_path=ckpt_b1,
        acceptor_checkpoint_path=acc,
        donor_best_score=0.80,
        acceptor_best_score=0.0,
        donor_pr_auc=0.80,
        acceptor_pr_auc=0.0,
        validation_signature="sig_b",
    )

    report = prune_species_model_checkpoints(
        data_root=data_root,
        species=species,
        model_name=model_name,
        top_k=1,
        dry_run=False,
    )
    assert report.deleted_count == 1
    assert not ckpt_a1.exists()
    assert ckpt_a2.exists()
    assert ckpt_b1.exists()
    assert (
        data_root / species / "tuning" / model_name / "checkpoint_prune_top1.json"
    ).exists()
    assert not (
        data_root / species / "tuning" / model_name / "leaderboard_top1.json"
    ).exists()


def test_prune_isolates_legacy_signature(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    species = "Dmel"
    model_name = "cnn"

    ckpt_new = model_root / species / "donor" / "new.pt"
    ckpt_legacy = model_root / species / "donor" / "legacy.pt"
    for path in (ckpt_new, ckpt_legacy):
        _write_checkpoint(path)
    acc = model_root / species / "acceptor" / "placeholder.pt"
    _write_checkpoint(acc)

    learning_metric = data_root / species / "learning_metric"
    _write_train_summary(
        path=learning_metric / "new.train.json",
        model_name=model_name,
        donor_checkpoint_path=ckpt_new,
        acceptor_checkpoint_path=acc,
        donor_best_score=0.60,
        acceptor_best_score=0.0,
        donor_pr_auc=0.60,
        acceptor_pr_auc=0.0,
        validation_signature="sig_new",
    )
    _write_train_summary(
        path=learning_metric / "legacy.train.json",
        model_name=model_name,
        donor_checkpoint_path=ckpt_legacy,
        acceptor_checkpoint_path=acc,
        donor_best_score=0.95,
        acceptor_best_score=0.0,
        donor_pr_auc=0.95,
        acceptor_pr_auc=0.0,
        validation_signature=None,
    )

    report = prune_species_model_checkpoints(
        data_root=data_root,
        species=species,
        model_name=model_name,
        top_k=1,
        dry_run=False,
    )
    assert report.deleted_count == 0
    assert ckpt_new.exists()
    assert ckpt_legacy.exists()


def test_prune_protects_checkpoint_referenced_by_best_config(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    species = "Dmel"
    model_name = "cnn"

    ckpt_keep = model_root / species / "donor" / "keep.pt"
    ckpt_drop = model_root / species / "donor" / "drop.pt"
    for path in (ckpt_keep, ckpt_drop):
        _write_checkpoint(path)
    acc = model_root / species / "acceptor" / "placeholder.pt"
    _write_checkpoint(acc)

    learning_metric = data_root / species / "learning_metric"
    _write_train_summary(
        path=learning_metric / "keep.train.json",
        model_name=model_name,
        donor_checkpoint_path=ckpt_keep,
        acceptor_checkpoint_path=acc,
        donor_best_score=0.10,
        acceptor_best_score=0.0,
        donor_pr_auc=0.10,
        acceptor_pr_auc=0.0,
        validation_signature="sig_keep",
    )
    _write_train_summary(
        path=learning_metric / "drop.train.json",
        model_name=model_name,
        donor_checkpoint_path=ckpt_drop,
        acceptor_checkpoint_path=acc,
        donor_best_score=0.90,
        acceptor_best_score=0.0,
        donor_pr_auc=0.90,
        acceptor_pr_auc=0.0,
        validation_signature="sig_keep",
    )

    tuning_dir = data_root / species / "tuning" / model_name / "donor" / "run01"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    (tuning_dir / "best_config.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "phase": "full",
                "trial_id": 0,
            }
        ),
        encoding="utf-8",
    )
    (tuning_dir / "full_trial_0000.metrics.json").write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(ckpt_keep),
                "acceptor_checkpoint_path": str(acc),
            }
        ),
        encoding="utf-8",
    )

    report = prune_species_model_checkpoints(
        data_root=data_root,
        species=species,
        model_name=model_name,
        top_k=1,
        dry_run=False,
    )
    assert report.deleted_count == 0
    assert ckpt_keep.exists()
    assert ckpt_drop.exists()


def test_prune_protects_checkpoint_referenced_by_metrics_json_field(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    species = "Dmel"
    model_name = "cnn"

    ckpt_keep = model_root / species / "donor" / "keep_metrics.pt"
    ckpt_drop = model_root / species / "donor" / "drop_metrics.pt"
    for path in (ckpt_keep, ckpt_drop):
        _write_checkpoint(path)
    acc = model_root / species / "acceptor" / "placeholder.pt"
    _write_checkpoint(acc)

    learning_metric = data_root / species / "learning_metric"
    _write_train_summary(
        path=learning_metric / "keep_metrics.train.json",
        model_name=model_name,
        donor_checkpoint_path=ckpt_keep,
        acceptor_checkpoint_path=acc,
        donor_best_score=0.10,
        acceptor_best_score=0.0,
        donor_pr_auc=0.10,
        acceptor_pr_auc=0.0,
        validation_signature="sig_keep_metrics",
    )
    _write_train_summary(
        path=learning_metric / "drop_metrics.train.json",
        model_name=model_name,
        donor_checkpoint_path=ckpt_drop,
        acceptor_checkpoint_path=acc,
        donor_best_score=0.90,
        acceptor_best_score=0.0,
        donor_pr_auc=0.90,
        acceptor_pr_auc=0.0,
        validation_signature="sig_keep_metrics",
    )

    tuning_root = data_root / species / "tuning" / model_name / "donor"
    tuning_root.mkdir(parents=True, exist_ok=True)
    metrics_path = tuning_root / "external.metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(ckpt_keep),
                "acceptor_checkpoint_path": str(acc),
            }
        ),
        encoding="utf-8",
    )
    (tuning_root / "best_config.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "metrics_json": str(metrics_path),
            }
        ),
        encoding="utf-8",
    )

    report = prune_species_model_checkpoints(
        data_root=data_root,
        species=species,
        model_name=model_name,
        top_k=1,
        dry_run=False,
    )
    assert report.deleted_count == 0
    assert ckpt_keep.exists()
    assert ckpt_drop.exists()
