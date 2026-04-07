from __future__ import annotations

import json
from pathlib import Path

import pytest

from util.versioned_artifacts import ensure_publication_seed
from util.versioned_artifacts import finalize_published_version_outputs
from util.versioned_artifacts import publish_latest_best_version
from util.versioned_artifacts import read_version_history
from util.versioned_artifacts import refresh_published_version_if_improved
from util.versioned_artifacts import resolve_latest_published_run_assets
from util.versioned_artifacts import write_version_history
from util.versioned_artifacts import VersionHistoryEntry


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _best_payload(
    *,
    checkpoint_path: Path,
    objective_metric: str,
    objective_score: float,
) -> dict[str, object]:
    return {
        "status": "ok",
        "objective_metric": objective_metric,
        "objective_score": objective_score,
        "metrics_json": "",
        "sampled_params": {},
        "pair": {
            "checkpoint": str(checkpoint_path),
        },
    }


def _site_best_payload(
    *,
    task: str,
    checkpoint_path: Path,
    objective_metric: str,
    objective_score: float,
) -> dict[str, object]:
    payload = {
        "status": "ok",
        "objective_metric": objective_metric,
        "objective_score": objective_score,
        "metrics_json": "",
        "sampled_params": {
            "train_target": task,
        },
        f"{task}_checkpoint_path": str(checkpoint_path),
    }
    return payload


def _touch_public_outputs(base_dir: Path, stem: str) -> None:
    for directory_name, suffix in (
        ("site_score", ".tsv"),
        ("intron_score", ".tsv"),
        ("trans_score", ".tsv"),
        ("eval_score", ".txt"),
    ):
        path = base_dir / directory_name / f"{stem}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{stem}\n", encoding="utf-8")
    learning_metric = base_dir / "learning_metric"
    learning_metric.mkdir(parents=True, exist_ok=True)
    (learning_metric / f"{stem}.train.json").write_text("{}", encoding="utf-8")
    (learning_metric / f"{stem}_learning_curve.png").write_bytes(b"png")


def test_ensure_publication_seed_migrates_live_cnn_v2_outputs(tmp_path: Path) -> None:
    donor_checkpoint = tmp_path / "model" / "SpX" / "donor" / "donor_raw.pt"
    acceptor_checkpoint = (
        tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw.pt"
    )
    donor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    acceptor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    donor_checkpoint.write_bytes(b"donor")
    acceptor_checkpoint.write_bytes(b"acceptor")

    donor_best = (
        tmp_path / "data" / "SpX" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    )
    acceptor_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_v2"
        / "acceptor"
        / "best_config.json"
    )
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_checkpoint,
            objective_metric="donor_pr_auc",
            objective_score=0.91,
        ),
    )
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=acceptor_checkpoint,
            objective_metric="acceptor_pr_auc",
            objective_score=0.87,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_v2")

    published_name = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
    )

    assert published_name == "cnn_v2.01"
    assert not donor_checkpoint.exists()
    assert not acceptor_checkpoint.exists()
    assert (tmp_path / "model" / "SpX" / "donor" / "cnn_v2.01.pt").exists()
    assert (tmp_path / "model" / "SpX" / "acceptor" / "cnn_v2.01.pt").exists()
    assert (tmp_path / "data" / "SpX" / "trans_score" / "cnn_v2.01.tsv").exists()
    assert not (tmp_path / "data" / "SpX" / "trans_score" / "cnn_v2.tsv").exists()
    history = read_version_history(tmp_path / "data", "SpX", "cnn_v2")
    assert len(history) == 1
    assert history[0].published_name == "cnn_v2.01"
    assert history[0].archive_status == "live"


def test_publish_latest_best_version_carries_forward_other_site_checkpoint(
    tmp_path: Path,
) -> None:
    donor_raw_1 = tmp_path / "model" / "SpX" / "donor" / "donor_raw_1.pt"
    acceptor_raw_1 = (
        tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw_1.pt"
    )
    donor_raw_1.parent.mkdir(parents=True, exist_ok=True)
    acceptor_raw_1.parent.mkdir(parents=True, exist_ok=True)
    donor_raw_1.write_bytes(b"donor-1")
    acceptor_raw_1.write_bytes(b"acceptor-1")
    donor_best = (
        tmp_path / "data" / "SpX" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    )
    acceptor_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_v2"
        / "acceptor"
        / "best_config.json"
    )
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_raw_1,
            objective_metric="donor_pr_auc",
            objective_score=0.91,
        ),
    )
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=acceptor_raw_1,
            objective_metric="acceptor_pr_auc",
            objective_score=0.87,
        ),
    )
    _ = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
    )

    donor_raw_2 = tmp_path / "model" / "SpX" / "donor" / "donor_raw_2.pt"
    donor_raw_2.write_bytes(b"donor-2")
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_raw_2,
            objective_metric="donor_pr_auc",
            objective_score=0.95,
        ),
    )
    entry = publish_latest_best_version(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
        updated_side="donor",
    )

    assert entry is not None
    assert entry.published_name == "cnn_v2.02"
    donor_v2 = tmp_path / "model" / "SpX" / "donor" / "cnn_v2.02.pt"
    acceptor_v2 = tmp_path / "model" / "SpX" / "acceptor" / "cnn_v2.02.pt"
    assert donor_v2.read_bytes() == b"donor-2"
    assert acceptor_v2.read_bytes() == b"acceptor-1"
    history = read_version_history(tmp_path / "data", "SpX", "cnn_v2")
    assert [row.published_name for row in history] == ["cnn_v2.01", "cnn_v2.02"]
    assert [row.archive_status for row in history] == ["live", "live"]
    assert history[-1].carry_forward_side == "acceptor"


def test_finalize_published_version_outputs_archives_stale_site_outputs(
    tmp_path: Path,
) -> None:
    donor_raw_1 = tmp_path / "model" / "SpX" / "donor" / "donor_raw_1.pt"
    acceptor_raw_1 = (
        tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw_1.pt"
    )
    donor_raw_1.parent.mkdir(parents=True, exist_ok=True)
    acceptor_raw_1.parent.mkdir(parents=True, exist_ok=True)
    donor_raw_1.write_bytes(b"donor-1")
    acceptor_raw_1.write_bytes(b"acceptor-1")
    donor_best = (
        tmp_path / "data" / "SpX" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    )
    acceptor_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_v2"
        / "acceptor"
        / "best_config.json"
    )
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_raw_1,
            objective_metric="donor_pr_auc",
            objective_score=0.91,
        ),
    )
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=acceptor_raw_1,
            objective_metric="acceptor_pr_auc",
            objective_score=0.87,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_v2")
    _ = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
    )

    donor_raw_2 = tmp_path / "model" / "SpX" / "donor" / "donor_raw_2.pt"
    donor_raw_2.write_bytes(b"donor-2")
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_raw_2,
            objective_metric="donor_pr_auc",
            objective_score=0.95,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_v2.02")

    published_entry = publish_latest_best_version(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
        updated_side="donor",
    )

    assert published_entry is not None
    history = read_version_history(tmp_path / "data", "SpX", "cnn_v2")
    assert [row.archive_status for row in history] == ["live", "live"]

    finalized_entry = finalize_published_version_outputs(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
        published_name="cnn_v2.02",
    )

    assert finalized_entry is not None
    history = read_version_history(tmp_path / "data", "SpX", "cnn_v2")
    assert [row.archive_status for row in history] == ["archived", "live"]
    assert not (tmp_path / "data" / "SpX" / "eval_score" / "cnn_v2.01.txt").exists()
    assert (tmp_path / "data" / "SpX" / "eval_score" / "cnn_v2.02.txt").exists()
    assert not (tmp_path / "model" / "SpX" / "donor" / "cnn_v2.01.pt").exists()
    assert (tmp_path / "model" / "SpX" / "donor" / "cnn_v2.02.pt").exists()
    archived_eval = (
        tmp_path
        / "archive"
        / "versioned_artifacts"
        / "SpX"
        / "cnn_v2"
        / "cnn_v2.01"
        / "data"
        / "eval_score"
        / "cnn_v2.01.txt"
    )
    archived_checkpoint = (
        tmp_path
        / "archive"
        / "versioned_artifacts"
        / "SpX"
        / "cnn_v2"
        / "cnn_v2.01"
        / "model"
        / "donor"
        / "cnn_v2.01.pt"
    )
    assert archived_eval.is_file()
    assert archived_checkpoint.is_file()


def test_publish_latest_best_version_acceptor_update_ignores_stale_donor_payload(
    tmp_path: Path,
) -> None:
    donor_raw_1 = tmp_path / "model" / "SpX" / "donor" / "donor_raw_1.pt"
    acceptor_raw_1 = (
        tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw_1.pt"
    )
    donor_raw_1.parent.mkdir(parents=True, exist_ok=True)
    acceptor_raw_1.parent.mkdir(parents=True, exist_ok=True)
    donor_raw_1.write_bytes(b"donor-1")
    acceptor_raw_1.write_bytes(b"acceptor-1")
    donor_best = (
        tmp_path / "data" / "SpX" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    )
    acceptor_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_v2"
        / "acceptor"
        / "best_config.json"
    )
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_raw_1,
            objective_metric="donor_pr_auc",
            objective_score=0.91,
        ),
    )
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=acceptor_raw_1,
            objective_metric="acceptor_pr_auc",
            objective_score=0.87,
        ),
    )
    _ = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
    )

    stale_donor_payload = _site_best_payload(
        task="donor",
        checkpoint_path=tmp_path / "model" / "SpX" / "donor" / "stale_raw.pt",
        objective_metric="donor_pr_auc",
        objective_score=0.91,
    )
    new_acceptor_raw = tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw_2.pt"
    new_acceptor_raw.write_bytes(b"acceptor-2")
    _write_json(donor_best, stale_donor_payload)
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=new_acceptor_raw,
            objective_metric="acceptor_pr_auc",
            objective_score=0.92,
        ),
    )

    entry = publish_latest_best_version(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
        updated_side="acceptor",
    )

    assert entry is not None
    assert entry.published_name == "cnn_v2.02"
    donor_v2 = tmp_path / "model" / "SpX" / "donor" / "cnn_v2.02.pt"
    acceptor_v2 = tmp_path / "model" / "SpX" / "acceptor" / "cnn_v2.02.pt"
    assert donor_v2.read_bytes() == b"donor-1"
    assert acceptor_v2.read_bytes() == b"acceptor-2"


def test_publish_latest_best_version_acceptor_update_uses_archived_donor(
    tmp_path: Path,
) -> None:
    donor_raw_1 = tmp_path / "model" / "SpX" / "donor" / "donor_raw_1.pt"
    acceptor_raw_1 = (
        tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw_1.pt"
    )
    donor_raw_1.parent.mkdir(parents=True, exist_ok=True)
    acceptor_raw_1.parent.mkdir(parents=True, exist_ok=True)
    donor_raw_1.write_bytes(b"donor-1")
    acceptor_raw_1.write_bytes(b"acceptor-1")
    donor_best = (
        tmp_path / "data" / "SpX" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    )
    acceptor_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_v2"
        / "acceptor"
        / "best_config.json"
    )
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_raw_1,
            objective_metric="donor_pr_auc",
            objective_score=0.91,
        ),
    )
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=acceptor_raw_1,
            objective_metric="acceptor_pr_auc",
            objective_score=0.87,
        ),
    )
    _ = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
    )

    live_donor = tmp_path / "model" / "SpX" / "donor" / "cnn_v2.01.pt"
    archived_donor = (
        tmp_path
        / "archive"
        / "versioned_artifacts"
        / "SpX"
        / "cnn_v2"
        / "cnn_v2.01"
        / "model"
        / "donor"
        / "cnn_v2.01.pt"
    )
    archived_donor.parent.mkdir(parents=True, exist_ok=True)
    live_donor.replace(archived_donor)

    new_acceptor_raw = tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw_2.pt"
    new_acceptor_raw.write_bytes(b"acceptor-2")
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=new_acceptor_raw,
            objective_metric="acceptor_pr_auc",
            objective_score=0.92,
        ),
    )

    entry = publish_latest_best_version(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
        updated_side="acceptor",
    )

    assert entry is not None
    assert entry.published_name == "cnn_v2.02"
    donor_v2 = tmp_path / "model" / "SpX" / "donor" / "cnn_v2.02.pt"
    acceptor_v2 = tmp_path / "model" / "SpX" / "acceptor" / "cnn_v2.02.pt"
    assert donor_v2.read_bytes() == b"donor-1"
    assert acceptor_v2.read_bytes() == b"acceptor-2"


def test_ensure_publication_seed_promotes_legacy_pair_outputs(tmp_path: Path) -> None:
    pair_checkpoint = tmp_path / "model" / "SpX" / "pair" / "pair_raw.pt"
    pair_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pair_checkpoint.write_bytes(b"pair")
    pair_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_pair_v2"
        / "pair"
        / "best_config.json"
    )
    _write_json(
        pair_best,
        _best_payload(
            checkpoint_path=pair_checkpoint,
            objective_metric="pair_pr_auc",
            objective_score=0.88,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_pair_v2")

    published_name = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_pair_v2",
    )

    assert published_name == "cnn_pair_v2.01"
    assert (
        tmp_path / "model" / "SpX" / "pair" / "cnn_pair_v2.01.pt"
    ).read_bytes() == b"pair"
    assert (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_pair_v2"
        / "pair"
        / "best_config.json"
    ).is_file()
    assert (
        tmp_path / "data" / "SpX" / "trans_score" / "cnn_pair_v2.01.tsv"
    ).exists()


def test_ensure_publication_seed_promotes_cnn_v3_site_outputs(
    tmp_path: Path,
) -> None:
    donor_checkpoint = tmp_path / "model" / "SpX" / "donor" / "donor_raw.pt"
    acceptor_checkpoint = tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw.pt"
    donor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    acceptor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    donor_checkpoint.write_bytes(b"donor-v3")
    acceptor_checkpoint.write_bytes(b"acceptor-v3")
    donor_best = (
        tmp_path / "data" / "SpX" / "tuning" / "cnn_v3" / "donor" / "best_config.json"
    )
    acceptor_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_v3"
        / "acceptor"
        / "best_config.json"
    )
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_checkpoint,
            objective_metric="donor_pr_auc",
            objective_score=0.89,
        ),
    )
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=acceptor_checkpoint,
            objective_metric="acceptor_pr_auc",
            objective_score=0.87,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_v3")

    published_name = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v3",
    )

    assert published_name == "cnn_v3.01"
    assert (tmp_path / "model" / "SpX" / "donor" / "cnn_v3.01.pt").read_bytes() == (
        b"donor-v3"
    )
    assert (
        tmp_path / "model" / "SpX" / "acceptor" / "cnn_v3.01.pt"
    ).read_bytes() == b"acceptor-v3"
    assert (tmp_path / "data" / "SpX" / "trans_score" / "cnn_v3.01.tsv").exists()
    history = read_version_history(tmp_path / "data", "SpX", "cnn_v3")
    assert [row.published_name for row in history] == ["cnn_v3.01"]


def test_publish_latest_best_version_for_cnn_pair_v3_uses_its_own_namespace(
    tmp_path: Path,
) -> None:
    pair_raw_1 = tmp_path / "model" / "SpX" / "pair" / "pair_raw_1.pt"
    pair_raw_1.parent.mkdir(parents=True, exist_ok=True)
    pair_raw_1.write_bytes(b"pair-v3-1")
    pair_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_pair_v3"
        / "pair"
        / "best_config.json"
    )
    _write_json(
        pair_best,
        _best_payload(
            checkpoint_path=pair_raw_1,
            objective_metric="pair_pr_auc",
            objective_score=0.89,
        ),
    )
    _ = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_pair_v3",
    )

    pair_raw_2 = tmp_path / "model" / "SpX" / "pair" / "pair_raw_2.pt"
    pair_raw_2.write_bytes(b"pair-v3-2")
    _write_json(
        pair_best,
        _best_payload(
            checkpoint_path=pair_raw_2,
            objective_metric="pair_pr_auc",
            objective_score=0.93,
        ),
    )

    entry = publish_latest_best_version(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_pair_v3",
        updated_side="pair",
    )

    assert entry is not None
    assert entry.published_name == "cnn_pair_v3.02"
    assert (
        tmp_path / "model" / "SpX" / "pair" / "cnn_pair_v3.02.pt"
    ).read_bytes() == (
        b"pair-v3-2"
    )
    assert not (
        tmp_path / "data" / "SpX" / "tuning" / "cnn_pair_v2" / "pair" / "best_config.json"
    ).exists()
    history = read_version_history(tmp_path / "data", "SpX", "cnn_pair_v3")
    assert [row.published_name for row in history] == [
        "cnn_pair_v3.01",
        "cnn_pair_v3.02",
    ]
    assert [row.archive_status for row in history] == ["live", "live"]


def test_finalize_published_version_outputs_archives_stale_pair_outputs(
    tmp_path: Path,
) -> None:
    pair_raw_1 = tmp_path / "model" / "SpX" / "pair" / "pair_raw_1.pt"
    pair_raw_1.parent.mkdir(parents=True, exist_ok=True)
    pair_raw_1.write_bytes(b"pair-v3-1")
    pair_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_pair_v3"
        / "pair"
        / "best_config.json"
    )
    _write_json(
        pair_best,
        _best_payload(
            checkpoint_path=pair_raw_1,
            objective_metric="pair_pr_auc",
            objective_score=0.89,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_pair_v3")
    _ = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_pair_v3",
    )

    pair_raw_2 = tmp_path / "model" / "SpX" / "pair" / "pair_raw_2.pt"
    pair_raw_2.write_bytes(b"pair-v3-2")
    _write_json(
        pair_best,
        _best_payload(
            checkpoint_path=pair_raw_2,
            objective_metric="pair_pr_auc",
            objective_score=0.93,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_pair_v3.02")

    published_entry = publish_latest_best_version(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_pair_v3",
        updated_side="pair",
    )

    assert published_entry is not None
    history = read_version_history(tmp_path / "data", "SpX", "cnn_pair_v3")
    assert [row.archive_status for row in history] == ["live", "live"]

    finalized_entry = finalize_published_version_outputs(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_pair_v3",
        published_name="cnn_pair_v3.02",
    )

    assert finalized_entry is not None
    history = read_version_history(tmp_path / "data", "SpX", "cnn_pair_v3")
    assert [row.archive_status for row in history] == ["archived", "live"]
    assert not (
        tmp_path / "data" / "SpX" / "eval_score" / "cnn_pair_v3.01.txt"
    ).exists()
    assert (
        tmp_path / "data" / "SpX" / "eval_score" / "cnn_pair_v3.02.txt"
    ).exists()
    assert not (tmp_path / "model" / "SpX" / "pair" / "cnn_pair_v3.01.pt").exists()
    assert (tmp_path / "model" / "SpX" / "pair" / "cnn_pair_v3.02.pt").exists()
    archived_eval = (
        tmp_path
        / "archive"
        / "versioned_artifacts"
        / "SpX"
        / "cnn_pair_v3"
        / "cnn_pair_v3.01"
        / "data"
        / "eval_score"
        / "cnn_pair_v3.01.txt"
    )
    archived_checkpoint = (
        tmp_path
        / "archive"
        / "versioned_artifacts"
        / "SpX"
        / "cnn_pair_v3"
        / "cnn_pair_v3.01"
        / "model"
        / "pair"
        / "cnn_pair_v3.01.pt"
    )
    assert archived_eval.is_file()
    assert archived_checkpoint.is_file()


def test_publish_latest_best_version_uses_root_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    external_data_root = tmp_path / "external_data"
    external_model_root = tmp_path / "external_model"
    donor_raw = external_model_root / "SpX" / "donor" / "donor_raw.pt"
    acceptor_raw = external_model_root / "SpX" / "acceptor" / "acceptor_raw.pt"
    donor_raw.parent.mkdir(parents=True, exist_ok=True)
    acceptor_raw.parent.mkdir(parents=True, exist_ok=True)
    donor_raw.write_bytes(b"donor")
    acceptor_raw.write_bytes(b"acceptor")

    donor_best = (
        external_data_root
        / "SpX"
        / "tuning"
        / "cnn_v3"
        / "donor"
        / "best_config.json"
    )
    acceptor_best = (
        external_data_root
        / "SpX"
        / "tuning"
        / "cnn_v3"
        / "acceptor"
        / "best_config.json"
    )
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_raw,
            objective_metric="donor_pr_auc",
            objective_score=0.89,
        ),
    )
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=acceptor_raw,
            objective_metric="acceptor_pr_auc",
            objective_score=0.87,
        ),
    )
    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", str(external_data_root))
    monkeypatch.setenv("INTRONMODEL_MODEL_ROOT", str(external_model_root))

    published_name = ensure_publication_seed(
        project_root=project_root,
        species="SpX",
        model_name="cnn_v3",
    )

    assert published_name == "cnn_v3.01"
    assert not donor_raw.exists()
    assert not acceptor_raw.exists()
    assert (external_model_root / "SpX" / "donor" / "cnn_v3.01.pt").exists()
    assert (external_model_root / "SpX" / "acceptor" / "cnn_v3.01.pt").exists()
    history = read_version_history(external_data_root, "SpX", "cnn_v3")
    assert [row.published_name for row in history] == ["cnn_v3.01"]
    assert not Path(history[0].donor_checkpoint_path).is_absolute()
    assert not Path(history[0].acceptor_checkpoint_path).is_absolute()


def test_resolve_latest_published_run_assets_handles_pair_version(
    tmp_path: Path,
) -> None:
    pair_checkpoint = tmp_path / "model" / "SpX" / "pair" / "cnn_pair_v3.02.pt"
    pair_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pair_checkpoint.write_bytes(b"pair-v3")
    write_version_history(
        tmp_path / "data",
        "SpX",
        "cnn_pair_v3",
        [
            VersionHistoryEntry(
                version=1,
                published_name="cnn_pair_v3.01",
                published_at="2026-03-29T00:00:00Z",
                source_best_config="data/SpX/tuning/cnn_pair_v3/pair/best_config.json",
                objective_metric="pair_max_f1",
                objective_score="0.80",
                updated_side="pair",
                carry_forward_side="",
                donor_checkpoint_path="",
                acceptor_checkpoint_path="",
                pair_checkpoint_path="model/SpX/pair/cnn_pair_v3.01.pt",
                metrics_json="data/SpX/learning_metric/cnn_pair_v3.01.train.json",
                archive_status="archived",
            ),
            VersionHistoryEntry(
                version=2,
                published_name="cnn_pair_v3.02",
                published_at="2026-03-29T01:00:00Z",
                source_best_config="data/SpX/tuning/cnn_pair_v3/pair/best_config.json",
                objective_metric="pair_max_f1",
                objective_score="0.81",
                updated_side="pair",
                carry_forward_side="",
                donor_checkpoint_path="",
                acceptor_checkpoint_path="",
                pair_checkpoint_path="model/SpX/pair/cnn_pair_v3.02.pt",
                metrics_json="data/SpX/learning_metric/cnn_pair_v3.02.train.json",
                archive_status="live",
            ),
        ],
    )

    assets = resolve_latest_published_run_assets(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_pair_v3",
    )

    assert assets is not None
    assert assets["published_name"] == "cnn_pair_v3.02"
    assert assets["pair_checkpoint_path"] == str(pair_checkpoint.resolve())
    assert assets["transcript_output_tsv"].endswith("cnn_pair_v3.02.tsv")


def test_ensure_publication_seed_supports_generic_site_model(
    tmp_path: Path,
) -> None:
    donor_checkpoint = tmp_path / "model" / "SpX" / "donor" / "donor_raw.pt"
    acceptor_checkpoint = tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw.pt"
    donor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    acceptor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    donor_checkpoint.write_bytes(b"donor-generic")
    acceptor_checkpoint.write_bytes(b"acceptor-generic")
    donor_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_resdil"
        / "donor"
        / "best_config.json"
    )
    acceptor_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_resdil"
        / "acceptor"
        / "best_config.json"
    )
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_checkpoint,
            objective_metric="donor_pr_auc",
            objective_score=0.84,
        ),
    )
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=acceptor_checkpoint,
            objective_metric="acceptor_pr_auc",
            objective_score=0.83,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_resdil")

    published_name = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_resdil",
    )

    assert published_name == "cnn_resdil.01"
    assert (
        tmp_path / "model" / "SpX" / "donor" / "cnn_resdil.01.pt"
    ).read_bytes() == b"donor-generic"
    assert (
        tmp_path / "model" / "SpX" / "acceptor" / "cnn_resdil.01.pt"
    ).read_bytes() == b"acceptor-generic"
    history = read_version_history(tmp_path / "data", "SpX", "cnn_resdil")
    assert [row.published_name for row in history] == ["cnn_resdil.01"]


def test_ensure_publication_seed_supports_generic_pair_model(
    tmp_path: Path,
) -> None:
    pair_checkpoint = tmp_path / "model" / "SpX" / "pair" / "pair_raw.pt"
    pair_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pair_checkpoint.write_bytes(b"pair-generic")
    pair_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "bilstm_pair"
        / "pair"
        / "best_config.json"
    )
    _write_json(
        pair_best,
        _best_payload(
            checkpoint_path=pair_checkpoint,
            objective_metric="pair_pr_auc",
            objective_score=0.77,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "bilstm_pair")

    published_name = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="bilstm_pair",
    )

    assert published_name == "bilstm_pair.01"
    assert (
        tmp_path / "model" / "SpX" / "pair" / "bilstm_pair.01.pt"
    ).read_bytes() == b"pair-generic"
    history = read_version_history(tmp_path / "data", "SpX", "bilstm_pair")
    assert [row.published_name for row in history] == ["bilstm_pair.01"]


def test_refresh_published_version_if_improved_updates_same_site_version(
    tmp_path: Path,
) -> None:
    donor_raw = tmp_path / "model" / "SpX" / "donor" / "donor_raw.pt"
    acceptor_raw = tmp_path / "model" / "SpX" / "acceptor" / "acceptor_raw.pt"
    donor_raw.parent.mkdir(parents=True, exist_ok=True)
    acceptor_raw.parent.mkdir(parents=True, exist_ok=True)
    donor_raw.write_bytes(b"donor-seed")
    acceptor_raw.write_bytes(b"acceptor-seed")
    donor_best = (
        tmp_path / "data" / "SpX" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    )
    acceptor_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_v2"
        / "acceptor"
        / "best_config.json"
    )
    _write_json(
        donor_best,
        _site_best_payload(
            task="donor",
            checkpoint_path=donor_raw,
            objective_metric="pr_auc",
            objective_score=0.81,
        ),
    )
    _write_json(
        acceptor_best,
        _site_best_payload(
            task="acceptor",
            checkpoint_path=acceptor_raw,
            objective_metric="pr_auc",
            objective_score=0.79,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_v2")
    _ = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
    )

    donor_full = tmp_path / "model" / "SpX" / "donor" / "donor_full.pt"
    donor_full.write_bytes(b"donor-full")
    refreshed = refresh_published_version_if_improved(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_v2",
        published_name="cnn_v2.01",
        task_payloads={
            "donor": {
                "donor_checkpoint_path": str(donor_full),
                "objective_metric": "pr_auc",
                "objective_score": 0.93,
            }
        },
        metrics_json="data/SpX/learning_metric/cnn_v2_full.train.json",
    )

    assert refreshed is not None
    assert refreshed.published_name == "cnn_v2.01"
    assert (tmp_path / "model" / "SpX" / "donor" / "cnn_v2.01.pt").read_bytes() == (
        b"donor-full"
    )
    assert (
        tmp_path / "model" / "SpX" / "acceptor" / "cnn_v2.01.pt"
    ).read_bytes() == b"acceptor-seed"
    history = read_version_history(tmp_path / "data", "SpX", "cnn_v2")
    assert [row.published_name for row in history] == ["cnn_v2.01"]
    donor_payload = json.loads(donor_best.read_text(encoding="utf-8"))
    assert float(donor_payload["objective_score"]) == pytest.approx(0.93)
    assert donor_payload["published_name"] == "cnn_v2.01"


def test_refresh_published_version_if_improved_noops_without_improvement(
    tmp_path: Path,
) -> None:
    pair_raw = tmp_path / "model" / "SpX" / "pair" / "pair_raw.pt"
    pair_raw.parent.mkdir(parents=True, exist_ok=True)
    pair_raw.write_bytes(b"pair-seed")
    pair_best = (
        tmp_path
        / "data"
        / "SpX"
        / "tuning"
        / "cnn_pair_v3"
        / "pair"
        / "best_config.json"
    )
    _write_json(
        pair_best,
        _best_payload(
            checkpoint_path=pair_raw,
            objective_metric="pr_auc",
            objective_score=0.88,
        ),
    )
    _touch_public_outputs(tmp_path / "data" / "SpX", "cnn_pair_v3")
    _ = ensure_publication_seed(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_pair_v3",
    )

    pair_full = tmp_path / "model" / "SpX" / "pair" / "pair_full.pt"
    pair_full.write_bytes(b"pair-full")
    refreshed = refresh_published_version_if_improved(
        project_root=tmp_path,
        species="SpX",
        model_name="cnn_pair_v3",
        published_name="cnn_pair_v3.01",
        task_payloads={
            "pair": {
                "pair_checkpoint_path": str(pair_full),
                "objective_metric": "pr_auc",
                "objective_score": 0.88,
            }
        },
    )

    assert refreshed is None
    assert (
        tmp_path / "model" / "SpX" / "pair" / "cnn_pair_v3.01.pt"
    ).read_bytes() == b"pair-seed"
    history = read_version_history(tmp_path / "data", "SpX", "cnn_pair_v3")
    assert [row.published_name for row in history] == ["cnn_pair_v3.01"]
