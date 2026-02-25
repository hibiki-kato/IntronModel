from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from util.model_task_paths import (
    resolve_required_checkpoint_paths,
    resolve_tasks_to_train,
    resolve_train_target,
)


def test_resolve_required_checkpoint_paths_returns_both_paths(
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor.pt"
    donor.write_bytes(b"d")
    acceptor = tmp_path / "acceptor.pt"
    acceptor.write_bytes(b"a")
    args = Namespace(
        donor_checkpoint_path=str(donor),
        acceptor_checkpoint_path=str(acceptor),
    )

    resolved = resolve_required_checkpoint_paths(args, require_exists=True)

    assert resolved["donor"] == str(donor)
    assert resolved["acceptor"] == str(acceptor)


def test_resolve_required_checkpoint_paths_rejects_missing_value() -> None:
    args = Namespace(donor_checkpoint_path="", acceptor_checkpoint_path="x.pt")

    with pytest.raises(ValueError, match="Missing donor checkpoint path"):
        _ = resolve_required_checkpoint_paths(args, require_exists=False)


def test_resolve_required_checkpoint_paths_rejects_missing_file(
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor.pt"
    donor.write_bytes(b"d")
    missing = tmp_path / "missing.pt"
    args = Namespace(
        donor_checkpoint_path=str(donor),
        acceptor_checkpoint_path=str(missing),
    )

    with pytest.raises(FileNotFoundError, match="Acceptor checkpoint not found"):
        _ = resolve_required_checkpoint_paths(args, require_exists=True)


def test_resolve_train_target_and_tasks_to_train() -> None:
    args = Namespace(train_target=" donor ")
    target = resolve_train_target(args)
    tasks = resolve_tasks_to_train(target)

    assert target == "donor"
    assert tasks == ["donor"]


def test_resolve_train_target_rejects_invalid_value() -> None:
    args = Namespace(train_target="invalid")

    with pytest.raises(ValueError, match="--train_target must be one of"):
        _ = resolve_train_target(args)
