"""Shared helpers for model task selection and checkpoint path resolution."""

from __future__ import annotations

import os
from typing import Literal, cast

TaskName = Literal["donor", "acceptor"]
TrainTarget = Literal["both", "donor", "acceptor"]


def resolve_required_checkpoint_paths(
    common_args: object,
    *,
    require_exists: bool,
) -> dict[TaskName, str]:
    """Resolve donor/acceptor checkpoint paths from runtime args.

    Parameters
    ----------
    common_args : object
        Namespace-like object that may provide checkpoint path attributes.
    require_exists : bool
        When ``True``, the function validates file existence.

    Returns
    -------
    dict[TaskName, str]
        Resolved checkpoint path mapping for donor and acceptor.

    Raises
    ------
    ValueError
        If required checkpoint path values are missing.
    FileNotFoundError
        If ``require_exists=True`` and checkpoint files are absent.
    """
    donor_checkpoint_path = str(
        getattr(common_args, "donor_checkpoint_path", "")
    ).strip()
    acceptor_checkpoint_path = str(
        getattr(common_args, "acceptor_checkpoint_path", "")
    ).strip()
    if donor_checkpoint_path == "":
        raise ValueError("Missing donor checkpoint path in common_args.")
    if acceptor_checkpoint_path == "":
        raise ValueError("Missing acceptor checkpoint path in common_args.")

    if require_exists:
        if not os.path.exists(donor_checkpoint_path):
            raise FileNotFoundError(
                f"Donor checkpoint not found: {donor_checkpoint_path}"
            )
        if not os.path.exists(acceptor_checkpoint_path):
            raise FileNotFoundError(
                f"Acceptor checkpoint not found: {acceptor_checkpoint_path}"
            )

    return {
        "donor": donor_checkpoint_path,
        "acceptor": acceptor_checkpoint_path,
    }


def resolve_train_target(model_args: object) -> TrainTarget:
    """Resolve and validate train-target setting from model args.

    Parameters
    ----------
    model_args : object
        Namespace-like object that may provide ``train_target``.

    Returns
    -------
    TrainTarget
        One of ``both``, ``donor``, or ``acceptor``.

    Raises
    ------
    ValueError
        If the value is outside the supported set.
    """
    train_target = str(getattr(model_args, "train_target", "both")).strip().lower()
    if train_target not in {"both", "donor", "acceptor"}:
        raise ValueError("--train_target must be one of: both, donor, acceptor.")
    return cast(TrainTarget, train_target)


def resolve_tasks_to_train(train_target: TrainTarget) -> list[TaskName]:
    """Convert train-target mode to explicit task list.

    Parameters
    ----------
    train_target : TrainTarget
        Resolved train target mode.

    Returns
    -------
    list[TaskName]
        Ordered task list for training.
    """
    if train_target == "both":
        return ["donor", "acceptor"]
    return [train_target]
