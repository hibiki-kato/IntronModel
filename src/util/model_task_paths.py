"""Shared helpers for model task selection and checkpoint path resolution."""

from __future__ import annotations

import os
from typing import Sequence

DEFAULT_TASKS: tuple[str, ...] = ("donor", "acceptor")
_MODEL_TASK_OVERRIDES: dict[str, tuple[str, ...]] = {
    "cnn_pair": ("pair",),
    "markov_xgboost": ("pair",),
    "dnabert_pair": ("pair",),
    "dnabert2_pair": ("pair",),
    "dnabert6_pair": ("pair",),
    "dnaberts_pair": ("pair",),
}


def checkpoint_tasks_for_model(model_name: str) -> tuple[str, ...]:
    """Resolve checkpoint task names for one model.

    Parameters
    ----------
    model_name : str
        Model key from registry.

    Returns
    -------
    tuple[str, ...]
        Ordered checkpoint task names.
    """
    return _MODEL_TASK_OVERRIDES.get(model_name, DEFAULT_TASKS)


def resolve_required_checkpoint_paths(
    common_args: object,
    *,
    require_exists: bool,
    tasks: Sequence[str] | None = None,
) -> dict[str, str]:
    """Resolve checkpoint paths from runtime args for specified tasks.

    Parameters
    ----------
    common_args : object
        Namespace-like object that may provide ``<task>_checkpoint_path`` fields.
    require_exists : bool
        When ``True``, validate file existence.
    tasks : Sequence[str] | None, default=None
        Task names. Defaults to donor/acceptor.

    Returns
    -------
    dict[str, str]
        Mapping from task name to checkpoint path.

    Raises
    ------
    ValueError
        If required checkpoint path values are missing.
    FileNotFoundError
        If ``require_exists=True`` and checkpoint files are absent.
    """
    task_names = tuple(tasks) if tasks is not None else DEFAULT_TASKS
    if not task_names:
        raise ValueError("tasks must contain at least one task name.")

    resolved: dict[str, str] = {}
    for task in task_names:
        key_name = f"{task}_checkpoint_path"
        checkpoint_path = str(getattr(common_args, key_name, "")).strip()
        if checkpoint_path == "":
            raise ValueError(f"Missing {task} checkpoint path in common_args.")
        if require_exists and not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"{task.capitalize()} checkpoint not found: {checkpoint_path}"
            )
        resolved[task] = checkpoint_path
    return resolved


def resolve_train_target(
    model_args: object,
    *,
    allowed_targets: Sequence[str] | None = None,
) -> str:
    """Resolve and validate train-target setting from model args.

    Parameters
    ----------
    model_args : object
        Namespace-like object that may provide ``train_target``.
    allowed_targets : Sequence[str] | None, default=None
        Allowed train-target values. Default is ``both, donor, acceptor``.

    Returns
    -------
    str
        Resolved train-target value.

    Raises
    ------
    ValueError
        If the value is outside the supported set.
    """
    allowed = tuple(allowed_targets) if allowed_targets is not None else (
        "both",
        "donor",
        "acceptor",
    )
    train_target = str(getattr(model_args, "train_target", "both")).strip().lower()
    if train_target not in allowed:
        allowed_text = ", ".join(allowed)
        raise ValueError(f"--train_target must be one of: {allowed_text}.")
    return train_target


def resolve_tasks_to_train(
    train_target: str,
    *,
    both_tasks: Sequence[str] | None = None,
) -> list[str]:
    """Convert train-target mode to explicit task list.

    Parameters
    ----------
    train_target : str
        Resolved train-target mode.
    both_tasks : Sequence[str] | None, default=None
        Task list to use when ``train_target == 'both'``.

    Returns
    -------
    list[str]
        Ordered task list for training.
    """
    expanded_both = list(both_tasks) if both_tasks is not None else list(DEFAULT_TASKS)
    if train_target == "both":
        return expanded_both
    return [train_target]
