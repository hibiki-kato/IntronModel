"""Training-control utilities shared by model modules.

This module centralizes validation and normalization for runtime training
controls such as epoch budget, early-stopping parameters, and validation
metric selection.
"""

from __future__ import annotations

from typing import Mapping

VALIDATION_METRIC_CHOICES: tuple[str, ...] = (
    "pr_auc",
    "roc_auc",
    "max_f1",
    "acc@0.5",
)
_VALIDATION_METRIC_ALIASES: dict[str, str] = {
    "f1_max": "max_f1",
}
_VALIDATION_METRIC_FALLBACKS: dict[str, tuple[str, ...]] = {
    "pr_auc": ("pr_auc", "roc_auc", "acc@0.5"),
    "roc_auc": ("roc_auc", "pr_auc", "acc@0.5"),
    "max_f1": ("max_f1", "acc@0.5"),
    "acc@0.5": ("acc@0.5",),
}


def resolve_training_epoch_budget(
    epochs_arg: object,
    max_epochs: int,
) -> tuple[int, bool]:
    """Resolve fixed/auto epoch configuration into concrete settings.

    Parameters
    ----------
    epochs_arg : object
        User-provided ``--epochs`` value. Accepts a positive integer-like value
        or the string ``"auto"`` (case-insensitive).
    max_epochs : int
        Upper epoch bound used when ``epochs_arg`` is ``"auto"``.

    Returns
    -------
    tuple[int, bool]
        A tuple of ``(resolved_epochs, epochs_auto)``.

    Raises
    ------
    ValueError
        If ``max_epochs`` is not positive or ``epochs_arg`` is invalid.
    """
    if max_epochs <= 0:
        raise ValueError("--max_epochs must be a positive integer.")

    epochs_text = str(epochs_arg).strip().lower()
    if epochs_text == "auto":
        return max_epochs, True

    if not epochs_text.isdigit():
        raise ValueError("--epochs must be a positive integer or 'auto'.")

    resolved_epochs = int(epochs_text)
    if resolved_epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")
    return resolved_epochs, False


def resolve_early_stopping_params(
    patience_arg: object,
    min_delta_arg: object,
) -> tuple[int, float]:
    """Validate and normalize early-stopping parameters.

    Parameters
    ----------
    patience_arg : object
        User-provided ``--early_stop_patience`` value.
    min_delta_arg : object
        User-provided ``--early_stop_min_delta`` value.

    Returns
    -------
    tuple[int, float]
        A tuple of ``(patience, min_delta)``.

    Raises
    ------
    ValueError
        If either value is out of allowed range.
    """
    patience = int(patience_arg)
    if patience < 0:
        raise ValueError("--early_stop_patience must be >= 0.")

    min_delta = float(min_delta_arg)
    if min_delta < 0.0:
        raise ValueError("--early_stop_min_delta must be >= 0.")

    return patience, min_delta


def resolve_validation_metric(metric_arg: object) -> str:
    """Validate and normalize one validation metric name.

    Parameters
    ----------
    metric_arg : object
        User-provided validation metric name.

    Returns
    -------
    str
        Normalized validation metric name.

    Raises
    ------
    ValueError
        If the requested metric is empty or unsupported.
    """
    metric = str(metric_arg).strip().lower()
    if metric == "":
        raise ValueError("--validation_metric must not be empty.")
    metric = _VALIDATION_METRIC_ALIASES.get(metric, metric)
    if metric not in VALIDATION_METRIC_CHOICES:
        joined = "|".join(VALIDATION_METRIC_CHOICES)
        raise ValueError(f"--validation_metric must be one of: {joined}.")
    return metric


def select_validation_score(
    metrics: Mapping[str, object],
    validation_metric: object,
) -> tuple[float, str]:
    """Select one validation score for checkpointing and early stopping.

    Parameters
    ----------
    metrics : Mapping[str, object]
        Validation metrics collected for the current epoch.
    validation_metric : object
        Requested primary validation metric.

    Returns
    -------
    tuple[float, str]
        Selected score and the metric name that produced it.

    Raises
    ------
    ValueError
        If none of the compatible metrics are available.
    """
    normalized_metric = resolve_validation_metric(validation_metric)
    fallback_order = _VALIDATION_METRIC_FALLBACKS[normalized_metric]
    for metric_name in fallback_order:
        value = metrics.get(metric_name)
        if value is None:
            continue
        return float(value), metric_name
    joined = ", ".join(fallback_order)
    raise ValueError(
        "Validation metrics are missing all compatible scoring keys for "
        f"--validation_metric={normalized_metric}: {joined}."
    )
