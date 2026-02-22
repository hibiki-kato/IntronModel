"""Training-control utilities shared by model modules.

This module centralizes validation and normalization for runtime training
controls such as epoch budget and early-stopping parameters.
"""

from __future__ import annotations


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
