"""Shared scientific-notation formatting helpers for score text outputs."""

from __future__ import annotations

import math

SCORE_TEXT_DECIMAL_DIGITS: int = 14


def format_score_text(value: float) -> str:
    """Format one numeric score for plain-text score tables.

    Parameters
    ----------
    value : float
        Score value to serialize.

    Returns
    -------
    str
        Scientific-notation text with 14 digits after the decimal point.
    """
    numeric_value = float(value)
    if math.isinf(numeric_value):
        return "-inf" if numeric_value < 0.0 else "inf"
    return f"{numeric_value:.{SCORE_TEXT_DECIMAL_DIGITS}e}"
