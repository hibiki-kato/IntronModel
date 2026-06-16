"""Shared model capability flags used by orchestration layers."""

from __future__ import annotations

QUICK_FULL_WARM_START_SUPPORTED_MODELS: frozenset[str] = frozenset(
    {
        "cnn",
        "cnn_pair",
        "cnn_v2",
        "cnn_pair_v2",
        "cnn_v3",
        "cnn_pair_v3",
        "cnn_v3_meta",
        "bilstm_pair",
        "tcn",
        "bert",
        "dnabert",
        "dnabert2",
        "dnabert6",
        "dnaberts",
        "dnabert_pair",
        "dnabert2_pair",
        "dnabert6_pair",
        "dnaberts_pair",
        "reservoir_legacy",
        "cnn_resdil",
    }
)


def supports_quick_full_warm_start(model_name: object) -> bool:
    """Return whether one model supports quick-to-full checkpoint continuation.

    Parameters
    ----------
    model_name : object
        Candidate model name.

    Returns
    -------
    bool
        ``True`` when the model can reuse quick-phase checkpoints as the
        initialization for the remaining full-phase epoch budget.
    """
    if not isinstance(model_name, str):
        return False
    normalized_model_name = model_name.strip().lower()
    if normalized_model_name == "":
        return False
    return normalized_model_name in QUICK_FULL_WARM_START_SUPPORTED_MODELS
