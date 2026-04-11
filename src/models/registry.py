"""Model registry and capability validation.

This module centralizes model name -> module resolution for the unified CLI.
Each model module is expected to expose a common contract:

- add_train_args(parser)
- add_infer_args(parser)
- train(common_args, model_args) -> dict[str, object]
- infer_site(common_args, model_args) -> list[dict[str, object]]
"""

from __future__ import annotations

import argparse
import importlib
from types import ModuleType
from typing import Protocol, cast


class ModelModuleProtocol(Protocol):
    """Protocol for model modules used by ``run_model.py``."""

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        """Register model-specific training arguments."""

    def add_infer_args(self, parser: argparse.ArgumentParser) -> None:
        """Register model-specific inference arguments."""

    def train(
        self, common_args: argparse.Namespace, model_args: argparse.Namespace
    ) -> dict[str, object]:
        """Train donor/acceptor models and return a metrics summary."""

    def infer_site(
        self, common_args: argparse.Namespace, model_args: argparse.Namespace
    ) -> list[dict[str, object]]:
        """Infer site-level rows with fixed schema."""


_MODEL_TO_MODULE: dict[str, str] = {
    "cnn": "models.cnn",
    # cnn_pair is an alias for cnn_v2 (cnn_pair.py archived)
    "cnn_pair": "models.cnn_v2",
    "cnn_v2": "models.cnn_v2",
    "cnn_pair_v2": "models.cnn_v2",
    "cnn_v3": "models.cnn_v3",
    "cnn_pair_v3": "models.cnn_pair_v3",
    "cnn_v3_meta": "models.cnn_v3_meta",
    "bilstm_pair": "models.bilstm_pair",
    "markov_xgboost": "models.markov_xgboost",
    "tcn": "models.tcn",
    "bert": "models.bert",
    "dnabert": "models.dnabert",
    "dnabert2": "models.dnabert",
    "dnabert6": "models.dnabert",
    "dnaberts": "models.dnabert",
    "dnabert_pair": "models.dnabert",
    "dnabert2_pair": "models.dnabert",
    "dnabert6_pair": "models.dnabert",
    "dnaberts_pair": "models.dnabert",
    "reservoir": "models.reservoir",
    "cnn_resdil": "models.cnn_resdil",
    "spliceformer_sc": "models.spliceformer_sc",
}


def available_models() -> list[str]:
    """Return available model names."""
    return sorted(_MODEL_TO_MODULE.keys())


def _validate_model_contract(module: ModuleType, model_name: str) -> None:
    """Validate required API functions on a model module.

    Parameters
    ----------
    module : ModuleType
        Imported model module.
    model_name : str
        Model name from the registry.

    Raises
    ------
    TypeError
        If any required symbol is missing or not callable.
    """
    required = (
        "add_train_args",
        "add_infer_args",
        "train",
        "infer_site",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        missing_text = ", ".join(missing)
        raise TypeError(f"Model '{model_name}' is missing required API: {missing_text}")

    non_callable = [name for name in required if not callable(getattr(module, name))]
    if non_callable:
        bad_text = ", ".join(non_callable)
        raise TypeError(
            f"Model '{model_name}' has non-callable API symbols: {bad_text}"
        )


def load_model_module(model_name: str) -> ModelModuleProtocol:
    """Load a model module by name and validate the unified API contract.

    Parameters
    ----------
    model_name : str
        Model key registered in this module.

    Returns
    -------
    ModelModuleProtocol
        Imported module object cast to the required protocol.

    Raises
    ------
    ValueError
        If model name is unknown.
    RuntimeError
        If importing fails due to missing optional dependencies.
    TypeError
        If model module does not match the expected contract.
    """
    if model_name not in _MODEL_TO_MODULE:
        known = ", ".join(available_models())
        raise ValueError(f"Unknown model '{model_name}'. Available: {known}")

    module_name = _MODEL_TO_MODULE[model_name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                f"PyTorch is required for model '{model_name}'. "
                "Activate the correct environment."
            ) from exc
        raise

    _validate_model_contract(module=module, model_name=model_name)
    return cast(ModelModuleProtocol, module)
