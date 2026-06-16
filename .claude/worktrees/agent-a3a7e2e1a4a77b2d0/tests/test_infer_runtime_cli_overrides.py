from __future__ import annotations

import argparse
from types import ModuleType

import pytest

from models import cnn_resdil, tcn


@pytest.mark.parametrize(
    "module",
    [cnn_resdil, tcn],
    ids=["cnn_resdil", "tcn"],
)
def test_add_infer_args_registers_runtime_override_flags(
    module: ModuleType,
) -> None:
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    module.add_train_args(parser)
    module.add_infer_args(parser)

    args = parser.parse_args([])

    assert args.infer_batch_size is None
    assert args.infer_use_amp is None
    assert args.infer_amp_dtype is None
    assert args.infer_compile is None
    assert args.infer_compile_mode is None


@pytest.mark.parametrize(
    "module",
    [cnn_resdil, tcn],
    ids=["cnn_resdil", "tcn"],
)
def test_add_infer_args_parses_runtime_override_values(
    module: ModuleType,
) -> None:
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    module.add_train_args(parser)
    module.add_infer_args(parser)

    args = parser.parse_args(
        [
            "--infer_batch_size",
            "1024",
            "--infer_use_amp",
            "1",
            "--infer_amp_dtype",
            "bf16",
            "--infer_compile",
            "0",
            "--infer_compile_mode",
            "auto",
        ]
    )

    assert args.infer_batch_size == 1024
    assert args.infer_use_amp == 1
    assert args.infer_amp_dtype == "bf16"
    assert args.infer_compile == 0
    assert args.infer_compile_mode == "auto"
