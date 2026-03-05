from __future__ import annotations

import argparse

import pytest

_ = pytest.importorskip("torch")

from models import cnn_pair


def test_add_train_args_accepts_branch_conv_overrides() -> None:
    parser = argparse.ArgumentParser()
    cnn_pair.add_train_args(parser)
    args = parser.parse_args(
        [
            "--conv_channels",
            "64,128,256",
            "--donor_conv_channels",
            "128,256,512",
            "--acceptor_conv_channels",
            "96,192,384",
        ]
    )

    assert args.conv_channels == "64,128,256"
    assert args.donor_conv_channels == "128,256,512"
    assert args.acceptor_conv_channels == "96,192,384"


def test_resolve_pair_train_params_prefers_branch_overrides() -> None:
    parser = argparse.ArgumentParser()
    cnn_pair.add_train_args(parser)
    args = parser.parse_args(
        [
            "--conv_channels",
            "64,128,256",
            "--donor_conv_channels",
            "128,256,512",
        ]
    )

    resolved = cnn_pair._resolve_pair_train_params(args)

    assert resolved.donor_conv_channels == [128, 256, 512]
    assert resolved.acceptor_conv_channels == [64, 128, 256]


def test_add_train_args_accepts_branch_kernel_overrides() -> None:
    parser = argparse.ArgumentParser()
    cnn_pair.add_train_args(parser)
    args = parser.parse_args(
        [
            "--kernel_sizes",
            "11,7,5",
            "--donor_kernel_sizes",
            "13,9,5",
            "--acceptor_kernel_sizes",
            "9,7,5",
        ]
    )

    assert args.kernel_sizes == "11,7,5"
    assert args.donor_kernel_sizes == "13,9,5"
    assert args.acceptor_kernel_sizes == "9,7,5"


def test_resolve_pair_train_params_prefers_branch_kernel_overrides() -> None:
    parser = argparse.ArgumentParser()
    cnn_pair.add_train_args(parser)
    args = parser.parse_args(
        [
            "--conv_channels",
            "64,128,256",
            "--kernel_sizes",
            "11,7,5",
            "--donor_kernel_sizes",
            "13,9,5",
        ]
    )

    resolved = cnn_pair._resolve_pair_train_params(args)

    assert resolved.donor_kernel_sizes == [13, 9, 5]
    assert resolved.acceptor_kernel_sizes == [11, 7, 5]


def test_pair_splice_cnn_accepts_mismatched_kernel_size_depth() -> None:
    model = cnn_pair.PairSpliceCNN(
        donor_conv_channels=[64, 128, 256],
        acceptor_conv_channels=[64, 128, 256],
        donor_kernel_sizes=[11, 7],
        acceptor_kernel_sizes=[11, 7, 5, 3],
    )
    assert isinstance(model, cnn_pair.PairSpliceCNN)


def test_resolve_pair_train_params_accepts_f1_lambda() -> None:
    parser = argparse.ArgumentParser()
    cnn_pair.add_train_args(parser)
    args = parser.parse_args(["--f1_lambda", "0.3"])

    resolved = cnn_pair._resolve_pair_train_params(args)
    assert resolved.f1_lambda == pytest.approx(0.3)
