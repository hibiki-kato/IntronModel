from __future__ import annotations

import argparse

import pytest
import torch

from models import cnn_pair_v3


def test_resolve_pair_arch_params_prefers_branch_overrides() -> None:
    args = argparse.Namespace(
        conv_channels="64,128,256",
        donor_conv_channels="96,160,256,384",
        acceptor_conv_channels=None,
        kernel_sizes="9,7,5",
        donor_kernel_sizes="11,9,7,5",
        acceptor_kernel_sizes=None,
        block_dilations="1,2,4",
        donor_block_dilations="1,2,4,8",
        acceptor_block_dilations=None,
        residual_channels="32,64,96",
        donor_residual_channels="48,80,128,160",
        acceptor_residual_channels=None,
        head_type="gap",
        fc_hidden=192,
    )

    resolved = cnn_pair_v3._resolve_pair_arch_params(args)

    assert resolved.donor.channels == [96, 160, 256, 384]
    assert resolved.acceptor.channels == [64, 128, 256]
    assert resolved.donor.kernel_sizes == [11, 9, 7, 5]
    assert resolved.acceptor.kernel_sizes == [9, 7, 5]
    assert resolved.donor.dilations == [1, 2, 4, 8]
    assert resolved.acceptor.dilations == [1, 2, 4]
    assert resolved.donor.residual_channels == [48, 80, 128, 160]
    assert resolved.acceptor.residual_channels == [32, 64, 96]


def test_pair_organic_resdil_cnn_forward_onehot() -> None:
    arch = cnn_pair_v3.PairOrganicArchParams(
        donor=cnn_pair_v3.OrganicBranchLayout(
            channels=[32, 64, 96],
            kernel_sizes=[9, 7, 5],
            dilations=[1, 2, 4],
            residual_channels=[16, 32, 48],
        ),
        acceptor=cnn_pair_v3.OrganicBranchLayout(
            channels=[32, 64, 96],
            kernel_sizes=[9, 7, 5],
            dilations=[1, 2, 4],
            residual_channels=[16, 32, 48],
        ),
        head_type="gap",
        fc_hidden=64,
    )
    model = cnn_pair_v3.PairOrganicResDilCNN(
        input_mode="onehot",
        pair_mode="pair",
        embedding_dim=32,
        vocab_size=None,
        arch_params=arch,
        dropout=0.1,
    )

    donor_x = torch.randn(3, 4, 100)
    acceptor_x = torch.randn(3, 4, 100)

    logits = model(donor_x, acceptor_x)

    assert logits.shape == (3,)


def test_pair_organic_resdil_cnn_supports_token_input() -> None:
    arch = cnn_pair_v3.PairOrganicArchParams(
        donor=cnn_pair_v3.OrganicBranchLayout(
            channels=[16, 24],
            kernel_sizes=[7, 5],
            dilations=[1, 2],
            residual_channels=[8, 12],
        ),
        acceptor=cnn_pair_v3.OrganicBranchLayout(
            channels=[16, 32, 32],
            kernel_sizes=[7, 5, 5],
            dilations=[1, 2, 4],
            residual_channels=[8, 16, 16],
        ),
        head_type="center",
        fc_hidden=48,
    )
    model = cnn_pair_v3.PairOrganicResDilCNN(
        input_mode="kmer3",
        pair_mode="pair",
        embedding_dim=12,
        vocab_size=65,
        arch_params=arch,
        dropout=0.1,
    )

    donor_x = torch.randint(0, 65, (2, 20))
    acceptor_x = torch.randint(0, 65, (2, 20))

    logits = model(donor_x, acceptor_x)

    assert logits.shape == (2,)


def test_train_pair_model_forwards_model_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _StopAfterArchResolution(Exception):
        """Stop the train path after architecture resolution."""

    def _fake_resolve_pair_arch_params(
        model_args: argparse.Namespace,
        *,
        lightweight: bool = False,
    ) -> cnn_pair_v3.PairOrganicArchParams:
        captured["model_args"] = model_args
        captured["lightweight"] = lightweight
        raise _StopAfterArchResolution

    monkeypatch.setattr(
        cnn_pair_v3,
        "_resolve_pair_arch_params",
        _fake_resolve_pair_arch_params,
    )
    train_params = cnn_v2_params()
    model_args = argparse.Namespace(
        conv_channels=None,
        donor_conv_channels="32,64,96",
        acceptor_conv_channels="32,64,128",
        kernel_sizes=None,
        donor_kernel_sizes="9,7,5",
        acceptor_kernel_sizes="9,7,5",
        block_dilations="1,2,4",
        donor_block_dilations=None,
        acceptor_block_dilations=None,
        residual_channels="16,32,48",
        donor_residual_channels=None,
        acceptor_residual_channels=None,
        head_type="gap",
        fc_hidden=64,
    )

    with pytest.raises(_StopAfterArchResolution):
        cnn_pair_v3.train_pair_model(
            pos_path="pos.tsv",
            neg_path="neg.tsv",
            checkpoint_path="pair.pt",
            donor_window_len=100,
            acceptor_window_len=100,
            donor_len=100,
            acceptor_len=100,
            model_args=model_args,
            train_params=train_params,
            epochs=1,
            early_stop_patience=0,
            early_stop_min_delta=0.0,
            sequence_transform="none",
            seed=1337,
            lightweight=False,
            compile_model=False,
            compile_mode="off",
            device="cpu",
            use_amp=0,
            amp_dtype="auto",
            allow_tf32=0,
            cudnn_benchmark=0,
            deterministic=1,
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=0,
            pin_memory=0,
            min_batch_size=1,
            max_oom_retries=0,
            quick_phase=False,
            init_checkpoint_path=None,
        )

    assert captured["model_args"] is model_args
    assert captured["lightweight"] is False


def cnn_v2_params() -> cnn_pair_v3.cnn_v2.PairTrainParams:
    """Build one pair-train parameter object for shared training tests."""
    return cnn_pair_v3.cnn_v2.PairTrainParams(
        batch_size=4,
        lr=1e-3,
        loss_name="focal",
        input_mode="onehot",
        pair_mode="pair",
        fusion_mode="late",
        embedding_dim=8,
        bpe_pretrained_model_name=cnn_pair_v3.BPE_DEFAULT_MODEL_NAME,
        bpe_pretrained_revision=None,
        bpe_trust_remote_code=False,
        dropout=0.1,
        weight_decay=0.01,
        eta_min_ratio=0.01,
        val_frac=0.2,
        grad_clip=5.0,
        pos_weight_cap=20.0,
        focal_gamma=2.0,
        focal_alpha_pos=None,
        asym_gamma_pos=0.0,
        asym_gamma_neg=4.0,
        asym_alpha_pos=None,
        f1_lambda=0.1,
    )
