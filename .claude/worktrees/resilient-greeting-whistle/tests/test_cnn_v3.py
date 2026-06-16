from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from models import cnn, cnn_v3


def test_resolve_task_arch_params_prefers_task_overrides() -> None:
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
        max_pool_size=2,
        donor_max_pool_size=3,
        acceptor_max_pool_size=4,
        pool_every=2,
        donor_pool_every=1,
        acceptor_pool_every=3,
        head_type="gap",
        donor_head_type="center",
        acceptor_head_type="gap",
        fc_hidden=192,
        donor_fc_hidden=320,
        acceptor_fc_hidden=224,
    )

    donor_arch = cnn_v3._resolve_task_arch_params("donor", args)
    acceptor_arch = cnn_v3._resolve_task_arch_params("acceptor", args)

    assert donor_arch.layout.channels == [96, 160, 256, 384]
    assert acceptor_arch.layout.channels == [64, 128, 256]
    assert donor_arch.layout.kernel_sizes == [11, 9, 7, 5]
    assert acceptor_arch.layout.kernel_sizes == [9, 7, 5]
    assert donor_arch.layout.dilations == [1, 2, 4, 8]
    assert acceptor_arch.layout.dilations == [1, 2, 4]
    assert donor_arch.layout.residual_channels == [48, 80, 128, 160]
    assert acceptor_arch.layout.residual_channels == [32, 64, 96]
    assert donor_arch.max_pool_size == 3
    assert acceptor_arch.max_pool_size == 4
    assert donor_arch.pool_every == 1
    assert acceptor_arch.pool_every == 3
    assert donor_arch.head_type == "center"
    assert acceptor_arch.head_type == "gap"
    assert donor_arch.fc_hidden == 320
    assert acceptor_arch.fc_hidden == 224


def test_add_train_args_accepts_task_specific_pool_every() -> None:
    parser = argparse.ArgumentParser()

    cnn_v3.add_train_args(parser)

    args = parser.parse_args(
        [
            "--pool_every",
            "2",
            "--donor_pool_every",
            "1",
            "--acceptor_pool_every",
            "3",
        ]
    )

    assert args.pool_every == 2
    assert args.donor_pool_every == 1
    assert args.acceptor_pool_every == 3


def test_organic_site_cnn_forward_onehot() -> None:
    arch = cnn_v3.TaskOrganicArchParams(
        layout=cnn_v3.OrganicBranchLayout(
            channels=[32, 64, 96],
            kernel_sizes=[9, 7, 5],
            dilations=[1, 2, 4],
            residual_channels=[16, 32, 48],
        ),
        max_pool_size=2,
        pool_every=2,
        head_type="gap",
        fc_hidden=64,
    )
    model = cnn_v3.OrganicSiteCNN(
        arch_params=arch,
        dropout=0.1,
    )

    x = torch.randn(3, 4, 100)
    logits = model(x)

    assert logits.shape == (3,)


def test_organic_site_cnn_rejects_invalid_rank() -> None:
    arch = cnn_v3.TaskOrganicArchParams(
        layout=cnn_v3.OrganicBranchLayout(
            channels=[16, 32],
            kernel_sizes=[7, 5],
            dilations=[1, 2],
            residual_channels=[8, 16],
        ),
        max_pool_size=2,
        pool_every=2,
        head_type="center",
        fc_hidden=32,
    )
    model = cnn_v3.OrganicSiteCNN(
        arch_params=arch,
        dropout=0.1,
    )

    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(4, 100))


def test_train_task_model_forwards_requested_task_to_arch_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _StopAfterArchResolution(Exception):
        """Stop after architecture resolution to keep the test lightweight."""

    def _fake_resolve_task_arch_params(
        task: str,
        model_args: argparse.Namespace,
        *,
        lightweight: bool = False,
    ) -> cnn_v3.TaskOrganicArchParams:
        captured["task"] = task
        captured["model_args"] = model_args
        captured["lightweight"] = lightweight
        raise _StopAfterArchResolution

    monkeypatch.setattr(
        cnn_v3,
        "_resolve_task_arch_params",
        _fake_resolve_task_arch_params,
    )
    task_params = build_task_train_params()
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
        max_pool_size=2,
        donor_max_pool_size=None,
        acceptor_max_pool_size=None,
        pool_every=2,
        donor_pool_every=None,
        acceptor_pool_every=None,
        head_type="gap",
        donor_head_type=None,
        acceptor_head_type=None,
        fc_hidden=64,
        donor_fc_hidden=None,
        acceptor_fc_hidden=None,
    )

    with pytest.raises(_StopAfterArchResolution):
        cnn_v3.train_task_model(
            task="acceptor",
            pos_path="pos.tsv",
            neg_path="neg.tsv",
            checkpoint_path="acceptor.pt",
            window_len=100,
            donor_len=100,
            acceptor_len=100,
            model_args=model_args,
            task_params=task_params,
            epochs=1,
            early_stop_patience=0,
            early_stop_min_delta=0.0,
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
        )

    assert captured["task"] == "acceptor"
    assert captured["model_args"] is model_args
    assert captured["lightweight"] is False


def test_train_task_model_logs_selected_validation_metric_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    examples = [("ACGT" * 25, 1), ("TGCA" * 25, 0)] * 10

    def _fake_load_task_examples_with_transform(
        *,
        pos_path: str,
        neg_path: str,
        task: str,
        donor_len: int | None,
        acceptor_len: int | None,
        sequence_transform: str,
    ) -> list[tuple[str, int]]:
        del pos_path, neg_path, task, donor_len, acceptor_len, sequence_transform
        return examples

    eval_calls = iter(
        [
            {"pr_auc": 0.9228, "max_f1": 0.8849, "acc@0.5": 0.90},
            {"pr_auc": 0.9491, "max_f1": 0.9012, "acc@0.5": 0.93},
        ]
    )

    def _fake_evaluate(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader[object],
        device: str,
        use_amp: bool,
        amp_dtype: torch.dtype | None,
    ) -> dict[str, float]:
        del model, loader, device, use_amp, amp_dtype
        return next(eval_calls)

    monkeypatch.setattr(
        cnn,
        "_load_task_examples_with_transform",
        _fake_load_task_examples_with_transform,
    )
    monkeypatch.setattr(cnn, "evaluate", _fake_evaluate)

    model_args = argparse.Namespace(
        conv_channels="16,32",
        donor_conv_channels=None,
        acceptor_conv_channels=None,
        kernel_sizes="7,5",
        donor_kernel_sizes=None,
        acceptor_kernel_sizes=None,
        block_dilations="1,2",
        donor_block_dilations=None,
        acceptor_block_dilations=None,
        residual_channels="8,16",
        donor_residual_channels=None,
        acceptor_residual_channels=None,
        max_pool_size=2,
        donor_max_pool_size=None,
        acceptor_max_pool_size=None,
        pool_every=1,
        donor_pool_every=None,
        acceptor_pool_every=None,
        head_type="gap",
        donor_head_type=None,
        acceptor_head_type=None,
        fc_hidden=32,
        donor_fc_hidden=None,
        acceptor_fc_hidden=None,
    )

    summary = cnn_v3.train_task_model(
        task="acceptor",
        pos_path="pos.tsv",
        neg_path="neg.tsv",
        checkpoint_path=str(tmp_path / "acceptor.pt"),
        window_len=100,
        donor_len=100,
        acceptor_len=100,
        model_args=model_args,
        task_params=build_task_train_params(),
        epochs=1,
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        validation_metric="max_f1",
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
    )

    captured = capsys.readouterr().out
    assert "score_metric=max_f1" in captured
    assert "train_score=0.9012" in captured
    assert "val_score=0.8849" in captured
    assert summary["epoch_history"][0]["train_score"] == pytest.approx(0.9012)


def build_task_train_params() -> cnn.TaskTrainParams:
    """Build one site-train parameter object for shared training tests."""
    return cnn.TaskTrainParams(
        batch_size=4,
        lr=1e-3,
        loss_name="focal",
        conv_channels=[32, 64, 96],
        kernel_sizes=[9, 7, 5],
        max_pool_size=2,
        conv_stride=1,
        head_type="gap",
        dropout=0.1,
        fc_hidden=64,
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
