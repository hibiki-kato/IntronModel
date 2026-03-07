from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from models import cnn


def test_add_train_args_includes_perf_flags() -> None:
    parser = argparse.ArgumentParser()
    cnn.add_train_args(parser)
    args = parser.parse_args(
        [
            "--use_amp",
            "0",
            "--amp_dtype",
            "fp16",
            "--allow_tf32",
            "0",
            "--cudnn_benchmark",
            "0",
            "--deterministic",
            "1",
            "--compile_mode",
            "off",
            "--max_pool_size",
            "1",
            "--conv_stride",
            "2",
            "--head_type",
            "center",
            "--num_workers",
            "2",
            "--prefetch_factor",
            "3",
            "--persistent_workers",
            "0",
            "--pin_memory",
            "0",
            "--report_train_metrics",
            "0",
        ]
    )
    assert args.use_amp == 0
    assert args.amp_dtype == "fp16"
    assert args.allow_tf32 == 0
    assert args.cudnn_benchmark == 0
    assert args.deterministic == 1
    assert args.compile_mode == "off"
    assert args.max_pool_size == 1
    assert args.conv_stride == 2
    assert args.head_type == "center"
    assert args.num_workers == "2"
    assert args.prefetch_factor == 3
    assert args.persistent_workers == 0
    assert args.pin_memory == 0
    assert args.report_train_metrics == 0


def test_basic_splice_cnn_supports_disabling_max_pool() -> None:
    model = cnn.BasicSpliceCNN(
        conv_channels=[32, 64],
        kernel_size=[7, 5],
        max_pool_size=1,
    )

    max_pool_layers = [
        layer for layer in model.conv_layers if isinstance(layer, torch.nn.MaxPool1d)
    ]
    assert max_pool_layers == []

    batch = torch.randn(4, 4, 101)
    logits = model(batch)
    assert logits.shape == (4,)


def test_basic_splice_cnn_supports_center_readout_with_stride() -> None:
    model = cnn.BasicSpliceCNN(
        conv_channels=[32, 64],
        kernel_size=[7, 5],
        max_pool_size=1,
        conv_stride=2,
        head_type="center",
    )

    batch = torch.randn(4, 4, 101)
    logits = model(batch)
    assert logits.shape == (4,)


def test_resolve_task_train_params_prefers_task_overrides() -> None:
    parser = argparse.ArgumentParser()
    cnn.add_train_args(parser)
    args = parser.parse_args(
        [
            "--batch_size",
            "512",
            "--lr",
            "0.0005",
            "--loss",
            "focal",
            "--f1_lambda",
            "0.1",
            "--conv_channels",
            "64,128,256",
            "--donor_batch_size",
            "1024",
            "--donor_lr",
            "0.001",
            "--donor_loss",
            "weighted_bce",
            "--donor_f1_lambda",
            "0.25",
            "--donor_conv_channels",
            "128,256,512",
            "--donor_conv_stride",
            "2",
            "--donor_head_type",
            "center",
        ]
    )

    shared_conv = cnn.parse_conv_channels(args.conv_channels)
    donor_conv = cnn.parse_conv_channels(
        args.donor_conv_channels,
        arg_name="--donor_conv_channels",
    )
    acceptor_conv = cnn.parse_conv_channels(
        args.acceptor_conv_channels,
        arg_name="--acceptor_conv_channels",
    )
    shared_kernel = cnn.parse_kernel_sizes(args.kernel_sizes)
    donor_kernel = cnn.parse_kernel_sizes(
        args.donor_kernel_sizes,
        arg_name="--donor_kernel_sizes",
    )
    acceptor_kernel = cnn.parse_kernel_sizes(
        args.acceptor_kernel_sizes,
        arg_name="--acceptor_kernel_sizes",
    )

    donor_params = cnn._resolve_task_train_params(
        task="donor",
        model_args=args,
        shared_conv_channels=shared_conv,
        donor_conv_channels=donor_conv,
        acceptor_conv_channels=acceptor_conv,
        shared_kernel_sizes=shared_kernel,
        donor_kernel_sizes=donor_kernel,
        acceptor_kernel_sizes=acceptor_kernel,
    )
    acceptor_params = cnn._resolve_task_train_params(
        task="acceptor",
        model_args=args,
        shared_conv_channels=shared_conv,
        donor_conv_channels=donor_conv,
        acceptor_conv_channels=acceptor_conv,
        shared_kernel_sizes=shared_kernel,
        donor_kernel_sizes=donor_kernel,
        acceptor_kernel_sizes=acceptor_kernel,
    )

    assert donor_params.batch_size == 1024
    assert donor_params.lr == pytest.approx(0.001)
    assert donor_params.loss_name == "weighted_bce"
    assert donor_params.f1_lambda == pytest.approx(0.25)
    assert donor_params.conv_channels == [128, 256, 512]
    assert donor_params.conv_stride == 2
    assert donor_params.head_type == "center"
    assert acceptor_params.batch_size == 512
    assert acceptor_params.lr == pytest.approx(0.0005)
    assert acceptor_params.loss_name == "focal"
    assert acceptor_params.f1_lambda == pytest.approx(0.1)
    assert acceptor_params.conv_channels == [64, 128, 256]
    assert acceptor_params.conv_stride == 1
    assert acceptor_params.head_type == "gap"


def test_resolve_task_train_params_prefers_kernel_size_overrides() -> None:
    parser = argparse.ArgumentParser()
    cnn.add_train_args(parser)
    args = parser.parse_args(
        [
            "--conv_channels",
            "64,128,256",
            "--kernel_sizes",
            "11,9,7",
            "--donor_kernel_sizes",
            "13,11",
        ]
    )

    shared_conv = cnn.parse_conv_channels(args.conv_channels)
    donor_conv = cnn.parse_conv_channels(
        args.donor_conv_channels,
        arg_name="--donor_conv_channels",
    )
    acceptor_conv = cnn.parse_conv_channels(
        args.acceptor_conv_channels,
        arg_name="--acceptor_conv_channels",
    )
    shared_kernel = cnn.parse_kernel_sizes(args.kernel_sizes)
    donor_kernel = cnn.parse_kernel_sizes(
        args.donor_kernel_sizes,
        arg_name="--donor_kernel_sizes",
    )
    acceptor_kernel = cnn.parse_kernel_sizes(
        args.acceptor_kernel_sizes,
        arg_name="--acceptor_kernel_sizes",
    )

    donor_params = cnn._resolve_task_train_params(
        task="donor",
        model_args=args,
        shared_conv_channels=shared_conv,
        donor_conv_channels=donor_conv,
        acceptor_conv_channels=acceptor_conv,
        shared_kernel_sizes=shared_kernel,
        donor_kernel_sizes=donor_kernel,
        acceptor_kernel_sizes=acceptor_kernel,
    )
    acceptor_params = cnn._resolve_task_train_params(
        task="acceptor",
        model_args=args,
        shared_conv_channels=shared_conv,
        donor_conv_channels=donor_conv,
        acceptor_conv_channels=acceptor_conv,
        shared_kernel_sizes=shared_kernel,
        donor_kernel_sizes=donor_kernel,
        acceptor_kernel_sizes=acceptor_kernel,
    )

    assert donor_params.kernel_sizes == [13, 11]
    assert acceptor_params.kernel_sizes == [11, 9, 7]


def test_resolve_compile_enabled_auto_quick_phase() -> None:
    enabled = cnn._resolve_compile_enabled(
        compile_mode="auto",
        compile_flag=False,
        quick_phase=True,
        device="cuda",
        epochs=20,
    )
    assert enabled is False


def test_resolve_compile_enabled_auto_requires_ptxas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    monkeypatch.delenv("TRITON_PTXAS_BLACKWELL_PATH", raising=False)
    monkeypatch.setattr(cnn.shutil, "which", lambda _: None)
    enabled = cnn._resolve_compile_enabled(
        compile_mode="auto",
        compile_flag=False,
        quick_phase=False,
        device="cuda",
        epochs=20,
    )
    assert enabled is False


def test_resolve_compile_enabled_auto_with_ptxas_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/tmp/ptxas")
    monkeypatch.delenv("TRITON_PTXAS_BLACKWELL_PATH", raising=False)
    monkeypatch.setattr(cnn.shutil, "which", lambda _: None)
    enabled = cnn._resolve_compile_enabled(
        compile_mode="auto",
        compile_flag=False,
        quick_phase=False,
        device="cuda",
        epochs=20,
    )
    assert enabled is True


def test_is_compile_runtime_error_detects_inductor() -> None:
    exc = RuntimeError("torch._inductor.exc.InductorError: ptxas missing")
    assert cnn._is_compile_runtime_error(exc) is True


def test_is_mps_oom_error_detects_message() -> None:
    exc = RuntimeError("MPS backend out of memory while allocating tensor")
    assert cnn._is_mps_oom_error(exc) is True


def test_resolve_mps_max_batch_size_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_MPS_MAX_BATCH_SIZE", "384")
    assert cnn._resolve_mps_max_batch_size() == 384


def test_resolve_mps_max_batch_size_invalid_env_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_MPS_MAX_BATCH_SIZE", "bad")
    assert (
        cnn._resolve_mps_max_batch_size() == cnn.DEFAULT_MPS_MAX_BATCH_SIZE
    )


def test_configure_triton_tool_paths_sets_blackwell_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    monkeypatch.delenv("TRITON_PTXAS_BLACKWELL_PATH", raising=False)
    monkeypatch.setattr(cnn.shutil, "which", lambda _: "/tmp/ptxas")
    cnn._configure_triton_tool_paths()
    assert cnn.os.environ["TRITON_PTXAS_PATH"] == "/tmp/ptxas"
    assert cnn.os.environ["TRITON_PTXAS_BLACKWELL_PATH"] == "/tmp/ptxas"


def test_export_model_state_dict_unwraps_orig_mod() -> None:
    class _CompiledLikeWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._orig_mod = torch.nn.Linear(4, 2)
            self.wrapper_only = torch.nn.Linear(4, 2)

    wrapped = _CompiledLikeWrapper()
    state = cnn._export_model_state_dict(wrapped)
    assert "weight" in state
    assert "bias" in state
    assert all(not key.startswith("_orig_mod.") for key in state)


def test_normalize_checkpoint_state_dict_strips_orig_mod_prefix() -> None:
    raw_state = {
        "_orig_mod.fc.weight": torch.zeros((2, 2)),
        "_orig_mod.fc.bias": torch.zeros((2,)),
    }
    normalized = cnn._normalize_checkpoint_state_dict(raw_state)
    assert "fc.weight" in normalized
    assert "fc.bias" in normalized
    assert all(not key.startswith("_orig_mod.") for key in normalized)


def test_train_task_model_includes_pr_auc_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples: list[tuple[str, int]] = []
    for _ in range(20):
        examples.append(("ACGT" * 16, 1))
        examples.append(("TGCA" * 16, 0))

    def _fake_read_examples_single_task(
        pos_path: str,
        neg_path: str,
        task: str,
        donor_len: int | None,
        acceptor_len: int | None,
    ) -> list[tuple[str, int]]:
        del pos_path, neg_path, task, donor_len, acceptor_len
        return examples

    monkeypatch.setattr(
        cnn,
        "read_examples_single_task",
        _fake_read_examples_single_task,
    )

    checkpoint_path = tmp_path / "donor.pt"
    summary = cnn.train_task_model(
        task="donor",
        pos_path="unused_pos",
        neg_path="unused_neg",
        checkpoint_path=str(checkpoint_path),
        window_len=32,
        donor_len=32,
        acceptor_len=32,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        seed=7,
        compile_mode="off",
        use_amp=0,
        num_workers=0,
        pin_memory=0,
        persistent_workers=0,
        min_batch_size=8,
    )

    assert "best_pr_auc" in summary
    assert "best_roc_auc" in summary
    assert "best_max_f1" in summary
    assert "best_acc_at_0_5" in summary
    assert "effective_batch_size" in summary
    assert "oom_retries" in summary


def test_train_task_model_can_skip_train_eval_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples: list[tuple[str, int]] = []
    for _ in range(16):
        examples.append(("ACGT" * 16, 1))
        examples.append(("TGCA" * 16, 0))

    def _fake_read_examples_single_task(
        pos_path: str,
        neg_path: str,
        task: str,
        donor_len: int | None,
        acceptor_len: int | None,
    ) -> list[tuple[str, int]]:
        del pos_path, neg_path, task, donor_len, acceptor_len
        return examples

    evaluate_calls: list[int] = []

    def _fake_evaluate(
        model: torch.nn.Module,
        loader: DataLoader,
        device: str,
        use_amp: bool,
        amp_dtype: torch.dtype | None,
    ) -> dict[str, float]:
        del model, loader, device, use_amp, amp_dtype
        evaluate_calls.append(1)
        return {"pr_auc": 0.5, "acc@0.5": 0.5}

    monkeypatch.setattr(
        cnn,
        "read_examples_single_task",
        _fake_read_examples_single_task,
    )
    monkeypatch.setattr(cnn, "evaluate", _fake_evaluate)

    checkpoint_path = tmp_path / "donor_skip_train_eval.pt"
    summary = cnn.train_task_model(
        task="donor",
        pos_path="unused_pos",
        neg_path="unused_neg",
        checkpoint_path=str(checkpoint_path),
        window_len=32,
        donor_len=32,
        acceptor_len=32,
        epochs=2,
        batch_size=8,
        lr=1e-3,
        seed=7,
        compile_mode="off",
        use_amp=0,
        num_workers=0,
        pin_memory=0,
        persistent_workers=0,
        report_train_metrics=0,
        min_batch_size=8,
    )

    assert len(evaluate_calls) == 2
    assert summary["report_train_metrics"] is False
    for row in summary["epoch_history"]:
        assert row["train_pr_auc"] is None


def test_evaluate_handles_bfloat16_logits() -> None:
    class _BFloat16Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            return torch.zeros(
                (batch_size,),
                dtype=torch.bfloat16,
                device=x.device,
            )

    features = torch.zeros((8, 4, 16), dtype=torch.float32)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    loader = DataLoader(TensorDataset(features, labels), batch_size=4)

    metrics = cnn.evaluate(
        model=_BFloat16Model(),
        loader=loader,
        device="cpu",
        use_amp=False,
        amp_dtype=None,
    )

    assert "acc@0.5" in metrics


def test_evaluate_fallback_metrics_without_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cnn, "average_precision_score", None)
    monkeypatch.setattr(cnn, "roc_auc_score", None)

    class _ScoreFromInputModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, 0, 0].float()

    features = torch.zeros((6, 4, 16), dtype=torch.float32)
    features[:, 0, 0] = torch.tensor([-2.0, -1.0, 0.2, 0.4, 0.0, 1.2])
    labels = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.float32)
    loader = DataLoader(TensorDataset(features, labels), batch_size=3)

    metrics = cnn.evaluate(
        model=_ScoreFromInputModel(),
        loader=loader,
        device="cpu",
        use_amp=False,
        amp_dtype=None,
    )

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "max_f1" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["max_f1"] <= 1.0


def test_sigmoid_np_handles_extreme_logits_without_overflow() -> None:
    logits = np.array([1000.0, 100.0, 0.0, -100.0, -1000.0], dtype=np.float32)
    with np.errstate(over="raise", invalid="raise"):
        probs = cnn.sigmoid_np(logits)

    assert probs.shape == logits.shape
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert probs[0] == pytest.approx(1.0, rel=0.0, abs=1e-7)
    assert probs[-1] == pytest.approx(0.0, rel=0.0, abs=1e-7)


def test_dna_dataset_preencode_matches_on_demand_encoding() -> None:
    examples = [("ACGTACGT", 1), ("TGCATGCA", 0)]
    ds_cached = cnn.DNADataset(
        examples=examples,
        window_len=8,
        preencode=True,
    )
    ds_plain = cnn.DNADataset(
        examples=examples,
        window_len=8,
        preencode=False,
    )

    assert len(ds_cached) == len(ds_plain) == 2
    for idx in range(2):
        x_cached, y_cached = ds_cached[idx]
        x_plain, y_plain = ds_plain[idx]
        assert torch.allclose(x_cached, x_plain)
        assert float(y_cached.item()) == float(y_plain.item())
