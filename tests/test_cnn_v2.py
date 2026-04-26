from __future__ import annotations

import argparse

import models
import numpy as np
import pytest
import torch

from models import cnn_v2


def test_onehot_encoder_shapes_and_n_mask() -> None:
    encoder = cnn_v2._build_sequence_encoder(
        mode="onehot",
        window_len=6,
        bpe_pretrained_model_name=cnn_v2.BPE_DEFAULT_MODEL_NAME,
        bpe_pretrained_revision=None,
        bpe_trust_remote_code=False,
    )
    encoded = encoder.encode("ACGTNN")
    assert encoded.shape == (4, 6)
    assert np.all(encoded[:, 4:] == 0.0)


def test_kmer3_encoder_shape_and_vocab_bound() -> None:
    encoder = cnn_v2._build_sequence_encoder(
        mode="kmer3",
        window_len=8,
        bpe_pretrained_model_name=cnn_v2.BPE_DEFAULT_MODEL_NAME,
        bpe_pretrained_revision=None,
        bpe_trust_remote_code=False,
    )
    encoded = encoder.encode("ACGTACGT")
    assert encoded.shape == (6,)
    assert int(encoded.min()) >= 0
    assert int(encoded.max()) <= 64


def test_pair_splice_cnn_forward_onehot_pair_mode() -> None:
    model = cnn_v2.PairSpliceCNN(
        input_mode="onehot",
        pair_mode="pair",
        embedding_dim=32,
        vocab_size=None,
        dropout=0.1,
    )
    donor_x = torch.randn(3, 4, 100)
    acceptor_x = torch.randn(3, 4, 100)
    logits = model(donor_x, acceptor_x)
    assert logits.shape == (3,)


def test_prepare_infer_model_passes_compile_mode_to_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_modes: list[str] = []
    model = torch.nn.Linear(4, 1)

    monkeypatch.setattr(cnn_v2, "_configure_triton_tool_paths", lambda: None)
    monkeypatch.setattr(cnn_v2, "_configure_torch_compile_runtime", lambda: None)

    def _fake_compile_model_with_fallback(
        module: torch.nn.Module,
        *,
        compile_mode: str = "auto",
    ) -> tuple[torch.nn.Module, bool, str | None, Exception | None]:
        captured_modes.append(compile_mode)
        return module, True, "reduce-overhead", None

    monkeypatch.setattr(
        cnn_v2,
        "_compile_model_with_fallback",
        _fake_compile_model_with_fallback,
    )

    compiled = cnn_v2._prepare_infer_model(
        model=model,
        task_name="pair",
        compile_enabled=True,
        compile_mode="on",
    )

    assert compiled is model
    assert captured_modes == ["on"]


def test_pair_splice_cnn_supports_token_early_fusion() -> None:
    model = cnn_v2.PairSpliceCNN(
        input_mode="kmer3",
        pair_mode="pair",
        embedding_dim=8,
        vocab_size=65,
        donor_conv_channels=[8, 16],
        acceptor_conv_channels=[8, 16],
        donor_kernel_sizes=[5, 3],
        acceptor_kernel_sizes=[5, 3],
        max_pool_size=2,
        conv_stride=1,
        head_type="gap",
        fusion_mode="early",
        dropout=0.1,
        fc_hidden=32,
    )
    donor_x = torch.randint(0, 65, (2, 12))
    acceptor_x = torch.randint(0, 65, (2, 12))
    logits = model(donor_x, acceptor_x)
    assert logits.shape == (2,)


def test_pair_splice_cnn_late_allows_asymmetric_branches() -> None:
    model = cnn_v2.PairSpliceCNN(
        input_mode="kmer3",
        pair_mode="pair",
        embedding_dim=8,
        vocab_size=65,
        donor_conv_channels=[8, 16],
        acceptor_conv_channels=[16, 32, 32],
        donor_kernel_sizes=[5, 3],
        acceptor_kernel_sizes=[7, 5, 3],
        max_pool_size=2,
        conv_stride=1,
        head_type="gap",
        fusion_mode="late",
        dropout=0.1,
        fc_hidden=32,
    )
    donor_x = torch.randint(0, 65, (2, 10))
    acceptor_x = torch.randint(0, 65, (2, 8))
    logits = model(donor_x, acceptor_x)
    assert logits.shape == (2,)


def test_pair_mode_off_alias_maps_to_independent() -> None:
    assert cnn_v2._normalize_pair_mode("off", arg_name="--pair_mode") == "independent"


def test_resolve_pair_train_params_v2_flags() -> None:
    args = argparse.Namespace(
        batch_size=64,
        lr=1e-3,
        loss="weighted_bce",
        input_mode="kmer3",
        pair_mode="pair",
        fusion_mode="early",
        embedding_dim=32,
        bpe_pretrained_model_name=cnn_v2.BPE_DEFAULT_MODEL_NAME,
        bpe_pretrained_revision=None,
        bpe_trust_remote_code=0,
        dropout=0.2,
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
    params = cnn_v2._resolve_pair_train_params(args)
    assert params.input_mode == "kmer3"
    assert params.pair_mode == "pair"
    assert params.fusion_mode == "early"
    assert params.embedding_dim == 32


def test_add_train_args_accepts_validation_metric() -> None:
    parser = argparse.ArgumentParser()
    cnn_v2.add_train_args(parser)
    args = parser.parse_args(["--validation_metric", "max_f1"])

    assert args.validation_metric == "max_f1"


def test_add_train_and_infer_args_accept_high_level_compile_modes() -> None:
    train_parser = argparse.ArgumentParser()
    cnn_v2.add_train_args(train_parser)
    train_args = train_parser.parse_args(["--compile_mode", "quick"])

    infer_parser = argparse.ArgumentParser()
    cnn_v2.add_infer_args(infer_parser)
    infer_args = infer_parser.parse_args(["--infer_compile_mode", "full"])

    assert train_args.compile_mode == "quick"
    assert infer_args.infer_compile_mode == "full"


def test_train_pair_model_forwards_model_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _StopAfterArchResolution(Exception):
        """Stop the train path after architecture resolution."""

    def _fake_resolve_pair_model_arch_params(
        model_args: argparse.Namespace,
        *,
        lightweight: bool = False,
    ) -> cnn_v2.PairModelArchParams:
        captured["model_args"] = model_args
        captured["lightweight"] = lightweight
        raise _StopAfterArchResolution

    monkeypatch.setattr(
        cnn_v2,
        "_resolve_pair_model_arch_params",
        _fake_resolve_pair_model_arch_params,
    )
    train_params = cnn_v2.PairTrainParams(
        batch_size=4,
        lr=1e-3,
        loss_name="focal",
        input_mode="onehot",
        pair_mode="pair",
        fusion_mode="late",
        embedding_dim=8,
        bpe_pretrained_model_name=cnn_v2.BPE_DEFAULT_MODEL_NAME,
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
    model_args = argparse.Namespace(
        fusion_mode="late",
        conv_channels=None,
        donor_conv_channels=[8, 16],
        acceptor_conv_channels=[8, 16],
        donor_kernel_sizes=[5, 3],
        acceptor_kernel_sizes=[5, 3],
        max_pool_size=2,
        conv_stride=1,
        head_type="gap",
        fc_hidden=32,
    )

    with pytest.raises(_StopAfterArchResolution):
        cnn_v2.train_pair_model(
            pos_path="pos.tsv",
            neg_path="neg.tsv",
            checkpoint_path="pair.pt",
            init_checkpoint_path=None,
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
            seed=1,
            lightweight=False,
            compile_model=False,
            compile_mode="off",
            device="cpu",
            use_amp=False,
            amp_dtype="auto",
            allow_tf32=False,
            cudnn_benchmark=False,
            deterministic=False,
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory=False,
            min_batch_size=1,
            max_oom_retries=0,
            quick_phase=False,
            gpu_id=None,
        )

    assert captured["model_args"] is model_args
    assert captured["lightweight"] is False


def test_train_pair_model_rejects_invalid_validation_metric() -> None:
    train_params = cnn_v2.PairTrainParams(
        batch_size=4,
        lr=1e-3,
        loss_name="focal",
        input_mode="onehot",
        pair_mode="pair",
        fusion_mode="late",
        embedding_dim=8,
        bpe_pretrained_model_name=cnn_v2.BPE_DEFAULT_MODEL_NAME,
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
    model_args = argparse.Namespace(
        fusion_mode="late",
        conv_channels=None,
        donor_conv_channels=[8, 16],
        acceptor_conv_channels=[8, 16],
        donor_kernel_sizes=[5, 3],
        acceptor_kernel_sizes=[5, 3],
        max_pool_size=2,
        conv_stride=1,
        head_type="gap",
        fc_hidden=32,
    )

    with pytest.raises(ValueError, match="validation_metric"):
        cnn_v2.train_pair_model(
            pos_path="pos.tsv",
            neg_path="neg.tsv",
            checkpoint_path="pair.pt",
            init_checkpoint_path=None,
            donor_window_len=100,
            acceptor_window_len=100,
            donor_len=100,
            acceptor_len=100,
            model_args=model_args,
            train_params=train_params,
            epochs=1,
            early_stop_patience=0,
            early_stop_min_delta=0.0,
            validation_metric="not_a_metric",
            sequence_transform="none",
            seed=1,
            lightweight=False,
            compile_model=False,
            compile_mode="off",
            device="cpu",
            use_amp=False,
            amp_dtype="auto",
            allow_tf32=False,
            cudnn_benchmark=False,
            deterministic=False,
            num_workers=0,
            prefetch_factor=1,
            persistent_workers=False,
            pin_memory=False,
            min_batch_size=1,
            max_oom_retries=0,
            quick_phase=False,
        )


def test_bpe_encoder_uses_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            del args
            captured_kwargs.update(kwargs)

            class _Tok:
                vocab_size = 321

                def __call__(self, text: str, **kw: object) -> dict[str, list[int]]:
                    del text
                    max_length = int(kw["max_length"])
                    return {"input_ids": [1] * max_length}

            return _Tok()

    monkeypatch.setattr(cnn_v2, "AutoTokenizer", _FakeAutoTokenizer)
    encoder = cnn_v2._build_sequence_encoder(
        mode="bpe",
        window_len=7,
        bpe_pretrained_model_name="fake/model",
        bpe_pretrained_revision=None,
        bpe_trust_remote_code=False,
    )
    encoded = encoder.encode("ACGT")
    assert encoded.shape == (7,)
    assert encoder.vocab_size == 321
    assert captured_kwargs["local_files_only"] is True
    assert captured_kwargs["trust_remote_code"] is False


def test_infer_site_independent_returns_site_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCnnModule:
        @staticmethod
        def infer_site(
            common_args: argparse.Namespace,
            model_args: argparse.Namespace,
        ) -> list[dict[str, object]]:
            _ = common_args
            assert model_args.sequence_transform == "none"
            assert not hasattr(model_args, "mask")
            return [
                {
                    "transcript_id": "tx1",
                    "intron_index": 1,
                    "site_type": "donor",
                    "score": 0.2,
                },
                {
                    "transcript_id": "tx1",
                    "intron_index": 1,
                    "site_type": "acceptor",
                    "score": 0.5,
                },
                {
                    "transcript_id": "tx2",
                    "intron_index": 3,
                    "site_type": "donor",
                    "score": 0.7,
                },
                {
                    "transcript_id": "tx2",
                    "intron_index": 3,
                    "site_type": "acceptor",
                    "score": 0.4,
                },
            ]

    monkeypatch.setattr(models, "cnn", _FakeCnnModule, raising=False)
    common_args = argparse.Namespace(species="Hsap")
    model_args = argparse.Namespace(
        pair_mode="independent",
        train_target="donor",
        sequence_transform="mask_outside_intron_n",
        mask="on",
    )
    out = cnn_v2.infer_site(common_args, model_args)
    assert out == [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "donor",
            "score": 0.2,
        },
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "site_type": "acceptor",
            "score": 0.5,
        },
        {
            "transcript_id": "tx2",
            "intron_index": 3,
            "site_type": "donor",
            "score": 0.7,
        },
        {
            "transcript_id": "tx2",
            "intron_index": 3,
            "site_type": "acceptor",
            "score": 0.4,
        },
    ]


def test_train_independent_fills_cnn_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCnnModule:
        @staticmethod
        def add_train_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--report_train_metrics", type=int, default=1)

        @staticmethod
        def add_infer_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--infer_batch_size", type=int, default=None)

        @staticmethod
        def train(
            common_args: argparse.Namespace,
            model_args: argparse.Namespace,
        ) -> dict[str, object]:
            _ = common_args
            assert hasattr(model_args, "report_train_metrics")
            assert int(model_args.report_train_metrics) == 1
            assert model_args.train_target == "donor"
            assert model_args.sequence_transform == "none"
            assert not hasattr(model_args, "mask")
            return {"model": "cnn", "donor": {"best_pr_auc": 0.5}}

    monkeypatch.setattr(models, "cnn", _FakeCnnModule, raising=False)
    common_args = argparse.Namespace(species="Hsap")
    model_args = argparse.Namespace(
        pair_mode="independent",
        train_target="donor",
        sequence_transform="mask_outside_intron_n",
        mask="on",
    )
    summary = cnn_v2.train(common_args, model_args)
    assert summary["model"] == "cnn_v2"
    assert summary["pair_mode"] == "independent"
    assert summary["train_target"] == "donor"
    assert summary["delegated_backend"] == "cnn"


def test_train_independent_preserves_single_site_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCnnModule:
        @staticmethod
        def add_train_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--report_train_metrics", type=int, default=1)

        @staticmethod
        def add_infer_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--infer_batch_size", type=int, default=None)

        @staticmethod
        def train(
            common_args: argparse.Namespace,
            model_args: argparse.Namespace,
        ) -> dict[str, object]:
            _ = common_args
            assert model_args.train_target == "donor"
            assert model_args.sequence_transform == "none"
            assert not hasattr(model_args, "mask")
            return {"model": "cnn", "donor": {"best_pr_auc": 0.5}}

    monkeypatch.setattr(models, "cnn", _FakeCnnModule, raising=False)
    common_args = argparse.Namespace(species="Hsap")
    model_args = argparse.Namespace(
        pair_mode="independent",
        train_target="donor",
        sequence_transform="mask_outside_intron_n",
        mask="on",
    )

    summary = cnn_v2.train(common_args, model_args)

    assert summary["model"] == "cnn_v2"
    assert summary["pair_mode"] == "independent"
    assert summary["train_target"] == "donor"
    assert summary["delegated_backend"] == "cnn"


def test_train_independent_rejects_pair_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = monkeypatch
    common_args = argparse.Namespace(species="Hsap")
    model_args = argparse.Namespace(
        pair_mode="independent",
        train_target="pair",
        sequence_transform="mask_outside_intron_n",
        mask="on",
    )

    with pytest.raises(ValueError, match="both, donor, or acceptor"):
        cnn_v2.train(common_args, model_args)


def test_train_independent_tolerates_duplicate_parser_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCnnModule:
        @staticmethod
        def add_train_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--batch_size", type=int, default=256)

        @staticmethod
        def add_infer_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--batch_size", type=int, default=512)
            parser.add_argument("--infer_batch_size", type=int, default=None)

        @staticmethod
        def train(
            common_args: argparse.Namespace,
            model_args: argparse.Namespace,
        ) -> dict[str, object]:
            _ = common_args
            assert hasattr(model_args, "batch_size")
            assert hasattr(model_args, "infer_batch_size")
            assert model_args.train_target == "donor"
            return {"model": "cnn", "donor": {"best_pr_auc": 0.5}}

    monkeypatch.setattr(models, "cnn", _FakeCnnModule, raising=False)
    common_args = argparse.Namespace(species="Hsap")
    model_args = argparse.Namespace(pair_mode="independent", train_target="donor")

    summary = cnn_v2.train(common_args, model_args)

    assert summary["model"] == "cnn_v2"
    assert summary["pair_mode"] == "independent"
    assert summary["train_target"] == "donor"


def test_train_independent_disables_sequence_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCnnModule:
        @staticmethod
        def add_train_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument(
                "--sequence_transform",
                choices=list(cnn_v2.SEQUENCE_TRANSFORM_CHOICES),
                default="none",
            )

        @staticmethod
        def add_infer_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument(
                "--sequence_transform",
                choices=list(cnn_v2.SEQUENCE_TRANSFORM_CHOICES),
                default="none",
            )

        @staticmethod
        def train(
            common_args: argparse.Namespace,
            model_args: argparse.Namespace,
        ) -> dict[str, object]:
            _ = common_args
            assert model_args.sequence_transform == "none"
            assert not hasattr(model_args, "mask")
            return {"model": "cnn", "donor": {"best_pr_auc": 0.5}}

    monkeypatch.setattr(models, "cnn", _FakeCnnModule, raising=False)
    common_args = argparse.Namespace(species="Hsap")
    model_args = argparse.Namespace(
        pair_mode="independent",
        train_target="acceptor",
        sequence_transform="mask_outside_intron_n",
        mask="on",
    )

    summary = cnn_v2.train(common_args, model_args)

    assert summary["model"] == "cnn_v2"
    assert summary["pair_mode"] == "independent"
