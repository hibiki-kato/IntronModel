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


def test_pair_mode_off_alias_maps_to_independent() -> None:
    assert cnn_v2._normalize_pair_mode("off", arg_name="--pair_mode") == "independent"


def test_resolve_pair_train_params_v2_flags() -> None:
    args = argparse.Namespace(
        batch_size=64,
        lr=1e-3,
        loss="weighted_bce",
        input_mode="kmer3",
        pair_mode="pair",
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
    assert params.embedding_dim == 32


def test_bpe_encoder_uses_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            class _Tok:
                vocab_size = 321

                def __call__(self, text: str, **kw: object) -> dict[str, list[int]]:
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


def test_infer_site_independent_multiplies_donor_acceptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCnnModule:
        @staticmethod
        def infer_site(
            common_args: argparse.Namespace,
            model_args: argparse.Namespace,
        ) -> list[dict[str, object]]:
            _ = common_args
            _ = model_args
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
    model_args = argparse.Namespace(pair_mode="independent")
    out = cnn_v2.infer_site(common_args, model_args)
    assert len(out) == 2
    assert out[0]["transcript_id"] == "tx1"
    assert out[0]["intron_index"] == 1
    assert out[0]["site_type"] == "pair"
    assert out[0]["score"] == pytest.approx(0.1)
    assert out[1]["transcript_id"] == "tx2"
    assert out[1]["intron_index"] == 3
    assert out[1]["site_type"] == "pair"
    assert out[1]["score"] == pytest.approx(0.28)


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
            assert model_args.train_target == "both"
            return {"model": "cnn", "donor": {"best_pr_auc": 0.5}}

    monkeypatch.setattr(models, "cnn", _FakeCnnModule, raising=False)
    common_args = argparse.Namespace(species="Hsap")
    model_args = argparse.Namespace(pair_mode="independent", train_target="both")
    summary = cnn_v2.train(common_args, model_args)
    assert summary["model"] == "cnn_v2"
    assert summary["pair_mode"] == "independent"
    assert summary["train_target"] == "both"
    assert summary["delegated_backend"] == "cnn"
