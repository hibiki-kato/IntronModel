from __future__ import annotations

import argparse
import importlib

import pytest

torch = pytest.importorskip("torch")
bilstm_pair = importlib.import_module("models.bilstm_pair")


def test_add_train_args_accepts_pair_arch_and_hidden_size() -> None:
    parser = argparse.ArgumentParser()
    bilstm_pair.add_train_args(parser)
    args = parser.parse_args(
        [
            "--pair_arch",
            "concat",
            "--hidden_size",
            "128",
            "--embedding_dim",
            "16",
            "--use_sep_token",
            "1",
        ]
    )

    assert args.pair_arch == "concat"
    assert args.hidden_size == 128
    assert args.embedding_dim == 16
    assert args.use_sep_token == 1


def test_add_train_args_accepts_input_mode_alias() -> None:
    parser = argparse.ArgumentParser()
    bilstm_pair.add_train_args(parser)
    args = parser.parse_args(
        [
            "--input_mode",
            "onehot",
            "--bpe_pretrained_model_name",
            "fake/model",
            "--bpe_trust_remote_code",
            "1",
        ]
    )

    assert args.input_mode == "onehot"
    assert args.bpe_pretrained_model_name == "fake/model"
    assert args.bpe_trust_remote_code == 1


def test_add_train_args_accepts_compile_mode_flags() -> None:
    parser = argparse.ArgumentParser()
    bilstm_pair.add_train_args(parser)
    args = parser.parse_args(["--compile", "--compile_mode", "auto"])

    assert args.compile is True
    assert args.compile_mode == "auto"


def test_resolve_pair_train_params_validates_hidden_size() -> None:
    parser = argparse.ArgumentParser()
    bilstm_pair.add_train_args(parser)
    args = parser.parse_args(["--hidden_size", "0"])

    with pytest.raises(ValueError, match="--hidden_size must be positive"):
        _ = bilstm_pair._resolve_pair_train_params(args)


def test_pair_bilstm_classifier_separate_forward_shape() -> None:
    model = bilstm_pair.PairBiLSTMClassifier(
        pair_arch="separate",
        use_sep_token=True,
        embedding_dim=16,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        fc_hidden=64,
    )
    donor_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 0, 0]], dtype=torch.long)
    donor_lengths = torch.tensor([4, 2], dtype=torch.long)
    acceptor_ids = torch.tensor([[4, 3, 2], [4, 0, 0]], dtype=torch.long)
    acceptor_lengths = torch.tensor([3, 1], dtype=torch.long)
    concat_ids = torch.tensor([[1, 2, 3, 6, 4, 3, 2], [1, 2, 6, 4, 0, 0, 0]])
    concat_lengths = torch.tensor([7, 4], dtype=torch.long)

    logits = model(
        donor_ids,
        donor_lengths,
        acceptor_ids,
        acceptor_lengths,
        concat_ids,
        concat_lengths,
    )

    assert logits.shape == (2,)


def test_pair_bilstm_classifier_concat_forward_shape() -> None:
    model = bilstm_pair.PairBiLSTMClassifier(
        pair_arch="concat",
        use_sep_token=True,
        embedding_dim=16,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        fc_hidden=64,
    )
    donor_ids = torch.tensor([[1, 2], [1, 0]], dtype=torch.long)
    donor_lengths = torch.tensor([2, 1], dtype=torch.long)
    acceptor_ids = torch.tensor([[4, 3], [4, 0]], dtype=torch.long)
    acceptor_lengths = torch.tensor([2, 1], dtype=torch.long)
    concat_ids = torch.tensor([[1, 2, 6, 4, 3], [1, 6, 4, 0, 0]], dtype=torch.long)
    concat_lengths = torch.tensor([5, 3], dtype=torch.long)

    logits = model(
        donor_ids,
        donor_lengths,
        acceptor_ids,
        acceptor_lengths,
        concat_ids,
        concat_lengths,
    )

    assert logits.shape == (2,)


def test_encode_dna_sequence_maps_unknown_to_n() -> None:
    tokens = bilstm_pair._encode_dna_sequence("ACGTXn")
    assert tokens == [1, 2, 3, 4, 5, 5]


def test_build_sequence_encoder_kmer3_returns_shifted_ids() -> None:
    encoder = bilstm_pair._build_sequence_encoder(
        mode="kmer3",
        bpe_pretrained_model_name="ignored/model",
        bpe_pretrained_revision=None,
        bpe_trust_remote_code=False,
    )

    tokens = encoder.encode("ACGTN", window_len=5)

    assert len(tokens) == 3
    assert min(tokens) >= 1
    assert encoder.vocab_size == 67
    assert encoder.sep_token_id == 66


def test_build_sequence_encoder_bpe_shifts_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTokenizer:
        vocab_size = 10

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[int]]:
            return {"input_ids": [2, 5, 9]}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeTokenizer:
            return _FakeTokenizer()

    monkeypatch.setattr(bilstm_pair, "AutoTokenizer", _FakeAutoTokenizer)
    bilstm_pair._TOKENIZER_CACHE.clear()

    encoder = bilstm_pair._build_sequence_encoder(
        mode="bpe",
        bpe_pretrained_model_name="fake/model",
        bpe_pretrained_revision=None,
        bpe_trust_remote_code=False,
    )
    tokens = encoder.encode("ACGT", window_len=4)

    assert tokens == [3, 6, 10]
    assert encoder.vocab_size == 12
    assert encoder.sep_token_id == 11
