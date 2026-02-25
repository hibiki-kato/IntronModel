from __future__ import annotations

from itertools import product
from typing import Mapping

from models import dnabert


class _TokenizerStub:
    """Tokenizer stub that exposes ``get_vocab``."""

    def __init__(self, vocab: Mapping[str, int]) -> None:
        self._vocab: dict[str, int] = dict(vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def _build_fixed_kmer_vocab(kmer_k: int) -> dict[str, int]:
    """Build a complete fixed k-mer vocabulary with special tokens."""
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab_tokens.extend("".join(chars) for chars in product("ACGT", repeat=kmer_k))
    return {token: idx for idx, token in enumerate(vocab_tokens)}


def test_resolve_tokenizer_input_kmer_detects_fixed_kmer_vocab() -> None:
    tokenizer = _TokenizerStub(_build_fixed_kmer_vocab(3))
    resolved = dnabert._resolve_tokenizer_input_kmer(tokenizer)
    assert resolved == 3


def test_resolve_tokenizer_input_kmer_returns_none_for_variable_vocab() -> None:
    tokenizer = _TokenizerStub(
        {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
            "A": 5,
            "AC": 6,
            "ACG": 7,
            "ACGT": 8,
        }
    )
    resolved = dnabert._resolve_tokenizer_input_kmer(tokenizer)
    assert resolved is None


def test_prepare_sequences_for_tokenizer_applies_kmer_text() -> None:
    prepared = dnabert._prepare_sequences_for_tokenizer(
        sequences=["acgtac"],
        input_kmer=3,
    )
    assert prepared == ["ACG CGT GTA TAC"]


def test_resolve_max_tokens_auto_matches_kmer_length() -> None:
    resolved = dnabert._resolve_max_tokens(
        raw="auto",
        window_len=50,
        tokenizer_limit=None,
        input_kmer=6,
    )
    assert resolved == 47


def test_resolve_max_tokens_auto_for_raw_sequence() -> None:
    resolved = dnabert._resolve_max_tokens(
        raw="auto",
        window_len=50,
        tokenizer_limit=None,
        input_kmer=None,
    )
    assert resolved == 52
