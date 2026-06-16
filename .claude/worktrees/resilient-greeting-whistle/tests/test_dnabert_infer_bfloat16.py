from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from models import dnabert


class _TokenizerEncodeStub:
    """Tokenizer stub that returns deterministic fixed-size token tensors."""

    def __call__(
        self,
        first: object,
        second: Optional[object] = None,
        **_: object,
    ) -> dict[str, object]:
        if not isinstance(first, list):
            raise TypeError("first must be a list")
        batch_size = len(first)
        if second is not None and (
            not isinstance(second, list) or len(second) != batch_size
        ):
            raise TypeError("second must be a list matching first length")
        return {
            "input_ids": [[1, 2, 3] for _ in range(batch_size)],
            "attention_mask": [[1, 1, 1] for _ in range(batch_size)],
        }


class _Bfloat16LogitModel(torch.nn.Module):
    """Model stub that returns bfloat16 logits for inference testing."""

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        batch_size = int(input_ids.shape[0])
        return torch.full(
            (batch_size,),
            0.5,
            dtype=torch.bfloat16,
            device=input_ids.device,
        )


def test_score_sequences_handles_bfloat16_logits() -> None:
    model = _Bfloat16LogitModel()
    tokenizer = _TokenizerEncodeStub()
    probs = dnabert.score_sequences(
        model=model,
        sequences=["ACGT", "TGCA", "GGGG"],
        tokenizer=tokenizer,
        max_tokens=16,
        device="cpu",
        batch_size=2,
        task_name="task",
        input_kmer=None,
        use_amp=False,
        amp_dtype=None,
    )
    assert probs.dtype == np.float32
    assert probs.shape == (3,)


def test_score_sequence_pairs_handles_bfloat16_logits() -> None:
    model = _Bfloat16LogitModel()
    tokenizer = _TokenizerEncodeStub()
    probs = dnabert.score_sequence_pairs(
        model=model,
        donor_sequences=["ACGT", "TGCA", "GGGG"],
        acceptor_sequences=["TTTT", "CCCC", "AAAA"],
        tokenizer=tokenizer,
        max_tokens=16,
        device="cpu",
        batch_size=2,
        task_name="pair",
        input_kmer=None,
        use_amp=False,
        amp_dtype=None,
    )
    assert probs.dtype == np.float32
    assert probs.shape == (3,)
