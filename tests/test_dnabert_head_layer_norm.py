from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from models import dnabert


class _BackboneOutput:
    """Minimal output object with ``last_hidden_state`` for classifier tests."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _DummyBackbone(nn.Module):
    """Backbone stub that returns deterministic hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> _BackboneOutput:
        del attention_mask
        batch_size, token_len = input_ids.shape
        base = torch.arange(
            self.hidden_size,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(1, 1, self.hidden_size)
        hidden = base.expand(batch_size, token_len, self.hidden_size).clone()
        return _BackboneOutput(last_hidden_state=self.proj(hidden))


def test_classifier_head_uses_layer_norm_when_enabled() -> None:
    model = dnabert.DnaBertBinaryClassifier(
        backbone=_DummyBackbone(hidden_size=8),
        hidden_size=8,
        dropout=0.1,
        head_layer_norm=True,
    )
    assert isinstance(model.head_norm, nn.LayerNorm)
    logits = model(
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
    )
    assert logits.shape == (2,)
    assert "head_norm.weight" in model.state_dict()
    assert "head_norm.bias" in model.state_dict()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert "classifier.weight" in trainable_names
    assert "classifier.bias" in trainable_names


def test_classifier_head_uses_identity_when_disabled() -> None:
    model = dnabert.DnaBertBinaryClassifier(
        backbone=_DummyBackbone(hidden_size=8),
        hidden_size=8,
        dropout=0.1,
        head_layer_norm=False,
    )
    assert isinstance(model.head_norm, nn.Identity)
    logits = model(
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
    )
    assert logits.shape == (2,)
    state = model.state_dict()
    assert "head_norm.weight" not in state
    assert "head_norm.bias" not in state
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert "classifier.weight" in trainable_names
    assert "classifier.bias" in trainable_names


def test_freeze_backbone_disables_backbone_gradients() -> None:
    model = dnabert.DnaBertBinaryClassifier(
        backbone=_DummyBackbone(hidden_size=8),
        hidden_size=8,
        dropout=0.1,
        head_layer_norm=True,
    )
    model.freeze_backbone()
    model.assert_backbone_frozen()

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert "classifier.weight" in trainable_names
    assert "classifier.bias" in trainable_names
    assert all(not name.startswith("backbone.") for name in trainable_names)


def test_add_train_args_includes_head_layer_norm_options() -> None:
    parser = argparse.ArgumentParser()
    dnabert.add_train_args(parser)
    args = parser.parse_args(
        [
            "--head_layer_norm",
            "0",
            "--donor_head_layer_norm",
            "1",
            "--acceptor_head_layer_norm",
            "0",
        ]
    )
    assert args.head_layer_norm == 0
    assert args.donor_head_layer_norm == 1
    assert args.acceptor_head_layer_norm == 0
