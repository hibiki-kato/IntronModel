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


def test_classifier_supports_linear_readout() -> None:
    model = dnabert.DnaBertBinaryClassifier(
        backbone=_DummyBackbone(hidden_size=8),
        hidden_size=8,
        dropout=0.1,
        head_layer_norm=True,
    )
    logits = model(
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
    )
    assert logits.shape == (2,)
    assert model.readout_type == "linear"

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


def test_add_train_args_omits_removed_readout_options() -> None:
    parser = argparse.ArgumentParser()
    dnabert.add_train_args(parser)
    args = parser.parse_args([])
    assert not hasattr(args, "readout_type")
    assert not hasattr(args, "readout_mlp_hidden_dim")
    assert not hasattr(args, "readout_mlp_layers")
    assert not hasattr(args, "donor_readout_type")
    assert not hasattr(args, "acceptor_readout_type")


def test_add_train_args_accepts_pair_train_target() -> None:
    parser = argparse.ArgumentParser()
    dnabert.add_train_args(parser)
    args = parser.parse_args(["--train_target", "pair"])
    assert args.train_target == "pair"


def test_add_train_args_includes_optimizer_schedule_options() -> None:
    parser = argparse.ArgumentParser()
    dnabert.add_train_args(parser)
    args = parser.parse_args(
        [
            "--lr_schedule",
            "linear",
            "--warmup_ratio",
            "0.02",
            "--adam_beta1",
            "0.9",
            "--adam_beta2",
            "0.97",
            "--adam_eps",
            "1e-8",
            "--donor_lr_schedule",
            "cosine",
            "--acceptor_warmup_ratio",
            "0.01",
        ]
    )
    assert args.lr_schedule == "linear"
    assert args.warmup_ratio == pytest.approx(0.02)
    assert args.adam_beta1 == pytest.approx(0.9)
    assert args.adam_beta2 == pytest.approx(0.97)
    assert args.adam_eps == pytest.approx(1e-8)
    assert args.donor_lr_schedule == "cosine"
    assert args.acceptor_warmup_ratio == pytest.approx(0.01)


def test_lr_schedule_multiplier_with_warmup_and_linear_decay() -> None:
    values = [
        dnabert._lr_schedule_multiplier(
            step_index=step_index,
            total_steps=10,
            warmup_steps=2,
            eta_min_ratio=0.1,
            lr_schedule="linear",
        )
        for step_index in range(10)
    ]
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)
    assert all(values[idx] >= values[idx + 1] for idx in range(1, len(values) - 1))


def test_lr_schedule_multiplier_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="eta_min_ratio"):
        dnabert._lr_schedule_multiplier(
            step_index=0,
            total_steps=10,
            warmup_steps=1,
            eta_min_ratio=1.5,
            lr_schedule="cosine",
        )
    with pytest.raises(ValueError, match="warmup_steps"):
        dnabert._lr_schedule_multiplier(
            step_index=0,
            total_steps=10,
            warmup_steps=11,
            eta_min_ratio=0.1,
            lr_schedule="cosine",
        )
