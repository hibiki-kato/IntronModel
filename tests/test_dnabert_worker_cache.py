from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models import dnabert


class _DummyBackbone(nn.Module):
    """Minimal backbone used to verify DNABERT template caching."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.proj = nn.Linear(4, 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> object:
        del attention_mask
        batch_size = int(input_ids.shape[0])
        hidden = torch.zeros((batch_size, 1, 4), dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=self.proj(hidden))


@pytest.fixture(autouse=True)
def _clear_dnabert_resource_cache() -> None:
    dnabert._clear_pretrained_resource_caches()
    yield
    dnabert._clear_pretrained_resource_caches()


def test_load_tokenizer_uses_process_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse one tokenizer object for repeated pretrained loads."""
    load_calls: list[str] = []
    sentinel = object()

    def _fake_load_tokenizer_uncached(
        pretrained_model_name: str,
        pretrained_revision: str | None,
        trust_remote_code: bool,
    ) -> object:
        del pretrained_revision, trust_remote_code
        load_calls.append(pretrained_model_name)
        return sentinel

    monkeypatch.setattr(
        dnabert,
        "_load_tokenizer_uncached",
        _fake_load_tokenizer_uncached,
    )

    first = dnabert._load_tokenizer(
        pretrained_model_name="dummy-model",
        pretrained_revision=None,
        trust_remote_code=True,
    )
    second = dnabert._load_tokenizer(
        pretrained_model_name="dummy-model",
        pretrained_revision=None,
        trust_remote_code=True,
    )

    assert first is sentinel
    assert second is sentinel
    assert load_calls == ["dummy-model"]


def test_build_dnabert_model_reuses_cached_backbone_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clone fresh classifiers from one cached backbone template."""
    load_count = 0

    def _fake_load_backbone_template_uncached(
        pretrained_model_name: str,
        pretrained_revision: str | None,
        trust_remote_code: bool,
    ) -> dnabert._CachedBackboneTemplate:
        del pretrained_model_name, pretrained_revision, trust_remote_code
        nonlocal load_count
        load_count += 1
        return dnabert._CachedBackboneTemplate(
            backbone=_DummyBackbone(),
            hidden_size=4,
        )

    monkeypatch.setattr(
        dnabert,
        "_load_backbone_template_uncached",
        _fake_load_backbone_template_uncached,
    )

    first = dnabert._build_dnabert_model(
        pretrained_model_name="dummy-model",
        pretrained_revision=None,
        trust_remote_code=True,
        dropout=0.1,
        head_layer_norm=False,
    )
    second = dnabert._build_dnabert_model(
        pretrained_model_name="dummy-model",
        pretrained_revision=None,
        trust_remote_code=True,
        dropout=0.1,
        head_layer_norm=False,
    )

    assert load_count == 1
    assert first is not second
    assert first.backbone is not second.backbone
    first_weight = first.backbone.proj.weight
    second_weight = second.backbone.proj.weight
    assert first_weight.data_ptr() != second_weight.data_ptr()
