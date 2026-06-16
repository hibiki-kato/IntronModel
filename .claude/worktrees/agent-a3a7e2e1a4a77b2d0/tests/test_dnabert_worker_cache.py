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


def test_load_tokenizer_uncached_uses_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load tokenizer from local cache without Hub network access."""
    captured: dict[str, object] = {}
    sentinel = object()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(
            pretrained_model_name: str,
            **kwargs: object,
        ) -> object:
            captured["pretrained_model_name"] = pretrained_model_name
            captured.update(kwargs)
            return sentinel

    monkeypatch.setattr(dnabert, "AutoTokenizer", _FakeAutoTokenizer)

    tokenizer = dnabert._load_tokenizer_uncached(
        pretrained_model_name="dummy-model",
        pretrained_revision="main",
        trust_remote_code=True,
    )

    assert tokenizer is sentinel
    assert captured["pretrained_model_name"] == "dummy-model"
    assert captured["local_files_only"] is True
    assert captured["revision"] == "main"
    assert captured["trust_remote_code"] is True


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


def test_load_backbone_template_uncached_uses_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load config and backbone from local cache without Hub network access."""
    config_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []

    class _FakeConfig:
        def __init__(self) -> None:
            self.pad_token_id = None

    fake_config = _FakeConfig()
    fake_backbone = _DummyBackbone()

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(
            pretrained_model_name: str,
            **kwargs: object,
        ) -> _FakeConfig:
            config_calls.append(
                {
                    "pretrained_model_name": pretrained_model_name,
                    **kwargs,
                }
            )
            return fake_config

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(
            pretrained_model_name: str,
            **kwargs: object,
        ) -> _DummyBackbone:
            model_calls.append(
                {
                    "pretrained_model_name": pretrained_model_name,
                    **kwargs,
                }
            )
            return fake_backbone

    monkeypatch.setattr(dnabert, "AutoConfig", _FakeAutoConfig)
    monkeypatch.setattr(dnabert, "AutoModel", _FakeAutoModel)
    monkeypatch.setattr(
        dnabert,
        "_patch_dnabert_alibi_meta_compat",
        lambda **_: None,
    )
    monkeypatch.setattr(
        dnabert,
        "_disable_dnabert_triton_flash_attention",
        lambda _backbone: None,
    )
    monkeypatch.setattr(
        dnabert,
        "_materialize_dnabert_meta_buffers",
        lambda _backbone: None,
    )
    monkeypatch.setattr(dnabert, "_resolve_hidden_size", lambda _backbone: 4)

    template = dnabert._load_backbone_template_uncached(
        pretrained_model_name="dummy-model",
        pretrained_revision="main",
        trust_remote_code=True,
    )

    assert template.hidden_size == 4
    assert template.backbone is fake_backbone
    assert config_calls == [
        {
            "pretrained_model_name": "dummy-model",
            "local_files_only": True,
            "revision": "main",
            "trust_remote_code": True,
        }
    ]
    assert len(model_calls) == 1
    assert model_calls[0]["pretrained_model_name"] == "dummy-model"
    assert model_calls[0]["local_files_only"] is True
    assert model_calls[0]["revision"] == "main"
    assert model_calls[0]["trust_remote_code"] is True
    assert model_calls[0]["config"] is fake_config
