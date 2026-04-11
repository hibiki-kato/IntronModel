"""Smoke tests for models.spliceformer_sc."""

import argparse
import math

import numpy as np
import pytest
import torch

from models.spliceformer_sc import (
    CandidateSelector,
    DilatedResidualBlock,
    FiLMLayer,
    FiLMLayerTransformer,
    GenomicPositionalEncoding,
    SpliceAIEncoder,
    SpliceTransformerEncoder,
    Spliceformer,
    SpliceformerConfig,
    SpliceformerDataset,
    SpliceformerExample,
    _build_model,
    _config_from_dict,
    _config_to_dict,
    add_train_args,
    one_hot_encode_dna_5ch,
)


# ---------------------------------------------------------------------------
# one_hot_encode_dna_5ch
# ---------------------------------------------------------------------------


def test_one_hot_shape():
    arr = one_hot_encode_dna_5ch("ACGTN", window_len=5)
    assert arr.shape == (5, 5)
    assert arr.dtype == np.float32


def test_one_hot_channels():
    arr = one_hot_encode_dna_5ch("ACGTN", window_len=5)
    assert arr[0, 0] == 1.0  # A
    assert arr[1, 1] == 1.0  # C
    assert arr[2, 2] == 1.0  # G
    assert arr[3, 3] == 1.0  # T
    assert arr[4, 4] == 1.0  # N


def test_one_hot_unknown_base_maps_to_n():
    arr = one_hot_encode_dna_5ch("X", window_len=1)
    assert arr[4, 0] == 1.0


def test_one_hot_truncation():
    arr = one_hot_encode_dna_5ch("AAAAAAAAAA", window_len=3)
    assert arr.shape == (5, 3)


def test_one_hot_zero_padding():
    arr = one_hot_encode_dna_5ch("A", window_len=5)
    assert arr[:, 1:].sum() == 0.0


# ---------------------------------------------------------------------------
# DilatedResidualBlock
# ---------------------------------------------------------------------------


def test_dilated_block_preserves_length():
    block = DilatedResidualBlock(
        in_channels=5, out_channels=32, kernel_size=11,
        dilation=2, dropout=0.0,
    )
    x = torch.randn(2, 5, 100)
    out = block(x)
    assert out.shape == (2, 32, 100)


def test_dilated_block_no_pooling_longer():
    block = DilatedResidualBlock(
        in_channels=32, out_channels=32, kernel_size=7,
        dilation=4, dropout=0.0,
    )
    x = torch.randn(1, 32, 1000)
    assert block(x).shape == (1, 32, 1000)


# ---------------------------------------------------------------------------
# SpliceAIEncoder
# ---------------------------------------------------------------------------


def test_encoder_preserves_length_short():
    enc = SpliceAIEncoder(in_channels=5, d_model=32, dilations=[1, 2, 4, 8], dropout=0.0)
    x = torch.randn(2, 5, 100)
    h = enc(x)
    assert h.shape == (2, 32, 100)


def test_encoder_preserves_length_long():
    enc = SpliceAIEncoder(in_channels=5, d_model=32, dilations=[1, 2, 4, 8], dropout=0.0)
    x = torch.randn(1, 5, 2000)
    assert enc(x).shape == (1, 32, 2000)


def test_encoder_with_film():
    enc = SpliceAIEncoder(
        in_channels=5, d_model=32, dilations=[1, 2], dropout=0.0,
        use_film=True, species_embed_dim=64,
    )
    x = torch.randn(3, 5, 100)
    sp = torch.randn(3, 64)
    assert enc(x, sp).shape == (3, 32, 100)


# ---------------------------------------------------------------------------
# FiLM layers
# ---------------------------------------------------------------------------


def test_film_cnn_shape():
    film = FiLMLayer(species_embed_dim=64, n_channels=32)
    x = torch.randn(2, 32, 100)
    emb = torch.randn(2, 64)
    assert film(x, emb).shape == (2, 32, 100)


def test_film_transformer_shape():
    film = FiLMLayerTransformer(species_embed_dim=64, d_model=32)
    x = torch.randn(2, 50, 32)
    emb = torch.randn(2, 64)
    assert film(x, emb).shape == (2, 50, 32)


def test_film_zero_init_is_identity():
    """FiLM with zero-init weights should leave features unchanged."""
    film = FiLMLayer(species_embed_dim=64, n_channels=32)
    x = torch.randn(2, 32, 100)
    emb = torch.randn(2, 64)
    out = film(x, emb)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# CandidateSelector
# ---------------------------------------------------------------------------


def test_selector_bypass_short():
    """100 positions with K=512 — all positions returned unchanged."""
    sel = CandidateSelector(d_model=32, k_donor=256, k_acceptor=256)
    h = torch.randn(2, 32, 100)
    h_sel, pos = sel(h)
    assert h_sel.shape == (2, 32, 100)
    assert pos.shape == (2, 100)


def test_selector_selects_long():
    """2000 positions with K=512 — exactly 512 selected."""
    sel = CandidateSelector(d_model=32, k_donor=256, k_acceptor=256)
    h = torch.randn(2, 32, 2000)
    h_sel, pos = sel(h)
    assert h_sel.shape[2] == 512
    assert pos.shape == (2, 512)


def test_selector_positions_sorted():
    sel = CandidateSelector(d_model=32, k_donor=256, k_acceptor=256)
    h = torch.randn(1, 32, 2000)
    _, pos = sel(h)
    diffs = pos[0, 1:] - pos[0, :-1]
    assert (diffs >= 0).all(), "Selected positions are not sorted."


# ---------------------------------------------------------------------------
# GenomicPositionalEncoding
# ---------------------------------------------------------------------------


def test_pos_enc_shape():
    pe = GenomicPositionalEncoding(d_model=32)
    x = torch.zeros(2, 100, 32)
    pos = torch.arange(100).unsqueeze(0).expand(2, -1)
    out = pe(x, pos)
    assert out.shape == (2, 100, 32)


# ---------------------------------------------------------------------------
# SpliceTransformerEncoder
# ---------------------------------------------------------------------------


def test_transformer_shape():
    tf = SpliceTransformerEncoder(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.0,
    )
    x = torch.randn(2, 50, 32)
    assert tf(x).shape == (2, 50, 32)


def test_transformer_with_film():
    tf = SpliceTransformerEncoder(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.0,
        use_film=True, species_embed_dim=64,
    )
    x = torch.randn(2, 50, 32)
    sp = torch.randn(2, 64)
    assert tf(x, sp).shape == (2, 50, 32)


# ---------------------------------------------------------------------------
# Full Spliceformer model
# ---------------------------------------------------------------------------


def _small_model(**kwargs) -> Spliceformer:
    """Lightweight model for fast tests."""
    defaults = dict(
        d_model=16,
        cnn_dilations=(1, 2),
        cnn_kernel_size=5,
        nhead=2,
        num_transformer_layers=2,
        dim_feedforward=32,
        num_species=4,
        species_embed_dim=32,
        use_film=False,
        k_donor=64,
        k_acceptor=64,
        dropout=0.0,
        mode="binary",
    )
    defaults.update(kwargs)
    return Spliceformer(**defaults)


def test_forward_short_sequence():
    """Short input (100 bp) — selector bypassed, returns all positions."""
    model = _small_model()
    x = torch.randn(2, 5, 100)
    sp = torch.tensor([0, 1])
    logits, pos = model(x, sp)
    assert logits.shape == (2, 100, 2)  # K_out == L (bypass)
    assert pos.shape == (2, 100)


def test_forward_long_sequence():
    """Long input (2000 bp) — selector active, K positions selected."""
    model = _small_model()
    x = torch.randn(1, 5, 2000)
    sp = torch.tensor([0])
    logits, pos = model(x, sp)
    assert logits.shape == (1, 128, 2)  # k_donor + k_acceptor = 128
    assert pos.shape == (1, 128)


def test_forward_binary_returns_scalar_per_sample():
    model = _small_model()
    x = torch.randn(3, 5, 100)
    sp = torch.tensor([0, 1, 2])
    donor_logits = model.forward_binary(x, sp, task="donor")
    accept_logits = model.forward_binary(x, sp, task="acceptor")
    assert donor_logits.shape == (3,)
    assert accept_logits.shape == (3,)


def test_forward_multiclass():
    model = _small_model(mode="multiclass")
    x = torch.randn(2, 5, 100)
    sp = torch.tensor([0, 1])
    logits, pos = model(x, sp)
    assert logits.shape[2] == 3


def test_forward_with_film():
    model = _small_model(use_film=True)
    x = torch.randn(2, 5, 100)
    sp = torch.tensor([0, 1])
    logits, pos = model(x, sp)
    assert logits.shape == (2, 100, 2)


# ---------------------------------------------------------------------------
# SpliceformerDataset
# ---------------------------------------------------------------------------


def test_dataset_getitem():
    sp_map = {"Hsap": 0, "Mmus": 1}
    examples = [
        SpliceformerExample("ACGTACGT", 1, "donor", "Hsap"),
        SpliceformerExample("TGCATGCA", 0, "acceptor", "Mmus"),
    ]
    ds = SpliceformerDataset(examples, window_len=8, species_to_idx=sp_map)
    x, label, sp, task = ds[0]
    assert x.shape == (5, 8)
    assert float(label) == 1.0
    assert int(sp) == 0   # Hsap
    assert int(task) == 0  # donor


def test_dataset_preencode():
    sp_map = {"Hsap": 0}
    examples = [SpliceformerExample("ACGT", 1, "donor", "Hsap")] * 4
    ds = SpliceformerDataset(examples, window_len=4, species_to_idx=sp_map, preencode=True)
    assert len(ds) == 4
    x, _, _, _ = ds[0]
    assert x.shape == (5, 4)


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_config_round_trip():
    cfg = SpliceformerConfig(d_model=48, cnn_dilations=(1, 4, 16), num_species=6)
    d = _config_to_dict(cfg)
    cfg2 = _config_from_dict(d)
    assert cfg2.d_model == 48
    assert cfg2.cnn_dilations == (1, 4, 16)
    assert cfg2.num_species == 6
    assert d["site_arch"] == "spliceformer_sc"


# ---------------------------------------------------------------------------
# add_train_args
# ---------------------------------------------------------------------------


def test_add_train_args_registers_params():
    """Parser must accept all spliceformer-specific arguments."""
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_train_args(parser)
    args = parser.parse_args([
        "--d_model", "64",
        "--cnn_dilations", "1,2,4,8,16",
        "--nhead", "8",
        "--num_transformer_layers", "4",
        "--dim_feedforward", "256",
        "--species_embed_dim", "128",
        "--use_film", "1",
        "--k_donor", "128",
        "--k_acceptor", "128",
        "--spliceformer_mode", "binary",
        "--species_list", "Athal,Hsap",
    ])
    assert args.d_model == 64
    assert args.use_film == 1
    assert args.species_list == "Athal,Hsap"
    assert args.spliceformer_mode == "binary"
