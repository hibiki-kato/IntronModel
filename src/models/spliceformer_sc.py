"""SpliceFormer-SC: Species-conditioned splice-site classifier.

Architecture overview
---------------------
Input (B, 5, L)
  └─ SpliceAIEncoder (dilated CNN, no pooling)  → (B, d_model, L)
      └─ FiLM conditioning per block             [optional, --use_film 1]
  └─ CandidateSelector (top-K by learned score)  → (B, d_model, K)
      • bypassed when L ≤ k_donor + k_acceptor
        (e.g. 100 bp short-context input)
  └─ GenomicPositionalEncoding (sinusoidal, position-indexed)
  └─ SpliceTransformerEncoder (pre-norm, GELU)   → (B, K, d_model)
      └─ FiLM conditioning per layer             [optional, --use_film 1]
  └─ Binary head: donor_head / acceptor_head     → (B, K, 2)
     forward_binary() adapter selects center-position logit → (B,)

Species conditioning (FiLM)
----------------------------
SpeciesEmbedding maps species_idx → (B, species_embed_dim).
When use_film=True each DilatedResidualBlock and SpliceTransformerLayer
applies:  x = x * (1 + gamma) + beta
where gamma, beta are linear projections of the species embedding.
Both projections are zero-initialised so FiLM starts as identity.

Multi-species training
-----------------------
Set --species_list "Athal,Dmel,Hsap,Mmus" to pool training data across
species.  Falls back to common_args.species for single-species runs.
A single checkpoint is saved for the shared model.

Binary / 3-class output
-----------------------
Default --spliceformer_mode binary: two BCE heads share one encoder.
forward_binary() extracts the center-position logit for one task,
keeping the model fully compatible with score_sequences / infer_site_scores.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
import random
import time
import warnings
from typing import ContextManager, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import cnn
from util.data_proc import (
    infer_default_train_paths,
    read_test_site_rows,
    resolve_effective_window_lengths,
    resolve_test_tsv,
    resolve_train_paths,
    species_data_dirs,
    validate_window_args,
)
from util.model_runtime import (
    bool_from_flag as _bool_from_flag,
    compile_model_with_fallback as _compile_model_with_fallback,
    empty_device_cache as _empty_device_cache,
    export_model_state_dict as _export_model_state_dict,
    is_cuda_oom_error as _is_cuda_oom_error,
    is_mps_oom_error as _is_mps_oom_error,
    normalize_checkpoint_state_dict as _normalize_checkpoint_state_dict,
    pick_device,
    resolve_amp_dtype as _resolve_amp_dtype,
    resolve_compile_enabled as _resolve_compile_enabled,
    resolve_num_workers as _resolve_num_workers,
    seed_worker as _seed_worker,
    set_seed,
    warm_start_model as _warm_start_model,
)
from util.model_task_paths import resolve_required_checkpoint_paths
from util.training_control import resolve_training_schedule
from util.transcript_eval import SCORE_SPACE_FIELD, SCORE_SPACE_LOG10


# ---------------------------------------------------------------------------
# One-hot encoding — 5 channels (A/C/G/T/N)
# ---------------------------------------------------------------------------

def one_hot_encode_dna_5ch(seq: str, window_len: int) -> np.ndarray:
    """One-hot encode a DNA sequence into 5 channels.

    Channels: A=0, C=1, G=2, T=3, N=4.  Unknown bases map to N.

    Parameters
    ----------
    seq : str
        DNA sequence (case-insensitive).
    window_len : int
        Output length; truncated or zero-padded as needed.

    Returns
    -------
    np.ndarray
        Shape ``(5, window_len)``, dtype float32.
    """
    _MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    enc = np.zeros((5, window_len), dtype=np.float32)
    for i, base in enumerate(seq[:window_len].upper()):
        enc[_MAP.get(base, 4), i] = 1.0
    return enc


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpliceformerExample:
    sequence: str
    label: int    # 0 = negative, 1 = positive
    task: str     # "donor" or "acceptor"
    species: str  # e.g. "Athal"


_TASK_TO_IDX: dict[str, int] = {"donor": 0, "acceptor": 1}


class SpliceformerDataset(Dataset):
    """Dataset returning (x_5ch, label, species_idx, task_idx) tuples.

    Shapes
    ------
    x           float32  (5, window_len)
    label       float32  scalar
    species_idx int64    scalar
    task_idx    int64    scalar   0=donor, 1=acceptor
    """

    def __init__(
        self,
        examples: List[SpliceformerExample],
        window_len: int,
        species_to_idx: Dict[str, int],
        preencode: bool = False,
    ) -> None:
        self.examples = examples
        self.window_len = window_len
        self.species_to_idx = species_to_idx
        self._cx: Optional[torch.Tensor] = None
        self._cy: Optional[torch.Tensor] = None
        self._csp: Optional[torch.Tensor] = None
        self._ct: Optional[torch.Tensor] = None
        if preencode:
            xs = np.stack(
                [one_hot_encode_dna_5ch(ex.sequence, window_len) for ex in examples]
            ).astype(np.float32, copy=False)
            self._cx = torch.from_numpy(xs)
            self._cy = torch.tensor([ex.label for ex in examples], dtype=torch.float32)
            self._csp = torch.tensor(
                [species_to_idx[ex.species] for ex in examples], dtype=torch.long
            )
            self._ct = torch.tensor(
                [_TASK_TO_IDX[ex.task] for ex in examples], dtype=torch.long
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cx is not None:
            return self._cx[idx], self._cy[idx], self._csp[idx], self._ct[idx]  # type: ignore[index]
        ex = self.examples[idx]
        return (
            torch.from_numpy(one_hot_encode_dna_5ch(ex.sequence, self.window_len)),
            torch.tensor(ex.label, dtype=torch.float32),
            torch.tensor(self.species_to_idx[ex.species], dtype=torch.long),
            torch.tensor(_TASK_TO_IDX[ex.task], dtype=torch.long),
        )


def _stratified_split_examples(
    examples: List[SpliceformerExample],
    val_frac: float,
    seed: int = 1337,
) -> Tuple[List[SpliceformerExample], List[SpliceformerExample]]:
    """Stratified train/val split that preserves positive-negative ratio."""
    rng = random.Random(seed)
    pos = [ex for ex in examples if ex.label == 1]
    neg = [ex for ex in examples if ex.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val_pos = max(1, int(len(pos) * val_frac))
    n_val_neg = max(1, int(len(neg) * val_frac))
    train = pos[n_val_pos:] + neg[n_val_neg:]
    val = pos[:n_val_pos] + neg[:n_val_neg]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# FiLM conditioning
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for CNN features (B, C, L).

    x = x * (1 + gamma) + beta,  gamma/beta from species embedding.
    Zero-initialised so FiLM starts as an identity transform.
    """

    def __init__(self, species_embed_dim: int, n_channels: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(species_embed_dim, n_channels)
        self.beta_proj = nn.Linear(species_embed_dim, n_channels)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, species_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L), species_emb: (B, embed_dim)
        gamma = self.gamma_proj(species_emb).unsqueeze(-1)  # (B, C, 1)
        beta = self.beta_proj(species_emb).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class FiLMLayerTransformer(nn.Module):
    """FiLM for Transformer features (B, L, C). Zero-initialised."""

    def __init__(self, species_embed_dim: int, d_model: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(species_embed_dim, d_model)
        self.beta_proj = nn.Linear(species_embed_dim, d_model)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, species_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C), species_emb: (B, embed_dim)
        gamma = self.gamma_proj(species_emb).unsqueeze(1)  # (B, 1, C)
        beta = self.beta_proj(species_emb).unsqueeze(1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# SpliceAI-style dilated CNN encoder (no pooling)
# ---------------------------------------------------------------------------

class DilatedResidualBlock(nn.Module):
    """Residual dilated 1D CNN block without pooling.

    Input/output: ``(B, C, L)`` → ``(B, out_channels, L)``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_film: bool = False,
        species_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        pad = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=pad, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, padding=pad, dilation=dilation
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj: nn.Module = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.film: Optional[FiLMLayer] = (
            FiLMLayer(species_embed_dim, out_channels) if use_film else None
        )

    def forward(
        self,
        x: torch.Tensor,
        species_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = self.proj(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.act(self.norm2(self.conv2(out)) + residual)
        if self.film is not None and species_emb is not None:
            out = self.film(out, species_emb)
        return out


class SpliceAIEncoder(nn.Module):
    """SpliceAI-style dilated CNN encoder.

    Preserves single-nucleotide resolution (no pooling).
    Input:  ``(B, in_channels, L)``
    Output: ``(B, d_model, L)``
    """

    def __init__(
        self,
        *,
        in_channels: int = 5,
        d_model: int = 32,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 11,
        dropout: float = 0.1,
        use_film: bool = False,
        species_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
                use_film=use_film,
                species_embed_dim=species_embed_dim,
            )
            for d in dilations
        ])

    def forward(
        self,
        x: torch.Tensor,
        species_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.stem(x)            # (B, d_model, L)
        for block in self.blocks:
            h = block(h, species_emb)
        return h


# ---------------------------------------------------------------------------
# Species embedding
# ---------------------------------------------------------------------------

class SpeciesEmbedding(nn.Module):
    """Learned species embedding table."""

    def __init__(self, num_species: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_species, embed_dim)

    def forward(self, species_idx: torch.Tensor) -> torch.Tensor:
        return self.embed(species_idx)  # (B,) → (B, embed_dim)


# ---------------------------------------------------------------------------
# Candidate selector
# ---------------------------------------------------------------------------

class CandidateSelector(nn.Module):
    """Learned top-K candidate position selector.

    Bypassed automatically when ``L <= k_donor + k_acceptor``
    (e.g., 100 bp short-context input).

    Input:  ``(B, d_model, L)``
    Returns: ``h_sel (B, d_model, K_out)``,  ``positions (B, K_out)`` int64
    where K_out == L in bypass mode, K_out == k_total otherwise.
    """

    def __init__(
        self,
        d_model: int,
        k_donor: int = 256,
        k_acceptor: int = 256,
    ) -> None:
        super().__init__()
        self.k_donor = k_donor
        self.k_acceptor = k_acceptor
        self.k_total = k_donor + k_acceptor
        self.donor_score = nn.Linear(d_model, 1)
        self.acceptor_score = nn.Linear(d_model, 1)

    def forward(
        self,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, L = h.shape
        if L <= self.k_total:
            # Bypass: return all positions unchanged
            pos = torch.arange(L, device=h.device).unsqueeze(0).expand(B, -1)
            return h, pos

        h_t = h.transpose(1, 2)  # (B, L, d_model)
        d_scores = self.donor_score(h_t).squeeze(-1)    # (B, L)
        a_scores = self.acceptor_score(h_t).squeeze(-1)  # (B, L)

        _, d_idx = torch.topk(d_scores, self.k_donor, dim=-1)      # (B, k_donor)
        _, a_idx = torch.topk(a_scores, self.k_acceptor, dim=-1)    # (B, k_acceptor)

        # Union, sorted by position
        pos, _ = torch.sort(torch.cat([d_idx, a_idx], dim=-1), dim=-1)  # (B, K)

        # Gather features
        pos_exp = pos.unsqueeze(1).expand(-1, C, -1)  # (B, d_model, K)
        h_sel = torch.gather(h, 2, pos_exp)
        return h_sel, pos


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------

class GenomicPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding indexed by genomic coordinate."""

    def __init__(self, d_model: int, max_len: int = 50000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (B, K, d_model)
        positions : (B, K) int64 — genomic position indices

        Returns
        -------
        (B, K, d_model)
        """
        pe: torch.Tensor = self.pe  # type: ignore[assignment]
        pos = positions.clamp(max=pe.shape[0] - 1)
        return x + pe[pos]


class SpliceTransformerLayer(nn.Module):
    """Pre-norm Transformer layer with optional FiLM conditioning."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        use_film: bool = False,
        species_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.film: Optional[FiLMLayerTransformer] = (
            FiLMLayerTransformer(species_embed_dim, d_model) if use_film else None
        )

    def forward(
        self,
        x: torch.Tensor,
        species_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        if self.film is not None and species_emb is not None:
            x = self.film(x, species_emb)
        return x


class SpliceTransformerEncoder(nn.Module):
    """Stack of SpliceTransformerLayer with a final LayerNorm."""

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_film: bool = False,
        species_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SpliceTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_film=use_film,
                species_embed_dim=species_embed_dim,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        species_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, species_emb)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Spliceformer(nn.Module):
    """Species-conditioned spliceformer splice-site classifier.

    In binary mode (default) two independent linear heads share one encoder.
    forward_binary() extracts the center-position logit for compatibility
    with the existing score_sequences / infer_site evaluation pipeline.
    """

    def __init__(
        self,
        *,
        in_channels: int = 5,
        d_model: int = 32,
        cnn_dilations: Sequence[int] = (1, 2, 4, 8),
        cnn_kernel_size: int = 11,
        nhead: int = 4,
        num_transformer_layers: int = 8,
        dim_feedforward: int = 512,
        num_species: int = 4,
        species_embed_dim: int = 256,
        use_film: bool = False,
        k_donor: int = 256,
        k_acceptor: int = 256,
        dropout: float = 0.1,
        mode: str = "binary",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.species_embedding = SpeciesEmbedding(num_species, species_embed_dim)
        self.cnn_encoder = SpliceAIEncoder(
            in_channels=in_channels,
            d_model=d_model,
            dilations=cnn_dilations,
            kernel_size=cnn_kernel_size,
            dropout=dropout,
            use_film=use_film,
            species_embed_dim=species_embed_dim,
        )
        self.candidate_selector = CandidateSelector(
            d_model=d_model,
            k_donor=k_donor,
            k_acceptor=k_acceptor,
        )
        self.pos_encoding = GenomicPositionalEncoding(d_model)
        self.transformer = SpliceTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_film=use_film,
            species_embed_dim=species_embed_dim,
        )
        if mode == "binary":
            self.donor_head = nn.Linear(d_model, 1)
            self.acceptor_head = nn.Linear(d_model, 1)
        else:
            self.head = nn.Linear(d_model, 3)

    def forward(
        self,
        x: torch.Tensor,
        species_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Parameters
        ----------
        x           : (B, 5, L)
        species_idx : (B,) int64

        Returns
        -------
        logits    : binary → (B, K, 2) [donor=0, acceptor=1]
                    multiclass → (B, K, 3) [donor/acceptor/neither]
        positions : (B, K) int64 — selected genomic positions
        """
        sp_emb = self.species_embedding(species_idx)           # (B, embed_dim)
        h = self.cnn_encoder(x, sp_emb)                        # (B, d_model, L)
        h_sel, positions = self.candidate_selector(h)          # (B, d_model, K), (B, K)
        h_t = self.pos_encoding(h_sel.transpose(1, 2), positions)  # (B, K, d_model)
        h_out = self.transformer(h_t, sp_emb)                  # (B, K, d_model)

        if self.mode == "binary":
            donor_l = self.donor_head(h_out).squeeze(-1)       # (B, K)
            accept_l = self.acceptor_head(h_out).squeeze(-1)   # (B, K)
            logits = torch.stack([donor_l, accept_l], dim=-1)  # (B, K, 2)
        else:
            logits = self.head(h_out)                          # (B, K, 3)

        return logits, positions

    def forward_binary(
        self,
        x: torch.Tensor,
        species_idx: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        """Single-site binary logit for the center position.

        Parameters
        ----------
        x           : (B, 5, L)
        species_idx : (B,) int64
        task        : "donor" or "acceptor"

        Returns
        -------
        (B,) raw logit for the selected position
        """
        logits, positions = self.forward(x, species_idx)  # (B, K, 2), (B, K)
        center = x.shape[-1] // 2

        # Find the selected position closest to center
        best_k = (positions - center).abs().argmin(dim=-1)  # (B,)

        task_dim = 0 if task == "donor" else 1
        B = x.shape[0]
        return logits[torch.arange(B, device=x.device), best_k, task_dim]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpliceformerConfig:
    in_channels: int = 5
    d_model: int = 32
    cnn_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    cnn_kernel_size: int = 11
    nhead: int = 4
    num_transformer_layers: int = 8
    dim_feedforward: int = 512
    num_species: int = 4
    species_embed_dim: int = 256
    use_film: bool = False
    k_donor: int = 256
    k_acceptor: int = 256
    dropout: float = 0.1
    mode: str = "binary"


def _build_model(cfg: SpliceformerConfig) -> Spliceformer:
    return Spliceformer(
        in_channels=cfg.in_channels,
        d_model=cfg.d_model,
        cnn_dilations=cfg.cnn_dilations,
        cnn_kernel_size=cfg.cnn_kernel_size,
        nhead=cfg.nhead,
        num_transformer_layers=cfg.num_transformer_layers,
        dim_feedforward=cfg.dim_feedforward,
        num_species=cfg.num_species,
        species_embed_dim=cfg.species_embed_dim,
        use_film=cfg.use_film,
        k_donor=cfg.k_donor,
        k_acceptor=cfg.k_acceptor,
        dropout=cfg.dropout,
        mode=cfg.mode,
    )


def _config_from_args(model_args: argparse.Namespace, num_species: int) -> SpliceformerConfig:
    dilations = tuple(int(d) for d in str(model_args.cnn_dilations).split(","))
    return SpliceformerConfig(
        d_model=int(model_args.d_model),
        cnn_dilations=dilations,
        cnn_kernel_size=int(getattr(model_args, "cnn_kernel_size", 11)),
        nhead=int(model_args.nhead),
        num_transformer_layers=int(model_args.num_transformer_layers),
        dim_feedforward=int(model_args.dim_feedforward),
        num_species=num_species,
        species_embed_dim=int(model_args.species_embed_dim),
        use_film=bool(int(getattr(model_args, "use_film", 0))),
        k_donor=int(model_args.k_donor),
        k_acceptor=int(model_args.k_acceptor),
        dropout=float(getattr(model_args, "dropout", 0.1)),
        mode=str(getattr(model_args, "spliceformer_mode", "binary")),
    )


def _config_to_dict(cfg: SpliceformerConfig) -> dict:
    return {
        "site_arch": "spliceformer_sc",
        "in_channels": cfg.in_channels,
        "d_model": cfg.d_model,
        "cnn_dilations": list(cfg.cnn_dilations),
        "cnn_kernel_size": cfg.cnn_kernel_size,
        "nhead": cfg.nhead,
        "num_transformer_layers": cfg.num_transformer_layers,
        "dim_feedforward": cfg.dim_feedforward,
        "num_species": cfg.num_species,
        "species_embed_dim": cfg.species_embed_dim,
        "use_film": cfg.use_film,
        "k_donor": cfg.k_donor,
        "k_acceptor": cfg.k_acceptor,
        "dropout": cfg.dropout,
        "mode": cfg.mode,
    }


def _config_from_dict(d: dict) -> SpliceformerConfig:
    return SpliceformerConfig(
        in_channels=int(d.get("in_channels", 5)),
        d_model=int(d["d_model"]),
        cnn_dilations=tuple(int(x) for x in d["cnn_dilations"]),
        cnn_kernel_size=int(d.get("cnn_kernel_size", 11)),
        nhead=int(d["nhead"]),
        num_transformer_layers=int(d["num_transformer_layers"]),
        dim_feedforward=int(d["dim_feedforward"]),
        num_species=int(d["num_species"]),
        species_embed_dim=int(d["species_embed_dim"]),
        use_film=bool(d["use_film"]),
        k_donor=int(d["k_donor"]),
        k_acceptor=int(d["k_acceptor"]),
        dropout=float(d.get("dropout", 0.1)),
        mode=str(d.get("mode", "binary")),
    )


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_with_species(
    model: Spliceformer,
    sequences: List[str],
    species_idxs: List[int],
    window_len: int,
    task: str,
    device: str,
    batch_size: int,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
) -> np.ndarray:
    """Score sequences with per-sample species indices, return sigmoid probs."""
    model.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        seqs = sequences[start : start + batch_size]
        sps = species_idxs[start : start + batch_size]
        real_n = len(seqs)
        # Pad the last (partial) batch to batch_size so compiled static-shape
        # CUDA graphs are not invalidated by a smaller final batch.
        if real_n < batch_size:
            pad = batch_size - real_n
            seqs = seqs + [seqs[0]] * pad
            sps = sps + [sps[0]] * pad
        x = torch.from_numpy(
            np.stack([one_hot_encode_dna_5ch(s, window_len) for s in seqs]).astype(np.float32)
        ).to(device)
        sp = torch.tensor(sps, dtype=torch.long, device=device)
        amp_ctx: ContextManager = (
            torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype)
            if use_amp and amp_dtype is not None
            else nullcontext()
        )
        with amp_ctx:
            logits = model.forward_binary(x, sp, task=task)
        out.append(torch.sigmoid(logits.float()).cpu().numpy()[:real_n])
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_examples_for_species(
    species: str,
    task: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    sequence_transform: str,
) -> List[SpliceformerExample]:
    pos_path, neg_path, _ = resolve_train_paths(
        species=species,
        train_pos_path=None,
        train_neg_path=None,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )
    raw = cnn._load_task_examples_with_transform(
        pos_path=pos_path,
        neg_path=neg_path,
        task=task,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        sequence_transform=sequence_transform,
    )
    return [
        SpliceformerExample(sequence=seq, label=label, task=task, species=species)
        for seq, label in raw
    ]


def _load_multi_species_examples(
    species_list: List[str],
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    sequence_transform: str,
) -> List[SpliceformerExample]:
    all_examples: List[SpliceformerExample] = []
    for species in species_list:
        for task in ("donor", "acceptor"):
            all_examples.extend(
                _load_examples_for_species(
                    species, task, donor_len, acceptor_len, sequence_transform
                )
            )
    return all_examples


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_spliceformer(
    *,
    species_list: List[str],
    donor_checkpoint_path: str,
    acceptor_checkpoint_path: str,
    init_checkpoint_path: Optional[str],
    window_len: int,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    model_args: argparse.Namespace,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    seed: int,
    compile_model: bool,
    compile_mode: str,
    device: str,
    use_amp: Union[bool, int],
    amp_dtype: str,
    allow_tf32: Union[bool, int],
    cudnn_benchmark: Union[bool, int],
    num_workers: Union[str, int],
    prefetch_factor: int,
    persistent_workers: Union[bool, int],
    pin_memory: Union[bool, int],
    min_batch_size: int,
    max_oom_retries: int,
    validation_metric: str = "pr_auc",
    sequence_transform: str = "none",
) -> Dict[str, object]:
    """Train one spliceformer_sc model on pooled multi-species data.

    Saves the shared checkpoint to both donor_checkpoint_path and
    acceptor_checkpoint_path so the existing pipeline can find it.
    """
    t0 = time.time()
    set_seed(seed)

    # Flash Attention's backward is non-deterministic by design; suppress the
    # one-time UserWarning that fires on first backward when determinism is off.
    warnings.filterwarnings(
        "ignore",
        message="Flash Attention defaults to a non-deterministic algorithm",
        category=UserWarning,
    )

    # AMP / hardware flags
    if _bool_from_flag(allow_tf32) and device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if _bool_from_flag(cudnn_benchmark) and device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    # Build species index (deterministic: sort species alphabetically)
    species_to_idx: Dict[str, int] = {s: i for i, s in enumerate(sorted(species_list))}
    num_species = len(species_to_idx)

    # Build model
    cfg = _config_from_args(model_args, num_species)
    model = _build_model(cfg).to(device)

    if init_checkpoint_path and os.path.exists(init_checkpoint_path):
        _warm_start_model(model, init_checkpoint_path, device)
        print(f"[spliceformer_sc] warm-started from {init_checkpoint_path}")

    # Load data
    print(f"[spliceformer_sc] loading data for {species_list} ...")
    all_examples = _load_multi_species_examples(
        species_list, donor_len, acceptor_len, sequence_transform
    )
    n_pos = sum(ex.label for ex in all_examples)
    n_neg = len(all_examples) - n_pos
    print(
        f"[spliceformer_sc] total={len(all_examples)} pos={n_pos} neg={n_neg} "
        f"species_to_idx={species_to_idx}"
    )
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Insufficient training examples: pos={n_pos} neg={n_neg}."
        )

    val_frac = float(getattr(model_args, "val_frac", 0.1))
    train_ex, val_ex = _stratified_split_examples(all_examples, val_frac, seed)

    train_ds = SpliceformerDataset(train_ex, window_len, species_to_idx)
    val_ds = SpliceformerDataset(val_ex, window_len, species_to_idx)

    # DataLoaders
    batch_size = int(getattr(model_args, "batch_size", 512))
    resolved_workers = _resolve_num_workers(num_workers, device)
    use_persist = _bool_from_flag(persistent_workers) and resolved_workers > 0
    use_pin = _bool_from_flag(pin_memory) and device.startswith("cuda")

    train_kw: dict = {
        "dataset": train_ds,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": resolved_workers,
        "pin_memory": use_pin,
        "drop_last": False,
        "worker_init_fn": _seed_worker,
    }
    if resolved_workers > 0:
        train_kw["prefetch_factor"] = prefetch_factor
        train_kw["persistent_workers"] = use_persist
    train_loader = DataLoader(**train_kw)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=resolved_workers,
        pin_memory=use_pin,
    )
    print(
        f"[spliceformer_sc] train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} batch_size={batch_size}"
    )

    # Optimiser + scheduler
    lr = float(getattr(model_args, "lr", 1e-3))
    weight_decay = float(getattr(model_args, "weight_decay", 0.01))
    eta_min_ratio = float(getattr(model_args, "eta_min_ratio", 0.01))
    grad_clip = float(getattr(model_args, "grad_clip", 5.0))

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            fused=device.startswith("cuda"),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * eta_min_ratio
    )

    # AMP
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype, device)
    use_amp_bool = _bool_from_flag(use_amp) and device.startswith("cuda")
    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler()
        if use_amp_bool and resolved_amp_dtype == torch.float16
        else None
    )

    # torch.compile
    compile_enabled = _resolve_compile_enabled(compile_mode, compile_model, False, device, epochs)
    if compile_enabled:
        model, _, _, _ = _compile_model_with_fallback(model, compile_mode=compile_mode)

    # Training state
    bce = nn.BCEWithLogitsLoss()
    best_metric = float("-inf")
    best_epoch = 0
    epochs_since = 0
    stopped_early = False
    epoch_history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, labels, sp_idx, task_idx in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            sp_idx = sp_idx.to(device)
            task_idx = task_idx.to(device)

            amp_ctx: ContextManager = (
                torch.autocast(device_type=device.split(":")[0], dtype=resolved_amp_dtype)
                if use_amp_bool and resolved_amp_dtype is not None
                else nullcontext()
            )
            with amp_ctx:
                logits, positions = model(x, sp_idx)  # (B, K, 2), (B, K)

                # Find center-position index for each sample
                center = x.shape[-1] // 2
                best_k = (positions - center).abs().argmin(dim=-1)  # (B,)
                B = x.shape[0]
                arange = torch.arange(B, device=device)

                donor_mask = (task_idx == 0)
                acceptor_mask = (task_idx == 1)

                d_logits = logits[arange, best_k, 0]  # (B,)  donor logits
                a_logits = logits[arange, best_k, 1]  # (B,)  acceptor logits

                loss = torch.zeros(1, device=device)
                if donor_mask.any():
                    loss = loss + bce(d_logits[donor_mask], labels[donor_mask])
                if acceptor_mask.any():
                    loss = loss + bce(a_logits[acceptor_mask], labels[acceptor_mask])

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validation: score val set per task using forward_binary
        model.eval()
        val_seqs_d = [ex.sequence for ex in val_ex if ex.task == "donor"]
        val_labs_d = np.array([ex.label for ex in val_ex if ex.task == "donor"], dtype=np.float32)
        val_sps_d = [species_to_idx[ex.species] for ex in val_ex if ex.task == "donor"]

        val_seqs_a = [ex.sequence for ex in val_ex if ex.task == "acceptor"]
        val_labs_a = np.array([ex.label for ex in val_ex if ex.task == "acceptor"], dtype=np.float32)
        val_sps_a = [species_to_idx[ex.species] for ex in val_ex if ex.task == "acceptor"]

        with torch.no_grad():
            scores_d = _score_with_species(
                model, val_seqs_d, val_sps_d, window_len, "donor",
                device, batch_size, use_amp_bool, resolved_amp_dtype,
            )
            scores_a = _score_with_species(
                model, val_seqs_a, val_sps_a, window_len, "acceptor",
                device, batch_size, use_amp_bool, resolved_amp_dtype,
            )

        all_scores = np.concatenate([scores_d, scores_a])
        all_labels = np.concatenate([val_labs_d, val_labs_a])

        def _safe_pr_auc(labels_arr: np.ndarray, scores_arr: np.ndarray) -> float:
            if len(np.unique(labels_arr)) < 2:
                return float("nan")
            from sklearn.metrics import average_precision_score
            return float(average_precision_score(labels_arr, scores_arr))

        def _safe_roc_auc(labels_arr: np.ndarray, scores_arr: np.ndarray) -> float:
            if len(np.unique(labels_arr)) < 2:
                return float("nan")
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(labels_arr, scores_arr))

        val_pr_auc = _safe_pr_auc(all_labels, all_scores)
        val_roc_auc = _safe_roc_auc(all_labels, all_scores)
        val_metric = {"pr_auc": val_pr_auc, "roc_auc": val_roc_auc}.get(
            validation_metric, val_pr_auc
        )

        elapsed = time.time() - t0
        print(
            f"[spliceformer_sc] epoch={epoch}/{epochs} "
            f"loss={avg_loss:.4f} val_pr_auc={val_pr_auc:.4f} "
            f"val_roc_auc={val_roc_auc:.4f} elapsed={elapsed:.0f}s"
        )

        epoch_history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc,
        })

        if not math.isnan(val_metric) and val_metric > best_metric + early_stop_min_delta:
            best_metric = val_metric
            best_epoch = epoch
            epochs_since = 0
            # Save checkpoint at both donor and acceptor paths for pipeline compat
            ckpt = {
                "task": "multi",
                "window_len": window_len,
                "model_config": _config_to_dict(cfg),
                "model_state": _export_model_state_dict(model),
                "species_to_idx": species_to_idx,
            }
            for ckpt_path in (donor_checkpoint_path, acceptor_checkpoint_path):
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(ckpt, ckpt_path)
        else:
            epochs_since += 1
            if early_stop_patience > 0 and epochs_since >= early_stop_patience:
                stopped_early = True
                print(
                    f"[spliceformer_sc] early stop at epoch {epoch} "
                    f"(best={best_epoch} {validation_metric}={best_metric:.4f})"
                )
                break

    return {
        "model": "spliceformer_sc",
        "species_list": species_list,
        "window_len": window_len,
        "best_epoch": best_epoch,
        f"best_{validation_metric}": best_metric,
        "validation_metric": validation_metric,
        "stopped_early": stopped_early,
        "epoch_history": epoch_history,
        "donor_checkpoint_path": donor_checkpoint_path,
        "acceptor_checkpoint_path": acceptor_checkpoint_path,
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_task_model(
    checkpoint_path: str,
    device: str,
) -> Tuple[Spliceformer, Dict[str, object]]:
    """Load a trained spliceformer_sc checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = _config_from_dict(ckpt["model_config"])
    model = _build_model(cfg)
    state = _normalize_checkpoint_state_dict(ckpt["model_state"])
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, ckpt


def infer_site_scores(
    site_rows: List[Dict[str, object]],
    checkpoint_path: str,
    species: str,
    device: str = "auto",
    batch_size: int = 512,
    infer_use_amp: Union[bool, int] = 0,
    infer_amp_dtype: str = "auto",
    infer_compile: Union[bool, int] = 0,
    infer_compile_mode: str = "off",
) -> List[Dict[str, object]]:
    """Score site rows with one spliceformer_sc checkpoint."""
    device_name = pick_device(device)
    model, ckpt = load_task_model(checkpoint_path, device_name)

    species_to_idx: Dict[str, int] = ckpt["species_to_idx"]
    if species not in species_to_idx:
        known = sorted(species_to_idx.keys())
        raise ValueError(
            f"Species '{species}' not in checkpoint. Available: {known}"
        )
    sp_idx_val = species_to_idx[species]
    window_len = int(ckpt["window_len"])

    resolved_amp_dtype = _resolve_amp_dtype(infer_amp_dtype, device_name)
    use_amp_bool = _bool_from_flag(infer_use_amp) and device_name.startswith("cuda")

    compile_enabled = _resolve_compile_enabled(
        infer_compile_mode, _bool_from_flag(infer_compile), False, device_name, 1
    )
    if compile_enabled:
        model, _, _, _ = _compile_model_with_fallback(model, compile_mode=infer_compile_mode)

    def _score_rows(rows: List[Dict], task: str) -> List[Dict]:
        if not rows:
            return []
        seqs = [str(r["seq"]) for r in rows]
        scores = _score_with_species(
            model, seqs, [sp_idx_val] * len(seqs), window_len, task,
            device_name, batch_size, use_amp_bool, resolved_amp_dtype,
        )
        result = []
        for row, score in zip(rows, scores):
            out = dict(row)
            out["score"] = float(score)
            out[SCORE_SPACE_FIELD] = SCORE_SPACE_LOG10
            result.append(out)
        return result

    donor_rows = [r for r in site_rows if r.get("site_type") == "donor"]
    acceptor_rows = [r for r in site_rows if r.get("site_type") == "acceptor"]
    return _score_rows(donor_rows, "donor") + _score_rows(acceptor_rows, "acceptor")


# ---------------------------------------------------------------------------
# Argument registration
# ---------------------------------------------------------------------------

def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register spliceformer_sc training arguments."""
    cnn.add_train_args(parser)
    parser.add_argument(
        "--species_list",
        type=str,
        default=None,
        help="Comma-separated species for multi-species training (e.g. 'Athal,Hsap').",
    )
    parser.add_argument("--d_model", type=int, default=32,
                        help="CNN and Transformer hidden width.")
    parser.add_argument("--cnn_dilations", type=str, default="1,2,4,8",
                        help="Comma-separated dilation factors for CNN blocks.")
    parser.add_argument("--cnn_kernel_size", type=int, default=11,
                        help="Kernel size for CNN blocks (must be odd).")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of Transformer attention heads.")
    parser.add_argument("--num_transformer_layers", type=int, default=8,
                        help="Number of Transformer encoder layers.")
    parser.add_argument("--dim_feedforward", type=int, default=512,
                        help="Transformer FFN hidden dimension.")
    parser.add_argument("--species_embed_dim", type=int, default=256,
                        help="Species embedding dimension.")
    parser.add_argument("--use_film", type=int, default=0, choices=[0, 1],
                        help="Enable FiLM species conditioning (0=off, 1=on).")
    parser.add_argument("--k_donor", type=int, default=256,
                        help="Candidate selector: max donor positions to keep.")
    parser.add_argument("--k_acceptor", type=int, default=256,
                        help="Candidate selector: max acceptor positions to keep.")
    parser.add_argument(
        "--spliceformer_mode",
        choices=["binary", "multiclass"],
        default="binary",
        help="Output mode: binary (two BCE heads) or multiclass (3-class CE).",
    )


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register spliceformer_sc inference arguments."""
    cnn.add_infer_args(parser)


# ---------------------------------------------------------------------------
# Public API — called by run_model.py
# ---------------------------------------------------------------------------

def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train spliceformer_sc with the unified pipeline runtime."""
    # Resolve species list
    species_list_str = getattr(model_args, "species_list", None)
    if species_list_str:
        species_list = [s.strip() for s in str(species_list_str).split(",") if s.strip()]
    else:
        species_list = [common_args.species]

    # Resolve window lengths from first species
    first_species = species_list[0]
    dirs = species_data_dirs(first_species)
    inferred_len: Optional[int] = None
    if common_args.donor_len is None and common_args.acceptor_len is None:
        try:
            _, _, inferred_len = infer_default_train_paths(
                train_dir=dirs["raw"],
                donor_len=None,
                acceptor_len=None,
            )
        except ValueError:
            pass

    donor_len, acceptor_len = resolve_effective_window_lengths(
        donor_len=common_args.donor_len,
        acceptor_len=common_args.acceptor_len,
        inferred_train_len=inferred_len,
    )
    validate_window_args(donor_len=donor_len, acceptor_len=acceptor_len)
    window_len = donor_len if donor_len is not None else (acceptor_len or 100)

    # Resolve checkpoint paths from common_args (set by run_pipeline)
    task_paths = resolve_required_checkpoint_paths(common_args, require_exists=False)
    donor_ckpt_path = task_paths["donor"]
    acceptor_ckpt_path = task_paths["acceptor"]

    # Resolve init checkpoint (warm start)
    init_ckpt: Optional[str] = None
    donor_init_paths = resolve_required_checkpoint_paths(
        _InitCheckpointProxy(common_args), require_exists=False
    ) if _has_init_checkpoint(common_args) else {}
    if donor_init_paths:
        init_ckpt = donor_init_paths.get("donor")

    # Training schedule
    schedule = resolve_training_schedule(
        epochs_arg=getattr(model_args, "epochs", "auto"),
        max_epochs=int(getattr(model_args, "max_epochs", 100)),
        patience_arg=getattr(model_args, "early_stop_patience", 5),
        min_delta_arg=getattr(model_args, "early_stop_min_delta", 0.001),
    )

    return train_spliceformer(
        species_list=species_list,
        donor_checkpoint_path=donor_ckpt_path,
        acceptor_checkpoint_path=acceptor_ckpt_path,
        init_checkpoint_path=init_ckpt,
        window_len=window_len,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        model_args=model_args,
        epochs=schedule.resolved_epochs,
        early_stop_patience=schedule.effective_early_stop_patience,
        early_stop_min_delta=schedule.early_stop_min_delta,
        seed=int(getattr(common_args, "seed", 1337)),
        compile_model=_bool_from_flag(getattr(model_args, "compile", False)),
        compile_mode=str(getattr(model_args, "compile_mode", "default")),
        device=pick_device(common_args.device),
        use_amp=getattr(model_args, "use_amp", 0),
        amp_dtype=str(getattr(model_args, "amp_dtype", "auto")),
        allow_tf32=getattr(model_args, "allow_tf32", 1),
        cudnn_benchmark=getattr(model_args, "cudnn_benchmark", 1),
        num_workers=getattr(common_args, "num_workers", "auto"),
        prefetch_factor=int(getattr(model_args, "prefetch_factor", 2)),
        persistent_workers=getattr(model_args, "persistent_workers", 1),
        pin_memory=getattr(model_args, "pin_memory", 1),
        min_batch_size=int(getattr(model_args, "min_batch_size", 8)),
        max_oom_retries=int(getattr(model_args, "max_oom_retries", 3)),
        validation_metric=str(getattr(model_args, "validation_metric", "pr_auc")),
        sequence_transform=str(getattr(model_args, "sequence_transform", "none")),
    )


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Run site-level inference with a trained spliceformer_sc checkpoint."""
    dirs = species_data_dirs(common_args.species)
    inferred_len: Optional[int] = None
    if common_args.donor_len is None and common_args.acceptor_len is None:
        try:
            _, _, inferred_len = infer_default_train_paths(
                train_dir=dirs["raw"],
                donor_len=None,
                acceptor_len=None,
            )
        except ValueError:
            pass

    donor_len, acceptor_len = resolve_effective_window_lengths(
        donor_len=common_args.donor_len,
        acceptor_len=common_args.acceptor_len,
        inferred_train_len=inferred_len,
    )
    validate_window_args(donor_len=donor_len, acceptor_len=acceptor_len)

    test_tsv = resolve_test_tsv(common_args.species, common_args.test_tsv)
    site_rows, skipped = read_test_site_rows(
        test_tsv=test_tsv,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )
    print(f"[spliceformer_sc] loaded {len(site_rows)} test sites (skipped {skipped} short)")

    # Spliceformer uses a single shared checkpoint; load from donor path
    task_paths = resolve_required_checkpoint_paths(common_args, require_exists=True)
    checkpoint_path = task_paths["donor"]

    infer_batch = (
        int(model_args.infer_batch_size)
        if getattr(model_args, "infer_batch_size", None) is not None
        else int(model_args.batch_size)
    )

    return infer_site_scores(
        site_rows=site_rows,
        checkpoint_path=checkpoint_path,
        species=common_args.species,
        device=common_args.device,
        batch_size=infer_batch,
        infer_use_amp=getattr(model_args, "infer_use_amp", 0),
        infer_amp_dtype=str(getattr(model_args, "infer_amp_dtype", "auto")),
        infer_compile=getattr(model_args, "infer_compile", 0),
        infer_compile_mode=str(getattr(model_args, "infer_compile_mode", "off")),
    )


# ---------------------------------------------------------------------------
# Internal helpers for checkpoint path resolution in train()
# ---------------------------------------------------------------------------

def _has_init_checkpoint(common_args: argparse.Namespace) -> bool:
    return bool(str(getattr(common_args, "donor_init_checkpoint_path", "") or "").strip())


class _InitCheckpointProxy:
    """Proxy to resolve init checkpoint paths via standard utility."""
    def __init__(self, common_args: argparse.Namespace) -> None:
        donor = str(getattr(common_args, "donor_init_checkpoint_path", "") or "")
        object.__setattr__(self, "donor_checkpoint_path", donor)
        object.__setattr__(self, "acceptor_checkpoint_path", donor)
