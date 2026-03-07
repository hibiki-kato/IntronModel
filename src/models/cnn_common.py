"""Shared CNN building blocks for splice-site models.

This module provides reusable components for CNN-based models:
- DNA one-hot encoding
- channel-list parser
- Conv1D -> GAP encoder
- site classifier head
- batched sequence scoring helper
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def one_hot_encode_dna(seq: str, window_len: int = 50) -> np.ndarray:
    """One-hot encode a DNA sequence.

    Parameters
    ----------
    seq : str
        Input DNA sequence.
    window_len : int, default=50
        Output sequence length. Input is truncated to this length.

    Returns
    -------
    np.ndarray
        Encoded array with shape ``(4, window_len)`` and dtype ``float32``.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((4, window_len), dtype=np.float32)

    for i, base in enumerate(seq[:window_len].upper()):
        if base in mapping:
            encoded[mapping[base], i] = 1.0

    return encoded


def parse_conv_channels(
    raw: Optional[str],
    arg_name: str = "--conv_channels",
) -> Optional[List[int]]:
    """Parse comma-separated convolution channel sizes.

    Parameters
    ----------
    raw : str | None
        Comma-separated channel sizes like ``"64,128,256"``.
    arg_name : str, default="--conv_channels"
        Argument name used in error messages.

    Returns
    -------
    list[int] | None
        Parsed positive channel sizes, or ``None`` when not specified.

    Raises
    ------
    ValueError
        If the string has invalid format or non-positive sizes.
    """
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{arg_name} must include at least one integer.")

    channels: List[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {arg_name} item '{part}'. Use integers like 64,128,256."
            ) from exc
        if value <= 0:
            raise ValueError(f"{arg_name} values must be positive.")
        channels.append(value)
    return channels


def parse_kernel_sizes(
    raw: Optional[str],
    arg_name: str = "--kernel_sizes",
) -> Optional[List[int]]:
    """Parse comma-separated kernel sizes.

    Parameters
    ----------
    raw : str | None
        Comma-separated kernel sizes like ``"11,7,5"``.
    arg_name : str, default="--kernel_sizes"
        Argument name used in validation errors.

    Returns
    -------
    list[int] | None
        Parsed positive kernel sizes, or ``None`` when not specified.

    Raises
    ------
    ValueError
        If values are invalid.
    """
    return parse_conv_channels(raw=raw, arg_name=arg_name)


class CnnGapEncoder(nn.Module):
    """Conv1D encoder with adaptive global average pooling.

    Parameters
    ----------
    in_channels : int, default=4
        Input channel count.
    conv_channels : Sequence[int] | None, default=None
        Convolution channel sizes. Uses ``[64, 128, 256]`` when ``None``.
    kernel_size : int, default=7
        Convolution kernel size.
    dropout : float, default=0.3
        Dropout rate in convolution blocks.
    max_pool_size : int, default=2
        Max-pooling width applied after each convolution block. Use ``1`` to
        skip pooling.

    Notes
    -----
    Runtime is linear in sequence length and channel sizes for each layer.
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: Optional[Sequence[int]] = None,
        kernel_size: int | Sequence[int] = 7,
        dropout: float = 0.3,
        max_pool_size: int = 2,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = [64, 128, 256]
        if not conv_channels:
            raise ValueError("conv_channels must not be empty.")
        channel_list = list(conv_channels)

        kernel_sizes: list[int]
        if isinstance(kernel_size, int):
            if kernel_size <= 0:
                raise ValueError("kernel_size must be positive.")
            kernel_sizes = [kernel_size] * len(channel_list)
        else:
            kernel_sizes = [int(value) for value in kernel_size]
            if len(kernel_sizes) == 1:
                kernel_sizes = kernel_sizes * len(channel_list)
            elif len(kernel_sizes) < len(channel_list):
                pad_value = kernel_sizes[-1]
                kernel_sizes.extend(
                    [pad_value] * (len(channel_list) - len(kernel_sizes))
                )
            elif len(kernel_sizes) > len(channel_list):
                kernel_sizes = kernel_sizes[: len(channel_list)]
            if any(value <= 0 for value in kernel_sizes):
                raise ValueError("kernel_size list values must be positive.")
        if max_pool_size <= 0:
            raise ValueError("max_pool_size must be positive.")

        layers: list[nn.Module] = []
        prev_ch = in_channels
        for ch, layer_kernel_size in zip(channel_list, kernel_sizes):
            layers.extend(
                [
                    nn.Conv1d(
                        prev_ch,
                        ch,
                        layer_kernel_size,
                        padding=layer_kernel_size // 2,
                    ),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(inplace=True),
                ]
            )
            if max_pool_size > 1:
                layers.append(nn.MaxPool1d(max_pool_size))
            layers.append(nn.Dropout(dropout))
            prev_ch = ch

        self.conv_layers = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.output_dim: int = int(channel_list[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode batched one-hot DNA into pooled features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, 4, length)``.

        Returns
        -------
        torch.Tensor
            Feature tensor with shape ``(batch, output_dim)``.
        """
        y = self.conv_layers(x)
        y = self.gap(y)
        return y.squeeze(-1)


class BasicSpliceCNN(nn.Module):
    """Site-level binary classifier built on top of ``CnnGapEncoder``."""

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: Optional[Sequence[int]] = None,
        kernel_size: int | Sequence[int] = 7,
        dropout: float = 0.3,
        fc_hidden: int = 128,
        max_pool_size: int = 2,
    ) -> None:
        super().__init__()
        encoder = CnnGapEncoder(
            in_channels=in_channels,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            max_pool_size=max_pool_size,
        )
        # Keep legacy attribute names for checkpoint compatibility.
        self.conv_layers = encoder.conv_layers
        self.gap = encoder.gap
        output_dim = encoder.output_dim
        self.fc = nn.Sequential(
            nn.Linear(output_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, 4, length)``.

        Returns
        -------
        torch.Tensor
            Logit tensor with shape ``(batch,)``.
        """
        y = self.conv_layers(x)
        y = self.gap(y)
        features = y.squeeze(-1)
        return self.fc(features).squeeze(-1)


@torch.no_grad()
def score_sequences(
    model: nn.Module,
    sequences: Sequence[str],
    window_len: int,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Score sequences using a binary model.

    Parameters
    ----------
    model : nn.Module
        Model returning one logit per sample.
    sequences : Sequence[str]
        DNA sequences.
    window_len : int
        Encoding length.
    device : str
        Torch device name.
    batch_size : int, default=512
        Batch size for inference.

    Returns
    -------
    np.ndarray
        Probability vector with shape ``(n_sequences,)``.
    """
    if not sequences:
        return np.array([])

    model.eval()
    encoded = [one_hot_encode_dna(seq, window_len) for seq in sequences]
    x = torch.from_numpy(np.stack(encoded)).to(device)

    all_probs: list[np.ndarray] = []
    for index in range(0, len(x), batch_size):
        batch_x = x[index : index + batch_size]
        logits = model(batch_x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)
