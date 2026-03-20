"""Shared CNN building blocks for splice-site models.

This module provides reusable components for CNN-based models:
- DNA one-hot encoding
- channel-list parser
- Conv1D encoder + readout
- site classifier head
- batched sequence scoring helper
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

CNN_HEAD_TYPE_CHOICES: tuple[str, ...] = ("gap", "center")


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


def normalize_cnn_head_type(raw: object, *, arg_name: str) -> str:
    """Normalize one CNN readout type name.

    Parameters
    ----------
    raw : object
        Raw argument value.
    arg_name : str
        Argument name used in error messages.

    Returns
    -------
    str
        Normalized head type.

    Raises
    ------
    ValueError
        If the head type is unsupported.
    """
    head_type = str(raw).strip().lower()
    if head_type in CNN_HEAD_TYPE_CHOICES:
        return head_type
    choices_text = ", ".join(CNN_HEAD_TYPE_CHOICES)
    raise ValueError(f"{arg_name} must be one of: {choices_text}.")


class CnnFeatureReadout(nn.Module):
    """Readout over Conv1D features with position-sensitive options.

    Parameters
    ----------
    output_channels : int
        Channel width of the convolution stack output.
    head_type : str, default="gap"
        Readout mode. ``"gap"`` uses global average pooling and ``"center"``
        reads the center position, averaging the two middle positions for
        even-length feature maps.
    """

    def __init__(
        self,
        output_channels: int,
        head_type: str = "gap",
    ) -> None:
        super().__init__()
        if output_channels <= 0:
            raise ValueError("output_channels must be positive.")
        self.head_type = normalize_cnn_head_type(
            head_type,
            arg_name="head_type",
        )
        self.gap: Optional[nn.AdaptiveAvgPool1d]
        if self.head_type == "gap":
            self.gap = nn.AdaptiveAvgPool1d(1)
        else:
            self.gap = None
        self.output_dim: int = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read one feature vector per sequence.

        Parameters
        ----------
        x : torch.Tensor
            Feature map with shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Encoded features with shape ``(batch, channels)``.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, length).")
        if x.shape[2] <= 0:
            raise ValueError("x length dimension must be positive.")
        if self.head_type == "gap":
            if self.gap is None:
                raise RuntimeError("GAP readout is not initialized.")
            return self.gap(x).squeeze(-1)

        center_right = x.shape[2] // 2
        if x.shape[2] % 2 == 1:
            return x[:, :, center_right]
        center_left = center_right - 1
        return 0.5 * (x[:, :, center_left] + x[:, :, center_right])


class CnnGapEncoder(nn.Module):
    """Conv1D encoder with configurable readout.

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
    conv_stride : int, default=1
        Shared stride used by all convolution layers.
    head_type : str, default="gap"
        Feature readout mode. ``"gap"`` applies global average pooling and
        ``"center"`` uses the center position after the conv stack.

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
        conv_stride: int = 1,
        head_type: str = "gap",
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
        if conv_stride <= 0:
            raise ValueError("conv_stride must be positive.")

        layers: list[nn.Module] = []
        prev_ch = in_channels
        for ch, layer_kernel_size in zip(channel_list, kernel_sizes):
            layers.extend(
                [
                    nn.Conv1d(
                        prev_ch,
                        ch,
                        layer_kernel_size,
                        stride=conv_stride,
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
        self.readout = CnnFeatureReadout(
            output_channels=int(channel_list[-1]),
            head_type=head_type,
        )
        self.gap = self.readout.gap
        self.output_dim: int = self.readout.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode batched one-hot DNA into readout features.

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
        return self.readout(y)


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
        conv_stride: int = 1,
        head_type: str = "gap",
    ) -> None:
        super().__init__()
        encoder = CnnGapEncoder(
            in_channels=in_channels,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            max_pool_size=max_pool_size,
            conv_stride=conv_stride,
            head_type=head_type,
        )
        # Keep legacy attribute names for checkpoint compatibility.
        self.conv_layers = encoder.conv_layers
        self.readout = encoder.readout
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
        features = self.readout(y)
        return self.fc(features).squeeze(-1)


@torch.no_grad()
def score_sequences(
    model: nn.Module,
    sequences: Sequence[str],
    window_len: int,
    device: str,
    batch_size: int = 512,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
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
    use_amp : bool, default=False
        Whether to run CUDA autocast during inference.
    amp_dtype : torch.dtype | None, default=None
        Autocast dtype used when ``use_amp`` is enabled.

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
        if use_amp and device == "cuda" and amp_dtype is not None:
            amp_context: ContextManager[object] = torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            )
        else:
            amp_context = nullcontext()
        with amp_context:
            logits = model(batch_x)
        probs = torch.sigmoid(logits).float().detach().cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)
