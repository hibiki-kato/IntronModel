"""Reservoir model implementation with a Torch-based ESN backbone.

This module keeps the unified ``run_model.py`` contract:
- ``add_train_args`` / ``add_infer_args`` for CLI integration
- ``train`` for donor/acceptor training
- ``infer_site`` for site-level inference

Core implementation follows the ``time series classification`` workflow pattern:
- DNA sequences are converted into multivariate time series ``[N, T, V]``.
- A fixed reservoir (ESN) maps each sequence to a compact representation.
- A readout classifier is trained on the sequence representation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import pickle
import random
import subprocess
import time
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, cast

import numpy as np
import torch

from util.data_proc import (
    build_run_name,
    infer_default_train_paths,
    read_examples_single_task,
    read_test_site_rows,
    resolve_effective_window_lengths,
    resolve_test_tsv,
    resolve_train_paths,
    species_data_dirs,
    validate_window_args,
)
from util.model_task_paths import (
    resolve_required_checkpoint_paths,
    resolve_tasks_to_train,
    resolve_train_target,
)
from util.process_title import (
    apply_eta_process_title_from_epoch_progress,
    apply_eta_process_title_placeholder,
)
from util.training_control import (
    resolve_early_stopping_params,
    resolve_training_epoch_budget,
)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ModuleNotFoundError:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

CHECKPOINT_FORMAT_VERSION: str = "torch-reservoir-v1"
LOSS_NAME_CHOICES: tuple[str, ...] = (
    "bce",
    "weighted_bce",
    "focal",
    "asymmetric_focal",
)
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]")
DNA_BASES: tuple[str, ...] = ("A", "C", "G", "T")
INPUT_MODE_CHOICES: tuple[str, ...] = ("onehot", "kmer")
READ_ORDER_CHOICES: tuple[str, ...] = ("auto", "forward", "reverse")
POOLING_CHOICES: tuple[str, ...] = (
    "last",
    "mean",
    "max",
    "mean_max",
    "attention",
    "logit_sum",
    "weighted_logit_sum",
)
MTS_REP_CHOICES: tuple[str, ...] = ("last", "mean", "output", "reservoir")
MTS_REP_ARG_CHOICES: tuple[str, ...] = ("auto", *MTS_REP_CHOICES)
DIMRED_METHOD_CHOICES: tuple[str, ...] = ("none", "pca", "tenpca")
READOUT_TYPE_CHOICES: tuple[str, ...] = ("lin", "mlp", "svm")
POOLING_TO_MTS_REP: Mapping[str, str] = {
    "last": "last",
    "mean": "mean",
    "max": "mean",
    "mean_max": "mean",
    "attention": "mean",
    "logit_sum": "output",
    "weighted_logit_sum": "output",
}
DEFAULT_RC_STATE_BUDGET_GB: float = 8.0
RC_STATE_BUDGET_ENV: str = "INTRONMODEL_RC_STATE_BUDGET_GB"
RESERVOIR_STATE_DTYPE_BYTES: int = int(np.dtype(np.float32).itemsize)
STATE_MEMORY_HEADROOM_FACTOR: float = 1.5
AUTO_STATE_BUDGET_FRACTION: float = 0.20
AUTO_STATE_BUDGET_MIN_GB: float = 1.0
AUTO_STATE_BUDGET_MAX_GB: float = 32.0


def _bool_from_flag(flag: bool | int) -> bool:
    """Convert integer/boolean flags from CLI to strict bool."""
    if isinstance(flag, bool):
        return flag
    return int(flag) != 0


def _binary_clf_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative false/true positives at score thresholds."""
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and scores must be 1-D arrays.")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length.")
    if labels.size == 0:
        raise ValueError("labels and scores must be non-empty.")

    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(np.int64, copy=False)
    scores_sorted = scores[order]

    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_indices = np.r_[distinct_indices, labels_sorted.size - 1]

    true_positives = np.cumsum(labels_sorted)[threshold_indices]
    false_positives = (threshold_indices + 1) - true_positives
    return (
        false_positives.astype(np.float64, copy=False),
        true_positives.astype(np.float64, copy=False),
    )


def _fallback_average_precision(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute binary average precision without scikit-learn."""
    positives = float(np.sum(labels == 1))
    if positives <= 0.0:
        raise ValueError("At least one positive label is required.")

    false_positives, true_positives = _binary_clf_curve(labels, probs)
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / positives

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _fallback_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute binary ROC-AUC without scikit-learn."""
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    if positives <= 0.0 or negatives <= 0.0:
        raise ValueError("Both positive and negative labels are required.")

    false_positives, true_positives = _binary_clf_curve(labels, probs)
    fpr = np.r_[0.0, false_positives / negatives, 1.0]
    tpr = np.r_[0.0, true_positives / positives, 1.0]
    return float(np.trapezoid(tpr, fpr))


def _fallback_max_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute binary max-F1 over thresholds without scikit-learn."""
    positives = float(np.sum(labels == 1))
    if positives <= 0.0:
        raise ValueError("At least one positive label is required.")

    false_positives, true_positives = _binary_clf_curve(labels, probs)
    false_negatives = positives - true_positives

    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / np.maximum(true_positives + false_negatives, 1.0)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
    if f1.size == 0:
        raise ValueError("At least one prediction is required.")
    return float(np.max(f1))


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReservoirClassifierProtocol(Protocol):
    """Protocol implemented by the local Torch RC classifier."""

    n_drop: int
    bidir: bool
    dimred_method: Optional[str]
    mts_rep: str
    _reservoir: object
    readout: object

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True) -> None:
        """Fit one RC model from ``X`` and one-hot labels ``Y``."""

    def predict(self, Xte: np.ndarray) -> np.ndarray:
        """Predict class labels for ``Xte``."""


class TorchReservoirCore:
    """Fixed random reservoir implemented with Torch tensor operations.

    Parameters
    ----------
    input_dim : int
        Input feature dimension ``V``.
    n_internal_units : int
        Number of reservoir units ``D``.
    spectral_radius : float
        Target spectral radius for recurrent weights.
    leak : float
        Leak rate in ``(0, 1]``.
    connectivity : float
        Fraction of non-zero recurrent weights in ``(0, 1]``.
    input_scaling : float
        Scaling factor for input weights.
    seed : int
        RNG seed for fixed random weights.
    device : str
        Torch device string.
    batch_size : int
        Maximum sample count processed per state-computation batch.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        n_internal_units: int,
        spectral_radius: float,
        leak: float,
        connectivity: float,
        input_scaling: float,
        seed: int,
        device: str,
        batch_size: int,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if n_internal_units <= 0:
            raise ValueError("n_internal_units must be positive.")
        if spectral_radius <= 0.0:
            raise ValueError("spectral_radius must be positive.")
        if leak <= 0.0 or leak > 1.0:
            raise ValueError("leak must satisfy 0 < leak <= 1.")
        if connectivity <= 0.0 or connectivity > 1.0:
            raise ValueError("connectivity must satisfy 0 < connectivity <= 1.")
        if input_scaling <= 0.0:
            raise ValueError("input_scaling must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._input_dim: int = input_dim
        self._n_internal_units: int = n_internal_units
        self._leak: float = leak
        self._batch_size: int = batch_size
        self._device: torch.device = torch.device(device)

        rng = np.random.default_rng(seed)
        w_in = rng.choice(
            (-1.0, 1.0),
            size=(n_internal_units, input_dim),
        ).astype(np.float32)
        w_in *= input_scaling

        w_res = rng.normal(
            0.0,
            1.0,
            size=(n_internal_units, n_internal_units),
        ).astype(np.float32)
        mask = rng.random((n_internal_units, n_internal_units)) <= connectivity
        w_res = np.where(mask, w_res, 0.0)
        w_res_t = torch.from_numpy(w_res)
        radius = self._estimate_spectral_radius(w_res_t)
        scale = spectral_radius / max(radius, 1e-6)
        w_res_t = w_res_t * scale

        self.w_in: torch.Tensor = torch.from_numpy(w_in).to(self._device)
        self.w_res: torch.Tensor = w_res_t.to(self._device)

    @staticmethod
    def _estimate_spectral_radius(w: torch.Tensor, iters: int = 64) -> float:
        """Estimate spectral radius via power iteration."""
        v = torch.randn(w.shape[0], 1, dtype=w.dtype, device=w.device)
        for _ in range(iters):
            v = w @ v
            norm = torch.linalg.vector_norm(v)
            if float(norm) <= 1e-9:
                break
            v = v / norm
        rayleigh = (v.t() @ w @ v).item()
        return abs(float(rayleigh))

    def _compute_forward_states(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Compute forward reservoir states for one batch tensor."""
        batch_size, seq_len, _ = x_batch.shape
        state = torch.zeros(
            batch_size,
            self._n_internal_units,
            dtype=torch.float32,
            device=self._device,
        )
        states = torch.empty(
            batch_size,
            seq_len,
            self._n_internal_units,
            dtype=torch.float32,
            device=self._device,
        )
        leak = self._leak
        for step in range(seq_len):
            token_step = x_batch[:, step, :]
            pre = token_step @ self.w_in.t()
            pre = pre + (state @ self.w_res.t())
            nonlin = torch.tanh(pre)
            if leak >= 1.0:
                state = nonlin
            else:
                state = (1.0 - leak) * state + leak * nonlin
            states[:, step, :] = state
        return states

    def get_states(self, X: np.ndarray, n_drop: int, bidir: bool) -> np.ndarray:
        """Compute state tensor for ``X`` with optional bidirectional mode."""
        if X.ndim != 3:
            raise ValueError("X must have shape (N, T, V).")
        if n_drop < 0:
            raise ValueError("n_drop must be >= 0.")
        if X.shape[2] != self._input_dim:
            raise ValueError(
                f"Expected input_dim={self._input_dim}, got {X.shape[2]}."
            )
        if X.shape[0] == 0:
            n_steps = max(0, X.shape[1] - n_drop)
            n_features = self._n_internal_units * (2 if bidir else 1)
            return np.zeros((0, n_steps, n_features), dtype=np.float32)

        all_states: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self._batch_size):
                end = min(start + self._batch_size, X.shape[0])
                np_batch = np.asarray(
                    X[start:end],
                    dtype=np.float32,
                    order="C",
                )
                x_batch = torch.from_numpy(np_batch).to(self._device)
                fw_states = self._compute_forward_states(x_batch)
                if n_drop > 0:
                    fw_states = fw_states[:, n_drop:, :]

                if bidir:
                    rev_x = torch.flip(x_batch, dims=(1,))
                    bw_states = self._compute_forward_states(rev_x)
                    if n_drop > 0:
                        bw_states = bw_states[:, n_drop:, :]
                    bw_states = torch.flip(bw_states, dims=(1,))
                    states = torch.cat((fw_states, bw_states), dim=2)
                else:
                    states = fw_states

                all_states.append(states.cpu().numpy())

        return np.concatenate(all_states, axis=0)


class TorchRCModel:
    """Torch-based RC model with an API close to ``reservoir_computing.RC_model``."""

    def __init__(
        self,
        *,
        input_dim: int,
        n_internal_units: int,
        spectral_radius: float,
        leak: float,
        connectivity: float,
        input_scaling: float,
        n_drop: int,
        bidir: bool,
        dimred_method: Optional[str],
        n_dim: Optional[int],
        mts_rep: str,
        readout_type: str,
        w_ridge_embedding: float,
        w_ridge: float,
        mlp_layout: tuple[int, ...],
        num_epochs: int,
        w_l2: float,
        nonlinearity: str,
        svm_gamma: float,
        svm_C: float,
        seed: int,
        device: str,
        batch_size: int,
    ) -> None:
        if n_drop < 0:
            raise ValueError("n_drop must be >= 0.")
        if mts_rep not in MTS_REP_CHOICES:
            raise ValueError(
                "mts_rep must be one of: " + ", ".join(MTS_REP_CHOICES)
            )
        if readout_type not in READOUT_TYPE_CHOICES:
            raise ValueError(
                "readout_type must be one of: " + ", ".join(READOUT_TYPE_CHOICES)
            )

        self.n_drop: int = n_drop
        self.bidir: bool = bidir
        self.dimred_method: Optional[str] = dimred_method
        self.mts_rep: str = mts_rep
        self.readout_type: str = readout_type
        self.svm_gamma: float = svm_gamma
        self._batch_size: int = batch_size
        self._seed: int = seed
        self._device: str = device

        self._reservoir = TorchReservoirCore(
            input_dim=input_dim,
            n_internal_units=n_internal_units,
            spectral_radius=spectral_radius,
            leak=leak,
            connectivity=connectivity,
            input_scaling=input_scaling,
            seed=seed,
            device=device,
            batch_size=batch_size,
        )

        self._dim_red: Optional[IncrementalPCA]
        if dimred_method is not None:
            if n_dim is None:
                raise ValueError("n_dim must be provided when dimred is enabled.")
            self._dim_red = IncrementalPCA(
                n_components=n_dim,
                batch_size=max(256, batch_size),
            )
        else:
            self._dim_red = None

        if mts_rep in {"output", "reservoir"}:
            self._ridge_embedding = Ridge(
                alpha=max(w_ridge_embedding, 1e-8),
                fit_intercept=True,
            )
        else:
            self._ridge_embedding = None

        if readout_type == "lin":
            self.readout = Ridge(alpha=max(w_ridge, 1e-8))
        elif readout_type == "svm":
            self.readout = SVC(C=svm_C, kernel="precomputed")
        else:
            self.readout = MLPClassifier(
                hidden_layer_sizes=mlp_layout,
                activation=nonlinearity,
                alpha=max(w_l2, 1e-8),
                batch_size=max(32, batch_size),
                learning_rate="adaptive",
                learning_rate_init=0.001,
                max_iter=max(1, num_epochs),
                early_stopping=False,
                validation_fraction=0.0,
                random_state=seed,
            )

        self.input_repr_tr: Optional[np.ndarray] = None
        self.input_repr: Optional[np.ndarray] = None

    def _iter_batches(self, x_data: np.ndarray) -> list[np.ndarray]:
        """Split ``x_data`` into deterministic batch slices."""
        if x_data.ndim != 3:
            raise ValueError("x_data must have shape (N, T, V).")
        return [
            x_data[start : start + self._batch_size]
            for start in range(0, x_data.shape[0], self._batch_size)
        ]

    def _states_to_representation(
        self,
        *,
        states: np.ndarray,
        x_batch: np.ndarray,
    ) -> np.ndarray:
        """Convert state tensor to sequence representation for readout."""
        if self.mts_rep == "last":
            return states[:, -1, :]
        if self.mts_rep == "mean":
            return np.mean(states, axis=1)

        ridge_embedding = self._ridge_embedding
        if ridge_embedding is None:
            raise RuntimeError("ridge embedding is required for this mts_rep.")
        if states.shape[1] < 2:
            raise RuntimeError(
                "mts_rep output/reservoir requires >=2 steps after washout."
            )

        coeffs: list[np.ndarray] = []
        biases: list[np.ndarray] = []

        if self.mts_rep == "output":
            x_eval = x_batch
            if self.bidir:
                x_eval = np.concatenate((x_eval, x_eval[:, ::-1, :]), axis=2)
            for i in range(x_eval.shape[0]):
                ridge_embedding.fit(
                    states[i, 0:-1, :],
                    x_eval[i, self.n_drop + 1 :, :],
                )
                coeffs.append(cast(np.ndarray, ridge_embedding.coef_).ravel())
                biases.append(cast(np.ndarray, ridge_embedding.intercept_).ravel())
        else:
            for i in range(states.shape[0]):
                ridge_embedding.fit(states[i, 0:-1, :], states[i, 1:, :])
                coeffs.append(cast(np.ndarray, ridge_embedding.coef_).ravel())
                biases.append(cast(np.ndarray, ridge_embedding.intercept_).ravel())

        return np.concatenate((np.vstack(coeffs), np.vstack(biases)), axis=1)

    def _collect_representation(
        self,
        x_data: np.ndarray,
        fit_dimred: bool,
    ) -> np.ndarray:
        """Compute readout representation from input tensor batches."""
        batches = self._iter_batches(x_data)
        if not batches:
            return np.zeros((0, 1), dtype=np.float32)

        if self._dim_red is not None and fit_dimred:
            for x_batch in batches:
                states = self._reservoir.get_states(
                    x_batch,
                    n_drop=self.n_drop,
                    bidir=self.bidir,
                )
                flattened = states.reshape(-1, states.shape[2])
                self._dim_red.partial_fit(flattened)

        repr_parts: list[np.ndarray] = []
        for x_batch in batches:
            states = self._reservoir.get_states(
                x_batch,
                n_drop=self.n_drop,
                bidir=self.bidir,
            )
            if self._dim_red is not None:
                flattened = states.reshape(-1, states.shape[2])
                transformed = self._dim_red.transform(flattened)
                states = transformed.reshape(states.shape[0], states.shape[1], -1)
            repr_parts.append(
                self._states_to_representation(states=states, x_batch=x_batch)
            )
        return np.vstack(repr_parts)

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True) -> None:
        """Fit RC model on ``X`` and one-hot labels ``Y``."""
        time_start = time.time()
        input_repr = self._collect_representation(X, fit_dimred=True)

        if self.readout_type == "lin":
            self.readout.fit(input_repr, Y)
        elif self.readout_type == "svm":
            kernel_tr = squareform(pdist(input_repr, metric="sqeuclidean"))
            kernel_tr = np.exp(-self.svm_gamma * kernel_tr)
            labels = np.argmax(Y, axis=1)
            self.readout.fit(kernel_tr, labels)
            self.input_repr_tr = input_repr
        else:
            self.readout.fit(input_repr, Y)

        self.input_repr = input_repr
        if verbose:
            elapsed_min = (time.time() - time_start) / 60.0
            print(f"Training completed in {elapsed_min:.2f} min")

    def _predict_input_repr(self, Xte: np.ndarray) -> np.ndarray:
        """Compute representation for out-of-sample data."""
        return self._collect_representation(Xte, fit_dimred=False)

    def predict(self, Xte: np.ndarray) -> np.ndarray:
        """Predict class labels for test inputs."""
        input_repr_te = self._predict_input_repr(Xte)

        if self.readout_type == "lin":
            logits = self.readout.predict(input_repr_te)
            return np.argmax(logits, axis=1).astype(np.int64)
        if self.readout_type == "svm":
            if self.input_repr_tr is None:
                raise RuntimeError("SVM readout requires fitted input_repr_tr.")
            kernel_te = cdist(input_repr_te, self.input_repr_tr, metric="sqeuclidean")
            kernel_te = np.exp(-self.svm_gamma * kernel_te)
            pred = self.readout.predict(kernel_te)
            return np.asarray(pred, dtype=np.int64)

        pred = self.readout.predict(input_repr_te)
        if isinstance(pred, np.ndarray) and pred.ndim == 2:
            return np.argmax(pred, axis=1).astype(np.int64)
        return np.asarray(pred, dtype=np.int64)

    def predict_proba_pos(self, Xte: np.ndarray) -> np.ndarray:
        """Predict positive-class probabilities for test inputs."""
        input_repr_te = self._predict_input_repr(Xte)

        if self.readout_type == "lin":
            raw = np.asarray(self.readout.predict(input_repr_te), dtype=np.float64)
            return _positive_class_probability(raw)
        if self.readout_type == "mlp" and hasattr(self.readout, "predict_proba"):
            probs = np.asarray(self.readout.predict_proba(input_repr_te))
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1].astype(np.float64)
        if self.readout_type == "svm" and hasattr(self.readout, "decision_function"):
            if self.input_repr_tr is None:
                raise RuntimeError("SVM readout requires fitted input_repr_tr.")
            kernel_te = cdist(input_repr_te, self.input_repr_tr, metric="sqeuclidean")
            kernel_te = np.exp(-self.svm_gamma * kernel_te)
            decision = np.asarray(self.readout.decision_function(kernel_te))
            if decision.ndim == 1:
                return _sigmoid(decision)
            return _softmax(decision)[:, 1]

        labels = self.predict(Xte)
        return labels.astype(np.float64)


@dataclass(frozen=True)
class TaskTrainParams:
    """Resolved train-time hyperparameters for one task."""

    batch_size: int
    lr: float
    loss_name: str
    input_mode: str
    kmer_k: int
    max_tokens: str
    input_dim: int
    reservoir_size: int
    spectral_radius: float
    leak: float
    sparsity: float
    input_scale: float
    pooling: str
    mts_rep: str
    dimred_method: str
    n_dim: Optional[int]
    readout_type: str
    readout_hidden: int
    readout_dropout: float
    washout: int
    preroll_steps: int
    read_order: str
    weight_decay: float
    eta_min_ratio: float
    val_frac: float
    grad_clip: float
    pos_weight_cap: float
    focal_gamma: float
    focal_alpha_pos: Optional[float]
    asym_gamma_pos: float
    asym_gamma_neg: float
    asym_alpha_pos: Optional[float]


def build_kmer_vocab(kmer_k: int) -> dict[str, int]:
    """Build deterministic k-mer vocabulary with special tokens.

    Parameters
    ----------
    kmer_k : int
        K-mer size.

    Returns
    -------
    dict[str, int]
        Token-to-index mapping including ``[PAD]`` and ``[UNK]``.

    Raises
    ------
    ValueError
        If ``kmer_k <= 0``.
    """
    if kmer_k <= 0:
        raise ValueError("--kmer_k must be positive.")

    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    next_index = len(vocab)

    def _extend(prefix: str, depth: int) -> int:
        nonlocal next_index
        if depth == kmer_k:
            vocab[prefix] = next_index
            next_index += 1
            return next_index
        for base in DNA_BASES:
            _extend(prefix + base, depth + 1)
        return next_index

    _extend("", 0)
    return vocab


def build_onehot_vocab() -> dict[str, int]:
    """Build vocabulary for base-wise tokenization.

    Returns
    -------
    dict[str, int]
        Token-to-index mapping for ``[PAD]``, ``[UNK]``, ``A/C/G/T``.
    """
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for base in DNA_BASES:
        vocab[base] = len(vocab)
    return vocab


def kmerize(seq: str, kmer_k: int) -> list[str]:
    """Convert one DNA sequence into overlapping k-mers.

    Parameters
    ----------
    seq : str
        Input DNA sequence.
    kmer_k : int
        K-mer size.

    Returns
    -------
    list[str]
        Overlapping k-mer tokens.

    Raises
    ------
    ValueError
        If ``kmer_k <= 0``.
    """
    if kmer_k <= 0:
        raise ValueError("kmer_k must be positive.")
    upper = seq.upper()
    if len(upper) < kmer_k:
        return []
    return [upper[i : i + kmer_k] for i in range(0, len(upper) - kmer_k + 1)]


def _resolve_read_order(
    *,
    task: str,
    read_order: str,
    donor_read_order: Optional[str],
    acceptor_read_order: Optional[str],
) -> str:
    """Resolve read-order choice for one task.

    Parameters
    ----------
    task : str
        ``"donor"`` or ``"acceptor"``.
    read_order : str
        Global read order.
    donor_read_order : str | None
        Donor-specific override.
    acceptor_read_order : str | None
        Acceptor-specific override.

    Returns
    -------
    str
        Resolved order in ``{"forward", "reverse"}``.

    Raises
    ------
    ValueError
        If configuration is invalid.
    """
    if task not in {"donor", "acceptor"}:
        raise ValueError(f"Unsupported task: {task}")
    if task == "donor" and donor_read_order is not None:
        resolved = donor_read_order
    elif task == "acceptor" and acceptor_read_order is not None:
        resolved = acceptor_read_order
    else:
        resolved = read_order
    normalized = resolved.strip().lower()
    if normalized not in READ_ORDER_CHOICES:
        choices_text = ", ".join(READ_ORDER_CHOICES)
        raise ValueError(f"read_order must be one of: {choices_text}")
    if normalized == "auto":
        return "forward" if task == "donor" else "reverse"
    return normalized


def _apply_read_order(seq: str, read_order: str) -> str:
    """Apply sequence orientation transform.

    Parameters
    ----------
    seq : str
        Sequence string.
    read_order : str
        ``"forward"`` or ``"reverse"``.

    Returns
    -------
    str
        Ordered sequence.

    Raises
    ------
    ValueError
        If ``read_order`` is unsupported.
    """
    if read_order == "forward":
        return seq
    if read_order == "reverse":
        return seq[::-1]
    raise ValueError("read_order must be forward or reverse after resolution.")


def _resolve_max_tokens(
    raw: Union[str, int],
    *,
    input_mode: str,
    window_len: int,
    kmer_k: int,
) -> int:
    """Resolve fixed token length from ``auto`` or integer.

    Parameters
    ----------
    raw : str | int
        Raw CLI value.
    input_mode : str
        ``"onehot"`` or ``"kmer"``.
    window_len : int
        Effective donor/acceptor window length.
    kmer_k : int
        K-mer length used in ``kmer`` mode.

    Returns
    -------
    int
        Positive token length.

    Raises
    ------
    ValueError
        If value is invalid.
    """
    if window_len <= 0:
        raise ValueError("window_len must be positive.")
    if kmer_k <= 0:
        raise ValueError("kmer_k must be positive.")
    if input_mode == "onehot":
        auto_tokens = window_len
    else:
        auto_tokens = max(1, window_len - kmer_k + 1)

    if isinstance(raw, int):
        resolved = raw
    else:
        text = str(raw).strip().lower()
        if text == "auto":
            return auto_tokens
        try:
            resolved = int(text)
        except ValueError as exc:
            raise ValueError("--max_tokens must be 'auto' or integer > 0.") from exc

    if resolved <= 0:
        raise ValueError("--max_tokens must be > 0.")
    return resolved


def _resolve_mts_rep(pooling: str, mts_rep_arg: str) -> str:
    """Resolve MTS representation from legacy pooling and explicit override.

    Parameters
    ----------
    pooling : str
        Legacy pooling argument kept for script compatibility.
    mts_rep_arg : str
        New RC representation argument. ``"auto"`` maps from pooling.

    Returns
    -------
    str
        One of ``{"last", "mean", "output", "reservoir"}``.

    Raises
    ------
    ValueError
        If values are unsupported.
    """
    normalized_pooling = pooling.strip().lower()
    if normalized_pooling not in POOLING_CHOICES:
        choices_text = ", ".join(POOLING_CHOICES)
        raise ValueError(f"--pooling must be one of: {choices_text}")

    normalized_rep = mts_rep_arg.strip().lower()
    if normalized_rep not in MTS_REP_ARG_CHOICES:
        choices_text = ", ".join(MTS_REP_ARG_CHOICES)
        raise ValueError(f"--mts_rep must be one of: {choices_text}")

    if normalized_rep == "auto":
        return POOLING_TO_MTS_REP[normalized_pooling]
    return normalized_rep


def _resolve_dimred_method(dimred_method: str) -> Optional[str]:
    """Normalize dimensionality-reduction method.

    Parameters
    ----------
    dimred_method : str
        One of ``none|pca|tenpca``.

    Returns
    -------
    str | None
        ``None`` when no reduction is used.

    Raises
    ------
    ValueError
        If value is unsupported.
    """
    normalized = dimred_method.strip().lower()
    if normalized not in DIMRED_METHOD_CHOICES:
        choices_text = ", ".join(DIMRED_METHOD_CHOICES)
        raise ValueError(f"--dimred_method must be one of: {choices_text}")
    if normalized == "none":
        return None
    return normalized


def _resolve_feature_projection(
    *,
    feature_dim: int,
    input_dim: int,
    seed: int,
) -> Optional[np.ndarray]:
    """Build optional fixed random projection for token features.

    Parameters
    ----------
    feature_dim : int
        Original token one-hot dimension ``V``.
    input_dim : int
        Target projected dimension.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray | None
        Matrix with shape ``(feature_dim, input_dim)`` when projection is used,
        otherwise ``None``.

    Raises
    ------
    ValueError
        If dimensions are invalid.
    """
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    if input_dim <= 0:
        raise ValueError("--input_dim must be positive.")
    if input_dim == feature_dim:
        return None

    rng = np.random.default_rng(seed)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(float(feature_dim)),
        size=(feature_dim, input_dim),
    ).astype(np.float32)
    return projection


def _encode_sequence_to_mts(
    *,
    seq: str,
    vocab: Mapping[str, int],
    input_mode: str,
    kmer_k: int,
    max_tokens: int,
    read_order: str,
) -> np.ndarray:
    """Encode one DNA sequence into one time-series sample.

    Parameters
    ----------
    seq : str
        Input sequence.
    vocab : Mapping[str, int]
        Token vocabulary.
    input_mode : str
        ``onehot`` or ``kmer``.
    kmer_k : int
        K-mer size.
    max_tokens : int
        Fixed sequence length ``T`` after truncation/padding.
    read_order : str
        ``forward`` or ``reverse``.

    Returns
    -------
    np.ndarray
        Encoded sample with shape ``(T, V)`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If arguments are invalid.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    ordered = _apply_read_order(seq.upper(), read_order=read_order)
    if input_mode == "onehot":
        tokens = list(ordered)
    elif input_mode == "kmer":
        tokens = kmerize(ordered, kmer_k)
    else:
        raise ValueError("input_mode must be onehot|kmer.")

    unk_id = vocab["[UNK]"]
    token_ids = [vocab.get(token, unk_id) for token in tokens[:max_tokens]]

    feature_dim = len(vocab)
    encoded = np.zeros((max_tokens, feature_dim), dtype=np.float32)
    if token_ids:
        rows = np.arange(len(token_ids), dtype=np.int64)
        cols = np.asarray(token_ids, dtype=np.int64)
        encoded[rows, cols] = 1.0
    return encoded


def _encode_sequence_batch(
    *,
    sequences: Sequence[str],
    vocab: Mapping[str, int],
    input_mode: str,
    kmer_k: int,
    max_tokens: int,
    read_order: str,
    projection: Optional[np.ndarray],
) -> np.ndarray:
    """Encode sequence batch into ``[N, T, V]`` or projected ``[N, T, D]``.

    Parameters
    ----------
    sequences : Sequence[str]
        Input sequences of length ``N``.
    vocab : Mapping[str, int]
        Token vocabulary.
    input_mode : str
        ``onehot`` or ``kmer``.
    kmer_k : int
        K-mer size.
    max_tokens : int
        Fixed token length ``T``.
    read_order : str
        ``forward`` or ``reverse``.
    projection : np.ndarray | None
        Optional feature projection ``(V, D)``.

    Returns
    -------
    np.ndarray
        Batch tensor with shape ``(N, T, V|D)`` and dtype ``float32``.
    """
    if not sequences:
        feature_dim = len(vocab)
        output_dim = (
            projection.shape[1]
            if projection is not None
            else feature_dim
        )
        return np.zeros((0, max_tokens, output_dim), dtype=np.float32)

    encoded_samples = [
        _encode_sequence_to_mts(
            seq=seq,
            vocab=vocab,
            input_mode=input_mode,
            kmer_k=kmer_k,
            max_tokens=max_tokens,
            read_order=read_order,
        )
        for seq in sequences
    ]
    batch = np.stack(encoded_samples).astype(np.float32, copy=False)

    if projection is not None:
        batch = np.einsum("ntv,vd->ntd", batch, projection, optimize=True)
    return batch


def _labels_to_one_hot(labels: np.ndarray) -> np.ndarray:
    """Convert binary labels to one-hot matrix.

    Parameters
    ----------
    labels : np.ndarray
        Binary labels with shape ``(N,)``.

    Returns
    -------
    np.ndarray
        One-hot labels with shape ``(N, 2)``.

    Raises
    ------
    ValueError
        If labels are not binary.
    """
    if labels.ndim != 1:
        raise ValueError("labels must have shape (N,).")
    unique = np.unique(labels)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("labels must be binary in {0, 1}.")
    one_hot = np.zeros((labels.shape[0], 2), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels.astype(np.int64)] = 1.0
    return one_hot


def stratified_split(
    examples: Sequence[Tuple[str, int]],
    val_frac: float = 0.1,
    seed: int = 1337,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split examples into train/validation while preserving class ratio.

    Parameters
    ----------
    examples : Sequence[tuple[str, int]]
        Sequence/label pairs.
    val_frac : float, default=0.1
        Validation fraction in ``(0, 1)``.
    seed : int, default=1337
        RNG seed.

    Returns
    -------
    tuple[list[tuple[str, int]], list[tuple[str, int]]]
        ``(train_examples, val_examples)``.

    Raises
    ------
    ValueError
        If ``val_frac`` is outside ``(0, 1)``.
    """
    if val_frac <= 0.0 or val_frac >= 1.0:
        raise ValueError("val_frac must satisfy 0 < val_frac < 1.")

    rng = random.Random(seed)
    positives = [(seq, label) for seq, label in examples if label == 1]
    negatives = [(seq, label) for seq, label in examples if label == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_val_pos = max(1, int(len(positives) * val_frac))
    n_val_neg = max(1, int(len(negatives) * val_frac))
    n_val_pos = min(n_val_pos, len(positives) - 1)
    n_val_neg = min(n_val_neg, len(negatives) - 1)

    train = positives[n_val_pos:] + negatives[n_val_neg:]
    val = positives[:n_val_pos] + negatives[:n_val_neg]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _detect_system_total_memory_bytes() -> Optional[int]:
    """Detect total system memory in bytes.

    Returns
    -------
    int | None
        Total system memory in bytes when detection succeeds, otherwise
        ``None``.
    """
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = pages * page_size
        if total > 0:
            return total
    except (AttributeError, OSError, ValueError):
        pass

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        if kb > 0:
                            return kb * 1024
                    break
    except (OSError, ValueError):
        pass

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        total = int(result.stdout.strip())
        if total > 0:
            return total
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass

    return None


def _auto_state_budget_from_total_bytes(total_bytes: int) -> float:
    """Compute auto RC state budget in GiB from total memory size.

    Parameters
    ----------
    total_bytes : int
        Total system memory in bytes.

    Returns
    -------
    float
        Auto-selected memory budget in GiB.
    """
    total_gib = float(total_bytes) / float(1024**3)
    proposed = total_gib * AUTO_STATE_BUDGET_FRACTION
    bounded = max(AUTO_STATE_BUDGET_MIN_GB, proposed)
    return min(AUTO_STATE_BUDGET_MAX_GB, bounded)


def _resolve_state_budget_gb() -> float:
    """Resolve memory budget in GiB used for RC state allocation.

    Returns
    -------
    float
        Budget in GiB. ``<= 0`` disables automatic sample capping.
    """
    raw = os.environ.get(RC_STATE_BUDGET_ENV, "").strip()
    if raw == "" or raw.lower() == "auto":
        detected_total = _detect_system_total_memory_bytes()
        if detected_total is None:
            print(
                "[reservoir] failed to detect system RAM; "
                f"using default {DEFAULT_RC_STATE_BUDGET_GB:g} GiB for "
                f"{RC_STATE_BUDGET_ENV}."
            )
            return DEFAULT_RC_STATE_BUDGET_GB
        auto_budget = _auto_state_budget_from_total_bytes(detected_total)
        total_gib = float(detected_total) / float(1024**3)
        print(
            f"[reservoir] {RC_STATE_BUDGET_ENV}=auto -> "
            f"{auto_budget:.2f} GiB (total={total_gib:.2f} GiB)"
        )
        return auto_budget
    try:
        parsed = float(raw)
    except ValueError:
        print(
            "[reservoir] invalid "
            f"{RC_STATE_BUDGET_ENV}={raw!r}; "
            f"using default {DEFAULT_RC_STATE_BUDGET_GB:g} GiB."
        )
        return DEFAULT_RC_STATE_BUDGET_GB
    return parsed


def _estimate_state_bytes(
    *,
    num_samples: int,
    num_steps: int,
    reservoir_size: int,
) -> int:
    """Estimate bytes for one RC internal state tensor.

    Parameters
    ----------
    num_samples : int
        Number of samples ``N``.
    num_steps : int
        Effective number of time steps ``T`` after washout.
    reservoir_size : int
        Number of internal units ``D``.

    Returns
    -------
    int
        Estimated bytes for array shape ``(N, T, D)`` in float32.
    """
    if num_samples <= 0 or num_steps <= 0 or reservoir_size <= 0:
        return 0
    return (
        int(num_samples)
        * int(num_steps)
        * int(reservoir_size)
        * RESERVOIR_STATE_DTYPE_BYTES
    )


def _bytes_to_gib(num_bytes: int) -> float:
    """Convert bytes to GiB."""
    return float(num_bytes) / float(1024**3)


def _stratified_subsample_examples(
    examples: Sequence[Tuple[str, int]],
    max_samples: int,
    seed: int,
) -> list[Tuple[str, int]]:
    """Subsample examples while preserving binary class balance.

    Parameters
    ----------
    examples : Sequence[tuple[str, int]]
        Sequence/label pairs.
    max_samples : int
        Upper bound for selected samples.
    seed : int
        RNG seed.

    Returns
    -------
    list[tuple[str, int]]
        Subsampled examples.
    """
    if max_samples <= 0 or len(examples) <= max_samples:
        return list(examples)

    rng = random.Random(seed)
    positives = [(seq, label) for seq, label in examples if label == 1]
    negatives = [(seq, label) for seq, label in examples if label == 0]
    if not positives or not negatives:
        selected = list(examples)
        rng.shuffle(selected)
        return selected[:max_samples]

    total = len(positives) + len(negatives)
    target_pos = int(round(max_samples * (len(positives) / total)))
    target_pos = max(1, min(len(positives), target_pos))
    target_neg = max_samples - target_pos
    target_neg = max(1, min(len(negatives), target_neg))

    while (target_pos + target_neg) > max_samples:
        if target_pos >= target_neg and target_pos > 1:
            target_pos -= 1
        elif target_neg > 1:
            target_neg -= 1
        else:
            break

    while (target_pos + target_neg) < max_samples:
        if target_pos < len(positives):
            target_pos += 1
            continue
        if target_neg < len(negatives):
            target_neg += 1
            continue
        break

    selected_pos = rng.sample(positives, target_pos)
    selected_neg = rng.sample(negatives, target_neg)
    selected = selected_pos + selected_neg
    rng.shuffle(selected)
    return selected


def _cap_examples_for_state_budget(
    *,
    task: str,
    split_name: str,
    examples: Sequence[Tuple[str, int]],
    max_tokens: int,
    washout: int,
    reservoir_size: int,
    budget_gib: float,
    seed: int,
) -> tuple[list[Tuple[str, int]], dict[str, float | int]]:
    """Cap sample count to fit RC state-memory budget.

    Parameters
    ----------
    task : str
        Task name for logging.
    split_name : str
        Split label, e.g., ``train`` or ``val``.
    examples : Sequence[tuple[str, int]]
        Candidate examples.
    max_tokens : int
        Sequence length used for encoding.
    washout : int
        Dropped initial steps.
    reservoir_size : int
        Number of internal units.
    budget_gib : float
        State-memory budget in GiB. ``<= 0`` disables capping.
    seed : int
        RNG seed.

    Returns
    -------
    tuple[list[tuple[str, int]], dict[str, float | int]]
        ``(possibly_capped_examples, diagnostics)``.
    """
    effective_steps = max(1, max_tokens - washout)
    before_count = len(examples)
    before_bytes = _estimate_state_bytes(
        num_samples=before_count,
        num_steps=effective_steps,
        reservoir_size=reservoir_size,
    )

    diagnostics: dict[str, float | int] = {
        "before_count": before_count,
        "after_count": before_count,
        "effective_steps": effective_steps,
        "estimated_gib_before": _bytes_to_gib(before_bytes),
        "estimated_gib_after": _bytes_to_gib(before_bytes),
    }

    if budget_gib <= 0.0 or before_count == 0:
        return list(examples), diagnostics

    budget_bytes = int(budget_gib * (1024**3))
    bytes_per_sample = _estimate_state_bytes(
        num_samples=1,
        num_steps=effective_steps,
        reservoir_size=reservoir_size,
    )
    if bytes_per_sample <= 0:
        return list(examples), diagnostics

    max_samples = int(budget_bytes / (bytes_per_sample * STATE_MEMORY_HEADROOM_FACTOR))
    max_samples = max(2, max_samples)
    if before_count <= max_samples:
        return list(examples), diagnostics

    capped = _stratified_subsample_examples(
        examples=examples,
        max_samples=max_samples,
        seed=seed,
    )
    after_count = len(capped)
    after_bytes = _estimate_state_bytes(
        num_samples=after_count,
        num_steps=effective_steps,
        reservoir_size=reservoir_size,
    )
    diagnostics["after_count"] = after_count
    diagnostics["estimated_gib_after"] = _bytes_to_gib(after_bytes)
    print(
        f"[{task}] {split_name} capped by {RC_STATE_BUDGET_ENV}="
        f"{budget_gib:g}GiB: {before_count} -> {after_count} "
        f"(state_est {diagnostics['estimated_gib_before']:.2f}GiB -> "
        f"{diagnostics['estimated_gib_after']:.2f}GiB)"
    )
    return capped, diagnostics


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax for ``(N, C)`` logits."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return exp / denom


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute logistic sigmoid elementwise."""
    return 1.0 / (1.0 + np.exp(-values))


def _positive_class_probability(raw_scores: np.ndarray) -> np.ndarray:
    """Convert raw readout outputs to positive-class probability.

    Parameters
    ----------
    raw_scores : np.ndarray
        Readout output with shape ``(N,)``, ``(N, 1)``, or ``(N, C>=2)``.

    Returns
    -------
    np.ndarray
        Positive-class probability with shape ``(N,)``.
    """
    if raw_scores.ndim == 1:
        return _sigmoid(raw_scores)
    if raw_scores.ndim != 2:
        flattened = raw_scores.reshape(raw_scores.shape[0], -1)
        return _positive_class_probability(flattened)
    if raw_scores.shape[1] == 1:
        return _sigmoid(raw_scores[:, 0])
    probs = _softmax(raw_scores)
    return probs[:, 1]


def _compute_input_representation(
    model: ReservoirClassifierProtocol,
    x_data: np.ndarray,
) -> np.ndarray:
    """Rebuild RC representation used by the readout for one batch.

    Parameters
    ----------
    model : ReservoirClassifierProtocol
        Fitted RC model.
    x_data : np.ndarray
        Input batch with shape ``(N, T, V)``.

    Returns
    -------
    np.ndarray
        Readout input representation with shape ``(N, D)``.

    Raises
    ------
    RuntimeError
        If model representation configuration is unsupported.
    """
    reservoir_obj = getattr(model, "_reservoir", None)
    if reservoir_obj is None or not hasattr(reservoir_obj, "get_states"):
        raise RuntimeError("RC model does not expose reservoir states API.")

    get_states = getattr(reservoir_obj, "get_states")
    res_states = cast(
        np.ndarray,
        get_states(x_data, n_drop=model.n_drop, bidir=model.bidir),
    )

    dimred_method = model.dimred_method
    if dimred_method is not None:
        normalized_dimred = dimred_method.lower()
        dimred_obj = getattr(model, "_dim_red", None)
        if dimred_obj is None:
            raise RuntimeError("dimred_method is set but dimred object is missing.")
        if normalized_dimred in {"pca", "tenpca"}:
            n_samples = res_states.shape[0]
            reshaped = res_states.reshape(-1, res_states.shape[2])
            transformed = cast(np.ndarray, dimred_obj.transform(reshaped))
            red_states = transformed.reshape(n_samples, -1, transformed.shape[1])
        else:
            raise RuntimeError(
                f"Unsupported dimred method for inference: {dimred_method}"
            )
    else:
        red_states = res_states

    mts_rep = model.mts_rep
    if mts_rep == "last":
        return red_states[:, -1, :]
    if mts_rep == "mean":
        return np.mean(red_states, axis=1)

    if mts_rep == "output":
        if red_states.shape[1] < 2:
            raise RuntimeError(
                "mts_rep='output' requires at least 2 valid time steps after washout."
            )
        x_eval = x_data
        if model.bidir:
            x_eval = np.concatenate((x_eval, x_eval[:, ::-1, :]), axis=2)
        ridge_embedding = getattr(model, "_ridge_embedding", None)
        if ridge_embedding is None:
            raise RuntimeError("Missing ridge embedding model for mts_rep='output'.")
        coeffs: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for i in range(x_eval.shape[0]):
            ridge_embedding.fit(
                red_states[i, 0:-1, :],
                x_eval[i, model.n_drop + 1 :, :],
            )
            coeffs.append(cast(np.ndarray, ridge_embedding.coef_).ravel())
            biases.append(cast(np.ndarray, ridge_embedding.intercept_).ravel())
        return np.concatenate((np.vstack(coeffs), np.vstack(biases)), axis=1)

    if mts_rep == "reservoir":
        if red_states.shape[1] < 2:
            raise RuntimeError(
                "mts_rep='reservoir' requires at least 2 valid states after washout."
            )
        ridge_embedding = getattr(model, "_ridge_embedding", None)
        if ridge_embedding is None:
            raise RuntimeError("Missing ridge embedding model for mts_rep='reservoir'.")
        coeffs = []
        biases = []
        for i in range(red_states.shape[0]):
            ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
            coeffs.append(cast(np.ndarray, ridge_embedding.coef_).ravel())
            biases.append(cast(np.ndarray, ridge_embedding.intercept_).ravel())
        return np.concatenate((np.vstack(coeffs), np.vstack(biases)), axis=1)

    raise RuntimeError(f"Unsupported mts_rep: {mts_rep}")


def _predict_labels_and_probs(
    *,
    model: ReservoirClassifierProtocol,
    x_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict class labels and positive-class probabilities.

    Parameters
    ----------
    model : ReservoirClassifierProtocol
        Fitted RC model.
    x_data : np.ndarray
        Input batch with shape ``(N, T, V)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(pred_labels, pred_probs)`` where both have shape ``(N,)``.
    """
    pred_labels = np.asarray(model.predict(x_data), dtype=np.int64)
    pred_probs = pred_labels.astype(np.float64)

    if hasattr(model, "predict_proba_pos"):
        try:
            raw_probs = getattr(model, "predict_proba_pos")(x_data)
            pred_probs = np.asarray(raw_probs, dtype=np.float64).reshape(-1)
        except Exception:
            pred_probs = pred_labels.astype(np.float64)

    if pred_probs.shape[0] == pred_labels.shape[0]:
        pred_probs = np.clip(pred_probs, 1e-7, 1.0 - 1e-7)
        return pred_labels, pred_probs

    readout_obj = getattr(model, "readout", None)
    if readout_obj is None or not hasattr(readout_obj, "predict"):
        return pred_labels, pred_probs

    try:
        input_repr = _compute_input_representation(model=model, x_data=x_data)
        raw_scores = np.asarray(readout_obj.predict(input_repr), dtype=np.float64)
        pred_probs = _positive_class_probability(raw_scores)
    except Exception:
        pred_probs = pred_labels.astype(np.float64)

    if pred_probs.shape[0] != pred_labels.shape[0]:
        pred_probs = pred_labels.astype(np.float64)

    pred_probs = np.clip(pred_probs, 1e-7, 1.0 - 1e-7)
    return pred_labels, pred_probs


def _evaluate_binary_predictions(
    labels: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    """Evaluate binary predictions from probabilities.

    Parameters
    ----------
    labels : np.ndarray
        Ground-truth labels with shape ``(N,)``.
    probs : np.ndarray
        Positive-class probabilities with shape ``(N,)``.

    Returns
    -------
    dict[str, float]
        Available metrics among ``acc@0.5``, ``max_f1``, ``roc_auc``,
        ``pr_auc``.
    """
    labels_i = labels.astype(np.int32)
    probs_f = np.clip(probs.astype(np.float64), 1e-7, 1.0 - 1e-7)

    metrics: dict[str, float] = {}
    if labels_i.size == 0:
        return metrics

    metrics["acc@0.5"] = float(np.mean((probs_f >= 0.5) == (labels_i == 1)))
    max_f1_value: Optional[float] = None
    try:
        max_f1_value = _fallback_max_f1(labels_i, probs_f)
    except ValueError:
        max_f1_value = None
    if max_f1_value is not None:
        metrics["max_f1"] = max_f1_value
    if len(np.unique(labels_i)) <= 1:
        return metrics

    roc_auc_value: Optional[float] = None
    if roc_auc_score is not None:
        try:
            roc_auc_value = float(roc_auc_score(labels_i, probs_f))
        except Exception:
            roc_auc_value = None
    if roc_auc_value is None:
        try:
            roc_auc_value = _fallback_roc_auc(labels_i, probs_f)
        except ValueError:
            roc_auc_value = None
    if roc_auc_value is not None:
        metrics["roc_auc"] = roc_auc_value

    pr_auc_value: Optional[float] = None
    if average_precision_score is not None:
        try:
            pr_auc_value = float(average_precision_score(labels_i, probs_f))
        except Exception:
            pr_auc_value = None
    if pr_auc_value is None:
        try:
            pr_auc_value = _fallback_average_precision(labels_i, probs_f)
        except ValueError:
            pr_auc_value = None
    if pr_auc_value is not None:
        metrics["pr_auc"] = pr_auc_value

    return metrics


def _resolve_runtime_device(device: str) -> str:
    """Resolve runtime device for Torch reservoir state computation.

    Parameters
    ----------
    device : str
        Requested device argument from CLI.

    Returns
    -------
    str
        One of ``cpu|cuda|mps``.
    """
    requested = device.strip().lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("--device must be one of: auto, cuda, mps, cpu.")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return "mps"

    return "cpu"


def _resolve_n_dim(n_dim: Optional[int], reservoir_size: int) -> Optional[int]:
    """Resolve dimensionality-reduction output size.

    Parameters
    ----------
    n_dim : int | None
        User-provided target dimension.
    reservoir_size : int
        Reservoir state dimension.

    Returns
    -------
    int | None
        Effective reduced dimension.

    Raises
    ------
    ValueError
        If value is invalid.
    """
    if n_dim is None:
        return min(128, reservoir_size)
    if n_dim <= 0:
        raise ValueError("--n_dim must be positive when provided.")
    if n_dim > reservoir_size:
        raise ValueError("--n_dim must be <= --reservoir_size.")
    return n_dim


def _build_rc_model(
    *,
    input_dim: int,
    reservoir_size: int,
    spectral_radius: float,
    leak: float,
    sparsity: float,
    input_scale: float,
    washout: int,
    mts_rep: str,
    dimred_method: Optional[str],
    n_dim: Optional[int],
    readout_type: str,
    readout_hidden: int,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: str,
    batch_size: int,
) -> ReservoirClassifierProtocol:
    """Instantiate one Torch-based RC classifier.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    reservoir_size : int
        Number of reservoir units.
    spectral_radius : float
        Reservoir spectral radius.
    leak : float
        Leak rate.
    sparsity : float
        Reservoir connectivity.
    input_scale : float
        Input scaling.
    washout : int
        Number of transient steps dropped.
    mts_rep : str
        MTS representation type.
    dimred_method : str | None
        Optional dimensionality reduction method.
    n_dim : int | None
        Target reduced dimension.
    readout_type : str
        Readout type.
    readout_hidden : int
        Hidden units used for ``mlp`` readout.
    weight_decay : float
        Regularization used as ridge/MLP L2 weight.
    epochs : int
        Epoch-like budget used to derive MLP iterations.
    seed : int
        Random seed.
    device : str
        Torch device used for reservoir state computation.
    batch_size : int
        Reservoir state-computation batch size.

    Returns
    -------
    ReservoirClassifierProtocol
        Instantiated RC model object.
    """
    mlp_epochs = max(50, epochs * 25)
    mlp_layout = (max(1, readout_hidden),)
    model_obj = TorchRCModel(
        input_dim=input_dim,
        n_internal_units=reservoir_size,
        spectral_radius=spectral_radius,
        leak=leak,
        connectivity=sparsity,
        input_scaling=input_scale,
        n_drop=washout,
        bidir=False,
        dimred_method=dimred_method,
        n_dim=n_dim,
        mts_rep=mts_rep,
        readout_type=readout_type,
        w_ridge_embedding=max(weight_decay, 1e-8),
        w_ridge=max(weight_decay, 1e-8),
        mlp_layout=mlp_layout,
        num_epochs=mlp_epochs,
        w_l2=max(weight_decay, 1e-8),
        nonlinearity="relu",
        svm_gamma=1.0,
        svm_C=1.0,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    return cast(ReservoirClassifierProtocol, model_obj)


def _resolve_task_train_params(
    *,
    task: str,
    model_args: argparse.Namespace,
) -> TaskTrainParams:
    """Resolve task-specific training parameters with fallback logic.

    Parameters
    ----------
    task : str
        ``"donor"`` or ``"acceptor"``.
    model_args : argparse.Namespace
        Parsed model arguments.

    Returns
    -------
    TaskTrainParams
        Resolved hyperparameters for one task.

    Raises
    ------
    ValueError
        If task is unsupported.
    """
    if task not in {"donor", "acceptor"}:
        raise ValueError(f"Unsupported task: {task}")

    prefix = f"{task}_"

    def _override_or_default(name: str, default: object) -> object:
        override = getattr(model_args, f"{prefix}{name}", None)
        return default if override is None else override

    return TaskTrainParams(
        batch_size=int(_override_or_default("batch_size", model_args.batch_size)),
        lr=float(_override_or_default("lr", model_args.lr)),
        loss_name=str(_override_or_default("loss", model_args.loss)),
        input_mode=str(_override_or_default("input_mode", model_args.input_mode)),
        kmer_k=int(_override_or_default("kmer_k", model_args.kmer_k)),
        max_tokens=str(_override_or_default("max_tokens", model_args.max_tokens)),
        input_dim=int(_override_or_default("input_dim", model_args.input_dim)),
        reservoir_size=int(
            _override_or_default("reservoir_size", model_args.reservoir_size)
        ),
        spectral_radius=float(
            _override_or_default("spectral_radius", model_args.spectral_radius)
        ),
        leak=float(_override_or_default("leak", model_args.leak)),
        sparsity=float(_override_or_default("sparsity", model_args.sparsity)),
        input_scale=float(_override_or_default("input_scale", model_args.input_scale)),
        pooling=str(_override_or_default("pooling", model_args.pooling)),
        mts_rep=str(_override_or_default("mts_rep", model_args.mts_rep)),
        dimred_method=str(
            _override_or_default("dimred_method", model_args.dimred_method)
        ),
        n_dim=cast(
            Optional[int],
            _override_or_default("n_dim", model_args.n_dim),
        ),
        readout_type=str(
            _override_or_default("readout_type", model_args.readout_type)
        ),
        readout_hidden=int(
            _override_or_default("readout_hidden", model_args.readout_hidden)
        ),
        readout_dropout=float(
            _override_or_default("readout_dropout", model_args.readout_dropout)
        ),
        washout=int(_override_or_default("washout", model_args.washout)),
        preroll_steps=int(
            _override_or_default("preroll_steps", model_args.preroll_steps)
        ),
        read_order=_resolve_read_order(
            task=task,
            read_order=str(_override_or_default("read_order", model_args.read_order)),
            donor_read_order=getattr(model_args, "donor_read_order", None),
            acceptor_read_order=getattr(model_args, "acceptor_read_order", None),
        ),
        weight_decay=float(
            _override_or_default("weight_decay", model_args.weight_decay)
        ),
        eta_min_ratio=float(
            _override_or_default("eta_min_ratio", model_args.eta_min_ratio)
        ),
        val_frac=float(_override_or_default("val_frac", model_args.val_frac)),
        grad_clip=float(_override_or_default("grad_clip", model_args.grad_clip)),
        pos_weight_cap=float(
            _override_or_default("pos_weight_cap", model_args.pos_weight_cap)
        ),
        focal_gamma=float(_override_or_default("focal_gamma", model_args.focal_gamma)),
        focal_alpha_pos=cast(
            Optional[float],
            _override_or_default("focal_alpha_pos", model_args.focal_alpha_pos),
        ),
        asym_gamma_pos=float(
            _override_or_default("asym_gamma_pos", model_args.asym_gamma_pos)
        ),
        asym_gamma_neg=float(
            _override_or_default("asym_gamma_neg", model_args.asym_gamma_neg)
        ),
        asym_alpha_pos=cast(
            Optional[float],
            _override_or_default("asym_alpha_pos", model_args.asym_alpha_pos),
        ),
    )


def train_task_model(
    task: str,
    pos_path: str,
    neg_path: str,
    checkpoint_path: str,
    window_len: int,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    epochs: int = 20,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    batch_size: int = 256,
    lr: float = 5e-4,
    seed: int = 1337,
    input_mode: str = "onehot",
    kmer_k: int = 3,
    max_tokens: Union[str, int] = "auto",
    input_dim: int = 128,
    reservoir_size: int = 1024,
    spectral_radius: float = 0.95,
    leak: float = 0.3,
    sparsity: float = 0.1,
    input_scale: float = 0.5,
    pooling: str = "mean_max",
    mts_rep: str = "auto",
    dimred_method: str = "none",
    n_dim: Optional[int] = None,
    readout_type: str = "lin",
    readout_hidden: int = 256,
    readout_dropout: float = 0.2,
    washout: int = 0,
    preroll_steps: int = 0,
    read_order: str = "forward",
    weight_decay: float = 0.01,
    eta_min_ratio: float = 0.01,
    val_frac: float = 0.1,
    grad_clip: float = 1.0,
    compile_model: bool = False,
    compile_mode: str = "auto",
    device: str = "auto",
    loss_name: str = "weighted_bce",
    pos_weight_cap: float = 20.0,
    focal_gamma: float = 2.0,
    focal_alpha_pos: Optional[float] = None,
    asym_gamma_pos: float = 0.0,
    asym_gamma_neg: float = 4.0,
    asym_alpha_pos: Optional[float] = None,
    use_amp: Union[bool, int] = 1,
    amp_dtype: str = "auto",
    allow_tf32: Union[bool, int] = 1,
    cudnn_benchmark: Union[bool, int] = 1,
    deterministic: Union[bool, int] = 0,
    num_workers: Union[str, int] = "auto",
    prefetch_factor: int = 4,
    persistent_workers: Union[bool, int] = 1,
    pin_memory: Union[bool, int] = 1,
    min_batch_size: int = 64,
    max_oom_retries: int = 8,
    quick_phase: bool = False,
    gpu_id: Optional[int] = None,
) -> Dict[str, object]:
    """Train one donor/acceptor RC model.

    Parameters
    ----------
    task : str
        Target task in ``{"donor", "acceptor"}``.
    pos_path : str
        Positive training file path.
    neg_path : str
        Negative training file path.
    checkpoint_path : str
        Output checkpoint path.
    window_len : int
        Effective sequence window length.
    donor_len : int | None
        Donor window argument used by shared parser.
    acceptor_len : int | None
        Acceptor window argument used by shared parser.

    Returns
    -------
    dict[str, object]
        Training summary for this task.

    Raises
    ------
    ValueError
        If configuration or data is invalid.
    RuntimeError
        If required dependency is missing.

    Notes
    -----
    Compatibility runtime arguments (compile/amp/loader flags) are accepted to
    keep wrapper scripts stable. Reservoir state computation uses the resolved
    Torch device, while readout fitting/prediction runs via scikit-learn.
    """
    if task not in {"donor", "acceptor"}:
        raise ValueError("task must be donor or acceptor.")
    if window_len <= 0:
        raise ValueError("window_len must be positive.")
    if epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if input_mode not in INPUT_MODE_CHOICES:
        choices_text = ", ".join(INPUT_MODE_CHOICES)
        raise ValueError(f"input_mode must be one of: {choices_text}")
    if kmer_k <= 0:
        raise ValueError("--kmer_k must be positive.")
    if input_dim <= 0:
        raise ValueError("--input_dim must be positive.")
    if reservoir_size <= 0:
        raise ValueError("--reservoir_size must be positive.")
    if spectral_radius <= 0.0:
        raise ValueError("--spectral_radius must be positive.")
    if leak <= 0.0 or leak > 1.0:
        raise ValueError("--leak must satisfy 0 < leak <= 1.")
    if sparsity <= 0.0 or sparsity > 1.0:
        raise ValueError("--sparsity must satisfy 0 < sparsity <= 1.")
    if input_scale <= 0.0:
        raise ValueError("--input_scale must be positive.")
    if weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative.")
    if val_frac <= 0.0 or val_frac >= 1.0:
        raise ValueError("--val_frac must satisfy 0 < val_frac < 1.")
    if washout < 0:
        raise ValueError("--washout must be >= 0.")
    if readout_hidden <= 0:
        raise ValueError("--readout_hidden must be positive.")

    normalized_readout = readout_type.strip().lower()
    if normalized_readout not in READOUT_TYPE_CHOICES:
        choices_text = ", ".join(READOUT_TYPE_CHOICES)
        raise ValueError(f"--readout_type must be one of: {choices_text}")

    runtime_device = _resolve_runtime_device(device)
    deterministic_bool = _bool_from_flag(deterministic)
    cudnn_benchmark_bool = _bool_from_flag(cudnn_benchmark)
    allow_tf32_bool = _bool_from_flag(allow_tf32)

    _seed_everything(seed=seed)

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    examples = read_examples_single_task(
        pos_path,
        neg_path,
        task,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )

    n_pos = sum(label for _, label in examples)
    n_neg = len(examples) - n_pos
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Insufficient training examples for {task}: pos={n_pos}, neg={n_neg}."
        )

    train_ex, val_ex = stratified_split(examples, val_frac=val_frac, seed=seed)
    _ = apply_eta_process_title_placeholder()
    task_started_at = time.perf_counter()
    print(
        f"[{task}] backend=torch-reservoir device={runtime_device} "
        f"total={len(examples)} (pos={n_pos}, neg={n_neg}) "
        f"train={len(train_ex)} val={len(val_ex)}"
    )

    max_tokens_effective = _resolve_max_tokens(
        max_tokens,
        input_mode=input_mode,
        window_len=window_len,
        kmer_k=kmer_k,
    )
    if washout >= max_tokens_effective:
        raise ValueError("--washout must be smaller than effective max_tokens.")

    resolved_mts_rep = _resolve_mts_rep(pooling=pooling, mts_rep_arg=mts_rep)
    normalized_dimred = _resolve_dimred_method(dimred_method)
    resolved_n_dim: Optional[int]
    if normalized_dimred is None:
        resolved_n_dim = None
    else:
        resolved_n_dim = _resolve_n_dim(n_dim=n_dim, reservoir_size=reservoir_size)

    if resolved_mts_rep in {"output", "reservoir"}:
        available_steps = max_tokens_effective - washout
        if available_steps < 2:
            raise ValueError(
                "mts_rep output/reservoir requires at least 2 time steps after washout."
            )

    state_budget_gib = _resolve_state_budget_gb()
    train_ex_used, train_mem_diag = _cap_examples_for_state_budget(
        task=task,
        split_name="train",
        examples=train_ex,
        max_tokens=max_tokens_effective,
        washout=washout,
        reservoir_size=reservoir_size,
        budget_gib=state_budget_gib,
        seed=seed,
    )
    val_ex_used, val_mem_diag = _cap_examples_for_state_budget(
        task=task,
        split_name="val",
        examples=val_ex,
        max_tokens=max_tokens_effective,
        washout=washout,
        reservoir_size=reservoir_size,
        budget_gib=state_budget_gib,
        seed=seed + 1,
    )
    train_ex = train_ex_used
    val_ex = val_ex_used

    train_pos_after_cap = sum(label for _, label in train_ex)
    train_neg_after_cap = len(train_ex) - train_pos_after_cap
    if train_pos_after_cap < 1 or train_neg_after_cap < 1:
        raise RuntimeError(
            "State-budget capping left only one class in training split. "
            "Increase INTRONMODEL_RC_STATE_BUDGET_GB or reduce "
            "--reservoir_size/--max_tokens."
        )

    vocab = build_onehot_vocab() if input_mode == "onehot" else build_kmer_vocab(kmer_k)
    projection = _resolve_feature_projection(
        feature_dim=len(vocab),
        input_dim=input_dim,
        seed=seed,
    )

    train_sequences = [seq for seq, _ in train_ex]
    val_sequences = [seq for seq, _ in val_ex]
    train_labels = np.asarray([label for _, label in train_ex], dtype=np.int64)
    val_labels = np.asarray([label for _, label in val_ex], dtype=np.int64)

    x_train = _encode_sequence_batch(
        sequences=train_sequences,
        vocab=vocab,
        input_mode=input_mode,
        kmer_k=kmer_k,
        max_tokens=max_tokens_effective,
        read_order=read_order,
        projection=projection,
    )
    x_val = _encode_sequence_batch(
        sequences=val_sequences,
        vocab=vocab,
        input_mode=input_mode,
        kmer_k=kmer_k,
        max_tokens=max_tokens_effective,
        read_order=read_order,
        projection=projection,
    )
    y_train = _labels_to_one_hot(train_labels)

    model = _build_rc_model(
        input_dim=int(x_train.shape[2]),
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        leak=leak,
        sparsity=sparsity,
        input_scale=input_scale,
        washout=washout,
        mts_rep=resolved_mts_rep,
        dimred_method=normalized_dimred,
        n_dim=resolved_n_dim,
        readout_type=normalized_readout,
        readout_hidden=readout_hidden,
        weight_decay=weight_decay,
        epochs=epochs,
        seed=seed,
        device=runtime_device,
        batch_size=batch_size,
    )

    try:
        model.fit(x_train, y_train, verbose=False)
        _, val_probs = _predict_labels_and_probs(model=model, x_data=x_val)
    except MemoryError as exc:
        train_est_gib = _bytes_to_gib(
            _estimate_state_bytes(
                num_samples=x_train.shape[0],
                num_steps=max(1, x_train.shape[1] - washout),
                reservoir_size=reservoir_size,
            )
        )
        raise RuntimeError(
            "NON_RETRYABLE_OOM: torch reservoir backend could not allocate state "
            f"matrix for {task}. estimated_state={train_est_gib:.2f}GiB. "
            "Set INTRONMODEL_RC_STATE_BUDGET_GB smaller (e.g., 2.0), disable "
            "tuned hparams, or reduce RESERVOIR_SIZE / MAX_TOKENS."
        ) from exc
    val_metrics = _evaluate_binary_predictions(labels=val_labels, probs=val_probs)
    _ = apply_eta_process_title_from_epoch_progress(
        task_started_at=task_started_at,
        completed_epochs=1,
        total_epochs=1,
    )

    pr_auc = val_metrics.get("pr_auc")
    roc_auc = val_metrics.get("roc_auc")
    max_f1 = val_metrics.get("max_f1")
    acc_at_0_5 = val_metrics.get("acc@0.5")

    if pr_auc is not None:
        best_score = pr_auc
        best_metric_name = "pr_auc"
    elif roc_auc is not None:
        best_score = roc_auc
        best_metric_name = "roc_auc"
    else:
        best_score = float(acc_at_0_5 or 0.0)
        best_metric_name = "acc@0.5"

    checkpoint_payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT_VERSION,
        "task": task,
        "window_len": window_len,
        "model_config": {
            "input_mode": input_mode,
            "kmer_k": kmer_k,
            "max_tokens": max_tokens_effective,
            "input_dim": input_dim,
            "feature_dim": len(vocab),
            "effective_input_dim": int(x_train.shape[2]),
            "reservoir_size": reservoir_size,
            "spectral_radius": spectral_radius,
            "leak": leak,
            "sparsity": sparsity,
            "input_scale": input_scale,
            "pooling": pooling,
            "mts_rep": resolved_mts_rep,
            "dimred_method": (
                normalized_dimred if normalized_dimred is not None else "none"
            ),
            "n_dim": resolved_n_dim,
            "readout_type": normalized_readout,
            "readout_hidden": readout_hidden,
            "washout": washout,
            "read_order": read_order,
            "weight_decay": weight_decay,
        },
        "vocab": dict(vocab),
        "projection": projection,
        "model": model,
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_payload, f)

    print(
        f"[{task}] done best_{best_metric_name}={best_score:.4f} "
        "(single-fit RC training)"
    )

    return {
        "task": task,
        "num_examples": len(examples),
        "num_pos": n_pos,
        "num_neg": n_neg,
        "best_metric": best_metric_name,
        "best_epoch": 1,
        "best_score": float(best_score),
        "best_pr_auc": pr_auc,
        "best_roc_auc": roc_auc,
        "best_max_f1": max_f1,
        "best_acc_at_0_5": acc_at_0_5,
        "epoch_history": [
            {
                "epoch": 1,
                "train_loss": None,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "max_f1": max_f1,
                "acc@0.5": acc_at_0_5,
                "objective_metric": best_metric_name,
                "objective_score": float(best_score),
                "improved": True,
                "best_metric": best_metric_name,
                "best_score": float(best_score),
                "best_epoch": 1,
            }
        ],
        "epochs_completed": 1,
        "stopped_early": False,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "checkpoint": checkpoint_path,
        "loss": loss_name,
        "pos_weight": None,
        "focal_gamma": None,
        "focal_alpha_pos": None,
        "asym_gamma_pos": None,
        "asym_gamma_neg": None,
        "asym_alpha_pos": None,
        "input_mode": input_mode,
        "kmer_k": kmer_k,
        "max_tokens": max_tokens_effective,
        "input_dim": input_dim,
        "reservoir_size": reservoir_size,
        "spectral_radius": spectral_radius,
        "leak": leak,
        "sparsity": sparsity,
        "input_scale": input_scale,
        "pooling": pooling,
        "mts_rep": resolved_mts_rep,
        "dimred_method": normalized_dimred or "none",
        "n_dim": resolved_n_dim,
        "readout_type": normalized_readout,
        "readout_hidden": readout_hidden,
        "readout_dropout": readout_dropout,
        "washout": washout,
        "preroll_steps": preroll_steps,
        "read_order": read_order,
        "weight_decay": weight_decay,
        "eta_min_ratio": eta_min_ratio,
        "val_frac": val_frac,
        "grad_clip": grad_clip,
        "state_budget_gib": state_budget_gib,
        "train_examples_before_cap": int(train_mem_diag["before_count"]),
        "train_examples_after_cap": int(train_mem_diag["after_count"]),
        "val_examples_before_cap": int(val_mem_diag["before_count"]),
        "val_examples_after_cap": int(val_mem_diag["after_count"]),
        "train_state_estimated_gib_before_cap": float(
            train_mem_diag["estimated_gib_before"]
        ),
        "train_state_estimated_gib_after_cap": float(
            train_mem_diag["estimated_gib_after"]
        ),
        "val_state_estimated_gib_before_cap": float(
            val_mem_diag["estimated_gib_before"]
        ),
        "val_state_estimated_gib_after_cap": float(
            val_mem_diag["estimated_gib_after"]
        ),
        "compile_enabled": False,
        "use_amp": False,
        "amp_dtype": None,
        "allow_tf32": allow_tf32_bool,
        "cudnn_benchmark": cudnn_benchmark_bool,
        "deterministic": deterministic_bool,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": bool(persistent_workers),
        "pin_memory": bool(pin_memory),
        "effective_batch_size": batch_size,
        "oom_retries": 0,
        "gpu_id": gpu_id,
        "quick_phase": quick_phase,
        "optimizer_impl": "rc_model_fit",
    }


def _int_from_checkpoint(
    mapping: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    """Read integer field from a mapping with fallback.

    Parameters
    ----------
    mapping : Mapping[str, object]
        Source mapping.
    key : str
        Field key.
    default : int
        Fallback value.

    Returns
    -------
    int
        Parsed integer value.
    """
    raw = mapping.get(key, default)
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        return default
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return default
    return default


def _array_from_checkpoint(
    payload: Mapping[str, object],
    key: str,
) -> Optional[np.ndarray]:
    """Read optional 2D float array from checkpoint payload.

    Parameters
    ----------
    payload : Mapping[str, object]
        Checkpoint payload mapping.
    key : str
        Field name.

    Returns
    -------
    np.ndarray | None
        Parsed matrix or ``None`` if absent.

    Raises
    ------
    ValueError
        If shape is invalid.
    """
    raw = payload.get(key)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Checkpoint field '{key}' must be a 2D array.")
    return arr


def load_task_model(
    checkpoint_path: str,
) -> Tuple[
    ReservoirClassifierProtocol,
    Dict[str, object],
    Dict[str, int],
    Optional[np.ndarray],
]:
    """Load one donor/acceptor RC checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Checkpoint file created by this module.

    Returns
    -------
    tuple[
        ReservoirClassifierProtocol,
        dict[str, object],
        dict[str, int],
        np.ndarray | None,
    ]
        ``(model, model_config, vocab, projection)``.

    Raises
    ------
    ValueError
        If checkpoint payload is invalid.
    RuntimeError
        If checkpoint format is unsupported.
    """
    try:
        with open(checkpoint_path, "rb") as f:
            payload_obj = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load reservoir checkpoint. "
            "Legacy PyTorch reservoir checkpoints are not supported by this backend."
        ) from exc

    if not isinstance(payload_obj, dict):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")

    payload = cast(dict[str, object], payload_obj)
    checkpoint_format = payload.get("format")
    if checkpoint_format != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            "Unsupported checkpoint format for torch reservoir backend: "
            f"{checkpoint_format!r}. Re-train with current implementation."
        )

    model_obj = payload.get("model")
    if model_obj is None:
        raise ValueError(f"Checkpoint missing model object: {checkpoint_path}")
    if not hasattr(model_obj, "predict"):
        raise ValueError(f"Checkpoint model object is invalid: {checkpoint_path}")
    model = cast(ReservoirClassifierProtocol, model_obj)

    model_config_obj = payload.get("model_config", {})
    model_config: Dict[str, object]
    if isinstance(model_config_obj, dict):
        model_config = dict(model_config_obj)
    else:
        model_config = {}

    vocab_obj = payload.get("vocab")
    if not isinstance(vocab_obj, dict):
        raise ValueError(f"Checkpoint missing vocab: {checkpoint_path}")

    vocab: Dict[str, int] = {}
    for key, value in vocab_obj.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            vocab[key] = value
        elif isinstance(value, float) and value.is_integer():
            vocab[key] = int(value)
        elif isinstance(value, str):
            try:
                vocab[key] = int(value)
            except ValueError:
                continue

    if not vocab:
        raise ValueError(f"Checkpoint vocab is empty: {checkpoint_path}")

    projection = _array_from_checkpoint(payload=payload, key="projection")

    resolved_config: Dict[str, object] = {
        "input_mode": str(model_config.get("input_mode", "onehot")),
        "kmer_k": _int_from_checkpoint(model_config, "kmer_k", 3),
        "max_tokens": _int_from_checkpoint(model_config, "max_tokens", 100),
        "read_order": str(model_config.get("read_order", "forward")),
        "mts_rep": str(model_config.get("mts_rep", "mean")),
    }
    return model, resolved_config, vocab, projection


def score_sequences(
    model: ReservoirClassifierProtocol,
    sequences: Sequence[str],
    vocab: Mapping[str, int],
    input_mode: str,
    kmer_k: int,
    max_tokens: int,
    read_order: str,
    projection: Optional[np.ndarray],
    batch_size: int = 512,
) -> np.ndarray:
    """Score sequence batch with one trained task model.

    Parameters
    ----------
    model : ReservoirClassifierProtocol
        Fitted RC model.
    sequences : Sequence[str]
        Sequence list of length ``N``.
    vocab : Mapping[str, int]
        Token vocabulary.
    input_mode : str
        ``onehot`` or ``kmer``.
    kmer_k : int
        K-mer size.
    max_tokens : int
        Fixed token length ``T``.
    read_order : str
        ``forward`` or ``reverse``.
    projection : np.ndarray | None
        Optional feature projection ``(V, D)``.
    batch_size : int, default=512
        Number of sequences scored per chunk.

    Returns
    -------
    np.ndarray
        Positive-class probabilities with shape ``(N,)``.

    Raises
    ------
    ValueError
        If ``batch_size <= 0``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not sequences:
        return np.array([], dtype=np.float64)

    all_probs: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        seq_batch = sequences[start : start + batch_size]
        x_batch = _encode_sequence_batch(
            sequences=seq_batch,
            vocab=vocab,
            input_mode=input_mode,
            kmer_k=kmer_k,
            max_tokens=max_tokens,
            read_order=read_order,
            projection=projection,
        )
        _, probs = _predict_labels_and_probs(model=model, x_data=x_batch)
        all_probs.append(probs)

    return np.concatenate(all_probs)


def infer_site_scores(
    site_rows: List[Dict[str, object]],
    donor_model_path: str,
    acceptor_model_path: str,
    device: str = "auto",
    batch_size: int = 512,
) -> List[Dict[str, object]]:
    """Run donor/acceptor inference and return normalized site rows.

    Parameters
    ----------
    site_rows : list[dict[str, object]]
        Input rows with ``seq`` and site metadata.
    donor_model_path : str
        Donor checkpoint path.
    acceptor_model_path : str
        Acceptor checkpoint path.
    device : str, default="auto"
        Device argument (ignored; backend is CPU).
    batch_size : int, default=512
        Inference batch size.

    Returns
    -------
    list[dict[str, object]]
        Output rows with fixed schema:
        ``transcript_id, intron_index, site_type, score``.
    """
    del device

    donor_model, donor_config, donor_vocab, donor_projection = load_task_model(
        donor_model_path
    )
    acceptor_model, acceptor_config, acceptor_vocab, acceptor_projection = (
        load_task_model(acceptor_model_path)
    )

    donor_input_mode = str(donor_config.get("input_mode", "onehot"))
    donor_k = _int_from_checkpoint(donor_config, "kmer_k", 3)
    donor_max_tokens = _int_from_checkpoint(donor_config, "max_tokens", 100)
    donor_read_order = str(donor_config.get("read_order", "forward"))

    acceptor_input_mode = str(acceptor_config.get("input_mode", "onehot"))
    acceptor_k = _int_from_checkpoint(acceptor_config, "kmer_k", 3)
    acceptor_max_tokens = _int_from_checkpoint(acceptor_config, "max_tokens", 100)
    acceptor_read_order = str(acceptor_config.get("read_order", "forward"))

    donor_seqs = [str(row["seq"]) for row in site_rows if row["site_type"] == "donor"]
    acceptor_seqs = [
        str(row["seq"])
        for row in site_rows
        if row["site_type"] == "acceptor"
    ]

    donor_scores = score_sequences(
        model=donor_model,
        sequences=donor_seqs,
        vocab=donor_vocab,
        input_mode=donor_input_mode,
        kmer_k=donor_k,
        max_tokens=donor_max_tokens,
        read_order=donor_read_order,
        projection=donor_projection,
        batch_size=batch_size,
    )
    acceptor_scores = score_sequences(
        model=acceptor_model,
        sequences=acceptor_seqs,
        vocab=acceptor_vocab,
        input_mode=acceptor_input_mode,
        kmer_k=acceptor_k,
        max_tokens=acceptor_max_tokens,
        read_order=acceptor_read_order,
        projection=acceptor_projection,
        batch_size=batch_size,
    )

    out_rows: List[Dict[str, object]] = []
    donor_idx = 0
    acceptor_idx = 0

    for row in site_rows:
        site_type = str(row["site_type"])
        if site_type == "donor":
            score = (
                float(donor_scores[donor_idx])
                if donor_idx < len(donor_scores)
                else 0.0
            )
            donor_idx += 1
        else:
            score = (
                float(acceptor_scores[acceptor_idx])
                if acceptor_idx < len(acceptor_scores)
                else 0.0
            )
            acceptor_idx += 1

        out_rows.append(
            {
                "transcript_id": row["transcript_id"],
                "intron_index": int(row["intron_index"]),
                "site_type": site_type,
                "score": score,
            }
        )

    return out_rows


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register reservoir-specific training arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Target parser instance.
    """
    parser.add_argument(
        "--epochs",
        type=str,
        default="20",
        help="Epoch count (positive integer) or auto for early-stop mode.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Upper epoch limit used when --epochs=auto.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=12,
        help="Accepted for compatibility; RC backend performs single-fit training.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Accepted for compatibility; RC backend performs single-fit training.",
    )
    parser.add_argument(
        "--train_target",
        choices=["both", "donor", "acceptor"],
        default="both",
        help=(
            "Training target. 'both' trains donor and acceptor. "
            "'donor'/'acceptor' train one task only (for tuning)."
        ),
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument(
        "--input_mode",
        choices=list(INPUT_MODE_CHOICES),
        default="onehot",
        help="Input tokenization mode.",
    )
    parser.add_argument("--kmer_k", type=int, default=3)
    parser.add_argument("--max_tokens", default="auto")
    parser.add_argument("--input_dim", type=int, default=128)
    parser.add_argument("--reservoir_size", type=int, default=1024)
    parser.add_argument("--spectral_radius", type=float, default=0.95)
    parser.add_argument("--leak", type=float, default=0.3)
    parser.add_argument("--sparsity", type=float, default=0.1)
    parser.add_argument("--input_scale", type=float, default=0.5)
    parser.add_argument(
        "--pooling",
        choices=list(POOLING_CHOICES),
        default="mean_max",
        help="Legacy alias used to derive --mts_rep when mts_rep=auto.",
    )
    parser.add_argument(
        "--mts_rep",
        choices=list(MTS_REP_ARG_CHOICES),
        default="auto",
        help="RC representation type. auto maps from --pooling.",
    )
    parser.add_argument(
        "--dimred_method",
        choices=list(DIMRED_METHOD_CHOICES),
        default="none",
        help="Reservoir-state dimensionality reduction method.",
    )
    parser.add_argument(
        "--n_dim",
        type=int,
        default=None,
        help="Target dimension for dimred method when enabled.",
    )
    parser.add_argument(
        "--readout_type",
        choices=list(READOUT_TYPE_CHOICES),
        default="lin",
        help="Readout classifier type for Torch RC backend.",
    )
    parser.add_argument("--readout_hidden", type=int, default=256)
    parser.add_argument("--readout_dropout", type=float, default=0.2)
    parser.add_argument("--washout", type=int, default=0)
    parser.add_argument("--preroll_steps", type=int, default=0)
    parser.add_argument(
        "--read_order",
        choices=list(READ_ORDER_CHOICES),
        default="auto",
        help="Task-wise sequence order; auto uses donor=forward, acceptor=reverse.",
    )
    parser.add_argument(
        "--donor_read_order",
        choices=list(READ_ORDER_CHOICES),
        default=None,
        help="Donor-only override for --read_order.",
    )
    parser.add_argument(
        "--acceptor_read_order",
        choices=list(READ_ORDER_CHOICES),
        default=None,
        help="Acceptor-only override for --read_order.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eta_min_ratio", type=float, default=0.01)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--donor_batch_size", type=int, default=None)
    parser.add_argument("--acceptor_batch_size", type=int, default=None)
    parser.add_argument("--donor_lr", type=float, default=None)
    parser.add_argument("--acceptor_lr", type=float, default=None)
    parser.add_argument(
        "--donor_loss",
        choices=list(LOSS_NAME_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--acceptor_loss",
        choices=list(LOSS_NAME_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--donor_input_mode",
        choices=list(INPUT_MODE_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--acceptor_input_mode",
        choices=list(INPUT_MODE_CHOICES),
        default=None,
    )
    parser.add_argument("--donor_kmer_k", type=int, default=None)
    parser.add_argument("--acceptor_kmer_k", type=int, default=None)
    parser.add_argument("--donor_max_tokens", default=None)
    parser.add_argument("--acceptor_max_tokens", default=None)
    parser.add_argument("--donor_input_dim", type=int, default=None)
    parser.add_argument("--acceptor_input_dim", type=int, default=None)
    parser.add_argument("--donor_reservoir_size", type=int, default=None)
    parser.add_argument("--acceptor_reservoir_size", type=int, default=None)
    parser.add_argument("--donor_spectral_radius", type=float, default=None)
    parser.add_argument("--acceptor_spectral_radius", type=float, default=None)
    parser.add_argument("--donor_leak", type=float, default=None)
    parser.add_argument("--acceptor_leak", type=float, default=None)
    parser.add_argument("--donor_sparsity", type=float, default=None)
    parser.add_argument("--acceptor_sparsity", type=float, default=None)
    parser.add_argument("--donor_input_scale", type=float, default=None)
    parser.add_argument("--acceptor_input_scale", type=float, default=None)
    parser.add_argument(
        "--donor_pooling",
        choices=list(POOLING_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--acceptor_pooling",
        choices=list(POOLING_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--donor_mts_rep",
        choices=list(MTS_REP_ARG_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--acceptor_mts_rep",
        choices=list(MTS_REP_ARG_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--donor_dimred_method",
        choices=list(DIMRED_METHOD_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--acceptor_dimred_method",
        choices=list(DIMRED_METHOD_CHOICES),
        default=None,
    )
    parser.add_argument("--donor_n_dim", type=int, default=None)
    parser.add_argument("--acceptor_n_dim", type=int, default=None)
    parser.add_argument(
        "--donor_readout_type",
        choices=list(READOUT_TYPE_CHOICES),
        default=None,
    )
    parser.add_argument(
        "--acceptor_readout_type",
        choices=list(READOUT_TYPE_CHOICES),
        default=None,
    )
    parser.add_argument("--donor_readout_hidden", type=int, default=None)
    parser.add_argument("--acceptor_readout_hidden", type=int, default=None)
    parser.add_argument("--donor_readout_dropout", type=float, default=None)
    parser.add_argument("--acceptor_readout_dropout", type=float, default=None)
    parser.add_argument("--donor_washout", type=int, default=None)
    parser.add_argument("--acceptor_washout", type=int, default=None)
    parser.add_argument("--donor_preroll_steps", type=int, default=None)
    parser.add_argument("--acceptor_preroll_steps", type=int, default=None)
    parser.add_argument("--donor_weight_decay", type=float, default=None)
    parser.add_argument("--acceptor_weight_decay", type=float, default=None)
    parser.add_argument("--donor_eta_min_ratio", type=float, default=None)
    parser.add_argument("--acceptor_eta_min_ratio", type=float, default=None)
    parser.add_argument("--donor_val_frac", type=float, default=None)
    parser.add_argument("--acceptor_val_frac", type=float, default=None)
    parser.add_argument("--donor_grad_clip", type=float, default=None)
    parser.add_argument("--acceptor_grad_clip", type=float, default=None)

    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=["off", "on", "auto"],
        default="auto",
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument(
        "--use_amp",
        type=int,
        choices=[0, 1],
        default=1,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument(
        "--allow_tf32",
        type=int,
        choices=[0, 1],
        default=1,
        help="Accepted for compatibility.",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        type=int,
        choices=[0, 1],
        default=1,
        help="Accepted for compatibility.",
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        choices=[0, 1],
        default=0,
        help="Control global deterministic seed behavior.",
    )
    parser.add_argument(
        "--num_workers",
        default="auto",
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument(
        "--persistent_workers",
        type=int,
        choices=[0, 1],
        default=1,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument(
        "--pin_memory",
        type=int,
        choices=[0, 1],
        default=1,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=64,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument(
        "--max_oom_retries",
        type=int,
        default=8,
        help="Accepted for compatibility; ignored by RC backend.",
    )

    parser.add_argument(
        "--loss",
        choices=list(LOSS_NAME_CHOICES),
        default="weighted_bce",
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument(
        "--pos_weight_cap",
        type=float,
        default=20.0,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument("--donor_pos_weight_cap", type=float, default=None)
    parser.add_argument("--acceptor_pos_weight_cap", type=float, default=None)
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument("--donor_focal_gamma", type=float, default=None)
    parser.add_argument("--acceptor_focal_gamma", type=float, default=None)
    parser.add_argument(
        "--focal_alpha_pos",
        type=float,
        default=None,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument("--donor_focal_alpha_pos", type=float, default=None)
    parser.add_argument("--acceptor_focal_alpha_pos", type=float, default=None)
    parser.add_argument(
        "--asym_gamma_pos",
        type=float,
        default=0.0,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument("--donor_asym_gamma_pos", type=float, default=None)
    parser.add_argument("--acceptor_asym_gamma_pos", type=float, default=None)
    parser.add_argument(
        "--asym_gamma_neg",
        type=float,
        default=4.0,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument("--donor_asym_gamma_neg", type=float, default=None)
    parser.add_argument("--acceptor_asym_gamma_neg", type=float, default=None)
    parser.add_argument(
        "--asym_alpha_pos",
        type=float,
        default=None,
        help="Accepted for compatibility; ignored by RC backend.",
    )
    parser.add_argument("--donor_asym_alpha_pos", type=float, default=None)
    parser.add_argument("--acceptor_asym_alpha_pos", type=float, default=None)
    parser.add_argument("--tag", default=None)


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register reservoir-specific inference arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Target parser instance.
    """
    parser.add_argument("--batch_size", type=int, default=512)


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train donor/acceptor RC models with unified argument interface.

    Parameters
    ----------
    common_args : argparse.Namespace
        Shared pipeline arguments from ``run_model.py``.
    model_args : argparse.Namespace
        Reservoir model-specific arguments.

    Returns
    -------
    dict[str, object]
        Combined training summary with donor/acceptor task blocks.
    """
    train_pos_path, train_neg_path, inferred_train_len = resolve_train_paths(
        species=common_args.species,
        train_pos_path=common_args.train_pos_path,
        train_neg_path=common_args.train_neg_path,
        donor_len=common_args.donor_len,
        acceptor_len=common_args.acceptor_len,
    )

    donor_len, acceptor_len = resolve_effective_window_lengths(
        donor_len=common_args.donor_len,
        acceptor_len=common_args.acceptor_len,
        inferred_train_len=inferred_train_len,
    )
    validate_window_args(
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )

    donor_window_len = donor_len if donor_len is not None else 50
    acceptor_window_len = acceptor_len if acceptor_len is not None else 50

    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=False,
    )
    donor_checkpoint_path = task_checkpoint_paths["donor"]
    acceptor_checkpoint_path = task_checkpoint_paths["acceptor"]
    train_target = resolve_train_target(model_args)

    resolved_epochs, epochs_auto = resolve_training_epoch_budget(
        epochs_arg=model_args.epochs,
        max_epochs=int(model_args.max_epochs),
    )
    early_stop_patience, early_stop_min_delta = resolve_early_stopping_params(
        patience_arg=model_args.early_stop_patience,
        min_delta_arg=model_args.early_stop_min_delta,
    )
    effective_early_stop_patience = early_stop_patience if epochs_auto else 0

    if getattr(common_args, "continue_train", False):
        print(
            "[reservoir] --continue_train is accepted, but RC backend performs "
            "full refit from data in a single fit pass."
        )

    tasks_to_train = resolve_tasks_to_train(train_target)
    task_window_len = {
        "donor": donor_window_len,
        "acceptor": acceptor_window_len,
    }

    task_hparams: dict[str, TaskTrainParams] = {}
    task_metrics: dict[str, Dict[str, object]] = {}
    for task in tasks_to_train:
        resolved = _resolve_task_train_params(task=task, model_args=model_args)
        task_hparams[task] = resolved
        task_metrics[task] = train_task_model(
            task=task,
            pos_path=train_pos_path,
            neg_path=train_neg_path,
            checkpoint_path=task_checkpoint_paths[task],
            window_len=task_window_len[task],
            donor_len=donor_len,
            acceptor_len=acceptor_len,
            epochs=resolved_epochs,
            early_stop_patience=effective_early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            batch_size=resolved.batch_size,
            lr=resolved.lr,
            seed=common_args.seed,
            input_mode=resolved.input_mode,
            kmer_k=resolved.kmer_k,
            max_tokens=resolved.max_tokens,
            input_dim=resolved.input_dim,
            reservoir_size=resolved.reservoir_size,
            spectral_radius=resolved.spectral_radius,
            leak=resolved.leak,
            sparsity=resolved.sparsity,
            input_scale=resolved.input_scale,
            pooling=resolved.pooling,
            mts_rep=resolved.mts_rep,
            dimred_method=resolved.dimred_method,
            n_dim=resolved.n_dim,
            readout_type=resolved.readout_type,
            readout_hidden=resolved.readout_hidden,
            readout_dropout=resolved.readout_dropout,
            washout=resolved.washout,
            preroll_steps=resolved.preroll_steps,
            read_order=resolved.read_order,
            weight_decay=resolved.weight_decay,
            eta_min_ratio=resolved.eta_min_ratio,
            val_frac=resolved.val_frac,
            grad_clip=resolved.grad_clip,
            compile_model=model_args.compile,
            compile_mode=model_args.compile_mode,
            device=common_args.device,
            loss_name=resolved.loss_name,
            pos_weight_cap=resolved.pos_weight_cap,
            focal_gamma=resolved.focal_gamma,
            focal_alpha_pos=resolved.focal_alpha_pos,
            asym_gamma_pos=resolved.asym_gamma_pos,
            asym_gamma_neg=resolved.asym_gamma_neg,
            asym_alpha_pos=resolved.asym_alpha_pos,
            use_amp=model_args.use_amp,
            amp_dtype=model_args.amp_dtype,
            allow_tf32=model_args.allow_tf32,
            cudnn_benchmark=model_args.cudnn_benchmark,
            deterministic=model_args.deterministic,
            num_workers=model_args.num_workers,
            prefetch_factor=model_args.prefetch_factor,
            persistent_workers=model_args.persistent_workers,
            pin_memory=model_args.pin_memory,
            min_batch_size=model_args.min_batch_size,
            max_oom_retries=model_args.max_oom_retries,
            quick_phase=bool(getattr(common_args, "quick_phase", False)),
            gpu_id=getattr(common_args, "gpu_id", None),
        )

    run_name_lr = model_args.lr
    run_name_batch_size = model_args.batch_size
    if train_target != "both":
        selected_params = task_hparams[tasks_to_train[0]]
        run_name_lr = selected_params.lr
        run_name_batch_size = selected_params.batch_size

    run_name = build_run_name(
        model_name="reservoir",
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        lr=run_name_lr,
        batch_size=run_name_batch_size,
        epochs=resolved_epochs,
        tag=model_args.tag,
    )

    task_hparams_summary: dict[str, Dict[str, object]] = {}
    for task, params in task_hparams.items():
        task_hparams_summary[task] = {
            "batch_size": params.batch_size,
            "lr": params.lr,
            "loss": params.loss_name,
            "input_mode": params.input_mode,
            "kmer_k": params.kmer_k,
            "max_tokens": params.max_tokens,
            "input_dim": params.input_dim,
            "reservoir_size": params.reservoir_size,
            "spectral_radius": params.spectral_radius,
            "leak": params.leak,
            "sparsity": params.sparsity,
            "input_scale": params.input_scale,
            "pooling": params.pooling,
            "mts_rep": params.mts_rep,
            "dimred_method": params.dimred_method,
            "n_dim": params.n_dim,
            "readout_type": params.readout_type,
            "readout_hidden": params.readout_hidden,
            "readout_dropout": params.readout_dropout,
            "washout": params.washout,
            "preroll_steps": params.preroll_steps,
            "read_order": params.read_order,
            "weight_decay": params.weight_decay,
            "eta_min_ratio": params.eta_min_ratio,
            "val_frac": params.val_frac,
            "grad_clip": params.grad_clip,
            "pos_weight_cap": params.pos_weight_cap,
            "focal_gamma": params.focal_gamma,
            "focal_alpha_pos": params.focal_alpha_pos,
            "asym_gamma_pos": params.asym_gamma_pos,
            "asym_gamma_neg": params.asym_gamma_neg,
            "asym_alpha_pos": params.asym_alpha_pos,
        }

    summary: Dict[str, object] = {
        "model": "reservoir",
        "species": common_args.species,
        "train_pos_path": train_pos_path,
        "train_neg_path": train_neg_path,
        "donor_len": donor_len,
        "acceptor_len": acceptor_len,
        "epochs": resolved_epochs,
        "epochs_config": str(model_args.epochs),
        "epochs_auto": epochs_auto,
        "max_epochs": model_args.max_epochs,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "batch_size": model_args.batch_size,
        "lr": model_args.lr,
        "train_target": train_target,
        "seed": common_args.seed,
        "device": common_args.device,
        "checkpoint_name": os.path.basename(donor_checkpoint_path),
        "donor_checkpoint_path": donor_checkpoint_path,
        "acceptor_checkpoint_path": acceptor_checkpoint_path,
        "input_mode": model_args.input_mode,
        "kmer_k": model_args.kmer_k,
        "max_tokens": model_args.max_tokens,
        "input_dim": model_args.input_dim,
        "reservoir_size": model_args.reservoir_size,
        "spectral_radius": model_args.spectral_radius,
        "leak": model_args.leak,
        "sparsity": model_args.sparsity,
        "input_scale": model_args.input_scale,
        "pooling": model_args.pooling,
        "mts_rep": model_args.mts_rep,
        "dimred_method": model_args.dimred_method,
        "n_dim": model_args.n_dim,
        "readout_type": model_args.readout_type,
        "readout_hidden": model_args.readout_hidden,
        "readout_dropout": model_args.readout_dropout,
        "washout": model_args.washout,
        "preroll_steps": model_args.preroll_steps,
        "read_order": model_args.read_order,
        "weight_decay": model_args.weight_decay,
        "eta_min_ratio": model_args.eta_min_ratio,
        "val_frac": model_args.val_frac,
        "grad_clip": model_args.grad_clip,
        "compile": model_args.compile,
        "compile_mode": model_args.compile_mode,
        "use_amp": bool(model_args.use_amp),
        "amp_dtype": model_args.amp_dtype,
        "allow_tf32": bool(model_args.allow_tf32),
        "cudnn_benchmark": bool(model_args.cudnn_benchmark),
        "deterministic": bool(model_args.deterministic),
        "num_workers": model_args.num_workers,
        "prefetch_factor": model_args.prefetch_factor,
        "persistent_workers": bool(model_args.persistent_workers),
        "pin_memory": bool(model_args.pin_memory),
        "min_batch_size": model_args.min_batch_size,
        "max_oom_retries": model_args.max_oom_retries,
        "loss": model_args.loss,
        "focal_gamma": model_args.focal_gamma,
        "focal_alpha_pos": model_args.focal_alpha_pos,
        "asym_gamma_pos": model_args.asym_gamma_pos,
        "asym_gamma_neg": model_args.asym_gamma_neg,
        "asym_alpha_pos": model_args.asym_alpha_pos,
        "run_name": run_name,
        "inferred_train_len": inferred_train_len,
        "task_hyperparameters": task_hparams_summary,
    }
    summary.update(task_metrics)
    return summary


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Run site-level inference and return rows with fixed schema.

    Parameters
    ----------
    common_args : argparse.Namespace
        Shared pipeline args.
    model_args : argparse.Namespace
        Reservoir model-specific args.

    Returns
    -------
    list[dict[str, object]]
        Site rows with schema required by the pipeline.
    """
    dirs = species_data_dirs(common_args.species)
    inferred_train_len: Optional[int] = None
    if common_args.donor_len is None and common_args.acceptor_len is None:
        try:
            _, _, inferred_train_len = infer_default_train_paths(
                train_dir=dirs["raw"],
                donor_len=None,
                acceptor_len=None,
            )
        except ValueError:
            inferred_train_len = None

    donor_len, acceptor_len = resolve_effective_window_lengths(
        donor_len=common_args.donor_len,
        acceptor_len=common_args.acceptor_len,
        inferred_train_len=inferred_train_len,
    )
    validate_window_args(
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )

    test_tsv = resolve_test_tsv(common_args.species, common_args.test_tsv)
    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=True,
    )
    donor_model_path = task_checkpoint_paths["donor"]
    acceptor_model_path = task_checkpoint_paths["acceptor"]

    site_rows, skipped_short = read_test_site_rows(
        test_tsv=test_tsv,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )
    print(f"Loaded test sites: {len(site_rows)}")
    if skipped_short:
        print(f"Skipped short sites: {skipped_short}")

    return infer_site_scores(
        site_rows=site_rows,
        donor_model_path=donor_model_path,
        acceptor_model_path=acceptor_model_path,
        device=common_args.device,
        batch_size=model_args.batch_size,
    )
