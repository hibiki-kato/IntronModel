"""Markov-order PWM features + XGBoost pair model for splice scoring.

This module implements the unified ``run_model.py`` contract:
- ``add_train_args`` / ``add_infer_args``
- ``train`` for pair-model training
- ``infer_site`` for pair-site inference

Training pipeline
-----------------
1. Build donor and acceptor Markov PWM models (positive and negative class).
2. Convert each donor/acceptor pair into Markov features
   (pair summary, per-base, or hybrid).
3. Train one XGBoost binary classifier on pair features.
4. Save Markov models + classifier into one pair checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
import random
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, cast

import numpy as np

from util.data_proc import (
    infer_default_train_paths,
    read_examples_pair_task_with_metadata,
    read_test_pair_rows,
    resolve_effective_window_lengths,
    resolve_test_tsv,
    resolve_train_paths,
    species_data_dirs,
    validate_window_args,
)
from util.model_runtime import (
    fallback_average_precision as _fallback_average_precision,
    fallback_max_f1 as _fallback_max_f1,
    fallback_roc_auc as _fallback_roc_auc,
)
from util.model_task_paths import (
    resolve_required_checkpoint_paths,
    resolve_tasks_to_train,
    resolve_train_target,
)
from util.sequence_transform import (
    SEQUENCE_TRANSFORM_CHOICES,
    PairSequenceRecord,
    apply_pair_sequence_transform,
)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ModuleNotFoundError:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

try:
    from xgboost import XGBClassifier as _ImportedXGBClassifier
except ModuleNotFoundError:  # pragma: no cover
    _ImportedXGBClassifier = None

CHECKPOINT_FORMAT_VERSION: str = "markov-xgboost-v1"
MARKOV_FEATURE_CACHE_FORMAT_VERSION: str = "markov-xgboost-cache-v1"
DNA_BASES: tuple[str, ...] = ("A", "C", "G", "T")
BASE_TO_INDEX: dict[str, int] = {base: idx for idx, base in enumerate(DNA_BASES)}
MARKOV_FEATURE_NAMES: tuple[str, str] = (
    "donor_markov_log_odds",
    "acceptor_markov_log_odds",
)
MARKOV_FEATURE_MODE_CHOICES: tuple[str, ...] = (
    "pair_summary",
    "per_base",
    "hybrid",
)
MARKOV_CACHE_MODE_CHOICES: tuple[str, ...] = ("off", "auto", "refresh")


class XGBClassifierProtocol(Protocol):
    """Protocol required by this module for pair feature classifier."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> object:
        """Fit model from feature matrix and labels."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities with shape ``(n_samples, 2)``."""


@dataclass(frozen=True)
class MarkovPwmModel:
    """Position-specific Markov PWM model.

    Attributes
    ----------
    order : int
        Maximum context length used for conditional probabilities.
    alpha : float
        Additive smoothing pseudo-count.
    seq_length : int
        Fixed sequence length for this model.
    probabilities_by_pos : tuple[dict[tuple[int, ...], np.ndarray], ...]
        One dictionary per position. Keys are context tuples of integer base IDs
        (varying context length ``0..order``), values are length-4 probability
        arrays over ``A/C/G/T``.
    """

    order: int
    alpha: float
    seq_length: int
    probabilities_by_pos: tuple[dict[tuple[int, ...], np.ndarray], ...]


@dataclass(frozen=True)
class PairMarkovModels:
    """Container for donor/acceptor positive-vs-negative Markov models."""

    donor_pos: MarkovPwmModel
    donor_neg: MarkovPwmModel
    acceptor_pos: MarkovPwmModel
    acceptor_neg: MarkovPwmModel


@dataclass(frozen=True)
class MarkovFeatureBundle:
    """Precomputed Markov-derived feature tensors for one train split.

    Attributes
    ----------
    train_features : np.ndarray
        Train feature matrix with shape ``(n_train, feature_dim)``.
    val_features : np.ndarray
        Validation feature matrix with shape ``(n_val, feature_dim)``.
    labels_train : np.ndarray
        Train labels with shape ``(n_train,)``.
    labels_val : np.ndarray
        Validation labels with shape ``(n_val,)``.
    final_features : np.ndarray
        Full-train feature matrix with shape ``(n_total, feature_dim)`` used for
        final classifier fitting.
    labels_all : np.ndarray
        Full labels with shape ``(n_total,)``.
    final_pair_models : PairMarkovModels
        Markov models fitted on the full examples.
    feature_names : tuple[str, ...]
        Ordered feature names matching columns of feature matrices.
    n_total : int
        Total number of training examples.
    n_train : int
        Number of train-split examples.
    n_val : int
        Number of validation-split examples.
    """

    train_features: np.ndarray
    val_features: np.ndarray
    labels_train: np.ndarray
    labels_val: np.ndarray
    final_features: np.ndarray
    labels_all: np.ndarray
    final_pair_models: PairMarkovModels
    feature_names: tuple[str, ...]
    n_total: int
    n_train: int
    n_val: int


def _validate_markov_params(order: int, alpha: float) -> None:
    """Validate Markov PWM hyperparameters."""
    if order < 0:
        raise ValueError("--markov_order must be >= 0.")
    if alpha <= 0.0:
        raise ValueError("--markov_alpha must be > 0.")


def _normalize_markov_feature_mode(raw_mode: object) -> str:
    """Normalize and validate Markov feature mode."""
    mode = str(raw_mode).strip().lower()
    if mode not in MARKOV_FEATURE_MODE_CHOICES:
        choices_text = ", ".join(MARKOV_FEATURE_MODE_CHOICES)
        raise ValueError(f"--markov_feature_mode must be one of: {choices_text}.")
    return mode


def _normalize_markov_cache_mode(raw_mode: object) -> str:
    """Normalize and validate Markov feature-cache mode."""
    mode = str(raw_mode).strip().lower()
    if mode not in MARKOV_CACHE_MODE_CHOICES:
        choices_text = ", ".join(MARKOV_CACHE_MODE_CHOICES)
        raise ValueError(f"--markov_cache_mode must be one of: {choices_text}.")
    return mode


def _validate_xgb_params(model_args: argparse.Namespace) -> None:
    """Validate XGBoost hyperparameters."""
    if int(model_args.xgb_n_estimators) <= 0:
        raise ValueError("--xgb_n_estimators must be positive.")
    if int(model_args.xgb_max_depth) <= 0:
        raise ValueError("--xgb_max_depth must be positive.")
    if float(model_args.xgb_learning_rate) <= 0.0:
        raise ValueError("--xgb_learning_rate must be > 0.")
    if float(model_args.xgb_subsample) <= 0.0 or float(model_args.xgb_subsample) > 1.0:
        raise ValueError("--xgb_subsample must satisfy 0 < value <= 1.")
    if (
        float(model_args.xgb_colsample_bytree) <= 0.0
        or float(model_args.xgb_colsample_bytree) > 1.0
    ):
        raise ValueError(
            "--xgb_colsample_bytree must satisfy 0 < value <= 1."
        )
    if float(model_args.xgb_min_child_weight) < 0.0:
        raise ValueError("--xgb_min_child_weight must be >= 0.")
    if float(model_args.xgb_reg_lambda) < 0.0:
        raise ValueError("--xgb_reg_lambda must be >= 0.")
    if float(model_args.xgb_reg_alpha) < 0.0:
        raise ValueError("--xgb_reg_alpha must be >= 0.")
    if int(model_args.xgb_n_jobs) == 0:
        raise ValueError("--xgb_n_jobs must be non-zero (use -1 for all cores).")


def _require_xgb_classifier_class() -> type[XGBClassifierProtocol]:
    """Resolve runtime XGBoost classifier class.

    Returns
    -------
    type[XGBClassifierProtocol]
        ``xgboost.XGBClassifier`` type.

    Raises
    ------
    RuntimeError
        If xgboost is not installed.
    """
    if _ImportedXGBClassifier is None:
        raise RuntimeError(
            "xgboost is required for model 'markov_xgboost'. "
            "Install it in the intronmodel environment, e.g. "
            "`conda install -n intronmodel -c conda-forge xgboost`."
        )
    return cast(type[XGBClassifierProtocol], _ImportedXGBClassifier)


def _encode_sequence(sequence: str) -> np.ndarray:
    """Encode a DNA sequence into integer base IDs.

    Unknown bases are encoded as ``-1`` and treated as neutral during scoring.

    Complexity
    ----------
    O(L) time and O(L) memory where ``L`` is sequence length.
    """
    encoded = np.full(len(sequence), -1, dtype=np.int16)
    for index, base in enumerate(sequence.upper()):
        base_idx = BASE_TO_INDEX.get(base)
        if base_idx is not None:
            encoded[index] = base_idx
    return encoded


def _fit_markov_pwm_model(
    sequences: Sequence[str],
    *,
    order: int,
    alpha: float,
) -> MarkovPwmModel:
    """Fit one position-specific Markov PWM model.

    The implementation stores conditional probabilities at each position for
    all context lengths ``0..order``. At inference, unseen long contexts are
    resolved by deterministic backoff to shorter contexts.

    Complexity
    ----------
    O(N * L * (order + 1)) time and O(U) memory, where ``N`` is number of
    sequences, ``L`` is sequence length, and ``U`` is number of observed
    ``(position, context)`` states.
    """
    if not sequences:
        raise ValueError("At least one sequence is required to fit Markov PWM.")

    seq_length = len(sequences[0])
    if seq_length <= 0:
        raise ValueError("Training sequences must be non-empty.")

    encoded_sequences: list[np.ndarray] = []
    for sequence in sequences:
        if len(sequence) != seq_length:
            raise ValueError("All sequences must share the same length.")
        encoded_sequences.append(_encode_sequence(sequence))

    counts_by_pos: list[dict[tuple[int, ...], np.ndarray]] = [
        {} for _ in range(seq_length)
    ]

    for encoded in encoded_sequences:
        for pos in range(seq_length):
            base_idx = int(encoded[pos])
            if base_idx < 0:
                continue
            max_context = min(order, pos)
            for context_len in range(max_context + 1):
                if context_len == 0:
                    context: tuple[int, ...] = ()
                else:
                    raw_context = encoded[pos - context_len : pos]
                    if np.any(raw_context < 0):
                        continue
                    context = tuple(int(item) for item in raw_context)
                table = counts_by_pos[pos]
                base_counts = table.get(context)
                if base_counts is None:
                    base_counts = np.zeros(4, dtype=np.float64)
                    table[context] = base_counts
                base_counts[base_idx] += 1.0

    probabilities_by_pos: list[dict[tuple[int, ...], np.ndarray]] = []
    for pos_table in counts_by_pos:
        probability_table: dict[tuple[int, ...], np.ndarray] = {}
        for context, raw_counts in pos_table.items():
            smoothed = raw_counts + alpha
            probability_table[context] = smoothed / float(np.sum(smoothed))

        if () not in probability_table:
            probability_table[()] = np.full(4, 0.25, dtype=np.float64)
        probabilities_by_pos.append(probability_table)

    return MarkovPwmModel(
        order=order,
        alpha=alpha,
        seq_length=seq_length,
        probabilities_by_pos=tuple(probabilities_by_pos),
    )


def _sequence_log_contributions(model: MarkovPwmModel, sequence: str) -> np.ndarray:
    """Return per-position Markov log-probability contributions.

    Complexity
    ----------
    O(L * (order + 1)) time and O(L) memory where ``L`` is sequence length.
    """
    if len(sequence) != model.seq_length:
        raise ValueError(
            "Sequence length mismatch. "
            f"Expected {model.seq_length}, got {len(sequence)}."
        )

    encoded = _encode_sequence(sequence)
    contributions = np.zeros(model.seq_length, dtype=np.float64)
    for pos in range(model.seq_length):
        base_idx = int(encoded[pos])
        if base_idx < 0:
            contributions[pos] = float(np.log(0.25))
            continue

        max_context = min(model.order, pos)
        pos_table = model.probabilities_by_pos[pos]
        chosen_probs: np.ndarray | None = None
        for context_len in range(max_context, -1, -1):
            if context_len == 0:
                context = ()
            else:
                raw_context = encoded[pos - context_len : pos]
                if np.any(raw_context < 0):
                    continue
                context = tuple(int(item) for item in raw_context)
            probs = pos_table.get(context)
            if probs is not None:
                chosen_probs = probs
                break

        if chosen_probs is None:
            chosen_probs = np.full(4, 0.25, dtype=np.float64)
        prob = max(float(chosen_probs[base_idx]), 1e-12)
        contributions[pos] = float(np.log(prob))
    return contributions


def _sequence_log_probability(model: MarkovPwmModel, sequence: str) -> float:
    """Return total Markov log-probability for one sequence."""
    return float(np.sum(_sequence_log_contributions(model, sequence)))


def _markov_log_odds_score(
    sequence: str,
    *,
    positive_model: MarkovPwmModel,
    negative_model: MarkovPwmModel,
) -> float:
    """Compute sequence log-odds between positive and negative Markov models."""
    positive_logp = _sequence_log_probability(positive_model, sequence)
    negative_logp = _sequence_log_probability(negative_model, sequence)
    return float(positive_logp - negative_logp)


def _markov_log_odds_per_position(
    sequence: str,
    *,
    positive_model: MarkovPwmModel,
    negative_model: MarkovPwmModel,
) -> np.ndarray:
    """Return per-position log-odds contributions for one sequence."""
    positive_log_contrib = _sequence_log_contributions(positive_model, sequence)
    negative_log_contrib = _sequence_log_contributions(negative_model, sequence)
    return positive_log_contrib - negative_log_contrib


def _build_pair_markov_models(
    donor_sequences: Sequence[str],
    acceptor_sequences: Sequence[str],
    labels: np.ndarray,
    *,
    order: int,
    alpha: float,
) -> PairMarkovModels:
    """Fit positive/negative donor+acceptor Markov models from labeled pairs."""
    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D array.")
    if labels.size != len(donor_sequences) or labels.size != len(acceptor_sequences):
        raise ValueError("labels length must match donor and acceptor sequence counts.")

    positive_mask = labels == 1
    negative_mask = labels == 0
    if int(np.sum(positive_mask)) == 0 or int(np.sum(negative_mask)) == 0:
        raise ValueError("Both positive and negative examples are required.")

    donor_pos = [seq for seq, is_pos in zip(donor_sequences, positive_mask) if is_pos]
    donor_neg = [seq for seq, is_neg in zip(donor_sequences, negative_mask) if is_neg]
    acceptor_pos = [
        seq for seq, is_pos in zip(acceptor_sequences, positive_mask) if is_pos
    ]
    acceptor_neg = [
        seq for seq, is_neg in zip(acceptor_sequences, negative_mask) if is_neg
    ]

    return PairMarkovModels(
        donor_pos=_fit_markov_pwm_model(donor_pos, order=order, alpha=alpha),
        donor_neg=_fit_markov_pwm_model(donor_neg, order=order, alpha=alpha),
        acceptor_pos=_fit_markov_pwm_model(acceptor_pos, order=order, alpha=alpha),
        acceptor_neg=_fit_markov_pwm_model(acceptor_neg, order=order, alpha=alpha),
    )


def _compute_pair_markov_features(
    donor_sequences: Sequence[str],
    acceptor_sequences: Sequence[str],
    *,
    pair_models: PairMarkovModels,
    feature_mode: str,
) -> np.ndarray:
    """Build pair Markov features.

    ``pair_summary`` mode:
    - one donor scalar log-odds
    - one acceptor scalar log-odds

    ``per_base`` mode:
    - donor per-position log-odds vector
    - acceptor per-position log-odds vector

    ``hybrid`` mode:
    - pair_summary + per_base concatenation

    Complexity
    ----------
    O(N * L * (order + 1)) time where ``N`` is sample count and ``L`` is
    sequence length.
    """
    if len(donor_sequences) != len(acceptor_sequences):
        raise ValueError("donor and acceptor sequence counts must match.")

    normalized_mode = _normalize_markov_feature_mode(feature_mode)
    feature_rows: list[np.ndarray] = []

    for donor_seq, acceptor_seq in zip(donor_sequences, acceptor_sequences):
        donor_per_base = _markov_log_odds_per_position(
            donor_seq,
            positive_model=pair_models.donor_pos,
            negative_model=pair_models.donor_neg,
        )
        acceptor_per_base = _markov_log_odds_per_position(
            acceptor_seq,
            positive_model=pair_models.acceptor_pos,
            negative_model=pair_models.acceptor_neg,
        )
        donor_total = float(np.sum(donor_per_base))
        acceptor_total = float(np.sum(acceptor_per_base))

        if normalized_mode == "pair_summary":
            feature_rows.append(
                np.asarray([donor_total, acceptor_total], dtype=np.float64)
            )
            continue
        if normalized_mode == "per_base":
            feature_rows.append(
                np.concatenate([donor_per_base, acceptor_per_base]).astype(
                    np.float64,
                    copy=False,
                )
            )
            continue
        feature_rows.append(
            np.concatenate(
                [
                    np.asarray([donor_total, acceptor_total], dtype=np.float64),
                    donor_per_base.astype(np.float64, copy=False),
                    acceptor_per_base.astype(np.float64, copy=False),
                ]
            )
        )

    if not feature_rows:
        donor_len = pair_models.donor_pos.seq_length
        acceptor_len = pair_models.acceptor_pos.seq_length
        if normalized_mode == "pair_summary":
            width = 2
        elif normalized_mode == "per_base":
            width = donor_len + acceptor_len
        else:
            width = 2 + donor_len + acceptor_len
        return np.zeros((0, width), dtype=np.float64)
    return np.vstack(feature_rows)


def _build_markov_feature_names(
    *,
    donor_seq_length: int,
    acceptor_seq_length: int,
    feature_mode: str,
) -> list[str]:
    """Build ordered feature-name list for one Markov feature mode."""
    normalized_mode = _normalize_markov_feature_mode(feature_mode)
    donor_names = [f"donor_pos{idx:03d}_log_odds" for idx in range(donor_seq_length)]
    acceptor_names = [
        f"acceptor_pos{idx:03d}_log_odds" for idx in range(acceptor_seq_length)
    ]
    if normalized_mode == "pair_summary":
        return list(MARKOV_FEATURE_NAMES)
    if normalized_mode == "per_base":
        return [*donor_names, *acceptor_names]
    return [*MARKOV_FEATURE_NAMES, *donor_names, *acceptor_names]


def _training_file_signature(path: str) -> dict[str, object]:
    """Build deterministic signature for one training file path."""
    resolved = os.path.realpath(path)
    stat_info = os.stat(resolved)
    return {
        "path": resolved,
        "mtime_ns": int(stat_info.st_mtime_ns),
        "size": int(stat_info.st_size),
    }


def _build_markov_cache_digest(
    *,
    pos_path: str,
    neg_path: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    sequence_transform: str,
    markov_order: int,
    markov_alpha: float,
    markov_feature_mode: str,
    val_frac: float,
    seed: int,
) -> str:
    """Build cache digest key for Markov-derived features."""
    key_payload: dict[str, object] = {
        "format": MARKOV_FEATURE_CACHE_FORMAT_VERSION,
        "train_pos": _training_file_signature(pos_path),
        "train_neg": _training_file_signature(neg_path),
        "donor_len": donor_len,
        "acceptor_len": acceptor_len,
        "sequence_transform": sequence_transform,
        "markov_order": markov_order,
        "markov_alpha": markov_alpha,
        "markov_feature_mode": markov_feature_mode,
        "val_frac": val_frac,
        "seed": seed,
    }
    payload_text = json.dumps(
        key_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload_text.encode("utf-8")).hexdigest()


def _load_markov_feature_bundle(cache_path: str) -> MarkovFeatureBundle:
    """Load Markov feature bundle from cache file."""
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Markov cache payload: {cache_path}")

    cache_format = payload.get("format")
    if cache_format != MARKOV_FEATURE_CACHE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported Markov cache format: "
            f"{cache_format!r}. Expected {MARKOV_FEATURE_CACHE_FORMAT_VERSION!r}."
        )

    bundle = payload.get("bundle")
    if not isinstance(bundle, MarkovFeatureBundle):
        raise ValueError("Markov cache payload is missing valid feature bundle.")
    return bundle


def _save_markov_feature_bundle(
    *,
    cache_path: str,
    bundle: MarkovFeatureBundle,
) -> None:
    """Save Markov feature bundle into a cache file."""
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    payload: dict[str, object] = {
        "format": MARKOV_FEATURE_CACHE_FORMAT_VERSION,
        "bundle": bundle,
    }
    with open(cache_path, "wb") as handle:
        pickle.dump(payload, handle)


def _compute_markov_feature_bundle(
    *,
    pos_path: str,
    neg_path: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    markov_order: int,
    markov_alpha: float,
    markov_feature_mode: str,
    val_frac: float,
    seed: int,
    sequence_transform: str,
) -> MarkovFeatureBundle:
    """Compute Markov models and derived train/val/full feature matrices."""
    raw_examples = read_examples_pair_task_with_metadata(
        pos_path=pos_path,
        neg_path=neg_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        negative_pair_only=True,
    )

    examples: list[tuple[str, str, int]] = []
    for item in raw_examples:
        transformed = apply_pair_sequence_transform(
            PairSequenceRecord(
                donor_seq=item.donor_sequence,
                acceptor_seq=item.acceptor_sequence,
            ),
            transform_mode=sequence_transform,
            intron_half_length=item.intron_half_length,
        )
        examples.append((transformed.donor_seq, transformed.acceptor_seq, item.label))

    if not examples:
        raise ValueError("No pair examples found for training.")

    positive_count = int(sum(label for _, _, label in examples))
    negative_count = len(examples) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Insufficient training examples for pair classification: "
            f"pos={positive_count}, neg={negative_count}."
        )

    train_examples, val_examples = _stratified_split_pair_examples(
        examples,
        val_frac=val_frac,
        seed=seed,
    )

    donor_train = [item[0] for item in train_examples]
    acceptor_train = [item[1] for item in train_examples]
    labels_train = np.asarray([item[2] for item in train_examples], dtype=np.int32)

    donor_val = [item[0] for item in val_examples]
    acceptor_val = [item[1] for item in val_examples]
    labels_val = np.asarray([item[2] for item in val_examples], dtype=np.int32)

    split_pair_models = _build_pair_markov_models(
        donor_sequences=donor_train,
        acceptor_sequences=acceptor_train,
        labels=labels_train,
        order=markov_order,
        alpha=markov_alpha,
    )

    train_features = _compute_pair_markov_features(
        donor_train,
        acceptor_train,
        pair_models=split_pair_models,
        feature_mode=markov_feature_mode,
    )
    val_features = _compute_pair_markov_features(
        donor_val,
        acceptor_val,
        pair_models=split_pair_models,
        feature_mode=markov_feature_mode,
    )

    donor_all = [item[0] for item in examples]
    acceptor_all = [item[1] for item in examples]
    labels_all = np.asarray([item[2] for item in examples], dtype=np.int32)
    final_pair_models = _build_pair_markov_models(
        donor_sequences=donor_all,
        acceptor_sequences=acceptor_all,
        labels=labels_all,
        order=markov_order,
        alpha=markov_alpha,
    )
    final_features = _compute_pair_markov_features(
        donor_all,
        acceptor_all,
        pair_models=final_pair_models,
        feature_mode=markov_feature_mode,
    )
    feature_names = tuple(
        _build_markov_feature_names(
            donor_seq_length=final_pair_models.donor_pos.seq_length,
            acceptor_seq_length=final_pair_models.acceptor_pos.seq_length,
            feature_mode=markov_feature_mode,
        )
    )

    return MarkovFeatureBundle(
        train_features=train_features,
        val_features=val_features,
        labels_train=labels_train,
        labels_val=labels_val,
        final_features=final_features,
        labels_all=labels_all,
        final_pair_models=final_pair_models,
        feature_names=feature_names,
        n_total=len(examples),
        n_train=len(train_examples),
        n_val=len(val_examples),
    )


def _load_or_build_markov_feature_bundle(
    *,
    pos_path: str,
    neg_path: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    markov_order: int,
    markov_alpha: float,
    markov_feature_mode: str,
    val_frac: float,
    seed: int,
    sequence_transform: str,
    markov_cache_mode: str,
    markov_cache_dir: Optional[str],
) -> tuple[MarkovFeatureBundle, bool, Optional[str]]:
    """Load Markov feature bundle from cache or compute it on demand."""
    normalized_cache_mode = _normalize_markov_cache_mode(markov_cache_mode)
    if normalized_cache_mode == "off":
        bundle = _compute_markov_feature_bundle(
            pos_path=pos_path,
            neg_path=neg_path,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
            markov_order=markov_order,
            markov_alpha=markov_alpha,
            markov_feature_mode=markov_feature_mode,
            val_frac=val_frac,
            seed=seed,
            sequence_transform=sequence_transform,
        )
        return bundle, False, None

    if markov_cache_dir is None or markov_cache_dir.strip() == "":
        raise ValueError("markov_cache_dir must be set when markov_cache_mode != off.")

    digest = _build_markov_cache_digest(
        pos_path=pos_path,
        neg_path=neg_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        sequence_transform=sequence_transform,
        markov_order=markov_order,
        markov_alpha=markov_alpha,
        markov_feature_mode=markov_feature_mode,
        val_frac=val_frac,
        seed=seed,
    )
    cache_path = os.path.join(markov_cache_dir, f"{digest}.pkl")

    if normalized_cache_mode == "auto" and os.path.exists(cache_path):
        try:
            bundle = _load_markov_feature_bundle(cache_path)
            print(f"[markov-cache] hit: {cache_path}")
            return bundle, True, cache_path
        except Exception as exc:
            print(f"[markov-cache] stale cache ignored: {cache_path} ({exc})")

    bundle = _compute_markov_feature_bundle(
        pos_path=pos_path,
        neg_path=neg_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        markov_order=markov_order,
        markov_alpha=markov_alpha,
        markov_feature_mode=markov_feature_mode,
        val_frac=val_frac,
        seed=seed,
        sequence_transform=sequence_transform,
    )
    _save_markov_feature_bundle(cache_path=cache_path, bundle=bundle)
    print(f"[markov-cache] saved: {cache_path}")
    return bundle, False, cache_path


def _stratified_split_pair_examples(
    examples: Sequence[tuple[str, str, int]],
    *,
    val_frac: float,
    seed: int,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    """Split pair examples into stratified train/validation subsets."""
    if val_frac <= 0.0 or val_frac >= 1.0:
        raise ValueError("--val_frac must satisfy 0 < val_frac < 1.")

    positive_examples = [item for item in examples if item[2] == 1]
    negative_examples = [item for item in examples if item[2] == 0]
    if len(positive_examples) < 2 or len(negative_examples) < 2:
        raise ValueError(
            "Need at least two positive and two negative examples for "
            "train/validation split."
        )

    rng = random.Random(seed)
    rng.shuffle(positive_examples)
    rng.shuffle(negative_examples)

    n_val_pos = max(1, int(round(len(positive_examples) * val_frac)))
    n_val_neg = max(1, int(round(len(negative_examples) * val_frac)))
    n_val_pos = min(n_val_pos, len(positive_examples) - 1)
    n_val_neg = min(n_val_neg, len(negative_examples) - 1)

    train_examples = (
        positive_examples[n_val_pos:] + negative_examples[n_val_neg:]
    )
    val_examples = positive_examples[:n_val_pos] + negative_examples[:n_val_neg]
    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    return train_examples, val_examples


def _build_xgb_classifier(
    model_args: argparse.Namespace,
    *,
    seed: int,
) -> XGBClassifierProtocol:
    """Build one configured XGBoost classifier instance."""
    xgb_class = _require_xgb_classifier_class()
    classifier = xgb_class(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(model_args.xgb_n_estimators),
        max_depth=int(model_args.xgb_max_depth),
        learning_rate=float(model_args.xgb_learning_rate),
        subsample=float(model_args.xgb_subsample),
        colsample_bytree=float(model_args.xgb_colsample_bytree),
        min_child_weight=float(model_args.xgb_min_child_weight),
        reg_lambda=float(model_args.xgb_reg_lambda),
        reg_alpha=float(model_args.xgb_reg_alpha),
        tree_method=str(model_args.xgb_tree_method),
        random_state=seed,
        n_jobs=int(model_args.xgb_n_jobs),
    )
    return cast(XGBClassifierProtocol, classifier)


def _predict_positive_probability(
    classifier: XGBClassifierProtocol,
    features: np.ndarray,
) -> np.ndarray:
    """Return positive-class probabilities from a trained classifier."""
    raw_probs = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    if raw_probs.ndim != 2 or raw_probs.shape[0] != features.shape[0]:
        raise ValueError(
            "predict_proba must return a 2-D array with one row per sample."
        )
    if raw_probs.shape[1] < 2:
        raise ValueError(
            "predict_proba must return two columns for binary classification."
        )
    return np.clip(raw_probs[:, 1], 1e-7, 1.0 - 1e-7)


def _evaluate_binary_scores(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Evaluate common binary classification metrics."""
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("labels and probabilities must be 1-D arrays.")
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("labels and probabilities must have the same length.")

    metrics: Dict[str, float] = {}
    if labels.size == 0:
        return metrics

    labels_i32 = labels.astype(np.int32, copy=False)
    metrics["acc@0.5"] = float(np.mean((probabilities >= 0.5) == (labels_i32 == 1)))

    try:
        metrics["max_f1"] = float(_fallback_max_f1(labels_i32, probabilities))
    except ValueError:
        pass

    unique_labels = np.unique(labels_i32)
    if unique_labels.size > 1:
        roc_auc_value: float | None = None
        if roc_auc_score is not None:
            try:
                roc_auc_value = float(roc_auc_score(labels_i32, probabilities))
            except Exception:
                roc_auc_value = None
        if roc_auc_value is None:
            try:
                roc_auc_value = float(_fallback_roc_auc(labels_i32, probabilities))
            except ValueError:
                roc_auc_value = None
        if roc_auc_value is not None:
            metrics["roc_auc"] = roc_auc_value

        pr_auc_value: float | None = None
        if average_precision_score is not None:
            try:
                pr_auc_value = float(
                    average_precision_score(labels_i32, probabilities)
                )
            except Exception:
                pr_auc_value = None
        if pr_auc_value is None:
            try:
                pr_auc_value = float(
                    _fallback_average_precision(labels_i32, probabilities)
                )
            except ValueError:
                pr_auc_value = None
        if pr_auc_value is not None:
            metrics["pr_auc"] = pr_auc_value

    return metrics


def _select_best_metric(metrics: Dict[str, float]) -> tuple[str, float]:
    """Select one primary metric for checkpoint ranking."""
    priority = ("pr_auc", "max_f1", "roc_auc", "acc@0.5")
    for key in priority:
        value = metrics.get(key)
        if value is not None:
            return key, float(value)
    raise ValueError("At least one validation metric is required.")


def _load_pair_checkpoint(
    checkpoint_path: str,
) -> tuple[PairMarkovModels, XGBClassifierProtocol, dict[str, object]]:
    """Load pair checkpoint payload and validate required objects."""
    try:
        with open(checkpoint_path, "rb") as handle:
            payload = pickle.load(handle)
    except ModuleNotFoundError as exc:
        if exc.name == "xgboost":
            raise RuntimeError(
                "xgboost is required to load markov_xgboost checkpoints. "
                "Install xgboost in the current environment."
            ) from exc
        raise

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")

    checkpoint_format = payload.get("format")
    if checkpoint_format != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format for markov_xgboost: "
            f"{checkpoint_format!r}."
        )

    pair_models = payload.get("pair_markov_models")
    if not isinstance(pair_models, PairMarkovModels):
        raise ValueError("Checkpoint is missing valid pair Markov models.")

    classifier_obj = payload.get("xgb_model")
    if classifier_obj is None:
        raise ValueError("Checkpoint is missing xgb_model.")
    if not hasattr(classifier_obj, "predict_proba"):
        raise ValueError("Checkpoint xgb_model does not expose predict_proba.")

    classifier = cast(XGBClassifierProtocol, classifier_obj)
    return pair_models, classifier, payload


def _train_pair_markov_xgboost(
    *,
    pos_path: str,
    neg_path: str,
    checkpoint_path: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_window_len: int,
    acceptor_window_len: int,
    markov_order: int,
    markov_alpha: float,
    markov_feature_mode: str,
    markov_cache_mode: str,
    markov_cache_dir: Optional[str],
    val_frac: float,
    seed: int,
    model_args: argparse.Namespace,
    sequence_transform: str,
) -> dict[str, object]:
    """Train Markov feature extractor + XGBoost pair classifier."""
    if sequence_transform not in SEQUENCE_TRANSFORM_CHOICES:
        raise ValueError(
            "Unsupported --sequence_transform: "
            f"{sequence_transform}. Supported: {SEQUENCE_TRANSFORM_CHOICES}"
        )

    feature_bundle, markov_cache_hit, markov_cache_path = (
        _load_or_build_markov_feature_bundle(
            pos_path=pos_path,
            neg_path=neg_path,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
            markov_order=markov_order,
            markov_alpha=markov_alpha,
            markov_feature_mode=markov_feature_mode,
            val_frac=val_frac,
            seed=seed,
            sequence_transform=sequence_transform,
            markov_cache_mode=markov_cache_mode,
            markov_cache_dir=markov_cache_dir,
        )
    )

    train_features = feature_bundle.train_features
    val_features = feature_bundle.val_features
    labels_train = feature_bundle.labels_train
    labels_val = feature_bundle.labels_val

    split_classifier = _build_xgb_classifier(model_args, seed=seed)
    split_classifier.fit(train_features, labels_train)

    train_probs = _predict_positive_probability(split_classifier, train_features)
    val_probs = _predict_positive_probability(split_classifier, val_features)
    train_metrics = _evaluate_binary_scores(labels_train, train_probs)
    val_metrics = _evaluate_binary_scores(labels_val, val_probs)
    best_metric, best_score = _select_best_metric(val_metrics)

    final_pair_models = feature_bundle.final_pair_models
    feature_names = feature_bundle.feature_names
    final_features = feature_bundle.final_features
    labels_all = feature_bundle.labels_all

    final_classifier = _build_xgb_classifier(model_args, seed=seed)
    final_classifier.fit(final_features, labels_all)

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT_VERSION,
        "pair_markov_models": final_pair_models,
        "xgb_model": final_classifier,
        "feature_names": feature_names,
        "donor_window_len": donor_window_len,
        "acceptor_window_len": acceptor_window_len,
        "model_config": {
            "markov_order": markov_order,
            "markov_alpha": markov_alpha,
            "markov_feature_mode": markov_feature_mode,
            "xgb_n_estimators": int(model_args.xgb_n_estimators),
            "xgb_max_depth": int(model_args.xgb_max_depth),
            "xgb_learning_rate": float(model_args.xgb_learning_rate),
            "xgb_subsample": float(model_args.xgb_subsample),
            "xgb_colsample_bytree": float(model_args.xgb_colsample_bytree),
            "xgb_min_child_weight": float(model_args.xgb_min_child_weight),
            "xgb_reg_lambda": float(model_args.xgb_reg_lambda),
            "xgb_reg_alpha": float(model_args.xgb_reg_alpha),
            "xgb_tree_method": str(model_args.xgb_tree_method),
            "xgb_n_jobs": int(model_args.xgb_n_jobs),
            "sequence_transform": sequence_transform,
            "seed": seed,
        },
    }
    with open(checkpoint_path, "wb") as handle:
        pickle.dump(checkpoint_payload, handle)

    return {
        "checkpoint": checkpoint_path,
        "best_metric": best_metric,
        "best_score": best_score,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "n_total": int(feature_bundle.n_total),
        "n_train": int(feature_bundle.n_train),
        "n_val": int(feature_bundle.n_val),
        "train_pos": int(np.sum(labels_train == 1)),
        "train_neg": int(np.sum(labels_train == 0)),
        "val_pos": int(np.sum(labels_val == 1)),
        "val_neg": int(np.sum(labels_val == 0)),
        "markov_cache_mode": markov_cache_mode,
        "markov_cache_hit": bool(markov_cache_hit),
        "markov_cache_path": markov_cache_path,
    }


def _resolve_markov_cache_dir(
    *,
    species: str,
    cache_mode: str,
    cache_dir: Optional[str],
) -> Optional[str]:
    """Resolve Markov cache directory with species-aware default."""
    normalized_mode = _normalize_markov_cache_mode(cache_mode)
    normalized_dir: Optional[str] = cache_dir
    if normalized_dir is not None:
        normalized_dir = normalized_dir.strip()
        if normalized_dir == "":
            normalized_dir = None
    if normalized_mode == "off":
        return None
    if normalized_dir is not None:
        return normalized_dir

    train_dir = species_data_dirs(species)["train"]
    return os.path.join(train_dir, "markov_xgboost_cache")


def _infer_pair_site_scores(
    *,
    pair_rows: Sequence[dict[str, object]],
    checkpoint_path: str,
    sequence_transform: str,
) -> list[dict[str, object]]:
    """Infer pair scores from donor/acceptor rows using one checkpoint."""
    if sequence_transform not in SEQUENCE_TRANSFORM_CHOICES:
        raise ValueError(
            "Unsupported --sequence_transform: "
            f"{sequence_transform}. Supported: {SEQUENCE_TRANSFORM_CHOICES}"
        )

    pair_models, classifier, payload = _load_pair_checkpoint(checkpoint_path)
    model_config_obj = payload.get("model_config", {})
    if isinstance(model_config_obj, dict):
        feature_mode_raw = model_config_obj.get("markov_feature_mode", "pair_summary")
    else:
        feature_mode_raw = "pair_summary"
    feature_mode = _normalize_markov_feature_mode(feature_mode_raw)

    transformed_pairs: list[tuple[str, str]] = []
    for row in pair_rows:
        transformed = apply_pair_sequence_transform(
            PairSequenceRecord(
                donor_seq=str(row["donor_seq"]),
                acceptor_seq=str(row["acceptor_seq"]),
            ),
            transform_mode=sequence_transform,
            intron_half_length=(
                int(row["intron_half_length"])
                if row.get("intron_half_length") is not None
                else None
            ),
        )
        transformed_pairs.append((transformed.donor_seq, transformed.acceptor_seq))

    donor_sequences = [item[0] for item in transformed_pairs]
    acceptor_sequences = [item[1] for item in transformed_pairs]
    pair_features = _compute_pair_markov_features(
        donor_sequences,
        acceptor_sequences,
        pair_models=pair_models,
        feature_mode=feature_mode,
    )
    probabilities = _predict_positive_probability(classifier, pair_features)

    out_rows: list[dict[str, object]] = []
    for row, score in zip(pair_rows, probabilities):
        out_rows.append(
            {
                "transcript_id": str(row["transcript_id"]),
                "intron_index": int(row["intron_index"]),
                "site_type": "pair",
                "score": float(score),
            }
        )
    return out_rows


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register Markov+XGBoost training arguments."""
    parser.add_argument("--train_target", choices=["pair"], default="pair")
    parser.add_argument(
        "--sequence_transform",
        choices=list(SEQUENCE_TRANSFORM_CHOICES),
        default="none",
    )
    parser.add_argument(
        "--markov_order",
        type=int,
        default=2,
        help=(
            "Markov order used in PWM scoring (how many previous bases are "
            "conditioned for each base)."
        ),
    )
    parser.add_argument(
        "--markov_alpha",
        type=float,
        default=0.5,
        help="Additive smoothing pseudo-count for Markov PWM probabilities.",
    )
    parser.add_argument(
        "--markov_feature_mode",
        choices=list(MARKOV_FEATURE_MODE_CHOICES),
        default="per_base",
        help=(
            "Feature mode passed to XGBoost: pair_summary (2-dim), per_base "
            "(donor+acceptor position-wise log-odds), or hybrid."
        ),
    )
    parser.add_argument(
        "--markov_cache_mode",
        choices=list(MARKOV_CACHE_MODE_CHOICES),
        default="auto",
        help=(
            "Markov-feature cache mode: off (always recompute), auto "
            "(reuse if available), or refresh (force recompute and overwrite)."
        ),
    )
    parser.add_argument(
        "--markov_cache_dir",
        default=None,
        help=(
            "Directory for serialized precomputed Markov features. "
            "If omitted and cache mode is not off, defaults to "
            "data/<species>/train/markov_xgboost_cache."
        ),
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Validation split fraction for checkpoint ranking.",
    )
    parser.add_argument("--xgb_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_max_depth", type=int, default=4)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=1.0)
    parser.add_argument("--xgb_min_child_weight", type=float, default=1.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    parser.add_argument(
        "--xgb_tree_method",
        choices=["auto", "hist", "exact"],
        default="hist",
    )
    parser.add_argument(
        "--xgb_n_jobs",
        type=int,
        default=-1,
        help="XGBoost thread count; use -1 to use all available cores.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Unused compatibility argument for generic hparam_search "
            "infrastructure."
        ),
    )
    parser.add_argument("--tag", default=None)


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register Markov+XGBoost inference arguments."""
    parser.add_argument(
        "--sequence_transform",
        choices=list(SEQUENCE_TRANSFORM_CHOICES),
        default="none",
    )


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train Markov+XGBoost pair model and save one pair checkpoint."""
    markov_feature_mode = _normalize_markov_feature_mode(
        getattr(model_args, "markov_feature_mode", "per_base")
    )
    markov_cache_mode = _normalize_markov_cache_mode(
        getattr(model_args, "markov_cache_mode", "auto")
    )
    _validate_markov_params(
        order=int(model_args.markov_order),
        alpha=float(model_args.markov_alpha),
    )
    _validate_xgb_params(model_args)

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
    validate_window_args(donor_len=donor_len, acceptor_len=acceptor_len)

    donor_window_len = donor_len if donor_len is not None else 50
    acceptor_window_len = acceptor_len if acceptor_len is not None else 50

    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=False,
        tasks=("pair",),
    )
    pair_checkpoint_path = task_checkpoint_paths["pair"]

    train_target = resolve_train_target(model_args, allowed_targets=("pair",))
    tasks_to_train = resolve_tasks_to_train(train_target, both_tasks=("pair",))
    if tasks_to_train != ["pair"]:
        raise ValueError("markov_xgboost expects train_target=pair.")
    markov_cache_dir = _resolve_markov_cache_dir(
        species=common_args.species,
        cache_mode=markov_cache_mode,
        cache_dir=getattr(model_args, "markov_cache_dir", None),
    )

    pair_summary = _train_pair_markov_xgboost(
        pos_path=train_pos_path,
        neg_path=train_neg_path,
        checkpoint_path=pair_checkpoint_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
        markov_order=int(model_args.markov_order),
        markov_alpha=float(model_args.markov_alpha),
        markov_feature_mode=markov_feature_mode,
        markov_cache_mode=markov_cache_mode,
        markov_cache_dir=markov_cache_dir,
        val_frac=float(model_args.val_frac),
        seed=int(common_args.seed),
        model_args=model_args,
        sequence_transform=str(model_args.sequence_transform),
    )

    return {
        "model": "markov_xgboost",
        "species": common_args.species,
        "train_pos_path": train_pos_path,
        "train_neg_path": train_neg_path,
        "donor_len": donor_len,
        "acceptor_len": acceptor_len,
        "train_target": train_target,
        "sequence_transform": model_args.sequence_transform,
        "seed": common_args.seed,
        "pair_checkpoint_path": pair_checkpoint_path,
        "checkpoint_name": os.path.basename(pair_checkpoint_path),
        "markov_order": int(model_args.markov_order),
        "markov_alpha": float(model_args.markov_alpha),
        "markov_feature_mode": markov_feature_mode,
        "markov_cache_mode": markov_cache_mode,
        "markov_cache_dir": markov_cache_dir,
        "val_frac": float(model_args.val_frac),
        "xgb_n_estimators": int(model_args.xgb_n_estimators),
        "xgb_max_depth": int(model_args.xgb_max_depth),
        "xgb_learning_rate": float(model_args.xgb_learning_rate),
        "xgb_subsample": float(model_args.xgb_subsample),
        "xgb_colsample_bytree": float(model_args.xgb_colsample_bytree),
        "xgb_min_child_weight": float(model_args.xgb_min_child_weight),
        "xgb_reg_lambda": float(model_args.xgb_reg_lambda),
        "xgb_reg_alpha": float(model_args.xgb_reg_alpha),
        "xgb_tree_method": str(model_args.xgb_tree_method),
        "xgb_n_jobs": int(model_args.xgb_n_jobs),
        "inferred_train_len": inferred_train_len,
        "pair": pair_summary,
    }


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Run pair-level inference using Markov+XGBoost checkpoint."""
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
    validate_window_args(donor_len=donor_len, acceptor_len=acceptor_len)

    test_tsv = resolve_test_tsv(common_args.species, common_args.test_tsv)
    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=True,
        tasks=("pair",),
    )
    pair_checkpoint_path = task_checkpoint_paths["pair"]

    pair_rows, skipped_short, skipped_unpaired = read_test_pair_rows(
        test_tsv=test_tsv,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )
    print(f"Loaded test pairs: {len(pair_rows)}")
    if skipped_short:
        print(f"Skipped short sites: {skipped_short}")
    if skipped_unpaired:
        print(f"Skipped unpaired introns: {skipped_unpaired}")

    return _infer_pair_site_scores(
        pair_rows=pair_rows,
        checkpoint_path=pair_checkpoint_path,
        sequence_transform=str(model_args.sequence_transform),
    )
