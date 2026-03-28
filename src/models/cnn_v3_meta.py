"""CNN v3 meta pair model using multiple base-model outputs as input.

This module implements a stacking-style pair model:
1. Load multiple pretrained pair checkpoints (typically ``cnn_v2`` outputs).
2. Score donor/acceptor pairs with each base model.
3. Train one small meta MLP on stacked base scores.

The resulting checkpoint stores the meta MLP and the base checkpoint list.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import cnn_v2
from util.data_proc import (
    build_run_name,
    read_examples_pair_task_with_metadata,
    read_test_pair_rows,
    resolve_effective_window_lengths,
    resolve_test_tsv,
    resolve_train_paths,
    validate_window_args,
)
from util.model_task_paths import (
    resolve_required_checkpoint_paths,
    resolve_train_target,
)
from util.model_runtime import log10_sigmoid_np
from util.sequence_transform import (
    PairSequenceRecord,
    apply_pair_sequence_transform,
)
from util.transcript_eval import SCORE_SPACE_FIELD, SCORE_SPACE_LOG10
from util.training_control import resolve_training_epoch_budget

try:
    from sklearn.metrics import average_precision_score
except ImportError:  # pragma: no cover
    average_precision_score = None


@dataclass(frozen=True)
class _PairExample:
    """Pair sequence example for meta learning."""

    donor_seq: str
    acceptor_seq: str
    label: int


class _MetaFeatureDataset(Dataset):
    """Dataset for meta-features and labels."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x = torch.from_numpy(x.astype(np.float32, copy=False))
        self._y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[index], self._y[index]


class _MetaPairMLP(nn.Module):
    """Small MLP for pair classification from base-model scores."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)[:, 0]


def _parse_checkpoint_csv(raw_value: str, *, arg_name: str) -> list[str]:
    """Parse comma-separated checkpoint paths."""
    items = [part.strip() for part in raw_value.split(",") if part.strip() != ""]
    if not items:
        raise ValueError(f"{arg_name} must contain at least one checkpoint path.")
    resolved: list[str] = []
    seen: set[str] = set()
    for item in items:
        path = str(Path(item).expanduser().resolve())
        if path in seen:
            continue
        seen.add(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Base checkpoint not found: {path}")
        resolved.append(path)
    return resolved


def _resolve_base_pair_checkpoints(model_args: argparse.Namespace) -> list[str]:
    """Resolve required base pair checkpoint paths from args."""
    raw = str(getattr(model_args, "base_pair_checkpoints", "")).strip()
    checkpoints = _parse_checkpoint_csv(raw, arg_name="--base_pair_checkpoints")
    incompatible: list[str] = []
    for checkpoint_path in checkpoints:
        if not _is_compatible_cnn_pair_v2_checkpoint(checkpoint_path):
            incompatible.append(checkpoint_path)
    if incompatible:
        incompatible_text = ", ".join(incompatible)
        raise ValueError(
            "--base_pair_checkpoints must reference cnn_v2-compatible pair "
            f"checkpoints. Incompatible paths: {incompatible_text}"
        )
    return checkpoints


def _normalize_model_state_key(key: str) -> str:
    """Normalize checkpoint model-state key prefixes."""
    if key.startswith("_orig_mod."):
        return key.removeprefix("_orig_mod.")
    return key


def _is_cnn_pair_v2_checkpoint_payload(payload: object) -> bool:
    """Return whether a checkpoint payload matches cnn_pair_v2 format."""
    if not isinstance(payload, dict):
        return False
    model_config_obj = payload.get("model_config")
    model_state_obj = payload.get("model_state")
    if not isinstance(model_config_obj, dict):
        return False
    if not isinstance(model_state_obj, dict):
        return False

    input_mode = str(model_config_obj.get("input_mode", "")).strip().lower()
    pair_mode = str(model_config_obj.get("pair_mode", "")).strip().lower()
    if input_mode not in cnn_v2.INPUT_MODE_CHOICES:
        return False
    if pair_mode not in {"pair", "independent"}:
        return False

    normalized_keys = [
        _normalize_model_state_key(str(key)) for key in model_state_obj.keys()
    ]
    has_donor_encoder = any(key.startswith("donor_encoder.") for key in normalized_keys)
    has_acceptor_encoder = any(
        key.startswith("acceptor_encoder.") for key in normalized_keys
    )
    has_fc_head = any(key.startswith("fc.") for key in normalized_keys)
    return has_donor_encoder and has_acceptor_encoder and has_fc_head


def _is_compatible_cnn_pair_v2_checkpoint(checkpoint_path: str) -> bool:
    """Return whether one checkpoint path is cnn_pair_v2-compatible for cnn_v3."""
    try:
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return False
    return _is_cnn_pair_v2_checkpoint_payload(payload)


def _stratified_split_indices(
    labels: np.ndarray,
    *,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stratified train/val indices."""
    if labels.ndim != 1:
        raise ValueError("labels must be 1D.")
    if val_frac <= 0.0 or val_frac >= 1.0:
        raise ValueError("val_frac must satisfy 0 < val_frac < 1.")
    pos_idx = np.where(labels > 0.5)[0]
    neg_idx = np.where(labels <= 0.5)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError("Both positive and negative labels are required.")
    rng = np.random.default_rng(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_val_pos = max(1, int(pos_idx.size * val_frac))
    n_val_neg = max(1, int(neg_idx.size * val_frac))
    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _load_pair_examples(
    *,
    pos_path: str,
    neg_path: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    sequence_transform: str,
) -> list[_PairExample]:
    """Load transformed pair examples for meta training."""
    raw_examples = read_examples_pair_task_with_metadata(
        pos_path=pos_path,
        neg_path=neg_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        negative_pair_only=True,
    )
    examples: list[_PairExample] = []
    for item in raw_examples:
        transformed = apply_pair_sequence_transform(
            PairSequenceRecord(
                donor_seq=item.donor_sequence,
                acceptor_seq=item.acceptor_sequence,
            ),
            transform_mode=sequence_transform,
            intron_half_length=item.intron_half_length,
        )
        examples.append(
            _PairExample(
                donor_seq=transformed.donor_seq,
                acceptor_seq=transformed.acceptor_seq,
                label=int(item.label),
            )
        )
    return examples


def _predict_with_base_checkpoint(
    *,
    checkpoint_path: str,
    pairs: Sequence[tuple[str, str]],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Predict pair probabilities using one base pair checkpoint."""
    model, payload = cnn_v2.load_pair_model(checkpoint_path, device)
    donor_window_len = int(payload.get("donor_window_len", 50))
    acceptor_window_len = int(payload.get("acceptor_window_len", 50))
    config_obj = payload.get("model_config", {})
    if not isinstance(config_obj, dict):
        raise TypeError("Invalid base checkpoint model_config.")
    input_mode = str(config_obj.get("input_mode", "onehot"))
    bpe_pretrained_model_name = str(
        config_obj.get("bpe_pretrained_model_name", cnn_v2.BPE_DEFAULT_MODEL_NAME)
    )
    bpe_pretrained_revision_obj = config_obj.get("bpe_pretrained_revision")
    bpe_pretrained_revision = (
        str(bpe_pretrained_revision_obj)
        if bpe_pretrained_revision_obj is not None
        else None
    )
    bpe_trust_remote_code = bool(config_obj.get("bpe_trust_remote_code", False))
    return cnn_v2.score_pair_sequences(
        model=model,
        pairs=pairs,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
        device=device,
        input_mode=input_mode,
        bpe_pretrained_model_name=bpe_pretrained_model_name,
        bpe_pretrained_revision=bpe_pretrained_revision,
        bpe_trust_remote_code=bpe_trust_remote_code,
        batch_size=batch_size,
        use_amp=False,
        amp_dtype=None,
    )


def _build_meta_features(
    *,
    base_checkpoints: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Build stacked feature matrix from multiple base model predictions."""
    if not base_checkpoints:
        raise ValueError("base_checkpoints must not be empty.")
    columns: list[np.ndarray] = []
    for checkpoint_path in base_checkpoints:
        scores = _predict_with_base_checkpoint(
            checkpoint_path=checkpoint_path,
            pairs=pairs,
            device=device,
            batch_size=batch_size,
        )
        if scores.shape[0] != len(pairs):
            raise ValueError("Base score length does not match pair count.")
        columns.append(scores.astype(np.float32, copy=False))
    return np.column_stack(columns).astype(np.float32, copy=False)


@torch.no_grad()
def _evaluate_meta_model(
    model: _MetaPairMLP,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluate meta model and return validation metrics."""
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for x, y in loader:
        logits = model(x.to(device))
        logits_list.append(logits.detach().cpu().numpy())
        labels_list.append(y.detach().cpu().numpy())
    if not logits_list:
        return {"acc@0.5": 0.0}
    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list).astype(np.int32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    metrics: dict[str, float] = {
        "acc@0.5": float(np.mean((probs >= 0.5) == (labels >= 0.5)))
    }
    max_f1: Optional[float] = None
    try:
        max_f1 = float(cnn_v2._fallback_max_f1(labels, probs))
    except ValueError:
        max_f1 = None
    if max_f1 is not None:
        metrics["max_f1"] = max_f1
    if labels.min() != labels.max():
        pr_auc: Optional[float] = None
        if average_precision_score is not None:
            try:
                pr_auc = float(average_precision_score(labels, probs))
            except ValueError:
                pr_auc = None
        if pr_auc is None:
            try:
                pr_auc = float(cnn_v2._fallback_average_precision(labels, probs))
            except ValueError:
                pr_auc = None
        if pr_auc is not None:
            metrics["pr_auc"] = pr_auc
    return metrics


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Register training arguments for ``cnn_v3`` meta model."""
    cnn_v2.add_train_args(parser)
    parser.add_argument(
        "--base_pair_checkpoints",
        type=str,
        default=None,
        help=(
            "Comma-separated pretrained pair checkpoint paths used as "
            "meta-model inputs."
        ),
    )
    parser.add_argument(
        "--meta_hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension for meta MLP.",
    )
    parser.add_argument(
        "--meta_dropout",
        type=float,
        default=0.2,
        help="Dropout for meta MLP.",
    )


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register inference arguments for ``cnn_v3``."""
    cnn_v2.add_infer_args(parser)


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> dict[str, object]:
    """Train meta pair model on stacked outputs from base pair models."""
    requested_pair_mode = str(getattr(model_args, "pair_mode", "pair")).strip().lower()
    if requested_pair_mode != "pair":
        raise ValueError("cnn_v3 currently supports only --pair_mode=pair.")

    requested_train_target = resolve_train_target(
        model_args,
        allowed_targets=("both", "donor", "acceptor", "pair"),
    )
    if requested_train_target != "pair":
        print("[cnn_v3] forcing --train_target=pair.")
    base_checkpoints = _resolve_base_pair_checkpoints(model_args)

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

    examples = _load_pair_examples(
        pos_path=train_pos_path,
        neg_path=train_neg_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        sequence_transform=str(model_args.sequence_transform),
    )
    if not examples:
        raise ValueError("No training examples available for cnn_v3.")
    pairs = [(row.donor_seq, row.acceptor_seq) for row in examples]
    labels = np.asarray([row.label for row in examples], dtype=np.float32)

    device = cnn_v2.pick_device(common_args.device)
    feature_batch_size = int(model_args.batch_size)
    features = _build_meta_features(
        base_checkpoints=base_checkpoints,
        pairs=pairs,
        device=device,
        batch_size=feature_batch_size,
    )
    train_idx, val_idx = _stratified_split_indices(
        labels,
        val_frac=float(model_args.val_frac),
        seed=int(common_args.seed),
    )

    train_ds = _MetaFeatureDataset(features[train_idx], labels[train_idx])
    val_ds = _MetaFeatureDataset(features[val_idx], labels[val_idx])
    train_loader = DataLoader(
        train_ds,
        batch_size=int(model_args.batch_size),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(model_args.batch_size),
        shuffle=False,
    )

    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=False,
        tasks=("pair",),
    )
    pair_checkpoint_path = task_checkpoint_paths["pair"]
    checkpoint_parent = Path(pair_checkpoint_path).parent
    checkpoint_parent.mkdir(parents=True, exist_ok=True)

    resolved_epochs, _epochs_auto = resolve_training_epoch_budget(
        epochs_arg=model_args.epochs,
        max_epochs=int(model_args.max_epochs),
    )
    model = _MetaPairMLP(
        input_dim=int(features.shape[1]),
        hidden_dim=int(model_args.meta_hidden_dim),
        dropout=float(model_args.meta_dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_args.lr),
        weight_decay=float(model_args.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    best_score = float("-inf")
    best_epoch = 0
    best_max_f1: Optional[float] = None
    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, resolved_epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            logits = model(x_batch.to(device))
            loss = criterion(logits, y_batch.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_metrics = _evaluate_meta_model(model, val_loader, device)
        pr_auc = val_metrics.get("pr_auc")
        max_f1 = val_metrics.get("max_f1")
        score = pr_auc if pr_auc is not None else float(val_metrics["acc@0.5"])
        if max_f1 is not None:
            best_max_f1 = (
                max_f1 if best_max_f1 is None else max(best_max_f1, max_f1)
            )
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "pr_auc": pr_auc,
                "max_f1": max_f1,
                "acc@0.5": float(val_metrics["acc@0.5"]),
                "objective_score": float(score),
                "improved": improved,
            }
        )
    if best_state is None:
        raise RuntimeError("Failed to train cnn_v3 meta model.")

    torch.save(
        {
            "task": "pair",
            "checkpoint_format": "cnn_v3_meta_v1",
            "meta_hidden_dim": int(model_args.meta_hidden_dim),
            "meta_dropout": float(model_args.meta_dropout),
            "input_dim": int(features.shape[1]),
            "base_pair_checkpoints": list(base_checkpoints),
            "model_state": best_state,
        },
        pair_checkpoint_path,
    )

    run_name = build_run_name(
        model_name="cnn_v3_meta",
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        lr=float(model_args.lr),
        batch_size=int(model_args.batch_size),
        epochs=resolved_epochs,
        tag=model_args.tag,
    )

    return {
        "model": "cnn_v3_meta",
        "species": common_args.species,
        "base_pair_checkpoints": list(base_checkpoints),
        "donor_len": donor_len,
        "acceptor_len": acceptor_len,
        "epochs": resolved_epochs,
        "epochs_config": str(model_args.epochs),
        "max_epochs": model_args.max_epochs,
        "batch_size": model_args.batch_size,
        "lr": model_args.lr,
        "train_target": "pair",
        "sequence_transform": model_args.sequence_transform,
        "seed": common_args.seed,
        "device": device,
        "checkpoint_name": Path(pair_checkpoint_path).name,
        "pair_checkpoint_path": pair_checkpoint_path,
        "pair_mode": "pair",
        "meta_hidden_dim": int(model_args.meta_hidden_dim),
        "meta_dropout": float(model_args.meta_dropout),
        "run_name": run_name,
        "inferred_train_len": inferred_train_len,
        "pair": {
            "best_metric": "pr_auc_or_acc",
            "best_epoch": best_epoch,
            "best_score": float(best_score),
            "best_max_f1": best_max_f1,
            "epoch_history": history,
            "checkpoint": pair_checkpoint_path,
            "num_examples": int(labels.shape[0]),
            "num_base_models": len(base_checkpoints),
        },
    }


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Infer pair scores by stacked predictions from base models."""
    task_checkpoint_paths = resolve_required_checkpoint_paths(
        common_args,
        require_exists=True,
        tasks=("pair",),
    )
    meta_checkpoint_path = task_checkpoint_paths["pair"]
    payload = torch.load(meta_checkpoint_path, map_location="cpu", weights_only=False)
    base_checkpoints_obj = payload.get("base_pair_checkpoints")
    if not isinstance(base_checkpoints_obj, list) or not base_checkpoints_obj:
        raise ValueError("cnn_v3 checkpoint is missing base_pair_checkpoints.")
    base_checkpoints: list[str] = [str(item) for item in base_checkpoints_obj]
    for path in base_checkpoints:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Base checkpoint not found for inference: {path}")

    test_tsv = resolve_test_tsv(common_args.species, common_args.test_tsv)
    donor_len, acceptor_len = resolve_effective_window_lengths(
        donor_len=common_args.donor_len,
        acceptor_len=common_args.acceptor_len,
        inferred_train_len=None,
    )
    validate_window_args(donor_len=donor_len, acceptor_len=acceptor_len)
    pair_rows, skipped_short, skipped_unpaired = read_test_pair_rows(
        test_tsv=test_tsv,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )
    if skipped_short:
        print(f"Skipped short sites: {skipped_short}")
    if skipped_unpaired:
        print(f"Skipped unpaired introns: {skipped_unpaired}")
    transformed_pairs: list[tuple[str, str]] = []
    for row in pair_rows:
        transformed = apply_pair_sequence_transform(
            PairSequenceRecord(
                donor_seq=str(row["donor_seq"]),
                acceptor_seq=str(row["acceptor_seq"]),
            ),
            transform_mode=str(model_args.sequence_transform),
            intron_half_length=(
                int(row["intron_half_length"])
                if row.get("intron_half_length") is not None
                else None
            ),
        )
        transformed_pairs.append((transformed.donor_seq, transformed.acceptor_seq))

    infer_batch_size = (
        int(model_args.infer_batch_size)
        if model_args.infer_batch_size is not None
        else int(model_args.batch_size)
    )
    device = cnn_v2.pick_device(common_args.device)
    features = _build_meta_features(
        base_checkpoints=base_checkpoints,
        pairs=transformed_pairs,
        device=device,
        batch_size=infer_batch_size,
    )
    model = _MetaPairMLP(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["meta_hidden_dim"]),
        dropout=float(payload["meta_dropout"]),
    ).to(device)
    state_dict_obj = payload.get("model_state")
    if not isinstance(state_dict_obj, dict):
        raise ValueError("cnn_v3 checkpoint is missing model_state.")
    model.load_state_dict(state_dict_obj)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features.astype(np.float32)).to(device))
        scores = log10_sigmoid_np(logits.float().detach().cpu().numpy())
    if len(scores) != len(pair_rows):
        raise ValueError(
            "Pair score count does not match pair row count: "
            f"{len(scores)} != {len(pair_rows)}"
        )

    out_rows: list[dict[str, object]] = []
    for row, score in zip(pair_rows, scores, strict=True):
        out_rows.append(
            {
                "transcript_id": str(row["transcript_id"]),
                "intron_index": int(row["intron_index"]),
                "site_type": "pair",
                "score": float(score),
                SCORE_SPACE_FIELD: SCORE_SPACE_LOG10,
            }
        )
    return out_rows
