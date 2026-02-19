"""CNN model implementation for site-level splice scoring.

This module contains CNN-specific components:
- model architecture
- training and validation loop
- checkpoint loading
- site-level inference for donor/acceptor sequences
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
from util.losses import LOSS_NAME_CHOICES, build_binary_classification_loss

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


def set_seed(seed: int = 1337) -> None:
    """Set deterministic random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader worker-local RNG states."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pick_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def one_hot_encode_dna(seq: str, window_len: int = 50) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((4, window_len), dtype=np.float32)

    for i, base in enumerate(seq[:window_len]):
        if base in mapping:
            encoded[mapping[base], i] = 1.0

    return encoded


def parse_conv_channels(raw: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated convolution channel sizes.

    Parameters
    ----------
    raw : str | None
        Comma-separated channel sizes like ``"64,128,256"``.

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
        raise ValueError("--conv_channels must include at least one integer.")

    channels: List[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --conv_channels item '{part}'. Use integers like 64,128,256."
            ) from exc
        if value <= 0:
            raise ValueError("--conv_channels values must be positive.")
        channels.append(value)
    return channels


class DNADataset(Dataset):
    def __init__(self, examples: Sequence[Tuple[str, int]], window_len: int = 50):
        self.examples = list(examples)
        self.window_len = window_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        x = one_hot_encode_dna(seq, self.window_len)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


class BasicSpliceCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: Optional[Sequence[int]] = None,
        kernel_size: int = 7,
        dropout: float = 0.3,
        fc_hidden: int = 128,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [64, 128, 256]
        if not conv_channels:
            raise ValueError("conv_channels must not be empty.")

        layers = []
        prev_ch = in_channels
        for ch in conv_channels:
            layers.extend(
                [
                    nn.Conv1d(prev_ch, ch, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            prev_ch = ch

        self.conv_layers = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        logits = self.fc(x).squeeze(-1)
        return logits


def stratified_split(
    examples: Sequence[Tuple[str, int]], val_frac: float = 0.1, seed: int = 1337
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    rng = random.Random(seed)
    pos = [(s, y) for s, y in examples if y == 1]
    neg = [(s, y) for s, y in examples if y == 0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_val_pos = max(1, int(len(pos) * val_frac))
    n_val_neg = max(1, int(len(neg) * val_frac))

    train = pos[n_val_pos:] + neg[n_val_neg:]
    val = pos[:n_val_pos] + neg[:n_val_neg]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    probs = sigmoid_np(logits) if logits.size else np.array([])

    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    labels = labels.astype(np.int32)

    metrics: Dict[str, float] = {}
    if labels.size:
        metrics["acc@0.5"] = float(np.mean((probs >= 0.5) == (labels >= 0.5)))

        if roc_auc_score and len(np.unique(labels)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(labels, probs))
            except Exception:
                pass

        if average_precision_score and len(np.unique(labels)) > 1:
            try:
                metrics["pr_auc"] = float(average_precision_score(labels, probs))
            except Exception:
                pass

    return metrics


def train_task_model(
    task: str,
    pos_path: str,
    neg_path: str,
    checkpoint_path: str,
    window_len: int,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 5e-4,
    seed: int = 1337,
    lightweight: bool = False,
    conv_channels: Optional[Sequence[int]] = None,
    kernel_size: int = 7,
    dropout: float = 0.3,
    fc_hidden: int = 128,
    weight_decay: float = 0.01,
    eta_min_ratio: float = 0.01,
    val_frac: float = 0.1,
    grad_clip: float = 5.0,
    compile_model: bool = False,
    device: str = "auto",
    loss_name: str = "weighted_bce",
    pos_weight_cap: float = 20.0,
    focal_gamma: float = 2.0,
    focal_alpha_pos: Optional[float] = None,
    asym_gamma_pos: float = 0.0,
    asym_gamma_neg: float = 4.0,
    asym_alpha_pos: Optional[float] = None,
) -> Dict[str, object]:
    if kernel_size <= 0:
        raise ValueError("--kernel_size must be positive.")
    if fc_hidden <= 0:
        raise ValueError("--fc_hidden must be positive.")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("--dropout must satisfy 0 <= dropout < 1.")
    if weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative.")
    if eta_min_ratio < 0.0:
        raise ValueError("--eta_min_ratio must be non-negative.")
    if val_frac <= 0.0 or val_frac >= 1.0:
        raise ValueError("--val_frac must satisfy 0 < val_frac < 1.")
    if grad_clip < 0.0:
        raise ValueError("--grad_clip must be non-negative.")

    set_seed(seed)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    device = pick_device(device)
    num_workers = 0 if device in ["mps", "cpu"] else 2
    pin_memory = device == "cuda"

    examples = read_examples_single_task(
        pos_path,
        neg_path,
        task,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )

    n_pos = sum(y for _, y in examples)
    n_neg = len(examples) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Insufficient training examples for {task}: pos={n_pos}, neg={n_neg}."
        )

    train_ex, val_ex = stratified_split(examples, val_frac=val_frac, seed=seed)
    print(
        f"[{task}] device={device} total={len(examples)} "
        f"(pos={n_pos}, neg={n_neg}) train={len(train_ex)} val={len(val_ex)}"
    )

    train_ds = DNADataset(train_ex, window_len=window_len)
    val_ds = DNADataset(val_ex, window_len=window_len)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if conv_channels is None:
        conv_channels = [64, 128] if lightweight else [64, 128, 256]
    else:
        conv_channels = list(conv_channels)
    model = BasicSpliceCNN(
        in_channels=4,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        fc_hidden=fc_hidden,
    ).to(device)

    if compile_model and hasattr(torch, "compile") and device != "mps":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    train_pos = sum(y for _, y in train_ex)
    train_neg = len(train_ex) - train_pos
    criterion, loss_meta = build_binary_classification_loss(
        loss_name=loss_name,
        train_pos=train_pos,
        train_neg=train_neg,
        device=device,
        pos_weight_cap=pos_weight_cap,
        focal_gamma=focal_gamma,
        focal_alpha_pos=focal_alpha_pos,
        asym_gamma_pos=asym_gamma_pos,
        asym_gamma_neg=asym_gamma_neg,
        asym_alpha_pos=asym_alpha_pos,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * eta_min_ratio,
    )

    best_score = -1e9
    best_metric_name = "acc@0.5"
    best_epoch = 0
    log_every = max(1, epochs // 5)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        val_metrics = evaluate(model, val_loader, device)
        if "pr_auc" in val_metrics:
            score = val_metrics["pr_auc"]
            score_name = "pr_auc"
        elif "roc_auc" in val_metrics:
            score = val_metrics["roc_auc"]
            score_name = "roc_auc"
        else:
            score = val_metrics.get("acc@0.5", 0.0)
            score_name = "acc@0.5"

        improved = score > best_score
        if improved:
            best_score = score
            best_metric_name = score_name
            best_epoch = epoch
            torch.save(
                {
                    "task": task,
                    "window_len": window_len,
                    "model_config": {
                        "conv_channels": list(conv_channels),
                        "kernel_size": kernel_size,
                        "dropout": dropout,
                        "fc_hidden": fc_hidden,
                    },
                    "model_state": model.state_dict(),
                },
                checkpoint_path,
            )

        should_log = (
            epoch == 1
            or epoch == epochs
            or epoch % log_every == 0
            or improved
        )
        if should_log:
            mark = "*" if improved else "-"
            print(
                f"[{task}] {mark} epoch {epoch}/{epochs} "
                f"loss={train_loss:.4f} {score_name}={score:.4f} "
                f"best={best_score:.4f} (ep {best_epoch})"
            )

    print(
        f"[{task}] done best_{best_metric_name}={best_score:.4f} at epoch {best_epoch}"
    )

    return {
        "task": task,
        "num_examples": len(examples),
        "num_pos": n_pos,
        "num_neg": n_neg,
        "best_metric": best_metric_name,
        "best_score": float(best_score),
        "checkpoint": checkpoint_path,
        "loss": loss_name,
        "pos_weight": loss_meta["pos_weight"],
        "focal_gamma": loss_meta["focal_gamma"],
        "focal_alpha_pos": loss_meta["focal_alpha_pos"],
        "asym_gamma_pos": loss_meta["asym_gamma_pos"],
        "asym_gamma_neg": loss_meta["asym_gamma_neg"],
        "asym_alpha_pos": loss_meta["asym_alpha_pos"],
        "conv_channels": list(conv_channels),
        "kernel_size": kernel_size,
        "dropout": dropout,
        "fc_hidden": fc_hidden,
        "weight_decay": weight_decay,
        "eta_min_ratio": eta_min_ratio,
        "val_frac": val_frac,
        "grad_clip": grad_clip,
    }


def load_task_model(checkpoint_path: str, device: str) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state"]

    model_config = ckpt.get("model_config", {})
    conv_channels = model_config.get("conv_channels")
    kernel_size = int(model_config.get("kernel_size", 7))
    dropout = float(model_config.get("dropout", 0.3))
    fc_hidden = int(model_config.get("fc_hidden", 128))

    if conv_channels is None:
        conv_channels = [64, 128, 256]
        if (
            "conv_layers.0.weight" in state_dict
            and "conv_layers.5.weight" not in state_dict
        ):
            conv_channels = [64, 128]

    model = BasicSpliceCNN(
        in_channels=4,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        fc_hidden=fc_hidden,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_sequences(
    model: nn.Module,
    sequences: Sequence[str],
    window_len: int,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    if not sequences:
        return np.array([])

    model.eval()
    encoded = [one_hot_encode_dna(seq.upper(), window_len) for seq in sequences]
    x = torch.from_numpy(np.stack(encoded)).to(device)

    all_probs = []
    for i in range(0, len(x), batch_size):
        batch_x = x[i : i + batch_size]
        logits = model(batch_x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)


def infer_site_scores(
    site_rows: List[Dict[str, object]],
    donor_model_path: str,
    acceptor_model_path: str,
    device: str = "auto",
    batch_size: int = 512,
) -> List[Dict[str, object]]:
    device = pick_device(device)

    donor_model, donor_ckpt = load_task_model(donor_model_path, device)
    acceptor_model, acceptor_ckpt = load_task_model(acceptor_model_path, device)

    donor_window_len = int(donor_ckpt.get("window_len", 50))
    acceptor_window_len = int(acceptor_ckpt.get("window_len", 50))

    donor_seqs = [str(r["seq"]) for r in site_rows if r["site_type"] == "donor"]
    acceptor_seqs = [str(r["seq"]) for r in site_rows if r["site_type"] == "acceptor"]

    donor_scores = score_sequences(
        donor_model,
        donor_seqs,
        donor_window_len,
        device,
        batch_size=batch_size,
    )
    acceptor_scores = score_sequences(
        acceptor_model,
        acceptor_seqs,
        acceptor_window_len,
        device,
        batch_size=batch_size,
    )

    out_rows: List[Dict[str, object]] = []
    donor_idx = 0
    acceptor_idx = 0
    for row in site_rows:
        site_type = str(row["site_type"])
        if site_type == "donor":
            if donor_idx < len(donor_scores):
                score = float(donor_scores[donor_idx])
            else:
                score = 0.0
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
    """Register CNN-specific training arguments."""
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument(
        "--conv_channels",
        type=str,
        default=None,
        help=(
            "Comma-separated convolution channels, e.g. 64,128,256. "
            "If omitted, default architecture is used."
        ),
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=7,
        help="Convolution kernel size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate used in convolution and fully-connected blocks.",
    )
    parser.add_argument(
        "--fc_hidden",
        type=int,
        default=128,
        help="Hidden units in the fully-connected block.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--eta_min_ratio",
        type=float,
        default=0.01,
        help="CosineAnnealingLR eta_min as lr * eta_min_ratio.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Validation split fraction for stratified split.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="Gradient clipping max norm. Use 0 to disable clipping.",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--loss",
        choices=list(LOSS_NAME_CHOICES),
        default="weighted_bce",
        help="Training loss type for donor/acceptor models.",
    )
    parser.add_argument(
        "--pos_weight_cap",
        type=float,
        default=20.0,
        help="Upper bound of positive-class weight for weighted_bce.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter used when --loss focal is selected.",
    )
    parser.add_argument(
        "--focal_alpha_pos",
        type=float,
        default=None,
        help=(
            "Positive-class alpha for focal loss (0 < alpha < 1). "
            "If omitted, it is inferred from class imbalance."
        ),
    )
    parser.add_argument(
        "--asym_gamma_pos",
        type=float,
        default=0.0,
        help="Positive-class gamma for --loss asymmetric_focal.",
    )
    parser.add_argument(
        "--asym_gamma_neg",
        type=float,
        default=4.0,
        help="Negative-class gamma for --loss asymmetric_focal.",
    )
    parser.add_argument(
        "--asym_alpha_pos",
        type=float,
        default=None,
        help=(
            "Positive-class alpha for --loss asymmetric_focal "
            "(0 < alpha < 1). If omitted, inferred from class imbalance."
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional run-name suffix for training summary.",
    )


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    """Register CNN-specific inference arguments."""
    parser.add_argument("--batch_size", type=int, default=512)


def train(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> Dict[str, object]:
    """Train donor/acceptor CNN models with unified argument interface."""
    conv_channels = parse_conv_channels(model_args.conv_channels)

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

    donor_checkpoint_path = str(
        getattr(common_args, "donor_checkpoint_path", "")
    ).strip()
    acceptor_checkpoint_path = str(
        getattr(common_args, "acceptor_checkpoint_path", "")
    ).strip()
    if not donor_checkpoint_path:
        raise ValueError("Missing donor checkpoint path in common_args.")
    if not acceptor_checkpoint_path:
        raise ValueError("Missing acceptor checkpoint path in common_args.")

    donor_metrics = train_task_model(
        task="donor",
        pos_path=train_pos_path,
        neg_path=train_neg_path,
        checkpoint_path=donor_checkpoint_path,
        window_len=donor_window_len,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        epochs=model_args.epochs,
        batch_size=model_args.batch_size,
        lr=model_args.lr,
        seed=common_args.seed,
        lightweight=model_args.lightweight,
        conv_channels=conv_channels,
        kernel_size=model_args.kernel_size,
        dropout=model_args.dropout,
        fc_hidden=model_args.fc_hidden,
        weight_decay=model_args.weight_decay,
        eta_min_ratio=model_args.eta_min_ratio,
        val_frac=model_args.val_frac,
        grad_clip=model_args.grad_clip,
        compile_model=model_args.compile,
        device=common_args.device,
        loss_name=model_args.loss,
        pos_weight_cap=model_args.pos_weight_cap,
        focal_gamma=model_args.focal_gamma,
        focal_alpha_pos=model_args.focal_alpha_pos,
        asym_gamma_pos=model_args.asym_gamma_pos,
        asym_gamma_neg=model_args.asym_gamma_neg,
        asym_alpha_pos=model_args.asym_alpha_pos,
    )
    acceptor_metrics = train_task_model(
        task="acceptor",
        pos_path=train_pos_path,
        neg_path=train_neg_path,
        checkpoint_path=acceptor_checkpoint_path,
        window_len=acceptor_window_len,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        epochs=model_args.epochs,
        batch_size=model_args.batch_size,
        lr=model_args.lr,
        seed=common_args.seed,
        lightweight=model_args.lightweight,
        conv_channels=conv_channels,
        kernel_size=model_args.kernel_size,
        dropout=model_args.dropout,
        fc_hidden=model_args.fc_hidden,
        weight_decay=model_args.weight_decay,
        eta_min_ratio=model_args.eta_min_ratio,
        val_frac=model_args.val_frac,
        grad_clip=model_args.grad_clip,
        compile_model=model_args.compile,
        device=common_args.device,
        loss_name=model_args.loss,
        pos_weight_cap=model_args.pos_weight_cap,
        focal_gamma=model_args.focal_gamma,
        focal_alpha_pos=model_args.focal_alpha_pos,
        asym_gamma_pos=model_args.asym_gamma_pos,
        asym_gamma_neg=model_args.asym_gamma_neg,
        asym_alpha_pos=model_args.asym_alpha_pos,
    )

    run_name = build_run_name(
        model_name="cnn",
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        lr=model_args.lr,
        batch_size=model_args.batch_size,
        epochs=model_args.epochs,
        tag=model_args.tag,
    )
    return {
        "model": "cnn",
        "species": common_args.species,
        "train_pos_path": train_pos_path,
        "train_neg_path": train_neg_path,
        "donor_len": donor_len,
        "acceptor_len": acceptor_len,
        "epochs": model_args.epochs,
        "batch_size": model_args.batch_size,
        "lr": model_args.lr,
        "seed": common_args.seed,
        "device": common_args.device,
        "checkpoint_name": os.path.basename(donor_checkpoint_path),
        "donor_checkpoint_path": donor_checkpoint_path,
        "acceptor_checkpoint_path": acceptor_checkpoint_path,
        "lightweight": model_args.lightweight,
        "conv_channels": conv_channels,
        "kernel_size": model_args.kernel_size,
        "dropout": model_args.dropout,
        "fc_hidden": model_args.fc_hidden,
        "weight_decay": model_args.weight_decay,
        "eta_min_ratio": model_args.eta_min_ratio,
        "val_frac": model_args.val_frac,
        "grad_clip": model_args.grad_clip,
        "compile": model_args.compile,
        "loss": model_args.loss,
        "focal_gamma": model_args.focal_gamma,
        "focal_alpha_pos": model_args.focal_alpha_pos,
        "asym_gamma_pos": model_args.asym_gamma_pos,
        "asym_gamma_neg": model_args.asym_gamma_neg,
        "asym_alpha_pos": model_args.asym_alpha_pos,
        "run_name": run_name,
        "inferred_train_len": inferred_train_len,
        "donor": donor_metrics,
        "acceptor": acceptor_metrics,
    }


def infer_site(
    common_args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Run site-level inference and return rows with fixed schema."""
    dirs = species_data_dirs(common_args.species)
    inferred_train_len: Optional[int] = None
    if common_args.donor_len is None and common_args.acceptor_len is None:
        try:
            _, _, inferred_train_len = infer_default_train_paths(
                train_dir=dirs["train"],
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
    donor_model_path = str(getattr(common_args, "donor_checkpoint_path", "")).strip()
    acceptor_model_path = str(
        getattr(common_args, "acceptor_checkpoint_path", "")
    ).strip()
    if not donor_model_path:
        raise ValueError("Missing donor checkpoint path in common_args.")
    if not acceptor_model_path:
        raise ValueError("Missing acceptor checkpoint path in common_args.")
    if not os.path.exists(donor_model_path):
        raise FileNotFoundError(f"Donor checkpoint not found: {donor_model_path}")
    if not os.path.exists(acceptor_model_path):
        raise FileNotFoundError(f"Acceptor checkpoint not found: {acceptor_model_path}")

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
