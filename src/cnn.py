"""
Basic CNN-based splice site scoring for Dmel
==================================================
1. Train independent CNN models for donor and acceptor sites
2. Score all introns in test_sites.tsv
3. Output transcript-level scores (min intron score)

Usage:
    python cnn_splice_scoring.py
"""

import argparse
import os
import re
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = None
    average_precision_score = None


# ==================== Main ====================
def main():
    ap = argparse.ArgumentParser(description="Basic CNN-based splice site scoring")

    # Species and data path specification
    ap.add_argument("--species", type=str, default="Dmel", help="Species name")
    ap.add_argument(
        "--train_pos_path",
        type=str,
        default=None,
        help="Path to positive training file (.err). If omitted, inferred from donor_len/acceptor_len",
    )
    ap.add_argument(
        "--train_neg_path",
        type=str,
        default=None,
        help="Path to negative training file (.neg.err). If omitted, inferred from donor_len/acceptor_len",
    )
    ap.add_argument(
        "--test_tsv",
        type=str,
        default=None,
        help="Path to test TSV. Defaults to data/{species}/raw/transcripts.tsv",
    )
    ap.add_argument(
        "--output_tsv",
        type=str,
        default=None,
        help="Path to output TSV. If omitted, auto-named from donor_len/acceptor_len",
    )
    ap.add_argument(
        "--donor_model_dir",
        type=str,
        default="../model/dirosophila/donar/cnn",
        help="Directory for donor model checkpoint (best.pt)",
    )
    ap.add_argument(
        "--acceptor_model_dir",
        type=str,
        default="../model/dirosophila/acceptor/cnn",
        help="Directory for acceptor model checkpoint (best.pt)",
    )

    # Site window configuration (same semantics as make_test_data_from_gtf.py)
    ap.add_argument(
        "--donor_len",
        type=int,
        default=None,
        help="Donor window length to extract from input sequences (optional)",
    )
    ap.add_argument(
        "--donor_left",
        type=int,
        default=3,
        help="Donor left context length used for extraction anchor",
    )
    ap.add_argument(
        "--acceptor_len",
        type=int,
        default=None,
        help="Acceptor window length to extract from input sequences (optional)",
    )
    ap.add_argument(
        "--acceptor_right",
        type=int,
        default=3,
        help="Acceptor right context length used for extraction anchor",
    )

    # Model params
    ap.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Optional override for model input length (if omitted, donor_len/acceptor_len are used)",
    )
    ap.add_argument("--epochs", type=int, default=20, help="Training epochs")
    ap.add_argument(
        "--batch_size", type=int, default=512, help="Batch size (larger=faster on GPU)"
    )
    ap.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)"
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight model (2 layers, faster training)",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (may be unstable on some setups)",
    )

    # Skip training (if models already exist)
    ap.add_argument(
        "--skip_training", action="store_true", help="Skip training, only score"
    )

    # Device selection
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (auto=CUDA>MPS>CPU)",
    )

    args = ap.parse_args()

    # Build paths from species (relative to src/ directory)
    species = args.species
    raw_dir = f"../data/{species}/raw"
    train_dir = f"../data/{species}/train"

    raw_refs = detect_raw_reference_files(raw_dir)
    print("Detected raw reference files:")
    print(f"  fna: {raw_refs.get('fna') or 'not found'}")
    print(f"  gff: {raw_refs.get('gff') or 'not found'}")
    print(f"  gtf: {raw_refs.get('gtf') or 'not found'}")

    inferred_train_len: Optional[int] = None
    default_pos_path = None
    default_neg_path = None
    if args.train_pos_path is None or args.train_neg_path is None:
        default_pos_path, default_neg_path, inferred_train_len = infer_default_train_paths(
            train_dir=train_dir,
            donor_len=args.donor_len,
            acceptor_len=args.acceptor_len,
        )

    default_test_tsv = f"../data/{species}/raw/transcripts.tsv"
    default_output_tsv = default_output_path(
        species,
        args.donor_len,
        args.acceptor_len,
        fallback_train_len=inferred_train_len,
    )

    pos_path = args.train_pos_path or default_pos_path
    neg_path = args.train_neg_path or default_neg_path
    if pos_path is None or neg_path is None:
        raise ValueError(
            "Could not infer default training file path from species. "
            "Specify --train_pos_path and --train_neg_path."
        )
    donor_model_dir = args.donor_model_dir
    acceptor_model_dir = args.acceptor_model_dir
    test_tsv = args.test_tsv or default_test_tsv
    output_tsv = args.output_tsv or default_output_tsv

    effective_donor_len = args.donor_len
    effective_acceptor_len = args.acceptor_len
    if inferred_train_len is not None:
        if effective_donor_len is None:
            effective_donor_len = inferred_train_len
        if effective_acceptor_len is None:
            effective_acceptor_len = inferred_train_len

    validate_window_args(
        donor_len=effective_donor_len,
        donor_left=args.donor_left,
        acceptor_len=effective_acceptor_len,
        acceptor_right=args.acceptor_right,
    )

    if args.max_len is not None:
        donor_model_max_len = args.max_len
        acceptor_model_max_len = args.max_len
    else:
        donor_model_max_len = effective_donor_len if effective_donor_len is not None else 50
        acceptor_model_max_len = (
            effective_acceptor_len if effective_acceptor_len is not None else 50
        )

    # Device auto-detection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(
                f"🚀 Auto-detected device: CUDA (GPU: {torch.cuda.get_device_name(0)})"
            )
        elif torch.backends.mps.is_available():
            device = "mps"
            print("🚀 Auto-detected device: MPS (Apple Silicon GPU)")
        else:
            device = "cpu"
            print("⚠️  WARNING: Using CPU (slow). Consider using a machine with GPU.")
    else:
        device = args.device
        if device == "cpu":
            print("⚠️  WARNING: CPU device explicitly selected (slow).")
        else:
            print(f"Device: {device} (manually selected)")

    # Train models
    if not args.skip_training:
        train_model(
            task="donor",
            pos_path=pos_path,
            neg_path=neg_path,
            out_dir=donor_model_dir,
            max_len=donor_model_max_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            lightweight=args.lightweight,
            compile_model=args.compile,
            donor_len=effective_donor_len,
            donor_left=args.donor_left,
            acceptor_len=effective_acceptor_len,
            acceptor_right=args.acceptor_right,
        )

        train_model(
            task="acceptor",
            pos_path=pos_path,
            neg_path=neg_path,
            out_dir=acceptor_model_dir,
            max_len=acceptor_model_max_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            lightweight=args.lightweight,
            compile_model=args.compile,
            donor_len=effective_donor_len,
            donor_left=args.donor_left,
            acceptor_len=effective_acceptor_len,
            acceptor_right=args.acceptor_right,
        )
    else:
        print("Skipping training (--skip_training flag set)")

    # Score test sites
    donor_model_path = os.path.join(donor_model_dir, "best.pt")
    acceptor_model_path = os.path.join(acceptor_model_dir, "best.pt")

    if not os.path.exists(donor_model_path) or not os.path.exists(acceptor_model_path):
        print("Error: Model checkpoints not found!")
        print(f"  Donor: {donor_model_path}")
        print(f"  Acceptor: {acceptor_model_path}")
        return

    score_test_sites(
        test_tsv=test_tsv,
        donor_model_path=donor_model_path,
        acceptor_model_path=acceptor_model_path,
        output_tsv=output_tsv,
        device=device,
        donor_len=effective_donor_len,
        donor_left=args.donor_left,
        acceptor_len=effective_acceptor_len,
        acceptor_right=args.acceptor_right,
    )


# ==================== Utilities ====================
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ==================== Data Parsing ====================
LINE_PAIR_RE = re.compile(
    r"^DEBUG\s+donor\s+([A-Za-z]+)\s+acceptor\s+([A-Za-z]+)(?:\s+[+-])?\s*$"
)
LINE_SINGLE_RE = re.compile(r"^DEBUG\s+(donor|acceptor)\s+([A-Za-z]+)(?:\s+[+-])?\s*$")


def validate_window_args(
    donor_len: Optional[int],
    donor_left: int,
    acceptor_len: Optional[int],
    acceptor_right: int,
):
    if donor_left < 0:
        raise ValueError("--donor_left must be >= 0")
    if acceptor_right < 0:
        raise ValueError("--acceptor_right must be >= 0")

    if donor_len is not None:
        if donor_len <= 0:
            raise ValueError("--donor_len must be > 0")
        if donor_left > donor_len:
            raise ValueError("--donor_left must satisfy donor_left <= donor_len")

    if acceptor_len is not None:
        if acceptor_len <= 0:
            raise ValueError("--acceptor_len must be > 0")
        if acceptor_right > acceptor_len:
            raise ValueError(
                "--acceptor_right must satisfy acceptor_right <= acceptor_len"
            )


def detect_raw_reference_files(raw_dir: str) -> Dict[str, Optional[str]]:
    """Detect single reference files in raw dir by extension family."""
    out: Dict[str, Optional[str]] = {"fna": None, "gff": None, "gtf": None}
    if not os.path.isdir(raw_dir):
        return out

    files = [
        os.path.join(raw_dir, name)
        for name in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, name))
    ]

    fna = [p for p in files if p.endswith(".fna")]
    gff = [
        p
        for p in files
        if p.endswith(".gff") or p.endswith(".gff3") or ".gff." in os.path.basename(p)
    ]
    gtf = [p for p in files if p.endswith(".gtf")]

    out["fna"] = fna[0] if len(fna) == 1 else None
    out["gff"] = gff[0] if len(gff) == 1 else None
    out["gtf"] = gtf[0] if len(gtf) == 1 else None
    return out


def list_available_train_lengths(train_dir: str) -> List[int]:
    if not os.path.isdir(train_dir):
        return []

    pos_lengths = set()
    neg_lengths = set()
    for name in os.listdir(train_dir):
        m_pos = re.match(r"^(\d+)bp\.err$", name)
        if m_pos:
            pos_lengths.add(int(m_pos.group(1)))
            continue
        m_neg = re.match(r"^(\d+)bp\.neg\.err$", name)
        if m_neg:
            neg_lengths.add(int(m_neg.group(1)))

    return sorted(pos_lengths & neg_lengths)


def infer_default_train_paths(
    train_dir: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
) -> Tuple[str, str, int]:
    available = list_available_train_lengths(train_dir)
    if not available:
        raise ValueError(
            f"No paired training files found in {train_dir}. Expected <N>bp.err and <N>bp.neg.err."
        )

    requested = [x for x in [donor_len, acceptor_len] if x is not None]
    if requested:
        required_len = max(requested)
        candidates = [x for x in available if x >= required_len]
        if not candidates:
            raise ValueError(
                f"No training length >= requested window ({required_len}) in {train_dir}. "
                f"Available lengths: {available}"
            )
        chosen = min(candidates)
    else:
        chosen = max(available)

    pos = os.path.join(train_dir, f"{chosen}bp.err")
    neg = os.path.join(train_dir, f"{chosen}bp.neg.err")
    print(f"Using training files (auto): {pos}, {neg}")
    return pos, neg, chosen


def default_output_path(
    species: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    fallback_train_len: Optional[int] = None,
) -> str:
    if donor_len is not None and acceptor_len is not None:
        if donor_len == acceptor_len:
            suffix = f"{donor_len}bp"
        else:
            suffix = f"d{donor_len}bp_a{acceptor_len}bp"
    elif donor_len is not None:
        suffix = f"d{donor_len}bp"
    elif acceptor_len is not None:
        suffix = f"a{acceptor_len}bp"
    elif fallback_train_len is not None:
        suffix = f"{fallback_train_len}bp"
    else:
        return f"../data/{species}/trans_score/cnn.tsv"
    return f"../data/{species}/trans_score/cnn{suffix}.tsv"


def reshape_site_sequence(
    seq: str,
    site_type: str,
    donor_len: Optional[int],
    donor_left: int,
    acceptor_len: Optional[int],
    acceptor_right: int,
) -> Optional[str]:
    """Reshape site sequence by window args.

    donor: keep [boundary-donor_left, boundary+donor_right) where boundary is at index donor_left
    acceptor: keep [boundary-acceptor_left, boundary+acceptor_right) where boundary is right-anchored
    """
    seq = seq.upper()

    if site_type == "donor":
        if donor_len is None:
            return seq
        donor_right = donor_len - donor_left
        start = donor_left - donor_left
        end = donor_left + donor_right
        if end > len(seq):
            return None
        return seq[start:end]

    if site_type == "acceptor":
        if acceptor_len is None:
            return seq
        acceptor_left = acceptor_len - acceptor_right
        boundary = len(seq) - acceptor_right
        start = boundary - acceptor_left
        end = boundary + acceptor_right
        if start < 0 or end > len(seq):
            return None
        return seq[start:end]

    return None


def read_examples_single_task(
    pos_path: str,
    neg_path: str,
    task: str,
    donor_len: Optional[int],
    donor_left: int,
    acceptor_len: Optional[int],
    acceptor_right: int,
) -> List[Tuple[str, int]]:
    """Read training examples for one task (donor or acceptor).

    Returns: list of (sequence, label) where label=1 for pos, label=0 for neg
    """
    examples: List[Tuple[str, int]] = []

    def read_one_set(path: str, label: int):
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("DEBUG"):
                    continue

                m_pair = LINE_PAIR_RE.match(line)
                if m_pair:
                    donor_seq, acceptor_seq = m_pair.groups()
                    raw_seq = donor_seq if task == "donor" else acceptor_seq
                    seq = reshape_site_sequence(
                        raw_seq,
                        task,
                        donor_len=donor_len,
                        donor_left=donor_left,
                        acceptor_len=acceptor_len,
                        acceptor_right=acceptor_right,
                    )
                    if seq:
                        examples.append((seq, label))
                    continue

                m_single = LINE_SINGLE_RE.match(line)
                if m_single:
                    tname, seq = m_single.groups()
                    if tname == task:
                        reshaped = reshape_site_sequence(
                            seq,
                            task,
                            donor_len=donor_len,
                            donor_left=donor_left,
                            acceptor_len=acceptor_len,
                            acceptor_right=acceptor_right,
                        )
                        if reshaped:
                            examples.append((reshaped, label))
                    continue

    read_one_set(pos_path, label=1)
    read_one_set(neg_path, label=0)
    return examples


# ==================== DNA One-Hot Encoding ====================
def one_hot_encode_dna(seq: str, max_len: int = 50) -> np.ndarray:
    """Convert DNA sequence to one-hot encoding (4, max_len).

    A=0, C=1, G=2, T=3, others=all zeros
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((4, max_len), dtype=np.float32)

    for i, base in enumerate(seq[:max_len]):
        if base in mapping:
            encoded[mapping[base], i] = 1.0

    return encoded


# ==================== Dataset ====================
class DNADataset(Dataset):
    def __init__(self, examples: List[Tuple[str, int]], max_len: int = 50):
        self.examples = examples
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        x = one_hot_encode_dna(seq, self.max_len)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


# ==================== Basic CNN Model ====================
class BasicSpliceCNN(nn.Module):
    """Basic 1D CNN for splice site classification (simple sequential design)."""

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: List[int] = [64, 128, 256],
        kernel_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

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
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, L) one-hot encoded DNA
        Returns:
            logits: (B,) binary classification logits
        """
        x = self.conv_layers(x)  # (B, C, L')
        x = self.gap(x)  # (B, C, 1)
        x = x.squeeze(-1)  # (B, C)
        logits = self.fc(x).squeeze(-1)  # (B,)
        return logits


# ==================== Training ====================
def stratified_split(
    examples: List[Tuple[str, int]], val_frac: float = 0.1, seed: int = 1337
):
    """Split data by label to ensure both train/val have pos and neg examples."""
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
def evaluate(model, loader, device):
    """Evaluate model and return metrics."""
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

    # Clip probs to prevent numerical instability
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    # Ensure labels are integers for sklearn metrics
    labels = labels.astype(np.int32)

    # Check for invalid values
    has_invalid = (
        np.isnan(labels).any()
        or np.isinf(labels).any()
        or np.isnan(probs).any()
        or np.isinf(probs).any()
    )

    metrics = {}
    if labels.size and not has_invalid:
        metrics["acc@0.5"] = float(np.mean((probs >= 0.5) == (labels >= 0.5)))

        # Compute sklearn metrics with error handling
        if roc_auc_score and len(np.unique(labels)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(labels, probs))
            except Exception:
                pass  # Skip if computation fails

        if average_precision_score and len(np.unique(labels)) > 1:
            try:
                metrics["pr_auc"] = float(average_precision_score(labels, probs))
            except Exception:
                pass  # Skip if computation fails

    return metrics


def train_model(
    task: str,
    pos_path: str,
    neg_path: str,
    out_dir: str,
    max_len: int = 50,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 5e-4,
    seed: int = 1337,
    lightweight: bool = False,
    compile_model: bool = False,
    donor_len: Optional[int] = None,
    donor_left: int = 3,
    acceptor_len: Optional[int] = None,
    acceptor_right: int = 3,
):
    """Train CNN model for one task (donor or acceptor)."""

    print(f"\n{'=' * 60}")
    print(f"Training {task.upper()} model {'(LIGHTWEIGHT)' if lightweight else ''}")
    print(f"{'=' * 60}")

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Device auto-detection (prioritize GPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Optimize DataLoader settings per device
    num_workers = 0 if device in ["mps", "cpu"] else 2  # MacOS multiprocessing issues
    pin_memory = device == "cuda"  # Only useful for CUDA

    # Load data
    examples = read_examples_single_task(
        pos_path,
        neg_path,
        task,
        donor_len=donor_len,
        donor_left=donor_left,
        acceptor_len=acceptor_len,
        acceptor_right=acceptor_right,
    )
    n_pos = sum(y for _, y in examples)
    n_neg = len(examples) - n_pos
    print(f"Total examples: {len(examples)} | pos={n_pos} | neg={n_neg}")
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Insufficient training examples for {task}: pos={n_pos}, neg={n_neg}. "
            "Check --train_pos_path/--train_neg_path and window arguments."
        )

    train_ex, val_ex = stratified_split(examples, val_frac=0.1, seed=seed)
    train_pos = sum(y for _, y in train_ex)
    train_neg = len(train_ex) - train_pos
    print(f"Train: {len(train_ex)} | pos={train_pos} | neg={train_neg}")
    print(
        f"Val:   {len(val_ex)} | pos={sum(y for _, y in val_ex)} | neg={len(val_ex) - sum(y for _, y in val_ex)}"
    )

    # Datasets
    train_ds = DNADataset(train_ex, max_len=max_len)
    val_ds = DNADataset(val_ex, max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Model
    conv_channels = [64, 128] if lightweight else [64, 128, 256]
    model = BasicSpliceCNN(
        in_channels=4,
        conv_channels=conv_channels,
        kernel_size=7,
        dropout=0.3,
    ).to(device)

    print(
        f"Model architecture: {len(conv_channels)} conv layers, channels={conv_channels}"
    )

    # Compile model for faster training (PyTorch 2.0+)
    if compile_model and hasattr(torch, "compile") and device != "mps":
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception:
            print("torch.compile() not available or failed, using eager mode")

    # Loss with pos_weight for class imbalance
    # Cap pos_weight to prevent extreme gradient values
    pos_weight_raw = (train_neg / max(1, train_pos)) if train_pos > 0 else 1.0
    pos_weight = min(pos_weight_raw, 20.0)  # Cap at 20 to avoid numerical instability
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t, reduction="mean")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    print(f"pos_weight (raw): {pos_weight_raw:.4f} -> (capped): {pos_weight:.4f}")
    print(f"Learning rate: {lr}")

    # Training loop
    best_score = -1e9

    print(f"\nStarting training: {len(train_loader)} batches per epoch")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Relaxed clip
            optimizer.step()

            running_loss += loss.item()

            # Progress indicator every 20% of batches
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] loss: {loss.item():.4f}"
                )

        scheduler.step()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device)

        # Prefer PR-AUC for imbalanced data
        if "pr_auc" in val_metrics:
            score = val_metrics["pr_auc"]
            score_name = "pr_auc"
        elif "roc_auc" in val_metrics:
            score = val_metrics["roc_auc"]
            score_name = "roc_auc"
        else:
            score = val_metrics.get("acc@0.5", 0.0)
            score_name = "acc@0.5"

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Val metrics: {val_metrics}")
        print(f"  Score ({score_name}): {score:.4f}")

        if score > best_score:
            best_score = score
            ckpt_path = os.path.join(out_dir, "best.pt")
            torch.save(
                {
                    "task": task,
                    "max_len": max_len,
                    "model_state": model.state_dict(),
                },
                ckpt_path,
            )
            print(f"  ✅ Saved checkpoint: {ckpt_path}")

    print(f"\nBest {task} {score_name}: {best_score:.4f}")
    return model


# ==================== Scoring ====================
def load_model(checkpoint_path: str, device: str) -> Tuple[nn.Module, Dict]:
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect conv_channels from checkpoint if available
    state_dict = ckpt["model_state"]
    # Try to infer architecture from state dict
    conv_channels = [64, 128, 256]  # default
    if "conv_layers.5.weight" in state_dict:  # 3 conv layers
        conv_channels = [64, 128, 256]
    elif (
        "conv_layers.0.weight" in state_dict
        and "conv_layers.5.weight" not in state_dict
    ):
        conv_channels = [64, 128]  # lightweight 2 layers

    model = BasicSpliceCNN(
        in_channels=4,
        conv_channels=conv_channels,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt


@torch.no_grad()
def score_sequences(
    model, sequences: List[str], max_len: int, device: str, batch_size: int = 512
) -> np.ndarray:
    """Score a list of DNA sequences.

    Returns: array of probabilities (0-1)
    """
    model.eval()

    # Encode sequences
    encoded = [one_hot_encode_dna(seq.upper(), max_len) for seq in sequences]
    x = torch.from_numpy(np.stack(encoded)).to(device)

    # Batch scoring
    all_probs = []
    for i in range(0, len(x), batch_size):
        batch_x = x[i : i + batch_size]
        logits = model(batch_x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)


def score_test_sites(
    test_tsv: str,
    donor_model_path: str,
    acceptor_model_path: str,
    output_tsv: str,
    device: str,
    donor_len: Optional[int],
    donor_left: int,
    acceptor_len: Optional[int],
    acceptor_right: int,
):
    """Score test sites and output transcript-level scores."""

    print(f"\n{'=' * 60}")
    print("Scoring test sites")
    print(f"{'=' * 60}")

    # Load models
    donor_model, donor_ckpt = load_model(donor_model_path, device)
    acceptor_model, acceptor_ckpt = load_model(acceptor_model_path, device)
    donor_max_len = donor_ckpt.get("max_len") or (
        donor_len if donor_len is not None else 50
    )
    acceptor_max_len = acceptor_ckpt.get("max_len") or (
        acceptor_len if acceptor_len is not None else 50
    )

    print(f"Loaded donor model: {donor_model_path}")
    print(f"Loaded acceptor model: {acceptor_model_path}")
    print(f"donor max_len: {donor_max_len}")
    print(f"acceptor max_len: {acceptor_max_len}")

    # Read test sites
    print(f"\nReading {test_tsv}...")
    data = []
    skipped_short = 0
    with open(test_tsv, "r") as f:
        _ = next(f)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 8:
                site_type = parts[2]
                reshaped = reshape_site_sequence(
                    parts[7],
                    site_type,
                    donor_len=donor_len,
                    donor_left=donor_left,
                    acceptor_len=acceptor_len,
                    acceptor_right=acceptor_right,
                )
                if not reshaped:
                    skipped_short += 1
                    continue
                data.append(
                    {
                        "transcript_id": parts[0],
                        "site_type": site_type,
                        "intron_index": int(parts[3]),
                        "seq": reshaped,
                    }
                )
    print(f"Total sites: {len(data)}")
    if skipped_short:
        print(f"Skipped sites due to short sequence for requested window: {skipped_short}")

    # Group by transcript and intron
    transcript_introns = defaultdict(lambda: defaultdict(dict))

    for row in data:
        tid = row["transcript_id"]
        iidx = row["intron_index"]
        stype = row["site_type"]
        seq = row["seq"]

        transcript_introns[tid][iidx][stype] = seq

    # Score each transcript - OPTIMIZED with batching
    results = []

    print("\nScoring transcripts...")

    # Phase 1: Collect all sequences to score
    all_donor_seqs = []
    all_acceptor_seqs = []
    transcript_keys = []  # Store (tid, iidx) pairs in order

    for tid, introns in transcript_introns.items():
        for iidx in sorted(introns.keys()):
            sites = introns[iidx]
            donor_seq = sites.get("donor", "")
            acceptor_seq = sites.get("acceptor", "")

            all_donor_seqs.append(donor_seq)
            all_acceptor_seqs.append(acceptor_seq)
            transcript_keys.append((tid, iidx))

    print(f"Total introns to score: {len(transcript_keys)}")
    if not transcript_keys:
        print("No valid introns to score after window reshaping.")
        outdir = os.path.dirname(output_tsv)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(output_tsv, "w") as f:
            f.write(
                "transcript_id\tmin_intron_index\tScore_donor\tScore_acceptor\tmin_donor_plus_acceptor\n"
            )
        return

    # Phase 2: Score all sequences in one batch per model
    print("Scoring donor sequences...")
    donor_scores = score_sequences(
        donor_model, all_donor_seqs, donor_max_len, device, batch_size=512
    )

    print("Scoring acceptor sequences...")
    acceptor_scores = score_sequences(
        acceptor_model, all_acceptor_seqs, acceptor_max_len, device, batch_size=512
    )

    print("Aggregating results...")

    # Phase 3: Aggregate by transcript
    transcript_intron_dict = defaultdict(dict)

    for idx, (tid, iidx) in enumerate(transcript_keys):
        donor_score = donor_scores[idx] if all_donor_seqs[idx] else 0.0
        acceptor_score = acceptor_scores[idx] if all_acceptor_seqs[idx] else 0.0
        total_score = donor_score + acceptor_score

        transcript_intron_dict[tid][iidx] = (donor_score, acceptor_score, total_score)

    # Phase 4: Find min intron per transcript
    for tid, introns_dict in transcript_intron_dict.items():
        if not introns_dict:
            continue

        # Find intron with minimum total score
        min_iidx = min(introns_dict.keys(), key=lambda iidx: introns_dict[iidx][2])
        donor_score, acceptor_score, total_score = introns_dict[min_iidx]

        results.append(
            {
                "transcript_id": tid,
                "min_intron_index": min_iidx,
                "Score_donor": donor_score,
                "Score_acceptor": acceptor_score,
                "min_donor_plus_acceptor": total_score,
            }
        )

    # Sort by transcript_id
    results.sort(key=lambda x: x["transcript_id"])

    # Write output
    print(f"\nWriting results to {output_tsv}...")
    outdir = os.path.dirname(output_tsv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(output_tsv, "w") as f:
        f.write(
            "transcript_id\tmin_intron_index\tScore_donor\tScore_acceptor\tmin_donor_plus_acceptor\n"
        )
        for r in results:
            f.write(
                f"{r['transcript_id']}\t{r['min_intron_index']}\t{r['Score_donor']:.6f}\t{r['Score_acceptor']:.6f}\t{r['min_donor_plus_acceptor']:.6f}\n"
            )

    print(f"✅ Done! Results saved to {output_tsv}")
    print(f"Total transcripts: {len(results)}")


if __name__ == "__main__":
    main()
