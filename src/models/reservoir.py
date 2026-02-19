"""
Reservoir computing (ESN) for Dmel splice site scoring.

- Train independent models for donor and acceptor
- Score test_sites.tsv and output transcript-level min intron scores

Design:
- k-mer tokenization (default k=3)
- Fixed random embedding + fixed ESN reservoir
- Trainable readout MLP only
"""

import argparse
import math
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    roc_auc_score = None
    average_precision_score = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


# --------------------------
# Main
# --------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Reservoir computing for splice site scoring"
    )

    ap.add_argument("--species", type=str, default="Dmel")
    ap.add_argument("--bp", type=int, default=50)

    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=32)

    ap.add_argument(
        "--model_type",
        type=str,
        default="esn",
        choices=["esn", "hf"],
        help="Model type: esn (classical reservoir) or hf (LLM reservoir)",
    )

    # Increased capacity for better representation
    ap.add_argument("--input_dim", type=int, default=256)
    ap.add_argument("--reservoir_size", type=int, default=512)
    ap.add_argument("--spectral_radius", type=float, default=0.9)
    ap.add_argument("--leak", type=float, default=0.7)
    ap.add_argument("--sparsity", type=float, default=0.02)
    ap.add_argument("--readout_hidden", type=int, default=512)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument(
        "--hf_models",
        type=str,
        default="",
        help="Comma-separated HF model names. Empty uses default list.",
    )
    ap.add_argument(
        "--hf_tokenization",
        type=str,
        default="auto",
        choices=["auto", "raw", "kmer"],
    )
    ap.add_argument("--hf_k", type=int, default=6)
    ap.add_argument("--hf_max_len", type=int, default=128)
    ap.add_argument(
        "--hf_pooling",
        type=str,
        default="mean",
        choices=["cls", "mean", "max", "mean_max"],
    )
    ap.add_argument("--hf_readout_hidden", type=int, default=512)
    ap.add_argument("--hf_batch_size", type=int, default=64)
    ap.add_argument("--hf_lr", type=float, default=None)
    ap.add_argument("--hf_epochs", type=int, default=None)
    ap.add_argument("--hf_unfreeze_backbone", action="store_true")
    ap.add_argument("--hf_trust_remote_code", action="store_true")

    ap.add_argument("--skip_training", action="store_true")

    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )

    args = ap.parse_args()

    species = args.species
    bp = args.bp
    pos_path = f"../data/{species}/train/{bp}bp.err"
    neg_path = f"../data/{species}/train/{bp}bp.neg.err"
    donor_model_dir = "../model/dirosophila/donar/reservoir"
    acceptor_model_dir = "../model/dirosophila/acceptor/reservoir"
    test_tsv = f"../data/{species}/raw/transcripts.tsv"
    output_tsv = f"../data/{species}/trans_score/reservoir{bp}bp.tsv"

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Auto-detected device: CUDA ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Auto-detected device: MPS")
        else:
            device = "cpu"
            print("WARNING: Using CPU (slow)")
    else:
        device = args.device
        print(f"Device: {device}")

    if not args.skip_training:
        if args.model_type == "esn":
            train_model(
                task="donor",
                pos_path=pos_path,
                neg_path=neg_path,
                out_dir=donor_model_dir,
                k=args.k,
                max_len=args.max_len,
                input_dim=args.input_dim,
                reservoir_size=args.reservoir_size,
                spectral_radius=args.spectral_radius,
                leak=args.leak,
                sparsity=args.sparsity,
                readout_hidden=args.readout_hidden,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=device,
            )

            train_model(
                task="acceptor",
                pos_path=pos_path,
                neg_path=neg_path,
                out_dir=acceptor_model_dir,
                k=args.k,
                max_len=args.max_len,
                input_dim=args.input_dim,
                reservoir_size=args.reservoir_size,
                spectral_radius=args.spectral_radius,
                leak=args.leak,
                sparsity=args.sparsity,
                readout_hidden=args.readout_hidden,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=device,
            )
        else:
            hf_models = parse_hf_models(args.hf_models)
            hf_epochs = args.hf_epochs if args.hf_epochs is not None else args.epochs
            hf_lr = args.hf_lr if args.hf_lr is not None else args.lr
            freeze_backbone = not args.hf_unfreeze_backbone

            for model_name in hf_models:
                tag = sanitize_model_tag(model_name)
                donor_dir = os.path.join(donor_model_dir, tag)
                acceptor_dir = os.path.join(acceptor_model_dir, tag)

                train_model_hf(
                    task="donor",
                    pos_path=pos_path,
                    neg_path=neg_path,
                    out_dir=donor_dir,
                    model_name=model_name,
                    tokenization=args.hf_tokenization,
                    k=args.hf_k,
                    max_len=args.hf_max_len,
                    pooling=args.hf_pooling,
                    readout_hidden=args.hf_readout_hidden,
                    freeze_backbone=freeze_backbone,
                    trust_remote_code=args.hf_trust_remote_code,
                    epochs=hf_epochs,
                    batch_size=args.hf_batch_size,
                    lr=hf_lr,
                    seed=args.seed,
                    device=device,
                )

                train_model_hf(
                    task="acceptor",
                    pos_path=pos_path,
                    neg_path=neg_path,
                    out_dir=acceptor_dir,
                    model_name=model_name,
                    tokenization=args.hf_tokenization,
                    k=args.hf_k,
                    max_len=args.hf_max_len,
                    pooling=args.hf_pooling,
                    readout_hidden=args.hf_readout_hidden,
                    freeze_backbone=freeze_backbone,
                    trust_remote_code=args.hf_trust_remote_code,
                    epochs=hf_epochs,
                    batch_size=args.hf_batch_size,
                    lr=hf_lr,
                    seed=args.seed,
                    device=device,
                )
    else:
        print("Skipping training (--skip_training)")

    if args.model_type == "esn":
        donor_model_path = os.path.join(donor_model_dir, "best.pt")
        acceptor_model_path = os.path.join(acceptor_model_dir, "best.pt")

        if not os.path.exists(donor_model_path) or not os.path.exists(
            acceptor_model_path
        ):
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
        )
    else:
        hf_models = parse_hf_models(args.hf_models)
        for model_name in hf_models:
            tag = sanitize_model_tag(model_name)
            donor_dir = os.path.join(donor_model_dir, tag)
            acceptor_dir = os.path.join(acceptor_model_dir, tag)
            donor_model_path = os.path.join(donor_dir, "best.pt")
            acceptor_model_path = os.path.join(acceptor_dir, "best.pt")

            if not os.path.exists(donor_model_path) or not os.path.exists(
                acceptor_model_path
            ):
                print("Error: Model checkpoints not found!")
                print(f"  Donor: {donor_model_path}")
                print(f"  Acceptor: {acceptor_model_path}")
                continue

            output_tsv = add_suffix_to_path(output_tsv, tag)
            score_test_sites_hf(
                test_tsv=test_tsv,
                donor_model_path=donor_model_path,
                acceptor_model_path=acceptor_model_path,
                output_tsv=output_tsv,
                device=device,
                batch_size=args.hf_batch_size,
            )


# --------------------------
# Repro utilities
# --------------------------


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# --------------------------
# Parsing training files
# --------------------------

LINE_PAIR_RE = re.compile(
    r"^DEBUG\s+donor\s+([A-Za-z]+)\s+acceptor\s+([A-Za-z]+)(?:\s+[+-])?\s*$"
)
LINE_SINGLE_RE = re.compile(r"^DEBUG\s+(donor|acceptor)\s+([A-Za-z]+)(?:\s+[+-])?\s*$")


def read_examples_single_task(
    pos_path: str,
    neg_path: str,
    task: str,
) -> List[Tuple[str, int]]:
    """Read training examples for one task (donor or acceptor)."""
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
                    if task == "donor":
                        examples.append((donor_seq.upper(), label))
                    else:
                        examples.append((acceptor_seq.upper(), label))
                    continue

                m_single = LINE_SINGLE_RE.match(line)
                if m_single:
                    tname, seq = m_single.groups()
                    if tname == task:
                        examples.append((seq.upper(), label))
                    continue

    read_one_set(pos_path, label=1)
    read_one_set(neg_path, label=0)
    return examples


# --------------------------
# k-mer tokenizer
# --------------------------

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]

DEFAULT_HF_MODELS = [
    "zhihan1996/DNABERT-2-117M",
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "LongSafari/hyenadna-tiny-1k-seqlen",
]


def build_kmer_vocab(k: int = 3) -> Dict[str, int]:
    bases = ["A", "C", "G", "T"]
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    idx = len(vocab)

    def rec(prefix: str, depth: int):
        nonlocal idx
        if depth == k:
            vocab[prefix] = idx
            idx += 1
            return
        for b in bases:
            rec(prefix + b, depth + 1)

    rec("", 0)
    return vocab


def kmerize(seq: str, k: int) -> List[str]:
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(0, len(seq) - k + 1)]


def encode_kmers(kmers: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab["[UNK]"]
    return [vocab.get(km, unk) for km in kmers]


def sanitize_model_tag(name: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return tag.strip("_.-") or "model"


def resolve_hf_tokenization(
    model_name: str, tokenization: str, k: int
) -> Tuple[str, int]:
    if tokenization != "auto":
        return tokenization, k
    lower = model_name.lower()
    if "dnabert" in lower:
        return "kmer", 6
    return "raw", k


def seq_to_kmer_text(seq: str, k: int) -> str:
    return " ".join(kmerize(seq, k))


# --------------------------
# Dataset
# --------------------------


class KmerDataset(Dataset):
    def __init__(
        self,
        examples: List[Tuple[str, int]],
        vocab: Dict[str, int],
        k: int = 3,
        max_len: int = 32,
    ):
        self.examples = examples
        self.vocab = vocab
        self.k = k
        self.max_len = max_len
        self.pad_id = vocab["[PAD]"]
        self.cls_id = vocab["[CLS]"]
        self.sep_id = vocab["[SEP]"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        kmers = kmerize(seq, self.k)
        ids = [self.cls_id] + encode_kmers(kmers, self.vocab) + [self.sep_id]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            ids[-1] = self.sep_id

        attn_mask = [1] * len(ids)
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.pad_id] * pad_len
            attn_mask += [0] * pad_len

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_fn(batch):
    input_ids, attn_mask, labels = zip(*batch)
    return (
        torch.stack(input_ids, dim=0),
        torch.stack(attn_mask, dim=0),
        torch.stack(labels, dim=0),
    )


class HFSequenceDataset(Dataset):
    def __init__(
        self,
        examples: List[Tuple[str, int]],
        tokenizer,
        max_len: int,
        tokenization: str,
        k: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tokenization = tokenization
        self.k = k

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        if self.tokenization == "kmer":
            text = seq_to_kmer_text(seq, self.k)
        else:
            text = seq

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return (
            input_ids,
            attention_mask,
            torch.tensor(label, dtype=torch.float32),
        )


# --------------------------
# ESN Reservoir Model
# --------------------------


class ESNReadout(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        input_dim: int,
        reservoir_size: int,
        spectral_radius: float,
        leak: float,
        sparsity: float,
        readout_hidden: int,
        seed: int = 1337,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak = leak
        self.sparsity = sparsity

        rng = np.random.default_rng(seed)

        # Fixed random embedding
        emb = rng.normal(
            0.0, 1.0 / math.sqrt(input_dim), size=(vocab_size, input_dim)
        ).astype(np.float32)
        self.register_buffer("tok_emb", torch.from_numpy(emb), persistent=True)

        # Fixed input weight
        w_in = rng.normal(
            0.0, 1.0 / math.sqrt(input_dim), size=(reservoir_size, input_dim)
        ).astype(np.float32)
        self.register_buffer("w_in", torch.from_numpy(w_in), persistent=True)

        # Fixed recurrent weight with sparsity
        w_res = rng.normal(0.0, 1.0, size=(reservoir_size, reservoir_size)).astype(
            np.float32
        )
        mask = rng.random((reservoir_size, reservoir_size))
        w_res[mask > sparsity] = 0.0

        # Scale to target spectral radius (power iteration)
        w_res_t = torch.from_numpy(w_res)
        radius = self._estimate_spectral_radius(w_res_t)
        scale = spectral_radius / max(radius, 1e-6)
        w_res_t = w_res_t * scale
        self.register_buffer("w_res", w_res_t, persistent=True)

        # Trainable readout MLP (deeper for better feature extraction)
        # Input will be reservoir_size * 2 due to avg+max pooling
        self.readout = nn.Sequential(
            nn.Linear(reservoir_size * 2, readout_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(readout_hidden, readout_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(readout_hidden // 2, 1),
        )

    @staticmethod
    def _estimate_spectral_radius(w: torch.Tensor, iters: int = 50) -> float:
        # Power iteration
        v = torch.randn(w.shape[0], 1)
        for _ in range(iters):
            v = w @ v
            v = v / (v.norm() + 1e-9)
        rayleigh = (v.t() @ w @ v).item()
        return abs(rayleigh)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L)
        # attn_mask: (B, L)
        bsz, seq_len = input_ids.shape

        # Embed tokens (fixed)
        u = self.tok_emb[input_ids]  # (B, L, input_dim)

        # ESN state update
        x = torch.zeros(bsz, self.reservoir_size, device=input_ids.device)
        states = []

        for t in range(seq_len):
            u_t = u[:, t, :]  # (B, input_dim)
            pre = torch.matmul(u_t, self.w_in.t()) + torch.matmul(x, self.w_res.t())
            x = (1.0 - self.leak) * x + self.leak * torch.tanh(pre)

            # only keep valid positions
            mask_t = attn_mask[:, t].unsqueeze(1)
            states.append(x * mask_t)

        # Pool states (mean + max over valid steps for richer representation)
        stacked = torch.stack(states, dim=1)  # (B, L, R)
        valid = attn_mask.sum(dim=1).unsqueeze(1).clamp(min=1.0)

        # Average pooling
        avg_pooled = stacked.sum(dim=1) / valid  # (B, R)

        # Max pooling (ignoring padded positions)
        stacked_masked = stacked.clone()
        mask_expanded = attn_mask.unsqueeze(2).expand_as(stacked)  # (B, L, R)
        stacked_masked[mask_expanded == 0] = -1e9
        max_pooled, _ = stacked_masked.max(dim=1)  # (B, R)

        # Concatenate avg and max
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # (B, 2*R)

        logits = self.readout(pooled).squeeze(-1)
        return logits


class HFReadout(nn.Module):
    def __init__(
        self,
        model_name: str,
        pooling: str,
        readout_hidden: int,
        freeze_backbone: bool = True,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError(
                "transformers is not installed. Install with: pip install transformers"
            )
        self.model_name = model_name
        self.pooling = pooling
        self.freeze_backbone = freeze_backbone

        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        hidden = int(self.backbone.config.hidden_size)

        if pooling == "mean_max":
            in_dim = hidden * 2
        else:
            in_dim = hidden

        self.readout = nn.Sequential(
            nn.Linear(in_dim, readout_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(readout_hidden, readout_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(readout_hidden // 2, 1),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).type_as(hidden)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean = (hidden * mask).sum(dim=1) / denom

        if self.pooling == "mean":
            return mean

        masked = hidden.masked_fill(mask == 0, -1e9)
        max_pooled, _ = masked.max(dim=1)

        if self.pooling == "max":
            return max_pooled

        return torch.cat([mean, max_pooled], dim=1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        hidden = outputs.last_hidden_state
        pooled = self._pool(hidden, attention_mask)
        logits = self.readout(pooled).squeeze(-1)
        return logits


# --------------------------
# Train / eval
# --------------------------


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    for input_ids, attn_mask, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attn_mask)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    probs = sigmoid_np(logits) if logits.size else np.array([])

    metrics = {}
    if labels.size:
        metrics["acc@0.5"] = float(np.mean((probs >= 0.5) == (labels >= 0.5)))
        metrics["loss_bce"] = float(
            np.mean(
                -(
                    labels * np.log(probs + 1e-9)
                    + (1 - labels) * np.log(1 - probs + 1e-9)
                )
            )
        )
        if roc_auc_score is not None and len(np.unique(labels)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        if average_precision_score is not None and len(np.unique(labels)) > 1:
            metrics["pr_auc"] = float(average_precision_score(labels, probs))

    return metrics


def stratified_split(
    examples: List[Tuple[str, int]], val_frac: float = 0.1, seed: int = 1337
):
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


def train_model(
    task: str,
    pos_path: str,
    neg_path: str,
    out_dir: str,
    k: int,
    max_len: int,
    input_dim: int,
    reservoir_size: int,
    spectral_radius: float,
    leak: float,
    sparsity: float,
    readout_hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
):
    print(f"\n{'=' * 60}")
    print(f"Training {task.upper()} ESN model")
    print(f"{'=' * 60}")

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    examples = read_examples_single_task(pos_path, neg_path, task)
    n_pos = sum(y for _, y in examples)
    n_neg = len(examples) - n_pos
    print(f"Total examples: {len(examples)} | pos={n_pos} | neg={n_neg}")

    train_ex, val_ex = stratified_split(examples, val_frac=0.1, seed=seed)
    train_pos = sum(y for _, y in train_ex)
    train_neg = len(train_ex) - train_pos
    print(f"Train: {len(train_ex)} | pos={train_pos} | neg={train_neg}")
    print(
        f"Val:   {len(val_ex)} | pos={sum(y for _, y in val_ex)} | neg={len(val_ex) - sum(y for _, y in val_ex)}"
    )

    vocab = build_kmer_vocab(k)
    print(f"Vocab size (k={k}): {len(vocab)}")

    train_ds = KmerDataset(train_ex, vocab=vocab, k=k, max_len=max_len)
    val_ds = KmerDataset(val_ex, vocab=vocab, k=k, max_len=max_len)

    num_workers = 0 if device in ["mps", "cpu"] else 2
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    model = ESNReadout(
        vocab_size=len(vocab),
        max_len=max_len,
        input_dim=input_dim,
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        leak=leak,
        sparsity=sparsity,
        readout_hidden=readout_hidden,
        seed=seed,
    ).to(device)

    pos_weight_raw = (train_neg / max(1, train_pos)) if train_pos > 0 else 1.0
    pos_weight = min(pos_weight_raw, 20.0)
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t, reduction="mean")

    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    print(f"pos_weight (raw): {pos_weight_raw:.4f} -> (capped): {pos_weight:.4f}")
    print(f"Learning rate: {lr}")
    print(f"Reservoir size: {reservoir_size}")

    best_score = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (input_ids, attn_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.readout.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] loss: {loss.item():.4f}"
                )

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
                    "k": k,
                    "max_len": max_len,
                    "input_dim": input_dim,
                    "reservoir_size": reservoir_size,
                    "spectral_radius": spectral_radius,
                    "leak": leak,
                    "sparsity": sparsity,
                    "readout_hidden": readout_hidden,
                    "vocab": build_kmer_vocab(k),
                    "model_state": model.state_dict(),
                },
                ckpt_path,
            )
            print(f"  ✅ Saved checkpoint: {ckpt_path}")

    print(f"\nBest {task} {score_name}: {best_score:.4f}")
    return model


def _ensure_hf_tokenizer(model_name: str, trust_remote_code: bool):
    if AutoTokenizer is None:
        raise RuntimeError(
            "transformers is not installed. Install with: pip install transformers"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def train_model_hf(
    task: str,
    pos_path: str,
    neg_path: str,
    out_dir: str,
    model_name: str,
    tokenization: str,
    k: int,
    max_len: int,
    pooling: str,
    readout_hidden: int,
    freeze_backbone: bool,
    trust_remote_code: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
):
    print(f"\n{'=' * 60}")
    print(f"Training {task.upper()} HF reservoir model")
    print(f"Backbone: {model_name}")
    print(f"{'=' * 60}")

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    examples = read_examples_single_task(pos_path, neg_path, task)
    n_pos = sum(y for _, y in examples)
    n_neg = len(examples) - n_pos
    print(f"Total examples: {len(examples)} | pos={n_pos} | neg={n_neg}")

    train_ex, val_ex = stratified_split(examples, val_frac=0.1, seed=seed)
    train_pos = sum(y for _, y in train_ex)
    train_neg = len(train_ex) - train_pos
    print(f"Train: {len(train_ex)} | pos={train_pos} | neg={train_neg}")
    print(
        f"Val:   {len(val_ex)} | pos={sum(y for _, y in val_ex)} | neg={len(val_ex) - sum(y for _, y in val_ex)}"
    )

    tokenizer = _ensure_hf_tokenizer(model_name, trust_remote_code)
    tokenization, k = resolve_hf_tokenization(model_name, tokenization, k)
    print(f"Tokenization: {tokenization} (k={k})")
    print(f"Max length: {max_len}")

    train_ds = HFSequenceDataset(
        train_ex,
        tokenizer=tokenizer,
        max_len=max_len,
        tokenization=tokenization,
        k=k,
    )
    val_ds = HFSequenceDataset(
        val_ex,
        tokenizer=tokenizer,
        max_len=max_len,
        tokenization=tokenization,
        k=k,
    )

    num_workers = 0 if device in ["mps", "cpu"] else 2
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    model = HFReadout(
        model_name=model_name,
        pooling=pooling,
        readout_hidden=readout_hidden,
        freeze_backbone=freeze_backbone,
        trust_remote_code=trust_remote_code,
    ).to(device)

    model.backbone.resize_token_embeddings(len(tokenizer))
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    pos_weight_raw = (train_neg / max(1, train_pos)) if train_pos > 0 else 1.0
    pos_weight = min(pos_weight_raw, 20.0)
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t, reduction="mean")

    if freeze_backbone:
        params = list(model.readout.parameters())
    else:
        params = list(model.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    print(f"pos_weight (raw): {pos_weight_raw:.4f} -> (capped): {pos_weight:.4f}")
    print(f"Learning rate: {lr}")
    print(f"Pooling: {pooling}")
    print(f"Freeze backbone: {freeze_backbone}")

    best_score = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (input_ids, attn_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] loss: {loss.item():.4f}"
                )

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

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Val metrics: {val_metrics}")
        print(f"  Score ({score_name}): {score:.4f}")

        if score > best_score:
            best_score = score
            ckpt_path = os.path.join(out_dir, "best.pt")
            payload = {
                "task": task,
                "hf_model_name": model_name,
                "hf_pooling": pooling,
                "hf_max_len": max_len,
                "hf_tokenization": tokenization,
                "hf_k": k,
                "readout_hidden": readout_hidden,
                "freeze_backbone": freeze_backbone,
                "hf_trust_remote_code": trust_remote_code,
                "readout_state": model.readout.state_dict(),
            }
            if not freeze_backbone:
                payload["backbone_state"] = model.backbone.state_dict()
            torch.save(payload, ckpt_path)
            print(f"  ✅ Saved checkpoint: {ckpt_path}")

    print(f"\nBest {task} {score_name}: {best_score:.4f}")
    return model


# --------------------------
# Scoring
# --------------------------


@torch.no_grad()
def load_model(checkpoint_path: str, device: str) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ESNReadout(
        vocab_size=len(ckpt["vocab"]),
        max_len=ckpt["max_len"],
        input_dim=ckpt["input_dim"],
        reservoir_size=ckpt["reservoir_size"],
        spectral_radius=ckpt["spectral_radius"],
        leak=ckpt["leak"],
        sparsity=ckpt["sparsity"],
        readout_hidden=ckpt["readout_hidden"],
        seed=1337,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def load_model_hf(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trust_remote_code = ckpt.get("hf_trust_remote_code", False)

    model = HFReadout(
        model_name=ckpt["hf_model_name"],
        pooling=ckpt["hf_pooling"],
        readout_hidden=ckpt["readout_hidden"],
        freeze_backbone=ckpt.get("freeze_backbone", True),
        trust_remote_code=trust_remote_code,
    ).to(device)

    tokenizer = _ensure_hf_tokenizer(ckpt["hf_model_name"], trust_remote_code)
    model.backbone.resize_token_embeddings(len(tokenizer))
    if ckpt.get("freeze_backbone", True):
        for p in model.backbone.parameters():
            p.requires_grad = False

    model.readout.load_state_dict(ckpt["readout_state"], strict=True)
    if "backbone_state" in ckpt:
        model.backbone.load_state_dict(ckpt["backbone_state"], strict=True)
    model.eval()
    tokenization = ckpt.get("hf_tokenization", "raw")
    k = int(ckpt.get("hf_k", 6))
    max_len = int(ckpt["hf_max_len"])
    return model, tokenizer, tokenization, k, max_len, ckpt


@torch.no_grad()
def score_sequences(
    model,
    sequences: List[str],
    vocab: Dict[str, int],
    k: int,
    max_len: int,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    dataset = KmerDataset(
        [(s, 0) for s in sequences], vocab=vocab, k=k, max_len=max_len
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    all_probs = []
    for input_ids, attn_mask, _ in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        logits = model(input_ids, attn_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs) if all_probs else np.array([])


@torch.no_grad()
def score_sequences_hf(
    model,
    sequences: List[str],
    tokenizer,
    tokenization: str,
    k: int,
    max_len: int,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    dataset = HFSequenceDataset(
        [(s, 0) for s in sequences],
        tokenizer=tokenizer,
        max_len=max_len,
        tokenization=tokenization,
        k=k,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    all_probs = []
    for input_ids, attn_mask, _ in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        logits = model(input_ids, attn_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs) if all_probs else np.array([])


def score_test_sites(
    test_tsv: str,
    donor_model_path: str,
    acceptor_model_path: str,
    output_tsv: str,
    device: str,
):
    print(f"\n{'=' * 60}")
    print("Scoring test sites")
    print(f"{'=' * 60}")

    donor_model, donor_ckpt = load_model(donor_model_path, device)
    acceptor_model, acceptor_ckpt = load_model(acceptor_model_path, device)

    if (
        donor_ckpt["k"] != acceptor_ckpt["k"]
        or donor_ckpt["max_len"] != acceptor_ckpt["max_len"]
    ):
        raise RuntimeError("Donor/acceptor config mismatch")

    k = donor_ckpt["k"]
    max_len = donor_ckpt["max_len"]
    vocab = donor_ckpt["vocab"]

    print(f"Loaded donor model: {donor_model_path}")
    print(f"Loaded acceptor model: {acceptor_model_path}")
    print(f"k={k}, max_len={max_len}")

    print(f"\nReading {test_tsv}...")
    data = []
    with open(test_tsv, "r") as f:
        _ = next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 8:
                data.append(
                    {
                        "transcript_id": parts[0],
                        "site_type": parts[2],
                        "intron_index": int(parts[3]),
                        "seq": parts[7],
                    }
                )

    print(f"Total sites: {len(data)}")

    transcript_introns = defaultdict(lambda: defaultdict(dict))
    for row in data:
        tid = row["transcript_id"]
        iidx = row["intron_index"]
        stype = row["site_type"]
        seq = row["seq"]
        transcript_introns[tid][iidx][stype] = seq

    results = []
    print("\nScoring transcripts...")

    all_donor_seqs = []
    all_acceptor_seqs = []
    transcript_keys = []

    for tid, introns in transcript_introns.items():
        for iidx in sorted(introns.keys()):
            sites = introns[iidx]
            all_donor_seqs.append(sites.get("donor", ""))
            all_acceptor_seqs.append(sites.get("acceptor", ""))
            transcript_keys.append((tid, iidx))

    print(f"Total introns to score: {len(transcript_keys)}")

    print("Scoring donor sequences...")
    donor_scores = score_sequences(
        donor_model, all_donor_seqs, vocab, k, max_len, device, batch_size=512
    )

    print("Scoring acceptor sequences...")
    acceptor_scores = score_sequences(
        acceptor_model, all_acceptor_seqs, vocab, k, max_len, device, batch_size=512
    )

    print("Aggregating results...")
    transcript_intron_dict = defaultdict(dict)

    for idx, (tid, iidx) in enumerate(transcript_keys):
        donor_score = donor_scores[idx] if all_donor_seqs[idx] else 0.0
        acceptor_score = acceptor_scores[idx] if all_acceptor_seqs[idx] else 0.0
        total_score = donor_score + acceptor_score
        transcript_intron_dict[tid][iidx] = (donor_score, acceptor_score, total_score)

    for tid, introns_dict in transcript_intron_dict.items():
        if not introns_dict:
            continue
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

    results.sort(key=lambda x: x["transcript_id"])

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


def score_test_sites_hf(
    test_tsv: str,
    donor_model_path: str,
    acceptor_model_path: str,
    output_tsv: str,
    device: str,
    batch_size: int,
):
    print(f"\n{'=' * 60}")
    print("Scoring test sites (HF reservoir)")
    print(f"{'=' * 60}")

    donor_model, donor_tok, donor_tok_mode, donor_k, donor_max_len, donor_ckpt = (
        load_model_hf(donor_model_path, device)
    )
    (
        acceptor_model,
        acc_tok,
        acc_tok_mode,
        acc_k,
        acc_max_len,
        acceptor_ckpt,
    ) = load_model_hf(acceptor_model_path, device)

    if donor_ckpt["hf_model_name"] != acceptor_ckpt["hf_model_name"]:
        raise RuntimeError("Donor/acceptor backbone mismatch")
    if donor_max_len != acc_max_len:
        raise RuntimeError("Donor/acceptor max_len mismatch")
    if donor_tok_mode != acc_tok_mode or donor_k != acc_k:
        raise RuntimeError("Donor/acceptor tokenization mismatch")

    print(f"Loaded donor model: {donor_model_path}")
    print(f"Loaded acceptor model: {acceptor_model_path}")
    print(f"Backbone: {donor_ckpt['hf_model_name']}")
    print(f"Tokenization: {donor_tok_mode} (k={donor_k})")
    print(f"Max length: {donor_max_len}")

    print(f"\nReading {test_tsv}...")
    data = []
    with open(test_tsv, "r") as f:
        _ = next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 8:
                data.append(
                    {
                        "transcript_id": parts[0],
                        "site_type": parts[2],
                        "intron_index": int(parts[3]),
                        "seq": parts[7],
                    }
                )

    print(f"Total sites: {len(data)}")

    transcript_introns = defaultdict(lambda: defaultdict(dict))
    for row in data:
        tid = row["transcript_id"]
        iidx = row["intron_index"]
        stype = row["site_type"]
        seq = row["seq"]
        transcript_introns[tid][iidx][stype] = seq

    results = []
    print("\nScoring transcripts...")

    all_donor_seqs = []
    all_acceptor_seqs = []
    transcript_keys = []

    for tid, introns in transcript_introns.items():
        for iidx in sorted(introns.keys()):
            sites = introns[iidx]
            all_donor_seqs.append(sites.get("donor", ""))
            all_acceptor_seqs.append(sites.get("acceptor", ""))
            transcript_keys.append((tid, iidx))

    print(f"Total introns to score: {len(transcript_keys)}")

    print("Scoring donor sequences...")
    donor_scores = score_sequences_hf(
        donor_model,
        all_donor_seqs,
        donor_tok,
        donor_tok_mode,
        donor_k,
        donor_max_len,
        device,
        batch_size=batch_size,
    )

    print("Scoring acceptor sequences...")
    acceptor_scores = score_sequences_hf(
        acceptor_model,
        all_acceptor_seqs,
        acc_tok,
        acc_tok_mode,
        acc_k,
        acc_max_len,
        device,
        batch_size=batch_size,
    )

    print("Aggregating results...")
    transcript_intron_dict = defaultdict(dict)

    for idx, (tid, iidx) in enumerate(transcript_keys):
        donor_score = donor_scores[idx] if all_donor_seqs[idx] else 0.0
        acceptor_score = acceptor_scores[idx] if all_acceptor_seqs[idx] else 0.0
        total_score = donor_score + acceptor_score
        transcript_intron_dict[tid][iidx] = (donor_score, acceptor_score, total_score)

    for tid, introns_dict in transcript_intron_dict.items():
        if not introns_dict:
            continue
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

    results.sort(key=lambda x: x["transcript_id"])

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


def parse_hf_models(arg: str) -> List[str]:
    if arg:
        return [m.strip() for m in arg.split(",") if m.strip()]
    return DEFAULT_HF_MODELS


def add_suffix_to_path(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}.{suffix}{ext or '.tsv'}"


if __name__ == "__main__":
    main()
