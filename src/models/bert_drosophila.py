"""
BERT-based splice site scoring for Dmel
=============================================
1. Train independent BERT models for donor and acceptor sites
2. Score all introns in transcripts.tsv
3. Output transcript-level scores (min intron score)

Usage:
    python bert_Dmel.py
"""

import argparse
import os
import re
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = None
    average_precision_score = None


# --------------------------
# Repro utilities
# --------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------
# Parsing your files
# --------------------------
# Supports both formats:
#   DEBUG donor <SEQ> acceptor <SEQ> +
#   DEBUG donor <SEQ> +
#   DEBUG acceptor <SEQ> -
#
# The final + or - is strand and ignored.
LINE_PAIR_RE = re.compile(
    r"^DEBUG\s+donor\s+([A-Za-z]+)\s+acceptor\s+([A-Za-z]+)(?:\s+[+-])?\s*$"
)
LINE_SINGLE_RE = re.compile(r"^DEBUG\s+(donor|acceptor)\s+([A-Za-z]+)(?:\s+[+-])?\s*$")

TASK2ID = {"donor": 0, "acceptor": 1}


def read_examples_single_task(
    pos_paths: List[str],
    neg_paths: List[str],
    task: str,
) -> List[Tuple[str, int]]:
    """
    Returns list of (sequence, label) for ONE task only (donor or acceptor).

    Label convention:
      - pos_paths => label 1
      - neg_paths => label 0

    Strand (+/- at end of line) is ignored.
    """
    assert task in TASK2ID
    want_task = task  # "donor" or "acceptor"
    examples: List[Tuple[str, int]] = []

    def read_one_set(paths: List[str], label: int):
        nonlocal examples
        for path in paths:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("DEBUG"):
                        continue

                    # Paired format: contains both donor and acceptor sequences
                    m_pair = LINE_PAIR_RE.match(line)
                    if m_pair:
                        donor_seq, acceptor_seq = m_pair.groups()
                        if want_task == "donor":
                            examples.append((donor_seq.upper(), label))
                        else:
                            examples.append((acceptor_seq.upper(), label))
                        continue

                    # Single format: one of donor/acceptor
                    m_single = LINE_SINGLE_RE.match(line)
                    if m_single:
                        tname, seq = m_single.groups()
                        if tname == want_task:
                            examples.append((seq.upper(), label))
                        continue

                    # Otherwise ignore unrecognized DEBUG formats

    read_one_set(pos_paths, label=1)
    read_one_set(neg_paths, label=0)
    return examples


# --------------------------
# k-mer tokenizer
# --------------------------
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]


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


# --------------------------
# Dataset + Batch
# --------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, L) token ids
    attn_mask: torch.Tensor  # (B, L) 1 real, 0 pad
    labels: torch.Tensor  # (B,) 0 = neg example, 1 = pos example


class SpliceDataset(Dataset):
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
        seq, y = self.examples[idx]
        kmers = kmerize(seq, self.k)
        ids = [self.cls_id] + encode_kmers(kmers, self.vocab) + [self.sep_id]

        # truncate
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            ids[-1] = self.sep_id

        attn_mask = [1] * len(ids)

        # pad
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.pad_id] * pad_len
            attn_mask += [0] * pad_len

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


def collate_fn(batch) -> Batch:
    input_ids, attn_mask, labels = zip(*batch)
    return Batch(
        input_ids=torch.stack(input_ids, dim=0),
        attn_mask=torch.stack(attn_mask, dim=0),
        labels=torch.stack(labels, dim=0),
    )


# --------------------------
# Small BERT-like encoder + single head
# --------------------------
class SmallBertEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Transformer expects True for PAD positions
        key_padding_mask = attn_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x  # (B, L, D)


class SingleTaskSpliceModel(nn.Module):
    def __init__(self, encoder: SmallBertEncoder, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.encoder(input_ids, attn_mask)  # (B, L, D)
        cls = x[:, 0, :]  # (B, D) [CLS] token
        logits = self.head(cls).squeeze(-1)  # (B,)
        return logits


# --------------------------
# Train / eval
# --------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attn_mask = batch.attn_mask.to(device)
        labels = batch.labels.to(device)

        logits = model(input_ids, attn_mask)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])

    probs = sigmoid_np(logits) if logits.size else np.array([])

    out = {}
    if labels.size:
        # we will use pr_auc prefarably,
        # fallback to roc_auc if pr_auc not available,
        # else fallback to acc@0.5 (not great for imbalanced but better than nothing)

        out["acc@0.5"] = float(np.mean((probs >= 0.5) == (labels >= 0.5)))
        out["loss_proxy_bce"] = float(
            np.mean(
                -(
                    labels * np.log(probs + 1e-9)
                    + (1 - labels) * np.log(1 - probs + 1e-9)
                )
            )
        )
        if roc_auc_score is not None and len(np.unique(labels)) > 1:
            out["roc_auc"] = float(roc_auc_score(labels, probs))
        if average_precision_score is not None and len(np.unique(labels)) > 1:
            out["pr_auc"] = float(average_precision_score(labels, probs))
        out["n"] = int(labels.size)
        out["pos"] = int(labels.sum())
        out["neg"] = int(labels.size - labels.sum())
    return out


def stratified_split_by_label(examples: List[Tuple[str, int]], val_frac=0.1, seed=1337):
    """
    Stratify by label (0/1) so train/val both have positives and negatives.

    val_frac: fraction of examples to put in validation set (e.g. 0.1 for 90% train, 10% val)
    seed: random seed for shuffling
    """
    rng = random.Random(seed)
    buckets = {0: [], 1: []}
    for seq, y in examples:
        buckets[int(y)].append((seq, int(y)))

    train, val = [], []
    for y, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac)) if len(items) > 0 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

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
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_frac: float,
    seed: int,
    num_workers: int,
    pos_weight: float | None,
    device: str,
):
    print(f"\n{'=' * 60}")
    print(f"Training {task.upper()} BERT model")
    print(f"{'=' * 60}")

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    pin = device == "cuda"

    examples = read_examples_single_task(
        pos_paths=[pos_path],
        neg_paths=[neg_path],
        task=task,
    )
    if len(examples) == 0:
        raise RuntimeError("No examples parsed. Check file formats / regex in script.")

    n_pos = sum(y for _, y in examples)
    n_neg = len(examples) - n_pos
    print(f"Total examples: {len(examples)} | pos={n_pos} | neg={n_neg}")

    train_ex, val_ex = stratified_split_by_label(examples, val_frac=val_frac, seed=seed)
    train_pos = sum(y for _, y in train_ex)
    train_neg = len(train_ex) - train_pos
    val_pos = sum(y for _, y in val_ex)
    val_neg = len(val_ex) - val_pos
    print(f"Train: {len(train_ex)} | pos={train_pos} | neg={train_neg}")
    print(f"Val:   {len(val_ex)} | pos={val_pos} | neg={val_neg}")

    vocab = build_kmer_vocab(k)
    print(f"Vocab size (k={k}): {len(vocab)} (incl specials)")

    train_ds = SpliceDataset(train_ex, vocab=vocab, k=k, max_len=max_len)
    val_ds = SpliceDataset(val_ex, vocab=vocab, k=k, max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        drop_last=False,
    )

    encoder = SmallBertEncoder(
        vocab_size=len(vocab),
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
    model = SingleTaskSpliceModel(encoder, d_model=d_model).to(device)

    if pos_weight is not None:
        pos_w = float(pos_weight)
    else:
        pos_w = (train_neg / max(1, train_pos)) if train_pos > 0 else 1.0
    pos_weight_t = torch.tensor([pos_w], dtype=torch.float32, device=device)
    print(f"Using pos_weight={pos_w:.4f} in BCEWithLogitsLoss")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * max(1, len(train_loader))

    def lr_at(step):
        min_lr = lr * 0.1
        return min_lr + 0.5 * (lr - min_lr) * (
            1 + math.cos(math.pi * step / total_steps)
        )

    best_val_score = -1e9
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = torch.zeros((), dtype=torch.float64)

        for batch in train_loader:
            input_ids = batch.input_ids.to(device)
            attn_mask = batch.attn_mask.to(device)
            labels = batch.labels.to(device)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            lr_now = lr_at(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            running_loss = running_loss + loss.detach().to(
                device="cpu",
                dtype=torch.float64,
            )

        train_loss = float(running_loss / max(1, len(train_loader)))
        val_metrics = evaluate(model, val_loader, device=device)

        if "pr_auc" in val_metrics:
            score = val_metrics["pr_auc"]
            score_name = "pr_auc"
        elif "roc_auc" in val_metrics:
            score = val_metrics["roc_auc"]
            score_name = "roc_auc"
        else:
            score = val_metrics.get("acc@0.5", 0.0)
            score_name = "acc@0.5"

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        if val_metrics:
            print(f"  Val metrics: {val_metrics}")
        print(f"  Model score ({score_name}): {score:.4f}")

        if score > best_val_score:
            best_val_score = score
            ckpt = {
                "args": {
                    "task": task,
                    "k": k,
                    "max_len": max_len,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "dropout": dropout,
                },
                "vocab": vocab,
                "model_state": model.state_dict(),
            }
            out_path = os.path.join(out_dir, "best.pt")
            torch.save(ckpt, out_path)
            print(f"  ✅ Saved best checkpoint to: {out_path}")

    print("\nDone.")
    print(f"Best {task} model score: {best_val_score:.4f}")
    print(f"Checkpoint: {os.path.join(out_dir, 'best.pt')}")


def load_model(
    checkpoint_path: str, device: str
) -> Tuple[nn.Module, Dict, Dict[str, int]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    vocab = ckpt.get("vocab")
    if vocab is None:
        raise RuntimeError("Checkpoint missing vocab. Re-train the model.")

    k = int(ckpt_args.get("k", 3))
    max_len = int(ckpt_args.get("max_len", 32))
    d_model = int(ckpt_args.get("d_model", 128))
    n_heads = int(ckpt_args.get("n_heads", 4))
    n_layers = int(ckpt_args.get("n_layers", 4))
    dropout = float(ckpt_args.get("dropout", 0.1))

    encoder = SmallBertEncoder(
        vocab_size=len(vocab),
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
    model = SingleTaskSpliceModel(encoder, d_model=d_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt_args, vocab


def encode_sequence(
    seq: str, vocab: Dict[str, int], k: int, max_len: int
) -> Tuple[List[int], List[int]]:
    pad_id = vocab["[PAD]"]
    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]

    kmers = kmerize(seq, k)
    ids = [cls_id] + encode_kmers(kmers, vocab) + [sep_id]

    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = sep_id

    attn_mask = [1] * len(ids)
    pad_len = max_len - len(ids)
    if pad_len > 0:
        ids += [pad_id] * pad_len
        attn_mask += [0] * pad_len

    return ids, attn_mask


@torch.no_grad()
def score_sequences(
    model: nn.Module,
    sequences: List[str],
    vocab: Dict[str, int],
    k: int,
    max_len: int,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    all_probs = []

    encoded_ids = []
    encoded_masks = []
    for seq in sequences:
        ids, mask = encode_sequence(seq.upper(), vocab, k, max_len)
        encoded_ids.append(ids)
        encoded_masks.append(mask)

    input_ids = torch.tensor(encoded_ids, dtype=torch.long)
    attn_mask = torch.tensor(encoded_masks, dtype=torch.long)

    for i in range(0, len(input_ids), batch_size):
        batch_ids = input_ids[i : i + batch_size].to(device)
        batch_mask = attn_mask[i : i + batch_size].to(device)
        logits = model(batch_ids, batch_mask)
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

    donor_model, donor_args, donor_vocab = load_model(donor_model_path, device)
    acceptor_model, acceptor_args, acceptor_vocab = load_model(
        acceptor_model_path, device
    )
    max_len = int(donor_args.get("max_len", 32))
    k = int(donor_args.get("k", 3))

    print(f"Loaded donor model: {donor_model_path}")
    print(f"Loaded acceptor model: {acceptor_model_path}")
    print(f"k: {k} | max_len: {max_len}")

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
        donor_model,
        all_donor_seqs,
        donor_vocab,
        k,
        max_len,
        device,
        batch_size=512,
    )

    print("Scoring acceptor sequences...")
    acceptor_scores = score_sequences(
        acceptor_model,
        all_acceptor_seqs,
        acceptor_vocab,
        k,
        max_len,
        device,
        batch_size=512,
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


def main():
    ap = argparse.ArgumentParser(description="BERT-based splice site scoring")

    ap.add_argument("--species", type=str, default="Dmel", help="Species name")
    ap.add_argument(
        "--bp", type=int, default=50, help="Base pair length for training data"
    )

    ap.add_argument("--max_len", type=int, default=50, help="Max tokenized length")
    ap.add_argument("--epochs", type=int, default=10, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--k", type=int, default=6, help="k-mer length")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--n_layers", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Optional: manual positive class weight for BCE",
    )

    ap.add_argument(
        "--skip_training", action="store_true", help="Skip training, only score"
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (auto=CUDA>MPS>CPU)",
    )

    args = ap.parse_args()

    species = args.species
    bp = args.bp
    pos_path = f"../data/{species}/train/{bp}bp.err"
    neg_path = f"../data/{species}/train/{bp}bp.neg.err"
    donor_model_dir = "../model/dirosophila/donar/bert"
    acceptor_model_dir = "../model/dirosophila/acceptor/bert"
    test_tsv = f"../data/{species}/raw/transcripts.tsv"
    output_tsv = f"../data/{species}/trans_score/bert{bp}bp.tsv"

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

    if not args.skip_training:
        train_model(
            task="donor",
            pos_path=pos_path,
            neg_path=neg_path,
            out_dir=donor_model_dir,
            k=args.k,
            max_len=args.max_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_frac=args.val_frac,
            seed=args.seed,
            num_workers=args.num_workers,
            pos_weight=args.pos_weight,
            device=device,
        )

        train_model(
            task="acceptor",
            pos_path=pos_path,
            neg_path=neg_path,
            out_dir=acceptor_model_dir,
            k=args.k,
            max_len=args.max_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_frac=args.val_frac,
            seed=args.seed,
            num_workers=args.num_workers,
            pos_weight=args.pos_weight,
            device=device,
        )
    else:
        print("Skipping training (--skip_training flag set)")

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
    )


if __name__ == "__main__":
    main()
