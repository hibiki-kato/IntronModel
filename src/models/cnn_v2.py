"""
Improved CNN-based splice site scoring (FULL TRAIN + SCORE)
============================================================
- Multi-scale first layer
- 2 pooling layers
- No global average pooling
- Center-focused branch
- Stable BCE
- Train donor & acceptor independently
- Score transcripts (min intron score)
"""

import argparse
import os
import re
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = None
    average_precision_score = None


# =========================
# Utilities
# =========================
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =========================
# Data parsing
# =========================
LINE_PAIR_RE = re.compile(
    r"^DEBUG\s+donor\s+([A-Za-z]+)\s+acceptor\s+([A-Za-z]+)(?:\s+[+-])?\s*$"
)
LINE_SINGLE_RE = re.compile(r"^DEBUG\s+(donor|acceptor)\s+([A-Za-z]+)(?:\s+[+-])?\s*$")


def read_examples_single_task(pos_path, neg_path, task):
    examples = []

    def read_one(path, label):
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("DEBUG"):
                    continue

                m_pair = LINE_PAIR_RE.match(line)
                if m_pair:
                    donor_seq, acceptor_seq = m_pair.groups()
                    seq = donor_seq if task == "donor" else acceptor_seq
                    examples.append((seq.upper(), label))
                    continue

                m_single = LINE_SINGLE_RE.match(line)
                if m_single:
                    tname, seq = m_single.groups()
                    if tname == task:
                        examples.append((seq.upper(), label))

    read_one(pos_path, 1)
    read_one(neg_path, 0)
    return examples


# =========================
# Encoding
# =========================
def one_hot_encode_dna(seq, max_len):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, max_len), dtype=np.float32)
    for i, b in enumerate(seq[:max_len]):
        j = mapping.get(b)
        if j is not None:
            arr[j, i] = 1.0
    return arr


class DNADataset(Dataset):
    def __init__(self, examples, max_len):
        self.examples = examples
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        x = one_hot_encode_dna(seq, self.max_len)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


# =========================
# Improved CNN
# =========================
class ImprovedSpliceCNN(nn.Module):
    def __init__(self, max_len=50, center_len=21, dropout=0.15):
        super().__init__()
        self.max_len = max_len
        self.center_len = center_len

        self.conv5 = nn.Conv1d(4, 48, 5, padding=2)
        self.conv9 = nn.Conv1d(4, 48, 9, padding=4)
        self.bn1 = nn.BatchNorm1d(96)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(96, 128, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 192, 7, padding=3)
        self.bn3 = nn.BatchNorm1d(192)

        main_L = (max_len // 2) // 2
        self.pos_bias = nn.Parameter(torch.zeros(1, 192, main_L))

        self.center_branch = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        center_L = center_len // 2
        in_dim = 192 * main_L + 64 * center_L

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        orig = x

        a = self.conv5(x)
        b = self.conv9(x)
        x = torch.cat([a, b], dim=1)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = x + self.pos_bias
        main_flat = torch.flatten(x, 1)

        L = self.max_len
        c0 = (L - self.center_len) // 2
        c1 = c0 + self.center_len
        xc = orig[:, :, c0:c1]
        c = self.center_branch(xc)
        center_flat = torch.flatten(c, 1)

        z = torch.cat([main_flat, center_flat], dim=1)
        return self.fc(z).squeeze(-1)


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all = []
    labels_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    probs = torch.sigmoid(logits)

    metrics = {}
    metrics["acc@0.5"] = float(((probs >= 0.5) == labels).float().mean())

    metrics["loss_bce"] = float(
        F.binary_cross_entropy_with_logits(logits, labels).item()
    )

    if roc_auc_score and len(torch.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels.numpy(), probs.numpy()))

    if average_precision_score and len(torch.unique(labels)) > 1:
        metrics["pr_auc"] = float(
            average_precision_score(labels.numpy(), probs.numpy())
        )

    return metrics


# =========================
# Training
# =========================
def train_model(
    task, pos_path, neg_path, out_dir, max_len, epochs, batch_size, lr, seed, device
):

    print("\nTraining", task)
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    examples = read_examples_single_task(pos_path, neg_path, task)
    random.shuffle(examples)

    split = int(0.9 * len(examples))
    train_ex = examples[:split]
    val_ex = examples[split:]

    train_loader = DataLoader(
        DNADataset(train_ex, max_len), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(DNADataset(val_ex, max_len), batch_size=batch_size)

    model = ImprovedSpliceCNN(max_len=max_len).to(device)

    pos = sum(y for _, y in train_ex)
    neg = len(train_ex) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_score = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        score = val_metrics.get("pr_auc", val_metrics.get("roc_auc", 0))
        print(f"Epoch {epoch}: {val_metrics}")

        if score > best_score:
            best_score = score
            torch.save(
                {"model_state": model.state_dict(), "max_len": max_len},
                os.path.join(out_dir, "best.pt"),
            )

    return model


# =========================
# Scoring
# =========================
@torch.no_grad()
def score_sequences(model, sequences, max_len, device):
    model.eval()
    encoded = [one_hot_encode_dna(s, max_len) for s in sequences]
    x = torch.from_numpy(np.stack(encoded)).to(device)
    return torch.sigmoid(model(x)).cpu().numpy()


def score_test_sites(
    test_tsv, donor_model_path, acceptor_model_path, output_tsv, device
):

    donor_ckpt = torch.load(donor_model_path, map_location=device)
    acceptor_ckpt = torch.load(acceptor_model_path, map_location=device)

    max_len = donor_ckpt["max_len"]

    donor_model = ImprovedSpliceCNN(max_len=max_len).to(device)
    donor_model.load_state_dict(donor_ckpt["model_state"])

    acceptor_model = ImprovedSpliceCNN(max_len=max_len).to(device)
    acceptor_model.load_state_dict(acceptor_ckpt["model_state"])

    data = []
    with open(test_tsv) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 8:
                data.append(
                    {
                        "tid": parts[0],
                        "stype": parts[2],
                        "iidx": int(parts[3]),
                        "seq": parts[7],
                    }
                )

    transcript_introns = defaultdict(lambda: defaultdict(dict))
    for row in data:
        transcript_introns[row["tid"]][row["iidx"]][row["stype"]] = row["seq"]

    results = []

    for tid, introns in transcript_introns.items():
        intron_scores = []
        for iidx in introns:
            donor_seq = introns[iidx].get("donor", "")
            acceptor_seq = introns[iidx].get("acceptor", "")

            d_score = score_sequences(donor_model, [donor_seq], max_len, device)[0]
            a_score = score_sequences(acceptor_model, [acceptor_seq], max_len, device)[
                0
            ]

            intron_scores.append((iidx, d_score + a_score))

        min_iidx, min_score = min(intron_scores, key=lambda x: x[1])
        results.append((tid, min_iidx, min_score))

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    with open(output_tsv, "w") as f:
        f.write("transcript_id\tmin_intron_index\tmin_donor_plus_acceptor\n")
        for tid, iidx, score in sorted(results):
            f.write(f"{tid}\t{iidx}\t{score:.6f}\n")

    print("Scoring complete. Saved to", output_tsv)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="Dmel")
    ap.add_argument("--bp", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)

    species = args.species
    bp = args.bp

    pos_path = f"../data/{species}/train/{bp}bp.err"
    neg_path = f"../data/{species}/train/{bp}bp.neg.err"

    donor_dir = f"../model/{species}/donor_cnn"
    acceptor_dir = f"../model/{species}/acceptor_cnn"

    donor_model = train_model(
        "donor",
        pos_path,
        neg_path,
        donor_dir,
        bp,
        args.epochs,
        args.batch_size,
        args.lr,
        1337,
        device,
    )

    acceptor_model = train_model(
        "acceptor",
        pos_path,
        neg_path,
        acceptor_dir,
        bp,
        args.epochs,
        args.batch_size,
        args.lr,
        1337,
        device,
    )

    test_tsv = f"../data/{species}/raw/transcripts.tsv"
    output_tsv = f"../data/{species}/trans_score/cnn{bp}bp.tsv"

    score_test_sites(
        test_tsv,
        os.path.join(donor_dir, "best.pt"),
        os.path.join(acceptor_dir, "best.pt"),
        output_tsv,
        device,
    )


if __name__ == "__main__":
    main()
