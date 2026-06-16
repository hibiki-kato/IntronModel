"""Evaluate whether PyTorch DataLoader is a throughput bottleneck.

This script compares two execution modes using NVIDIA DALI LoaderEvaluator:
- ``log``: normal DataLoader behavior
- ``replay``: idealized data loading replay

If replay throughput is much higher than log throughput, data loading is a
bottleneck in the current training setup.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.tcn import DNADataset  # noqa: E402
from util.data_proc import read_examples_single_task, resolve_train_paths  # noqa: E402
from util.model_runtime import resolve_auto_num_workers  # noqa: E402

try:
    from nvidia.dali.plugin.pytorch.loader_evaluator import LoaderEvaluator
except ImportError:  # pragma: no cover
    LoaderEvaluator = None


@dataclass(frozen=True)
class EvalConfig:
    """Validated configuration for loader bottleneck evaluation."""

    species: str
    target: str
    train_pos_path: Optional[str]
    train_neg_path: Optional[str]
    donor_len: Optional[int]
    acceptor_len: Optional[int]
    batch_size: int
    num_workers: str
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    device: str
    seed: int
    max_examples: Optional[int]
    warmup_steps: int
    measure_steps: int
    compute_mode: str


class TinyStepModel(nn.Module):
    """Small model for synthetic train-step timing."""

    def __init__(self, window_len: int) -> None:
        super().__init__()
        input_dim = 4 * window_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary logits for ``x`` with shape ``[B, 4, L]``."""
        return self.net(x).squeeze(-1)


class CachedBatchLoader:
    """In-memory batched iterable used as an idealized loader fallback.

    Parameters
    ----------
    batches : list[tuple[torch.Tensor, torch.Tensor]]
        Pre-batched CPU tensors ``(x, y)`` where ``x`` shape is ``[B, 4, L]``
        and ``y`` shape is ``[B]``.
    """

    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        if not batches:
            raise ValueError("CachedBatchLoader requires at least one batch.")
        self._batches: list[tuple[torch.Tensor, torch.Tensor]] = batches

    def __iter__(self):
        for batch in self._batches:
            yield batch


def parse_args(argv: list[str]) -> EvalConfig:
    """Parse CLI arguments and return validated evaluation config."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare log/replay DataLoader throughput using DALI "
            "LoaderEvaluator."
        )
    )
    parser.add_argument("--species", required=True, help="Species directory name.")
    parser.add_argument(
        "--target",
        required=True,
        choices=["donor", "acceptor"],
        help="Training target.",
    )
    parser.add_argument("--train_pos_path", default=None)
    parser.add_argument("--train_neg_path", default=None)
    parser.add_argument("--donor_len", type=int, default=None)
    parser.add_argument("--acceptor_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--num_workers",
        default="auto",
        help="DataLoader workers: integer or auto.",
    )
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument(
        "--persistent_workers",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--pin_memory",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=200_000,
        help="Cap number of examples for faster diagnostic; <=0 means all.",
    )
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--measure_steps", type=int, default=200)
    parser.add_argument(
        "--compute_mode",
        choices=["transfer", "train_step"],
        default="transfer",
        help="transfer=copy only, train_step=copy+forward/backward/step.",
    )
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be > 0.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be >= 0.")
    if args.measure_steps <= 0:
        raise ValueError("--measure_steps must be > 0.")
    if args.donor_len is not None and args.donor_len <= 0:
        raise ValueError("--donor_len must be > 0 when provided.")
    if args.acceptor_len is not None and args.acceptor_len <= 0:
        raise ValueError("--acceptor_len must be > 0 when provided.")

    max_examples: Optional[int]
    if args.max_examples is None or args.max_examples <= 0:
        max_examples = None
    else:
        max_examples = int(args.max_examples)

    return EvalConfig(
        species=str(args.species),
        target=str(args.target),
        train_pos_path=(
            None if args.train_pos_path in (None, "", "none") else args.train_pos_path
        ),
        train_neg_path=(
            None if args.train_neg_path in (None, "", "none") else args.train_neg_path
        ),
        donor_len=args.donor_len,
        acceptor_len=args.acceptor_len,
        batch_size=int(args.batch_size),
        num_workers=str(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
        pin_memory=bool(args.pin_memory),
        device=str(args.device),
        seed=int(args.seed),
        max_examples=max_examples,
        warmup_steps=int(args.warmup_steps),
        measure_steps=int(args.measure_steps),
        compute_mode=str(args.compute_mode),
    )


def pick_device(preference: str) -> str:
    """Resolve runtime device from user preference."""
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return "cuda"
    if preference == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_num_workers(raw: str, device: str) -> int:
    """Resolve DataLoader worker count from integer text or ``auto``."""
    text = raw.strip().lower()
    if text == "auto":
        if device != "cuda":
            return 0
        return resolve_auto_num_workers()
    parsed = int(text)
    if parsed < 0:
        raise ValueError("--num_workers must be >= 0.")
    return parsed


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducible benchmark setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    dataset: DNADataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    """Build DataLoader with stable worker seeding and optional prefetch."""
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": loader_generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**kwargs)


def maybe_sync(device: str) -> None:
    """Synchronize CUDA stream for accurate wall-clock timing."""
    if device == "cuda":
        torch.cuda.synchronize()


def timed_loop(
    loader: object,
    device: str,
    warmup_steps: int,
    measure_steps: int,
    compute_mode: str,
    window_len: int,
) -> tuple[float, int]:
    """Run warmup+measurement loop and return elapsed seconds and samples."""
    model: Optional[TinyStepModel]
    optimizer: Optional[torch.optim.Optimizer]
    if compute_mode == "train_step":
        model = TinyStepModel(window_len=window_len).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    else:
        model = None
        optimizer = None

    seen_samples = 0
    measured_batches = 0
    measured_samples = 0
    maybe_sync(device)
    started = 0.0
    ended = 0.0
    while measured_batches < measure_steps:
        for batch in loader:
            x: torch.Tensor
            y: torch.Tensor
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, y = batch
            elif isinstance(batch, dict):
                if "data" in batch and "label" in batch:
                    x = batch["data"]
                    y = batch["label"]
                else:
                    raise TypeError(
                        "Dict batch must contain 'data' and 'label' keys."
                    )
            else:
                raise TypeError(
                    "Expected batch as (x, y), [x, y], or {'data','label'}."
                )
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise TypeError("Expected tensor batch for both inputs and labels.")

            x = x.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))

            if model is not None and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                optimizer.step()

            if seen_samples == warmup_steps:
                maybe_sync(device)
                started = time.perf_counter()
            if seen_samples >= warmup_steps:
                measured_batches += 1
                measured_samples += int(x.shape[0])
            seen_samples += 1
            if measured_batches >= measure_steps:
                break
    maybe_sync(device)
    ended = time.perf_counter()
    elapsed = max(0.0, ended - started)
    return elapsed, measured_samples


def main(argv: list[str]) -> int:
    """Run loader bottleneck evaluation and print a compact report."""
    cfg = parse_args(argv)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    num_workers = resolve_num_workers(cfg.num_workers, device=device)
    use_pin_memory = cfg.pin_memory and device == "cuda"

    pos_path, neg_path, _ = resolve_train_paths(
        species=cfg.species,
        train_pos_path=cfg.train_pos_path,
        train_neg_path=cfg.train_neg_path,
        donor_len=cfg.donor_len,
        acceptor_len=cfg.acceptor_len,
    )
    examples = read_examples_single_task(
        pos_path=pos_path,
        neg_path=neg_path,
        task=cfg.target,
        donor_len=cfg.donor_len,
        acceptor_len=cfg.acceptor_len,
    )
    if cfg.max_examples is not None:
        examples = examples[: cfg.max_examples]
    if not examples:
        raise ValueError("No examples loaded. Check species/target/path settings.")

    window_len = len(examples[0][0])
    dataset = DNADataset(examples=examples, window_len=window_len, preencode=False)

    base_loader = build_loader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        pin_memory=use_pin_memory,
        seed=cfg.seed,
    )
    replay_loader = build_loader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        pin_memory=use_pin_memory,
        seed=cfg.seed,
    )

    if LoaderEvaluator is not None:
        wrapped_log: object = LoaderEvaluator(base_loader, mode="log")
        replay_mode_label = "dali_replay"
    else:
        wrapped_log = base_loader
        replay_mode_label = "ideal_cached"
    log_elapsed, log_samples = timed_loop(
        loader=wrapped_log,
        device=device,
        warmup_steps=cfg.warmup_steps,
        measure_steps=cfg.measure_steps,
        compute_mode=cfg.compute_mode,
        window_len=window_len,
    )
    if LoaderEvaluator is not None:
        wrapped_replay: object = LoaderEvaluator(replay_loader, mode="replay")
    else:
        cached_ds = DNADataset(examples=examples, window_len=window_len, preencode=True)
        if cached_ds._cached_x is None or cached_ds._cached_y is None:
            raise RuntimeError("Cached dataset tensors are unexpectedly None.")
        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, len(cached_ds), cfg.batch_size):
            end = min(start + cfg.batch_size, len(cached_ds))
            batches.append((cached_ds._cached_x[start:end], cached_ds._cached_y[start:end]))
        wrapped_replay = CachedBatchLoader(batches=batches)
    replay_elapsed, replay_samples = timed_loop(
        loader=wrapped_replay,
        device=device,
        warmup_steps=cfg.warmup_steps,
        measure_steps=cfg.measure_steps,
        compute_mode=cfg.compute_mode,
        window_len=window_len,
    )

    if log_elapsed <= 0.0 or replay_elapsed <= 0.0:
        raise RuntimeError("Measured elapsed time is zero. Increase measure_steps.")

    log_sps = log_samples / log_elapsed
    replay_sps = replay_samples / replay_elapsed
    speedup = replay_sps / log_sps if log_sps > 0.0 else float("inf")
    bottleneck_pct = max(0.0, 1.0 - (log_sps / replay_sps)) * 100.0

    print(
        "LoaderEvaluator report\n"
        f"  species={cfg.species} target={cfg.target} device={device}\n"
        f"  batch_size={cfg.batch_size} workers={num_workers} "
        f"prefetch={cfg.prefetch_factor}\n"
        f"  mode={cfg.compute_mode} examples={len(examples)} "
        f"warmup_steps={cfg.warmup_steps} measure_steps={cfg.measure_steps}\n"
        f"  log:    {log_sps:.2f} samples/s ({log_elapsed:.3f}s)\n"
        f"  replay[{replay_mode_label}]: {replay_sps:.2f} samples/s "
        f"({replay_elapsed:.3f}s)\n"
        f"  replay/log speedup={speedup:.3f}x "
        f"(estimated loader bottleneck={bottleneck_pct:.1f}%)"
    )
    if LoaderEvaluator is None:
        print(
            "Note: nvidia.dali.plugin.pytorch.loader_evaluator was unavailable; "
            "used idealized in-memory cached replay fallback."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
