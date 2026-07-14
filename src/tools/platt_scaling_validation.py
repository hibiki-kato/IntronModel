"""Apply validation-only Platt scaling to independent site classifiers.

The calibration set is reconstructed from the exact stratified validation
split used by a site-model training run.  Unlike ``platt_scaling_test.py``,
it never reads labels from the transcript evaluation pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dev.IntronModel.src.evaluate_scores import evaluate_score_file
from dev.IntronModel.src.models import cnn, cnn_v4
from dev.IntronModel.src.util.data_proc import resolve_train_paths
from dev.IntronModel.src.util.transcript_eval import (
    SCORE_SPACE_FIELD,
    SCORE_SPACE_PROBABILITY,
    aggregate_transcript_scores,
    coerce_score_to_probability,
    read_site_scores,
    write_transcript_scores,
)
from dev.IntronModel.src.util.unique_intron import UNIQUE_MAP_TSV_NAME, load_unique_map

EPS = 1e-12


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    out = np.empty_like(values, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def fit_platt(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit ``sigmoid(a * logit(p) + b)`` with binary log loss."""
    x_values = logit(probabilities)

    def objective(params: np.ndarray) -> float:
        logits = params[0] * x_values + params[1]
        return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))

    result = minimize(objective, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Platt fitting failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def brier_and_logloss(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    brier = float(np.mean((clipped - labels) ** 2))
    logloss = float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))
    return brier, logloss


def validation_examples(
    *,
    pos_path: str,
    neg_path: str,
    task: str,
    donor_len: int,
    acceptor_len: int,
    val_frac: float,
    seed: int,
) -> list[tuple[str, int]]:
    examples = cnn._load_task_examples_with_transform(
        pos_path=pos_path,
        neg_path=neg_path,
        task=task,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        sequence_transform="none",
    )
    _train_examples, val_examples = cnn.stratified_split(
        examples, val_frac=val_frac, seed=seed
    )
    return val_examples


def score_validation_examples(
    *,
    checkpoint_path: str,
    sequences: Iterable[str],
    device: str,
    batch_size: int,
) -> np.ndarray:
    model, checkpoint = cnn_v4.load_task_model(checkpoint_path, device=device)
    window_len = int(checkpoint["window_len"])
    log10_scores = cnn_v4.score_sequences(
        model=model,
        sequences=list(sequences),
        window_len=window_len,
        device=device,
        batch_size=batch_size,
        use_amp=device == "cuda",
    )
    return np.power(10.0, log10_scores)


def calibrate_site_rows(
    rows: list[dict[str, object]],
    *,
    donor_params: tuple[float, float],
    acceptor_params: tuple[float, float],
) -> list[dict[str, object]]:
    calibrated: list[dict[str, object]] = []
    for row in rows:
        site_type = str(row["site_type"])
        params = donor_params if site_type == "donor" else acceptor_params
        probability = coerce_score_to_probability(
            float(row["score"]), score_space=str(row.get(SCORE_SPACE_FIELD, ""))
        )
        score = float(sigmoid(params[0] * logit(np.array([probability])) + params[1])[0])
        calibrated.append({
            "transcript_id": row["transcript_id"],
            "intron_index": int(row["intron_index"]),
            "site_type": site_type,
            "score": score,
            SCORE_SPACE_FIELD: SCORE_SPACE_PROBABILITY,
        })
    return calibrated


def expand_unique_site_rows(
    rows: list[dict[str, object]], unique_map: dict[tuple[str, int], list]
) -> list[dict[str, object]]:
    expanded: list[dict[str, object]] = []
    for row in rows:
        members = unique_map.get((str(row["transcript_id"]), int(row["intron_index"])))
        if not members:
            continue
        for member in members:
            copied = dict(row)
            copied["transcript_id"] = member.transcript_id
            copied["intron_index"] = member.intron_index
            expanded.append(copied)
    return expanded


def resolve_ref_gff(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("*.gff")) + sorted(raw_dir.glob("*.gff3"))
    preferred = [path for path in candidates if path.name.endswith(".fix.gff")]
    if preferred:
        return preferred[0]
    if not candidates:
        raise FileNotFoundError(f"No reference GFF under {raw_dir}")
    return candidates[0]


def summarize_eval(lines: list[str]) -> dict[str, float]:
    best = max((tuple(map(float, line.split()[3:6])) for line in lines), key=lambda row: row[2])
    return {"sensitivity": best[0], "precision": best[1], "max_f1": best[2]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True)
    parser.add_argument("--model-version", default="cnn_v4.0")
    parser.add_argument("--donor-checkpoint", required=True)
    parser.add_argument("--acceptor-checkpoint", required=True)
    parser.add_argument("--donor-len", type=int, default=200)
    parser.add_argument("--acceptor-len", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output-trans-score", required=True)
    parser.add_argument("--output-eval", required=True)
    parser.add_argument("--output-metadata", required=True)
    args = parser.parse_args()

    pos_path, neg_path, _ = resolve_train_paths(
        species=args.species,
        train_pos_path=None,
        train_neg_path=None,
        donor_len=args.donor_len,
        acceptor_len=args.acceptor_len,
    )
    species_dir = PROJECT_ROOT / "data" / args.species
    parameters: dict[str, tuple[float, float]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for task, checkpoint_path in (("donor", args.donor_checkpoint), ("acceptor", args.acceptor_checkpoint)):
        val_examples = validation_examples(
            pos_path=pos_path, neg_path=neg_path, task=task,
            donor_len=args.donor_len, acceptor_len=args.acceptor_len,
            val_frac=args.val_frac, seed=args.seed,
        )
        sequences = [sequence for sequence, _label in val_examples]
        labels = np.array([label for _sequence, label in val_examples], dtype=float)
        probabilities = score_validation_examples(
            checkpoint_path=checkpoint_path, sequences=sequences,
            device=args.device, batch_size=args.batch_size,
        )
        params = fit_platt(probabilities, labels)
        calibrated = sigmoid(params[0] * logit(probabilities) + params[1])
        before = brier_and_logloss(probabilities, labels)
        after = brier_and_logloss(calibrated, labels)
        parameters[task] = params
        metrics[task] = {
            "n_validation": float(len(labels)),
            "brier_before": before[0], "brier_after": after[0],
            "logloss_before": before[1], "logloss_after": after[1],
        }
        print(f"[platt_validation] {task}: n={len(labels)} a={params[0]:.8g} b={params[1]:.8g} "
              f"brier {before[0]:.6g}->{after[0]:.6g} logloss {before[1]:.6g}->{after[1]:.6g}")

    raw_site_rows = read_site_scores(str(species_dir / "site_score" / f"{args.model_version}.tsv"))
    calibrated_rows = calibrate_site_rows(
        raw_site_rows, donor_params=parameters["donor"], acceptor_params=parameters["acceptor"]
    )
    mapped_rows = expand_unique_site_rows(
        calibrated_rows, load_unique_map(species_dir / "processed" / UNIQUE_MAP_TSV_NAME)
    )
    transcript_rows = aggregate_transcript_scores(mapped_rows)
    write_transcript_scores(args.output_trans_score, transcript_rows)

    ref_gff = resolve_ref_gff(species_dir / "raw")
    raw_eval = evaluate_score_file(
        class_file=str(species_dir / "raw" / "transcript_class.txt"),
        score_file=str(species_dir / "trans_score" / f"{args.model_version}.tsv"),
        ref_gff=str(ref_gff),
    )
    calibrated_eval = evaluate_score_file(
        class_file=str(species_dir / "raw" / "transcript_class.txt"),
        score_file=args.output_trans_score, ref_gff=str(ref_gff),
    )
    output_eval = Path(args.output_eval)
    output_eval.parent.mkdir(parents=True, exist_ok=True)
    output_eval.write_text("\n".join(calibrated_eval) + "\n", encoding="utf-8")
    payload = {
        "method": "validation_only_platt",
        "species": args.species,
        "model_version": args.model_version,
        "training_split": {"seed": args.seed, "val_frac": args.val_frac, "donor_len": args.donor_len, "acceptor_len": args.acceptor_len},
        "checkpoints": {"donor": args.donor_checkpoint, "acceptor": args.acceptor_checkpoint},
        "parameters": {task: {"a": values[0], "b": values[1]} for task, values in parameters.items()},
        "site_validation": metrics,
        "transcript": {"raw": summarize_eval(raw_eval), "platt": summarize_eval(calibrated_eval)},
        "output_trans_score": args.output_trans_score,
        "output_eval": args.output_eval,
    }
    output_metadata = Path(args.output_metadata)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[platt_validation] transcript raw={payload['transcript']['raw']} platt={payload['transcript']['platt']}")
    print(f"[platt_validation] wrote {args.output_trans_score} and {output_metadata}")


if __name__ == "__main__":
    main()
