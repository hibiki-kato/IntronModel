"""Test: fit Platt scaling on the Hsap cnn_v3 donor/acceptor site scores.

This is a one-off experiment script, not part of the production pipeline.
It reuses the already-computed cnn_v3.12 site scores for Hsap (no model
re-inference needed), fits one 2-parameter logistic calibration
(``sigmoid(a * logit(p) + b)``) per site type (donor, acceptor) with
L-BFGS-B, re-aggregates calibrated transcript scores through the same
``aggregate_transcript_scores`` path the production pipeline uses, and
compares transcript-level SN/PR/max-F1 before vs. after calibration.

Caveat: the labeled site set used to fit Platt scaling overlaps with the
transcripts used for the downstream transcript-level evaluation (both come
from the same Hsap eval pool), so the "after" numbers are optimistic
relative to a fully independent holdout. Good enough for a quick test of
whether calibration is worth pursuing further.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate_scores import evaluate_score_file
from util.transcript_eval import aggregate_transcript_scores
from util.transcript_eval import write_transcript_scores
from util.unique_intron import UNIQUE_MAP_TSV_NAME
from util.unique_intron import load_unique_map
from util.unique_intron import set_csv_field_limit_max

EPS = 1e-15


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(x_logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ``sigmoid(a*x + b)`` to binary labels ``y`` via L-BFGS-B."""

    def neg_log_likelihood(params: np.ndarray) -> float:
        a, b = params
        z = a * x_logit + b
        # log-sum-exp form of BCE for numerical stability
        loss = np.mean(np.log1p(np.exp(-z)) + (1 - y) * z)
        return float(loss)

    result = minimize(
        neg_log_likelihood,
        x0=np.array([1.0, 0.0]),
        method="L-BFGS-B",
    )
    a, b = result.x
    return float(a), float(b)


def brier_and_logloss(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p_clipped = np.clip(p, EPS, 1.0 - EPS)
    brier = float(np.mean((p_clipped - y) ** 2))
    logloss = float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))
    return brier, logloss


def load_labeled_site_scores(
    species_dir: Path,
    site_score_tsv: Path,
) -> dict[tuple[str, int], dict[str, float]]:
    """Join site_score.tsv probabilities with donor_label/acceptor_label."""
    set_csv_field_limit_max()

    labels: dict[tuple[str, int], tuple[int, int]] = {}
    labeled_path = species_dir / "processed" / "intron_eval_flank10.unique.tsv"
    with labeled_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            key = (str(raw["transcript_id"]).strip(), int(raw["intron_index"]))
            labels[key] = (int(raw["donor_label"]), int(raw["acceptor_label"]))

    joined: dict[tuple[str, int], dict[str, float]] = {}
    with site_score_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            key = (str(raw["transcript_id"]).strip(), int(raw["intron_index"]))
            label_pair = labels.get(key)
            if label_pair is None:
                continue
            donor_label, acceptor_label = label_pair
            joined[key] = {
                "donor_score": float(raw["donor_score"]),
                "acceptor_score": float(raw["acceptor_score"]),
                "donor_label": donor_label,
                "acceptor_label": acceptor_label,
            }
    return joined


def expand_unique_site_rows(
    rows: list[dict[str, object]],
    unique_map: dict[tuple[str, int], list],
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


def site_rows_to_long_format(
    joined: dict[tuple[str, int], dict[str, float]],
    *,
    donor_score_key: str,
    acceptor_score_key: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (transcript_id, intron_index), values in joined.items():
        rows.append(
            {
                "transcript_id": transcript_id,
                "intron_index": intron_index,
                "site_type": "donor",
                "score": values[donor_score_key],
            }
        )
        rows.append(
            {
                "transcript_id": transcript_id,
                "intron_index": intron_index,
                "site_type": "acceptor",
                "score": values[acceptor_score_key],
            }
        )
    return rows


def resolve_ref_gff(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("*.gff")) + sorted(raw_dir.glob("*.gff3"))
    preferred = [c for c in candidates if c.name.endswith(".fix.gff")]
    if preferred:
        return preferred[0]
    if not candidates:
        raise FileNotFoundError(f"No reference GFF found under {raw_dir}")
    return candidates[0]


def summarize_eval(output_lines: list[str]) -> dict[str, float]:
    best_f1 = -1.0
    best_row: tuple[float, float, float] | None = None
    for line in output_lines:
        fields = line.split()
        sensitivity, precision, f1 = (float(fields[3]), float(fields[4]), float(fields[5]))
        if f1 > best_f1:
            best_f1 = f1
            best_row = (sensitivity, precision, f1)
    if best_row is None:
        return {"max_f1": float("nan"), "sn_at_max_f1": float("nan"), "pr_at_max_f1": float("nan")}
    sensitivity, precision, f1 = best_row
    return {"max_f1": f1, "sn_at_max_f1": sensitivity, "pr_at_max_f1": precision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="Hsap")
    parser.add_argument("--model-version", default="cnn_v3.12")
    parser.add_argument("--fit-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-trans-score",
        default=None,
        help="Where to write the calibrated trans_score TSV (default: "
        "data/<species>/trans_score/<model-version>_platt.tsv)",
    )
    args = parser.parse_args()

    species_dir = PROJECT_ROOT / "data" / args.species
    site_score_tsv = species_dir / "site_score" / f"{args.model_version}.tsv"
    orig_trans_score_tsv = species_dir / "trans_score" / f"{args.model_version}.tsv"
    unique_map_path = species_dir / "processed" / UNIQUE_MAP_TSV_NAME
    class_file = species_dir / "raw" / "transcript_class.txt"
    ref_gff = resolve_ref_gff(species_dir / "raw")

    output_trans_score = (
        Path(args.output_trans_score)
        if args.output_trans_score
        else species_dir / "trans_score" / f"{args.model_version}_platt.tsv"
    )

    print(f"[platt_scaling_test] species={args.species} model={args.model_version}")
    print(f"[platt_scaling_test] site_score_tsv={site_score_tsv}")

    joined = load_labeled_site_scores(species_dir, site_score_tsv)
    print(f"[platt_scaling_test] joined labeled site rows: {len(joined)}")

    keys = list(joined.keys())
    fit_keys, holdout_keys = train_test_split(
        keys,
        train_size=args.fit_frac,
        random_state=args.seed,
        shuffle=True,
    )
    fit_key_set = set(fit_keys)
    holdout_key_set = set(holdout_keys)

    platt_params: dict[str, tuple[float, float]] = {}
    for site_type, score_key, label_key in (
        ("donor", "donor_score", "donor_label"),
        ("acceptor", "acceptor_score", "acceptor_label"),
    ):
        fit_p = np.array([joined[k][score_key] for k in fit_keys])
        fit_y = np.array([joined[k][label_key] for k in fit_keys], dtype=float)
        holdout_p = np.array([joined[k][score_key] for k in holdout_keys])
        holdout_y = np.array([joined[k][label_key] for k in holdout_keys], dtype=float)

        a, b = fit_platt(logit(fit_p), fit_y)
        platt_params[site_type] = (a, b)

        calibrated_holdout_p = sigmoid(a * logit(holdout_p) + b)
        brier_before, logloss_before = brier_and_logloss(holdout_p, holdout_y)
        brier_after, logloss_after = brier_and_logloss(calibrated_holdout_p, holdout_y)

        print(
            f"[platt_scaling_test] {site_type}: a={a:.6f} b={b:.6f} "
            f"(fit n={len(fit_keys)}, holdout n={len(holdout_keys)})"
        )
        print(
            f"[platt_scaling_test] {site_type} holdout brier: "
            f"{brier_before:.6f} -> {brier_after:.6f}  "
            f"logloss: {logloss_before:.6f} -> {logloss_after:.6f}"
        )

    # Apply calibration to every site row (not just the holdout) so the
    # downstream transcript aggregation covers the full transcript set.
    a_d, b_d = platt_params["donor"]
    a_a, b_a = platt_params["acceptor"]
    calibrated_joined: dict[tuple[str, int], dict[str, float]] = {}
    for key, values in joined.items():
        calibrated_joined[key] = {
            "donor_score": float(sigmoid(a_d * logit(np.array(values["donor_score"])) + b_d)),
            "acceptor_score": float(
                sigmoid(a_a * logit(np.array(values["acceptor_score"])) + b_a)
            ),
        }

    unique_map = load_unique_map(unique_map_path)

    uncalibrated_long = site_rows_to_long_format(
        joined, donor_score_key="donor_score", acceptor_score_key="acceptor_score"
    )
    calibrated_long = site_rows_to_long_format(
        calibrated_joined, donor_score_key="donor_score", acceptor_score_key="acceptor_score"
    )

    mapped_uncalibrated = expand_unique_site_rows(uncalibrated_long, unique_map)
    mapped_calibrated = expand_unique_site_rows(calibrated_long, unique_map)

    rebuilt_transcript_rows = aggregate_transcript_scores(site_score_rows=mapped_uncalibrated)
    calibrated_transcript_rows = aggregate_transcript_scores(site_score_rows=mapped_calibrated)

    rebuilt_tsv = species_dir / "trans_score" / f"{args.model_version}_rebuilt_check.tsv"
    write_transcript_scores(str(rebuilt_tsv), rebuilt_transcript_rows)
    write_transcript_scores(str(output_trans_score), calibrated_transcript_rows)
    print(f"[platt_scaling_test] wrote calibrated trans_score: {output_trans_score}")

    original_eval = evaluate_score_file(
        class_file=str(class_file), score_file=str(orig_trans_score_tsv), ref_gff=str(ref_gff)
    )
    rebuilt_eval = evaluate_score_file(
        class_file=str(class_file), score_file=str(rebuilt_tsv), ref_gff=str(ref_gff)
    )
    calibrated_eval = evaluate_score_file(
        class_file=str(class_file), score_file=str(output_trans_score), ref_gff=str(ref_gff)
    )

    original_summary = summarize_eval(original_eval)
    rebuilt_summary = summarize_eval(rebuilt_eval)
    calibrated_summary = summarize_eval(calibrated_eval)

    print("\n[platt_scaling_test] === Transcript-level results ===")
    print(f"published cnn_v3.12.tsv      : {original_summary}")
    print(f"rebuilt (sanity, uncalibrated): {rebuilt_summary}")
    print(f"platt-calibrated              : {calibrated_summary}")
    print(
        "\n[platt_scaling_test] NOTE: Platt params were fit on "
        f"{args.fit_frac:.0%} of the same labeled Hsap eval pool used for "
        "this transcript-level evaluation, so this is an optimistic test, "
        "not an independent holdout result."
    )


if __name__ == "__main__":
    main()
