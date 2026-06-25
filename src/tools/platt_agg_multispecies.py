"""Cross-species check: does noisy-OR transcript aggregation actually win
after Platt scaling, or was the Hsap result a fluke?

Exploratory script, not part of the production pipeline. Repeats the same
Platt-fit + transcript_score_agg sweep done for Hsap on Mmus, Athal, and
Dmel using each species' best published cnn_v3 site scores.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate_scores import count_reference_transcripts
from evaluate_scores import load_class_dict
from util.transcript_eval import TRANSCRIPT_SCORE_AGG_CHOICES
from util.transcript_eval import _combine_intron_score
from util.transcript_eval import aggregate_transcript_scores
from util.unique_intron import UNIQUE_MAP_TSV_NAME
from util.unique_intron import load_unique_map
from util.unique_intron import set_csv_field_limit_max

EPS = 1e-15


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    def nll(params: np.ndarray) -> float:
        a, b = params
        z = a * x + b
        return float(np.mean(np.log1p(np.exp(-z)) + (1 - y) * z))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def resolve_ref_gff(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("*.gff")) + sorted(raw_dir.glob("*.gff3"))
    preferred = [c for c in candidates if c.name.endswith(".fix.gff")]
    return preferred[0] if preferred else candidates[0]


def evaluate_rows_in_memory(transcript_scores, class_dict, reference_count):
    filtered = []
    running_total = 0
    running_good = 0
    for tid, score in transcript_scores.items():
        cc = class_dict.get(tid)
        if cc is None or cc == "c":
            continue
        filtered.append((tid, score, cc))
        running_total += 1
        if cc == "=":
            running_good += 1
    filtered.sort(key=lambda r: r[1])
    out = []
    for tid, score, cc in filtered:
        if cc == "=":
            running_good -= 1
        running_total -= 1
        if running_total <= 0:
            continue
        sn = 100.0 * running_good / reference_count if reference_count else 0.0
        pr = 100.0 * running_good / running_total if running_total else 0.0
        f1 = 2 * sn * pr / (sn + pr) if sn + pr > 0 else 0.0
        out.append((tid, score, cc, sn, pr, f1))
    return out


def run_for_species(species: str, model_version: str, seed: int = 0) -> None:
    species_dir = PROJECT_ROOT / "data" / species
    site_score_tsv = species_dir / "site_score" / f"{model_version}.tsv"
    class_file = species_dir / "raw" / "transcript_class.txt"
    ref_gff = resolve_ref_gff(species_dir / "raw")
    unique_map_path = species_dir / "processed" / UNIQUE_MAP_TSV_NAME
    labeled_path = species_dir / "processed" / "intron_eval_flank10.unique.tsv"

    set_csv_field_limit_max()
    class_dict = load_class_dict(str(class_file))
    reference_count = count_reference_transcripts(str(ref_gff))
    unique_map = load_unique_map(unique_map_path)

    labels: dict[tuple[str, int], tuple[int, int]] = {}
    with labeled_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for raw in reader:
            key = (raw["transcript_id"].strip(), int(raw["intron_index"]))
            labels[key] = (int(raw["donor_label"]), int(raw["acceptor_label"]))

    site_rows: dict[tuple[str, int], tuple[float, float]] = {}
    with site_score_tsv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for raw in reader:
            key = (raw["transcript_id"].strip(), int(raw["intron_index"]))
            site_rows[key] = (float(raw["donor_score"]), float(raw["acceptor_score"]))

    keys = [k for k in site_rows if k in labels]
    fit_keys, _ = train_test_split(keys, train_size=0.8, random_state=seed, shuffle=True)

    donor_p = np.array([site_rows[k][0] for k in fit_keys])
    donor_y = np.array([labels[k][0] for k in fit_keys], dtype=float)
    acceptor_p = np.array([site_rows[k][1] for k in fit_keys])
    acceptor_y = np.array([labels[k][1] for k in fit_keys], dtype=float)

    a_d, b_d = fit_platt(logit(donor_p), donor_y)
    a_a, b_a = fit_platt(logit(acceptor_p), acceptor_y)

    print(f"\n########## {species} ({model_version}) ##########")
    print(f"n_labeled_introns={len(keys)} donor_platt=(a={a_d:.6f}, b={b_d:.6f}) "
          f"acceptor_platt=(a={a_a:.6f}, b={b_a:.6f})")

    def build_mapped_long(calibrated: bool) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (tid, iidx), (d, a) in site_rows.items():
            if calibrated:
                d = float(sigmoid(a_d * logit(np.array(d)) + b_d))
                a = float(sigmoid(a_a * logit(np.array(a)) + b_a))
            members = unique_map.get((tid, iidx))
            if not members:
                continue
            for member in members:
                rows.append(
                    {"transcript_id": member.transcript_id, "intron_index": member.intron_index,
                     "site_type": "donor", "score": d}
                )
                rows.append(
                    {"transcript_id": member.transcript_id, "intron_index": member.intron_index,
                     "site_type": "acceptor", "score": a}
                )
        return rows

    for calibrated, label in ((False, "RAW"), (True, "CALIBRATED")):
        mapped = build_mapped_long(calibrated)
        print(f"  -- {label} --")
        results = {}
        for agg in TRANSCRIPT_SCORE_AGG_CHOICES:
            rows = aggregate_transcript_scores(site_score_rows=mapped, intron_score_op="+", transcript_score_agg=agg)
            scores = {r["transcript_id"]: r["trans_score"] for r in rows}
            ev = evaluate_rows_in_memory(scores, class_dict, reference_count)
            if not ev:
                continue
            best = max(ev, key=lambda r: r[5])
            results[agg] = best
            print(f"    {agg:14s} max_f1={best[5]:.4f}  sn={best[3]:.4f}  pr={best[4]:.4f}")
        if results:
            winner = max(results.items(), key=lambda kv: kv[1][5])
            print(f"    >>> best agg: {winner[0]} (max_f1={winner[1][5]:.4f})")


def main() -> None:
    targets = [
        ("Mmus", "cnn_v3.11"),
        ("Athal", "cnn_v3.13"),
        ("Dmel", "cnn_v3.10"),
    ]
    for species, version in targets:
        run_for_species(species, version)


if __name__ == "__main__":
    main()
