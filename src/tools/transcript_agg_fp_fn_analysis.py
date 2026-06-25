"""Investigate transcript-score aggregation: is `min` the bottleneck?

This is an exploratory analysis script (not part of the production
pipeline). It reuses the existing Hsap cnn_v3 site scores (no
re-inference) to:

1. Re-aggregate transcript scores under every supported
   ``transcript_score_agg`` choice and compare the resulting max-F1, to see
   whether swapping away from ``min`` actually helps.
2. At the current default (``min`` aggregation), characterize false
   positive / false negative transcripts at the max-F1 operating
   threshold: intron count, and how far the limiting (min-score) intron's
   score is from the transcript's other introns.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate_scores import count_reference_transcripts
from evaluate_scores import load_class_dict
from util.score_format import format_score_text
from util.transcript_eval import TRANSCRIPT_SCORE_AGG_CHOICES
from util.transcript_eval import _combine_intron_score
from util.transcript_eval import aggregate_transcript_scores
from util.unique_intron import UNIQUE_MAP_TSV_NAME
from util.unique_intron import load_unique_map
from util.unique_intron import set_csv_field_limit_max

SPECIES = "Hsap"
MODEL_VERSION = "cnn_v3.12"


def resolve_ref_gff(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("*.gff")) + sorted(raw_dir.glob("*.gff3"))
    preferred = [c for c in candidates if c.name.endswith(".fix.gff")]
    return preferred[0] if preferred else candidates[0]


def load_site_rows(site_score_tsv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with site_score_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            transcript_id = str(raw["transcript_id"]).strip()
            intron_index = int(raw["intron_index"])
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "intron_index": intron_index,
                    "site_type": "donor",
                    "score": float(raw["donor_score"]),
                }
            )
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "intron_index": intron_index,
                    "site_type": "acceptor",
                    "score": float(raw["acceptor_score"]),
                }
            )
    return rows


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


def evaluate_rows_in_memory(
    transcript_scores: dict[str, float],
    class_dict: dict[str, str],
    reference_count: int,
) -> list[tuple[str, float, str]]:
    """Replicate evaluate_scores.evaluate_score_file without touching disk."""
    filtered_data: list[tuple[str, float, str]] = []
    running_total = 0
    running_good = 0

    for transcript_id, score in transcript_scores.items():
        class_code = class_dict.get(transcript_id)
        if class_code is None or class_code == "c":
            continue
        filtered_data.append((transcript_id, score, class_code))
        running_total += 1
        if class_code == "=":
            running_good += 1

    filtered_data.sort(key=lambda row: row[1])

    output_rows: list[tuple[str, float, str, float, float, float]] = []
    for transcript_id, score, class_code in filtered_data:
        if class_code == "=":
            running_good -= 1
        running_total -= 1
        if running_total <= 0:
            continue
        sensitivity = 100.0 * running_good / reference_count if reference_count else 0.0
        precision = 100.0 * running_good / running_total if running_total else 0.0
        f1 = 0.0
        if sensitivity + precision > 0.0:
            f1 = 2.0 * sensitivity * precision / (sensitivity + precision)
        output_rows.append((transcript_id, score, class_code, sensitivity, precision, f1))
    return output_rows


def summarize(output_rows: list[tuple]) -> dict[str, float]:
    if not output_rows:
        return {"max_f1": float("nan"), "sn": float("nan"), "pr": float("nan"), "threshold": float("nan")}
    best = max(output_rows, key=lambda row: row[5])
    return {"max_f1": best[5], "sn": best[3], "pr": best[4], "threshold": best[1]}


def main() -> None:
    species_dir = PROJECT_ROOT / "data" / SPECIES
    site_score_tsv = species_dir / "site_score" / f"{MODEL_VERSION}.tsv"
    class_file = species_dir / "raw" / "transcript_class.txt"
    ref_gff = resolve_ref_gff(species_dir / "raw")
    unique_map_path = species_dir / "processed" / UNIQUE_MAP_TSV_NAME

    set_csv_field_limit_max()
    class_dict = load_class_dict(str(class_file))
    reference_count = count_reference_transcripts(str(ref_gff))

    unique_map = load_unique_map(unique_map_path)
    raw_site_rows = load_site_rows(site_score_tsv)
    mapped_site_rows = expand_unique_site_rows(raw_site_rows, unique_map)

    print(f"[agg] mapped site rows: {len(mapped_site_rows)}")

    # ---- Part 1: sweep transcript_score_agg, see if max F1 improves ----
    print("\n=== Part 1: max F1 by transcript_score_agg (intron_score_op='+') ===")
    agg_results: dict[str, dict[str, float]] = {}
    for agg in TRANSCRIPT_SCORE_AGG_CHOICES:
        transcript_rows = aggregate_transcript_scores(
            site_score_rows=mapped_site_rows,
            intron_score_op="+",
            transcript_score_agg=agg,
        )
        scores = {row["transcript_id"]: row["trans_score"] for row in transcript_rows}
        evaluated = evaluate_rows_in_memory(scores, class_dict, reference_count)
        summary = summarize(evaluated)
        agg_results[agg] = summary
        print(f"  {agg:14s} max_f1={summary['max_f1']:.4f}  sn={summary['sn']:.4f}  pr={summary['pr']:.4f}")

    # ---- Part 2: characterize FP/FN under the current default (min) ----
    print("\n=== Part 2: FP/FN characterization under default agg='min' ===")
    default_rows = aggregate_transcript_scores(
        site_score_rows=mapped_site_rows, intron_score_op="+", transcript_score_agg="min"
    )
    default_scores = {row["transcript_id"]: row["trans_score"] for row in default_rows}
    default_min_idx = {row["transcript_id"]: row["min_intron_index"] for row in default_rows}
    evaluated_default = evaluate_rows_in_memory(default_scores, class_dict, reference_count)
    default_summary = summarize(evaluated_default)
    threshold = default_summary["threshold"]
    print(f"  operating threshold (score at max F1): {threshold:.6g}")
    print(f"  max_f1={default_summary['max_f1']:.4f} sn={default_summary['sn']:.4f} pr={default_summary['pr']:.4f}")

    # group per-intron combined scores by transcript
    per_transcript_introns: dict[str, dict[int, float]] = defaultdict(dict)
    per_transcript_site: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for row in mapped_site_rows:
        pass
    # rebuild donor/acceptor per intron directly (mapped_site_rows is long format donor/acceptor separately)
    per_site_type: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in mapped_site_rows:
        key = (str(row["transcript_id"]), int(row["intron_index"]))
        per_site_type[key][str(row["site_type"])] = float(row["score"])

    for (tid, iidx), per_site in per_site_type.items():
        donor = per_site.get("donor")
        acceptor = per_site.get("acceptor")
        if donor is None or acceptor is None:
            continue
        intron_score = _combine_intron_score(donor, acceptor, op="+")
        per_transcript_introns[tid][iidx] = intron_score

    # classify each transcript
    groups: dict[str, list[str]] = {"TP": [], "FN": [], "FP": [], "TN": []}
    feature_rows: list[dict[str, object]] = []
    for transcript_id, score in default_scores.items():
        class_code = class_dict.get(transcript_id)
        if class_code is None or class_code == "c":
            continue
        is_true = class_code == "="
        is_pred_positive = score >= threshold
        if is_true and is_pred_positive:
            bucket = "TP"
        elif is_true and not is_pred_positive:
            bucket = "FN"
        elif not is_true and is_pred_positive:
            bucket = "FP"
        else:
            bucket = "TN"
        groups[bucket].append(transcript_id)

        intron_scores = per_transcript_introns.get(transcript_id, {})
        n_introns = len(intron_scores)
        scores_list = list(intron_scores.values())
        min_score = min(scores_list) if scores_list else float("nan")
        other_scores = sorted(scores_list)[1:] if len(scores_list) > 1 else []
        mean_other_log10 = (
            float(np.mean(np.log10(np.clip(other_scores, 1e-300, 1.0)))) if other_scores else float("nan")
        )
        min_log10 = float(np.log10(max(min_score, 1e-300))) if scores_list else float("nan")
        n_weak = sum(1 for s in scores_list if s < 0.5)
        feature_rows.append(
            {
                "transcript_id": transcript_id,
                "bucket": bucket,
                "class_code": class_code,
                "n_introns": n_introns,
                "trans_score": score,
                "min_intron_log10": min_log10,
                "mean_other_log10": mean_other_log10,
                "gap_log10": mean_other_log10 - min_log10 if other_scores else float("nan"),
                "n_weak_introns": n_weak,
            }
        )

    for bucket, ids in groups.items():
        print(f"  {bucket}: n={len(ids)}")

    print("\n=== Feature summary by bucket ===")
    import statistics

    for bucket in ("TP", "FN", "FP", "TN"):
        rows = [r for r in feature_rows if r["bucket"] == bucket]
        if not rows:
            continue
        n_introns_vals = [r["n_introns"] for r in rows]
        gap_vals = [r["gap_log10"] for r in rows if r["n_introns"] > 1 and not np.isnan(r["gap_log10"])]
        n_weak_vals = [r["n_weak_introns"] for r in rows]
        single_intron_frac = sum(1 for v in n_introns_vals if v == 1) / len(n_introns_vals)
        single_weak_frac = (
            sum(1 for r in rows if r["n_introns"] > 1 and r["n_weak_introns"] == 1) / max(1, sum(1 for v in n_introns_vals if v > 1))
        )
        print(
            f"  {bucket}: n={len(rows)}  mean_n_introns={statistics.mean(n_introns_vals):.2f}  "
            f"median_n_introns={statistics.median(n_introns_vals)}  "
            f"frac_single_intron={single_intron_frac:.3f}  "
            f"frac(multi-intron & exactly-1-weak<0.5)={single_weak_frac:.3f}"
        )
        if gap_vals:
            print(
                f"        (multi-intron only) mean_gap_log10(other - min)={statistics.mean(gap_vals):.2f}  "
                f"median_gap_log10={statistics.median(gap_vals):.2f}  n={len(gap_vals)}"
            )

    # dump a few concrete FN examples with multiple introns and a large gap (outlier-limited)
    fn_multi = [
        r
        for r in feature_rows
        if r["bucket"] == "FN" and r["n_introns"] > 1 and not np.isnan(r["gap_log10"])
    ]
    fn_multi.sort(key=lambda r: -r["gap_log10"])
    print("\n=== Top 10 FN transcripts where ONE outlier intron drags the score down ===")
    for r in fn_multi[:10]:
        print(
            f"  {r['transcript_id']}: n_introns={r['n_introns']} class={r['class_code']} "
            f"min_intron_log10={r['min_intron_log10']:.2f} mean_other_log10={r['mean_other_log10']:.2f} "
            f"gap={r['gap_log10']:.2f} n_weak={r['n_weak_introns']}"
        )

    fp_rows = [r for r in feature_rows if r["bucket"] == "FP"]
    fp_rows.sort(key=lambda r: -r["trans_score"])
    print("\n=== Top 10 FP transcripts by trans_score ===")
    for r in fp_rows[:10]:
        print(
            f"  {r['transcript_id']}: n_introns={r['n_introns']} class={r['class_code']} "
            f"trans_score={r['trans_score']:.3g} min_intron_log10={r['min_intron_log10']:.2f}"
        )


if __name__ == "__main__":
    main()
