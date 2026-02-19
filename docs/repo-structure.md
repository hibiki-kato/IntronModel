# Repository Structure

## Primary Runtime Paths

- `src/run_model.py`: single public CLI for train/infer/transcript/eval pipeline
- `src/models/registry.py`: model registration and contract validation
- `src/models/cnn.py`: only model connected to unified pipeline
- `src/util/`: shared utilities for data processing and transcript aggregation
- `src/evaluate_scores.py`: evaluation text generation and plotting
- `src/gffcompare_counts.py`: utility for gffcompare-derived count files

## Wrapper Scripts

- `run/cnn.sh`: thin wrapper for `src/run_model.py`
- `run/make_test_data.sh`: builds `transcripts.tsv` from FASTA/GTF
- `run/gffcompare_counts.sh`: builds `gffcompare_counts.txt`
- `run/eval_trans_score.sh`: evaluates existing transcript score TSV files
- `run/plot_eval.sh`: plots precision/sensitivity from eval outputs

## Non-Connected Legacy Modules

Legacy modules are intentionally kept under `src/models` for reference but are
not part of the public runtime path:

- `src/models/bert.py`
- `src/models/bert_drosophila.py`
- `src/models/cnn_v2.py`
- `src/models/reservoir.py`

## Data and Artifacts

- `data/` is externalized from Git tracking.
- `model/` stores local checkpoints (`*.pt`) and is not treated as source code.
- Only minimal deterministic fixtures should be kept under `tests/fixtures/`.
