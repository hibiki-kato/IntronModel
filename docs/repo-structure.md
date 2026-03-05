# Repository Structure

## Primary Runtime Paths

- `src/run_model.py`: single public CLI for train/infer/transcript/eval pipeline
- `src/models/registry.py`: model registration and contract validation
- `src/models/`: unified model modules currently in registry
  - `cnn.py`
  - `cnn_pair.py`
  - `cnn_resdil.py`
  - `tcn.py`
  - `bert.py`
  - `dnabert.py` (used by `dnabert`, `dnabert2`, `dnabert6` keys)
  - `reservoir.py`
  - `reservoir_legacy.py`
- `src/util/`: shared preprocessing, naming, and transcript aggregation
  utilities
- `src/evaluate_scores.py`: evaluation text generation and plotting

## Wrapper Scripts (Root)

Unified pipeline wrappers (config-only):

- `run/run_cnn.sh`
- `run/run_cnn_pair.sh`
- `run/run_cnn_resdil.sh`
- `run/run_tcn.sh`
- `run/run_bert.sh`
- `run/run_dnabert.sh`
- `run/reservoir.sh`

Tuning wrappers:

- `run/tune_cnn.sh`, `run/tune_cnn_time.sh`
- `run/tune_cnn_pair_time.sh`
- `run/tune_cnn_resdil.sh`, `run/tune_cnn_resdil_time.sh`
- `run/tune_tcn.sh`, `run/tune_tcn_time.sh`
- `run/tune_bert.sh`
- `run/tune_reservoir.sh`

Utilities:

- `run/make_test_data.sh`
- `run/make_intron_training_data.sh`
- `run/make_trimmed_pair_data.sh`
- `run/make_labeled_intron_eval_data.sh`
- `run/eval_trans_score.sh`
- `run/plot_eval.sh`

## Source Tooling and Data Scripts

- `src/scripts/`: data provisioning and lockfile update helpers
- `src/tools/`: Python tooling (doc figure generation, tuning runner, scans)

## Legacy or Non-Integrated Modules

These files are still present but are not wired to `run_model.py` registry:

- `src/models/bert_drosophila.py`
- `src/models/cnn_v2.py`
- `src/models/reservoir_rc.py`

## Data and Artifacts

- `data/` is externalized from Git tracking.
- `model/` stores local checkpoints (`*.pt`) and pretrained snapshots.
- Runtime path overrides: `INTRONMODEL_DATA_ROOT`,
  `INTRONMODEL_MODEL_ROOT`.
- Only minimal deterministic fixtures should be kept under `tests/fixtures/`.
