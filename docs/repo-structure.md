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

- `run/run_cnn_v2.sh`
- `run/run_cnn_v2_pair.sh`
- `run/run_isolated_mmus_rna60_pipeline.sh`

Archived wrappers:

- `archive/run/bert/run_bert.sh`
- `archive/run/bert/tune_bert.sh`
- `archive/run/bert/tune_bert_time.sh`
- `archive/run/bilstm_pair/run_bilstm_pair.sh`
- `archive/run/bilstm_pair/tune_bilstm_pair_time.sh`
- `archive/run/cnn/run_cnn_resdil.sh`
- `archive/run/cnn/tune_cnn_resdil.sh`
- `archive/run/cnn/tune_cnn_resdil_time.sh`
- `archive/run/dnabert/run_dnabert.sh`
- `archive/run/dnabert/run_dnabert_pair.sh`
- `archive/run/dnabert/tune_dnabert.sh`
- `archive/run/dnabert/tune_dnabert_time.sh`
- `archive/run/dnabert/tune_dnabert_pair.sh`
- `archive/run/dnabert/tune_dnabert_pair_time.sh`
- `archive/run/markov_xgboost/run_markov_xgboost.sh`
- `archive/run/markov_xgboost/tune_markov_xgboost.sh`
- `archive/run/reservoir/run_reservoir.sh`
- `archive/run/reservoir/tune_reservoir.sh`
- `archive/run/reservoir/tune_reservoir_time.sh`
- `archive/run/tcn/run_tcn.sh`
- `archive/run/tcn/tune_tcn.sh`
- `archive/run/tcn/tune_tcn_time.sh`

Active tuning wrappers:

- `run/tune_cnn_v2_time.sh`
- `run/tune_cnn_v2_pair_time.sh`

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
