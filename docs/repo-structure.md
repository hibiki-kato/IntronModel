# Repository Structure

## Read This First

- `README.md`: quick start and active model list.
- `docs/model-integration-contract.md`: model API contract.
- `docs/data-policy.md`: what belongs in Git vs local storage.
- `docs/legacy-model-status.md`: retired or partially integrated model status.

## Active Runtime

- `src/run_model.py`: public train/infer/transcript/eval CLI.
- `src/models/registry.py`: model names, aliases, and capability contract.
- `src/models/`: active model implementations:
  - `bert.py`
  - `bilstm_pair.py`
  - `cnn.py`
  - `cnn_v2.py`
  - `cnn_v3.py`
  - `cnn_v3_meta.py`
  - `cnn_pair_v3.py`
  - `cnn_resdil.py`
  - `dnabert.py`
  - `markov_xgboost.py`
  - `reservoir.py`
  - `spliceformer_sc.py`
  - `tcn.py`
- `src/models/cnn_common.py`: shared CNN helpers.
- `src/util/`: shared preprocessing, path, checkpoint, tuning, and scoring helpers.
- `src/tools/`: larger CLIs for tuning, scanning, plotting, and data builds.
- `src/scripts/`: reference-data setup scripts.

## Active Wrappers

Run wrappers are config-only. Edit the top `CONFIG (edit here)` block, then run
the script without CLI arguments.

Training/inference:

- `run/run_cnn_v2.sh`
- `run/run_cnn_pair_v2.sh`
- `run/run_cnn_v3.sh`
- `run/run_cnn_pair_v3.sh`
- `run/run_dnabert.sh`
- `run/run_dnabert_pair.sh`
- `run/run_spliceformer_sc.sh`

Grid/tuning:

- `run/grid_search_cnn_v2_flank.sh`
- `run/grid_search_dnabert2_flank.sh`
- `run/tune_cnn_v2_time.sh`
- `run/tune_cnn_pair_v2_time.sh`
- `run/tune_cnn_v3_time.sh`
- `run/tune_cnn_pair_v3_time.sh`
- `run/tune_dnabert_time.sh`
- `run/tune_dnabert_pair_time.sh`

Data and evaluation:

- `run/make_test_data.sh`
- `run/make_intron_training_data.sh`
- `run/make_trimmed_pair_data.sh`
- `run/make_labeled_intron_eval_data.sh`
- `run/make_mixed_pair_neg_data.sh`
- `run/make_random_intron_and_trans_scores.sh`
- `run/make_unique_intron_assets.sh`
- `run/eval_trans_score.sh`
- `run/eval_intron_pr_auc.sh`
- `run/scan_score_test_suite.sh`
- `run/scan_score_test_suite_integrated.sh`
- `run/scan_splice_candidates.sh`
- `run/share_eval_to_chirag.sh`

Shared shell helpers live in `run/lib/`:

- `run/lib/common.sh`: path, conda, model, GPU, tuning, and checkpoint helpers.
- `run/lib/wrapper_runtime.sh`: small wrapper-runtime helpers for args,
  `src/run_model.py` execution, and species/GPU dispatch.
- `run/lib/tuned_config.sh`: tuned-config extraction helpers.

Add shared wrapper behavior there before copying helper functions into another
wrapper.

## Data and Artifacts

- `data/`: local datasets, grid outputs, and generated analysis data.
- `model/`: checkpoints and pretrained model snapshots.
- `score_test_suite/`: small retained evaluation suite and expected outputs.
- `analysis/outputs/`, `docs/_build/`, `temp/`, `__pycache__/`, `.pytest_cache/`:
  generated local artifacts. Keep them out of source changes.
- `archive/`: retired scripts, historical outputs, and migration snapshots.
- `sub_repo/`: third-party or external source checkouts.

## Cleanup Rule

Prefer direct paths:

- wrapper -> shared shell helper/Python CLI -> `src/run_model.py`
- Python tool -> shared `src/util/` helper -> model/data code

Avoid adding new wrapper-only logic when existing `run/lib/` or `src/util/`
helpers can express it without runtime overhead.
