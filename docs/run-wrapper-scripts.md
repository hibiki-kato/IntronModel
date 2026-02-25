# Wrapper Script Configuration Guide

This page documents the config-only shell wrappers under `run/` and how they
map to `src/run_model.py`.

Runtime logic for the main training/inference wrappers is centralized in
`src/tools/run_wrapper_pipeline.py`. Each `run/*.sh` keeps only the editable
`CONFIG` block and delegates validation/argument assembly/execution to this
Python backend.

## 0. Editing Workflow (Top-First)

Wrapper scripts are intentionally organized so you can edit settings in one
place before reading implementation details.

- Edit only the top `CONFIG (edit here)` block.
- Run the wrapper without CLI arguments (for example, `bash run/tune_bert.sh`).
- For tuning wrappers, frequently changed knobs are placed first in `CONFIG`
  (for example: species, bp lengths, trial/epoch budget, target selection).
- The same top-first layout is applied to training/inference wrappers
  (`run/cnn.sh`, `run/bert.sh`, etc.).
- Embedded fallback search-space JSON is in the same `CONFIG` block but should
  be treated as an advanced/default section.

## 1. Pipeline Wrapper Inventory

Training/inference wrappers (edit CONFIG block, run without CLI args):

- `run/cnn.sh`
- `run/cnn_resdil.sh`
- `run/tcn.sh`
- `run/bert.sh`
- `run/dnabert.sh`
- `run/reservoir.sh`

Utility wrappers:

- `run/make_test_data.sh`
- `run/eval_trans_score.sh`
- `run/plot_eval.sh`

Utility wrappers keep editable defaults near the top as `CONFIG` or
`USER DEFAULTS` blocks.

Tuning wrappers:

- `run/tune_cnn.sh`, `run/tune_cnn_time.sh`
- `run/tune_cnn_resdil.sh`, `run/tune_cnn_resdil_time.sh`
- `run/tune_tcn.sh`, `run/tune_tcn_time.sh`
- `run/tune_bert.sh`
- `run/tune_dnabert.sh`, `run/tune_dnabert_time.sh`
- `run/tune_reservoir.sh`

## 2. Common Control Flags (CONFIG)

The train/infer wrappers share these controls.

- `SKIP_TRAINING=1` -> adds `--skip_train`
- `CONTINUE_TRAINING=1` -> adds `--continue_train`
- `TRAIN_ONLY=1` -> adds `--train_only`
- `PRECOMPUTED_SITE_SCORE_TSV=<path>` -> adds `--site_score_tsv <path>`

Validation rules in wrappers:

- `SKIP_TRAINING=1` and `CONTINUE_TRAINING=1` cannot be combined.
- `TRAIN_TARGET=donor|acceptor` requires `TRAIN_ONLY=1`.

## 3. Continue Learning Behavior

`--continue_train` is handled in `src/run_model.py` as:

1. Build checkpoint paths from current model + naming parameters.
2. Verify existing donor/acceptor checkpoint files exist.
3. Pass init checkpoint paths to model module training.

Current module behavior:

- `dnabert` uses init checkpoints explicitly in train flow.
- Other models currently keep this wrapper flag available, but do not yet
  consume init checkpoint paths inside task training functions.

Practical implication:

- To continue from a specific run, keep all naming-relevant parameters
  (`model`, bp lengths, hyperparameter fields in naming) consistent.

## 4. Tuned Hyperparameter Injection

Wrappers with tuned-config support (`cnn`, `cnn_resdil`, `tcn`, `bert`,
`reservoir`) provide:

- `USE_TUNED_HPARAMS=off|auto|required`
- `DONOR_TUNED_CONFIG_PATH`
- `ACCEPTOR_TUNED_CONFIG_PATH`
- `SHARED_TUNED_CONFIG_PATH`

Resolution order per task:

1. explicit task path
2. `data/<species>/tuning/<model>/<task>/best_config.json`
3. shared path
4. legacy shared file `data/<species>/tuning/<model>/best_config.json`

If `required`, missing/invalid config aborts execution.

When `SKIP_TRAINING=1` or `CONTINUE_TRAINING=1`, wrappers also try to resolve
checkpoint paths from the selected tuned `best_config.json` and materialize a
strict-name checkpoint alias if needed. This keeps tuning outputs reusable even
when strict checkpoint naming differs between tuning and normal runs.

## 5. DNABERT Variant Switching

`run/dnabert.sh` supports:

- `DNABERT_VARIANT="2"` -> `--model dnabert2`
- `DNABERT_VARIANT="6"` -> `--model dnabert6`

Both aliases are routed by `src/models/registry.py` to `models.dnabert`.
The wrapper also controls:

- `PRETRAINED_MODEL_NAME`
- `PRETRAINED_REVISION`
- `TRUST_REMOTE_CODE`

Runtime tokenization mode is resolved inside `src/models/dnabert.py` from the
loaded tokenizer vocabulary, so `dnabert2` and `dnabert6` can share the same
wrapper while using different sequence preprocessing.

## 6. Performance and Device Controls

Common runtime knobs exposed by wrappers include:

- `DEVICE` (`auto|cuda|mps|cpu`)
- `USE_AMP`, `AMP_DTYPE`
- `COMPILE_MODE` (`off|on|auto`)
- `ALLOW_TF32`, `CUDNN_BENCHMARK`, `DETERMINISTIC`
- DataLoader controls: `NUM_WORKERS`, `PREFETCH_FACTOR`,
  `PERSISTENT_WORKERS`, `PIN_MEMORY`
- OOM backoff controls: `MIN_BATCH_SIZE`, `MAX_OOM_RETRIES`
- MPS cap: `MPS_MAX_BATCH_SIZE` -> exported as
  `INTRONMODEL_MPS_MAX_BATCH_SIZE`

## 6.1 Auto tmux (SSH disconnect-safe)

All user-facing wrappers under `run/` (train/infer, tuning, and utility)
automatically bootstrap into `tmux` when launched from an SSH TTY and not
already inside `tmux`. This keeps long jobs running after SSH disconnect.

- Default mode: `INTRONMODEL_AUTO_TMUX=auto`
- Force always-on: `INTRONMODEL_AUTO_TMUX=on`
- Disable: `INTRONMODEL_AUTO_TMUX=off`
- Session prefix override: `INTRONMODEL_TMUX_SESSION_PREFIX=<prefix>`

When auto-bootstrapped, wrappers print the created session name and attach
command (for reconnect).

## 7. Recommended Usage Patterns

- First run: `SKIP_TRAINING=0`, `CONTINUE_TRAINING=0`, `TRAIN_ONLY=0`
- Score-only rerun from existing checkpoints: `SKIP_TRAINING=1`
- Donor-only or acceptor-only tuning: set `TRAIN_TARGET` and `TRAIN_ONLY=1`
- Prefer tuned configs: set `USE_TUNED_HPARAMS=required` after tuning outputs
  are available
- Adaptive epoch budget: set `EPOCHS=auto` and tune
  `MAX_EPOCHS` / `EARLY_STOP_PATIENCE` / `EARLY_STOP_MIN_DELTA`

## 8. Data/Model Path Overrides

Runtime roots can be overridden by environment variables:

- `INTRONMODEL_DATA_ROOT`: overrides `<repo>/data`
- `INTRONMODEL_MODEL_ROOT`: overrides `<repo>/model`

Impacted paths include:

- dataset and score files under `data/<species>/...`
- tuning outputs and `best_config.json` under
  `data/<species>/tuning/<model>/<target>/`
- donor/acceptor checkpoints under `model/<species>/...`
