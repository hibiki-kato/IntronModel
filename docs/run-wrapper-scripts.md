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
  (`run/run_cnn.sh`, `run/run_bert.sh`, etc.).
- Embedded fallback search-space JSON is in the same `CONFIG` block but should
  be treated as an advanced/default section.

## 1. Pipeline Wrapper Inventory

Training/inference wrappers (edit CONFIG block, run without CLI args):

- `run/run_cnn.sh`
- `run/run_cnn_pair.sh`
- `run/run_cnn_resdil.sh`
- `run/run_tcn.sh`
- `run/run_bert.sh`
- `run/run_dnabert.sh`
- `run/reservoir.sh`

Utility wrappers:

- `run/make_test_data.sh`
- `run/make_intron_training_data.sh`
- `run/make_trimmed_pair_data.sh`
- `run/make_labeled_intron_eval_data.sh`
- `run/eval_intron_pr_auc.sh`
- `run/eval_trans_score.sh`
- `run/plot_eval.sh`

Utility wrappers keep editable defaults near the top as `CONFIG` or
`USER DEFAULTS` blocks.

Data-generation notes:

- `run/make_test_data.sh` supports `--clip-short-intron` to keep donor/acceptor
  intronic context inside intron length for short introns.
- `run/make_intron_training_data.sh` converts `100bp.err` to
  full-intron-positive TSV (default:
  `data/<species>/raw/intron_full_flank10.pos.tsv`).
- `run/make_trimmed_pair_data.sh` creates variable-length pair datasets from
  `100bp.err` and `100bp.neg.err` using intron half-length metadata
  (`100bp_trimmed.err` and `100bp_trimmed.neg.err` by default).
- `run/eval_intron_pr_auc.sh` evaluates intron-level PR-AUC by joining
  `intron_eval_flank10.tsv` labels with `site_score/*.tsv` outputs.

Tuning wrappers:

- `run/tune_cnn.sh`, `run/tune_cnn_time.sh`
- `run/tune_cnn_pair_time.sh`
- `run/tune_cnn_resdil.sh`, `run/tune_cnn_resdil_time.sh`
- `run/tune_tcn.sh`, `run/tune_tcn_time.sh`
- `run/tune_bert.sh`
- `run/tune_dnabert.sh`, `run/tune_dnabert_time.sh`
- `run/tune_reservoir.sh`

Archived wrappers:

- `archive/cnn_v3_20260325_183046/run/run_cnn_v3.sh`
- `archive/cnn_v3_20260325_183046/run/tune_cnn_v3_time.sh`

CNN-family tuning search-space conventions:

- `cnn` / `cnn_resdil` tune scripts sample architecture from independent keys:
  `conv_depth` (CNN stage count), `channel_candidates`, `kernel_candidates`.
- `cnn_pair` uses branch-specific variants:
  `donor_conv_depth`, `acceptor_conv_depth`,
  `donor_channel_candidates`, `acceptor_channel_candidates`,
  `donor_kernel_candidates`, `acceptor_kernel_candidates`.
- The sampled architecture is materialized to run-time args
  (`conv_channels`, `kernel_sizes`, branch overrides) before each trial.
- CNN and CNN-pair tuning can also search `max_pool_size` to compare no-pool
  (`1`) vs. wider pooling windows, `conv_stride` for learned downsampling, and
  `head_type=gap|center` for readout style.
- Shape-invalid CNN / CNN-pair samples (for example, pooling that collapses the
  sequence length after conv/pool downsampling, or `cnn_pair` early/mid fusion
  with mismatched input lengths) are discarded and resampled before a trial is
  launched.
- `run/tune_cnn_v2_time.sh` and `run/tune_cnn_v2_pair_time.sh` intentionally
  omit `max_model_params` and rely on OOM backoff instead. Their search spaces
  now expose a binary `mask=off|on` hparam, which the wrappers translate to
  `sequence_transform=none|mask_outside_intron_n` during model execution.
  They are also widened to include more batch sizes and architecture
  candidates.
- `run/run_cnn_v2_pair.sh` and `run/tune_cnn_v2_pair_time.sh` also expose
  `SYNTHESIZE_MODE=off|on`. When enabled, the pair pipeline defaults the
  negative training file to `processed/100bp_mixed_one_side.neg.err` and
  appends a `synth` suffix through `tag`-based naming. Synth runs also move to
  the `cnn_v2_pair_synth` tuning subtree and use a separate
  `best_synth_config.json` there, while the run wrapper keeps its own `TAG`
  control authoritative so tuned-config tags do not leak into an `off` run.
- `run/run_dnabert_pair.sh`, `run/tune_dnabert_pair.sh`, and
  `run/tune_dnabert_pair_time.sh` also expose `SYNTHESIZE_MODE=off|on`. When
  enabled, they use the same pair synth defaults for training inputs, append a
  `synth` suffix to output naming, and switch tuned-config lookups to the
  `dnabert_pair_synth` tuning subtree with `best_synth_config.json`.
- Older CNN-family wrappers may still support `MAX_MODEL_PARAMS=auto`; those
  scripts estimate a conservative cap from selected GPU VRAM (`GPU_IDS`) and
  write the resolved integer into `hparam_search_config.json`.

## 2. Common Control Flags (CONFIG)

The train/infer wrappers share these controls.

- `SKIP_TRAINING=1` -> adds `--skip_train`
- `CONTINUE_TRAINING=1` -> adds `--continue_train`
- `TRAIN_ONLY=1` -> adds `--train_only`
- `PRECOMPUTED_SITE_SCORE_TSV=<path>` -> adds `--site_score_tsv <path>`

Validation rules in wrappers:

- `SKIP_TRAINING=1` and `CONTINUE_TRAINING=1` cannot be combined.
- For multi-task models, `TRAIN_TARGET=<single task>` requires `TRAIN_ONLY=1`.
- `run/run_cnn_pair.sh` uses `TRAIN_TARGET=pair` and can run full pipeline.
- `SEQUENCE_TRANSFORM=none|mask_outside_intron_n` is available in
  `run/run_cnn.sh` and `run/run_cnn_pair.sh`.
- `run/run_cnn_v2.sh` and `run/run_cnn_v2_pair.sh` accept tuned configs that
  store `mask=off|on`; the wrapper converts that back to
  `sequence_transform=none|mask_outside_intron_n` when applying tuned values.
- `MAX_POOL_SIZE>=1` is available in `run/run_cnn.sh`,
  `run/run_cnn_pair.sh`, `run/tune_cnn.sh`, `run/tune_cnn_time.sh`, and
  `run/tune_cnn_pair_time.sh`. `1` disables pooling.
- `CONV_STRIDE>=1` and `HEAD_TYPE=gap|center` are available in the same CNN /
  CNN-pair wrappers and tune scripts.

## 3. Continue Learning Behavior

`--continue_train` is handled in `src/run_model.py` as:

1. Build checkpoint paths from current model + naming parameters.
2. Verify existing task checkpoint files exist.
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

Injection behavior note:

- Tuned values are applied only when the corresponding task-level field in
  wrapper `CONFIG` is empty.
- If a task-level field is non-empty, wrapper value wins and tuned value is
  kept but not applied.

When `SKIP_TRAINING=1` or `CONTINUE_TRAINING=1`, wrappers also try to resolve
checkpoint paths from the selected tuned `best_config.json` and materialize a
strict-name checkpoint alias if needed. This keeps tuning outputs reusable even
when strict checkpoint naming differs between tuning and normal runs.

### 4.1 Reservoir practical overrides

`run/reservoir.sh` also exposes:

- `INTRONMODEL_RC_STATE_BUDGET_GB=auto|<float>`
- `MTS_REP=auto|last|mean|output|reservoir`
- `DIMRED_METHOD=none|pca|tenpca`
- `READOUT_TYPE=lin|mlp|svm`

For one-hot-only operation with tuned configs:

- Set `DONOR_INPUT_MODE="onehot"` and `ACCEPTOR_INPUT_MODE="onehot"` in
  `run/reservoir.sh`.
- Or disable tuned injection with `USE_TUNED_HPARAMS=off`.

## 5. DNABERT Variant Switching

`run/run_dnabert.sh` supports:

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
- `INTRONMODEL_TORCH_COMPILE_STRATEGY`
  (`default-then-off|max-then-default-then-off|max-only|default-only|off`)
- `ALLOW_TF32`, `CUDNN_BENCHMARK`, `DETERMINISTIC`
- DataLoader controls: `NUM_WORKERS`, `PREFETCH_FACTOR`,
  `PERSISTENT_WORKERS`, `PIN_MEMORY`
- OOM backoff controls: `MIN_BATCH_SIZE`, `MAX_OOM_RETRIES`
- MPS cap: `MPS_MAX_BATCH_SIZE` -> exported as
  `INTRONMODEL_MPS_MAX_BATCH_SIZE`
- CNN-family layer-wise kernels:
  `KERNEL_SIZES`, `DONOR_KERNEL_SIZES`, `ACCEPTOR_KERNEL_SIZES`

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
- task checkpoints (`donor`, `acceptor`, `pair`) under `model/<species>/...`
