# IntronModel

Unified splice-site modeling and transcript scoring pipeline.

## Environment (Conda)

Python target: 3.12 (pinned)

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate intronmodel
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate intronmodel
```

### OS compatibility note

`environment.yml` does **not** require the same OS, but resolution is
platform-dependent and may fail if a dependency is unavailable on your OS.

## Quick Start

Direct CLI run (train + infer + transcript aggregation + evaluation):

```bash
python src/run_model.py \
  --model cnn \
  --species Dmel \
  --donor_len 100 \
  --acceptor_len 100
```

Wrapper run (config-only; edit `run/run_cnn.sh` CONFIG first):

```bash
bash run/run_cnn.sh
```

Optional data preparation helper:

```bash
bash src/scripts/prepare_species_data.sh \
  --species Dmel \
  --donor-len 100 \
  --acceptor-len 100
```

Generate full-intron positive training data
(`intron + 10bp flank` by default):

```bash
bash run/make_intron_training_data.sh --species Dmel,Mmus,Athal
```

Generate variable-length pair datasets trimmed by intron half-length:

```bash
bash run/make_trimmed_pair_data.sh --species Dmel,Mmus,Athal
```

Generate test-site TSV with short-intron clipping enabled
(keep donor/acceptor intronic context inside intron length):

```bash
bash run/make_test_data.sh \
  --species Dmel \
  --donor-len 100 \
  --acceptor-len 100 \
  --clip-short-intron
```

Build intron-candidate evaluation data with intron-level labels:

```bash
bash run/make_labeled_intron_eval_data.sh --species Dmel
```

Evaluate intron-level PR-AUC from labeled introns and model site scores:

```bash
bash run/eval_intron_pr_auc.sh --species Dmel
```

## Available Models

Registered in `src/models/registry.py`:

- `cnn`
- `cnn_pair`
- `cnn_resdil`
- `tcn`
- `bert`
- `dnabert`
- `dnabert2`
- `dnabert6`
- `reservoir`
- `reservoir_legacy` (previous implementation kept for compatibility)

## Wrapper Scripts

Config-only training/inference wrappers:

- `run/run_cnn.sh`
- `run/run_cnn_pair.sh`
- `run/run_cnn_resdil.sh`
- `run/run_tcn.sh`
- `run/run_bert.sh`
- `run/run_dnabert.sh`
- `run/reservoir.sh`

Common wrapper controls:

- `SKIP_TRAINING=1`
- `CONTINUE_TRAINING=1`
- `TRAIN_ONLY=1`
- `USE_TUNED_HPARAMS=off|auto|required` (except `run/run_dnabert.sh`)
- `EPOCHS=<int|auto>` with `MAX_EPOCHS`, `EARLY_STOP_PATIENCE`,
  `EARLY_STOP_MIN_DELTA`
- `SEQUENCE_TRANSFORM=none|mask_outside_intron_n` (`run/run_cnn.sh`,
  `run/run_cnn_pair.sh`)
- CNN-family kernel config uses layer-wise lists:
  `KERNEL_SIZES`, `DONOR_KERNEL_SIZES`, `ACCEPTOR_KERNEL_SIZES`

Transcript score TSV compatibility:

- Output schema is fixed to 5 columns:
  `transcript_id`, `min_intron_index`, `Score_donor`,
  `Score_acceptor`, `min_donor_plus_acceptor`.
- `cnn_pair` keeps this schema by writing the same pair score into both
  `Score_donor` and `Score_acceptor`.

Reservoir-specific notes:

- `run/reservoir.sh` defaults to `INTRONMODEL_RC_STATE_BUDGET_GB="auto"`.
- `auto` resolves a state-memory budget from detected system RAM.
- Reservoir training uses a Torch ESN state generator and a scikit-learn
  readout (`lin|mlp|svm`).
- `USE_TUNED_HPARAMS=auto|required` can override `INPUT_MODE` per task when
  task-specific fields are left empty.
- To force one-hot input with tuned configs, set
  `DONOR_INPUT_MODE="onehot"` and `ACCEPTOR_INPUT_MODE="onehot"` explicitly.

Data utility wrappers:

- `run/make_test_data.sh`
- `run/make_intron_training_data.sh`
- `run/make_trimmed_pair_data.sh`
- `run/make_labeled_intron_eval_data.sh`
- `run/eval_intron_pr_auc.sh`

`run/run_dnabert.sh` variant switch:

- `DNABERT_VARIANT="2"` -> `--model dnabert2`
- `DNABERT_VARIANT="6"` -> `--model dnabert6`

DNABERT tokenizer input mode is selected automatically in `src/models/dnabert.py`:

- `dnabert2` (DNABERT-2/BPE) keeps raw DNA sequence input.
- `dnabert6` (fixed 6-mer vocab) is converted to overlapping 6-mer text.

Training/inference wrappers follow a top-first workflow:

- edit the top `CONFIG (edit here)` block first
- then run without arguments (`bash run/run_cnn.sh`, etc.)

Tuning wrappers (`run/tune_*.sh`) follow the same pattern:

- edit the top `CONFIG (edit here)` block first
- then run without arguments (`bash run/tune_cnn.sh`, etc.)
- pair time-budget tuning: `bash run/tune_cnn_pair_time.sh`
- CNN-family tuning now samples architecture with independent pools:
  `conv_depth` (layer count), `channel_candidates`, and `kernel_candidates`
  (`donor_*` / `acceptor_*` variants for `cnn_pair`)
- OOM protection in tuning uses both batch-size backoff and
  `max_model_params` pre-filtering in the generated search config
- CNN-family tuning supports `MAX_MODEL_PARAMS=auto`; it estimates a safe
  parameter cap from selected GPU VRAM (`GPU_IDS`) using conservative runtime
  factors and falls back to model-specific defaults if detection fails

## Documentation

- [Documentation entry](docs/index.md)
- [Model architecture](docs/model-architecture.md)
- [Run wrapper guide](docs/run-wrapper-scripts.md)
- [Docs system](docs/docs-system.md)

Build docs locally:

```bash
python src/tools/generate_doc_figures.py
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`.

## GitHub Pages

Docs are auto-deployed by `.github/workflows/docs-pages.yml` when pushing to
`main` or `master` and files under `docs/**`,
`src/tools/generate_doc_figures.py`, `environment.yml`, or the workflow itself
change.

Repository setting requirement: Pages must use **GitHub Actions** as the
build/deploy source.

## Path Overrides

Default runtime roots are:

- data root: `<repo>/data`
- model root: `<repo>/model`

You can override them with environment variables:

```bash
export INTRONMODEL_DATA_ROOT=/path/to/data_root
export INTRONMODEL_MODEL_ROOT=/path/to/model_root
```

These overrides affect:

- training/inference input and output under `data/<species>/...`
- tuning outputs and `best_config.json` under
  `data/<species>/tuning/<model>/<target>/`
- checkpoint paths under `model/<species>/{donor,acceptor,pair}/`

## Development

Run tests:

```bash
pytest -q
```

Update the active environment when `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```
