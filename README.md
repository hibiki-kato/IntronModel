# IntronModel

Unified splice-site modeling and transcript scoring pipeline.

## Environment (Conda)

Python target: 3.12 (pinned)
012
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

Wrapper run (config-only; edit `run/cnn.sh` CONFIG first):

```bash
bash run/cnn.sh
```

Optional data preparation helper:

```bash
bash src/scripts/prepare_species_data.sh \
  --species Dmel \
  --donor-len 100 \
  --acceptor-len 100
```

## Available Models

Registered in `src/models/registry.py`:

- `cnn`
- `cnn_resdil`
- `tcn`
- `bert`
- `dnabert`
- `dnabert2`
- `dnabert6`
- `reservoir`

## Wrapper Scripts

Config-only training/inference wrappers:

- `run/cnn.sh`
- `run/cnn_resdil.sh`
- `run/tcn.sh`
- `run/bert.sh`
- `run/dnabert.sh`
- `run/reservoir.sh`

Common wrapper controls:

- `SKIP_TRAINING=1`
- `CONTINUE_TRAINING=1`
- `TRAIN_ONLY=1`
- `USE_TUNED_HPARAMS=off|auto|required` (except `run/dnabert.sh`)
- `EPOCHS=<int|auto>` with `MAX_EPOCHS`, `EARLY_STOP_PATIENCE`,
  `EARLY_STOP_MIN_DELTA`

`run/dnabert.sh` variant switch:

- `DNABERT_VARIANT="2"` -> `--model dnabert2`
- `DNABERT_VARIANT="6"` -> `--model dnabert6`

DNABERT tokenizer input mode is selected automatically in `src/models/dnabert.py`:

- `dnabert2` (DNABERT-2/BPE) keeps raw DNA sequence input.
- `dnabert6` (fixed 6-mer vocab) is converted to overlapping 6-mer text.

Training/inference wrappers follow a top-first workflow:

- edit the top `CONFIG (edit here)` block first
- then run without arguments (`bash run/cnn.sh`, etc.)

Tuning wrappers (`run/tune_*.sh`) follow the same pattern:

- edit the top `CONFIG (edit here)` block first
- then run without arguments (`bash run/tune_cnn.sh`, etc.)

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
- checkpoint paths under `model/<species>/{donor,acceptor}/`

## Development

Run tests:

```bash
pytest -q
```

Update the active environment when `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```
