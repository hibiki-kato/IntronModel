# IntronModel

Splice-site modeling and transcript scoring with a unified pipeline CLI.

## Status

- Supported pipeline model: `cnn`
- Unified entrypoint: `src/run_model.py`
- Legacy experimental modules are kept under `src/models/` but are **not**
  connected to the unified CLI:
  - `src/models/bert.py`
  - `src/models/bert_drosophila.py`
  - `src/models/cnn_v2.py`
  - `src/models/reservoir.py`

## Repository Layout

```text
.
├── docs/
│   ├── README.md
│   ├── data-policy.md
│   ├── legacy-model-status.md
│   ├── model-integration-contract.md
│   ├── repo-structure.md
│   └── reports/
├── run/
│   ├── cnn.sh
│   ├── eval_trans_score.sh
│   ├── gffcompare_counts.sh
│   ├── make_test_data.sh
│   ├── plot_eval.sh
│   └── tune_cnn.sh
├── scripts/
│   ├── fetch_reference_data.sh
│   └── prepare_species_data.sh
├── src/
│   ├── evaluate_scores.py
│   ├── gffcompare_counts.py
│   ├── run_model.py
│   ├── models/
│   │   ├── registry.py
│   │   ├── cnn.py
│   │   └── <legacy models>
│   └── util/
└── tools/
    ├── hparam_search.py
    └── scan_obsolete.py
```

## Unified Pipeline CLI

`src/run_model.py` executes a fixed pipeline in order:

1. train
2. infer (site-level scoring)
3. transcript (transcript-level aggregation)
4. eval (SN/PR/F1 output and optional plot)

Automatic skipping:

- `--skip_train` skips training.
- `--site_score_tsv <path>` skips infer and consumes an external site score TSV.
- `--train_only` runs only training and stops before infer/transcript/eval.

### Main Usage

```bash
python3 src/run_model.py \
  --model cnn \
  --species Dmel \
  --donor_len 100 \
  --acceptor_len 100 \
  --loss focal \
  --name_fields bp_avg,loss
```

### Inference-like Usage (skip training)

```bash
python3 src/run_model.py \
  --model cnn \
  --species Dmel \
  --donor_len 100 \
  --acceptor_len 100 \
  --loss focal \
  --name_fields bp_avg,loss \
  --skip_train
```

### Training-only Usage (donor/acceptor model tuning)

```bash
python3 src/run_model.py \
  --model cnn \
  --species Dmel \
  --donor_len 100 \
  --acceptor_len 100 \
  --loss focal \
  --epochs 5 \
  --train_only
```

### Use Precomputed Site Scores (skip infer)

```bash
python3 src/run_model.py \
  --model cnn \
  --species Dmel \
  --site_score_tsv data/Dmel/site_score/cnn100bp_lossfocal.tsv \
  --skip_train
```

## Transcript Aggregation Options

`--intron_score_op`:

- `+`
- `*`
- `harmonic`
- `min`

`--transcript_score_agg`:

- `min`
- `softmin`
- `softmin_wavg`
- `+`
- `*`
- `mean`
- `avg` (alias of `mean`)
- `median`
- `max`

`--softmin_tau` is used by `softmin` and `softmin_wavg` and must be positive.

## Wrapper Scripts

All wrappers are maintained and aligned to the unified CLI.

`run/cnn.sh` and `run/tune_cnn.sh` are config-only scripts.
Edit the `CONFIG` block in each script, then run without arguments:

```bash
bash run/cnn.sh
bash run/tune_cnn.sh
```

`run/tune_cnn.sh` executes two-phase random search:

1. quick phase (short epochs)
2. full phase (top-k re-train)

Outputs are written under:

`data/<species>/tuning/cnn/<timestamp>/`

The following wrappers keep CLI help:

```bash
bash run/gffcompare_counts.sh --help
bash run/make_test_data.sh --help
bash run/eval_trans_score.sh --help
bash run/plot_eval.sh --help
```

## Data Policy

Large data and generated artifacts are externalized from Git tracking.

- `data/` is ignored.
- Reproducibility depends on scripts and documented procedures.
- See `docs/data-policy.md` for details.

## Environment

This project targets Python 3.12+.

Suggested packages:

- `torch`
- `numpy`
- `matplotlib`
- `pytest`

For test dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

## Tests

```bash
python3 -m pytest -q
```

## Documentation Index

- `docs/README.md`
- `docs/repo-structure.md`
- `docs/model-integration-contract.md`
- `docs/legacy-model-status.md`
- `docs/data-policy.md`
- `docs/history-rewrite-playbook.md`
- `docs/reports/repo_scan_2026-02-19.md`
