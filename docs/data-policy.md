# Data Policy

## Goal

Keep the Git repository lightweight by externalizing large reference files and
generated artifacts while preserving reproducibility through scripts.

## Scope of Externalized Paths

The following directories are considered runtime data and must stay outside
source control:

- `data/<species>/raw/`
- `data/<species>/train/`
- `data/<species>/site_score/`
- `data/<species>/trans_score/`
- `data/<species>/eval_score/`

Path root can be overridden by `INTRONMODEL_DATA_ROOT`.

## Regeneration Workflow

1. Import raw references from external storage:

```bash
bash src/scripts/fetch_reference_data.sh \
  --species Dmel \
  --source-root /path/to/external_data_root
```

2. Prepare species directories and generate `transcripts.tsv`:

```bash
bash src/scripts/prepare_species_data.sh \
  --species Dmel \
  --donor-len 100 \
  --acceptor-len 100 \
  --source-root /path/to/external_data_root
```

3. Run pipeline or evaluation wrappers as needed.

## Testing Data Rule

Only minimal deterministic fixtures may be stored under `tests/fixtures/`.
Do not commit full-size reference or generated data.

## Current Limitation

`prepare_species_data.sh` does not synthesize training `.err` files. Those must
be provisioned from preprocessing outputs or other deterministic generators.
