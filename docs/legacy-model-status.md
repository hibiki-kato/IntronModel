# Legacy Model Status

This page tracks model files under `src/models/` that are currently outside the
unified `run_model.py` registry path.

## Current Inventory (not in registry)

- `src/models/bert_drosophila.py`
- `src/models/cnn_v2.py`
- `src/models/reservoir_rc.py`

## Integrated Models (in registry)

- `cnn`
- `cnn_resdil`
- `tcn`
- `bert`
- `dnabert`
- `dnabert2`
- `dnabert6`
- `reservoir`
- `reservoir_legacy`

## Policy

- Keep legacy files for reference and controlled porting.
- Do not advertise legacy files as supported wrappers unless registered.
- Add new registry entries only after contract compliance and pipeline tests.

## Porting Checklist

1. Implement `add_train_args`, `add_infer_args`, `train`, `infer_site`.
2. Pass registry contract validation in `src/models/registry.py`.
3. Verify pipeline compatibility via `src/run_model.py`.
4. Add/adjust wrapper scripts and documentation.
