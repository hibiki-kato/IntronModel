# Legacy Model Status

These modules are retained as legacy experimental code. They are intentionally
not connected to the unified `run_model.py` path.

## Inventory

- `src/models/bert.py`
- `src/models/bert_drosophila.py`
- `src/models/cnn_v2.py`
- `src/models/reservoir.py`

## Current Policy

- Keep files for reference and future porting.
- Do not register them in `src/models/registry.py` yet.
- Do not claim production support in README.
- Port incrementally by conforming to the model integration contract.

## Rationale

- They use independent CLI flows and path conventions.
- They do not currently expose the unified `add_* / train / infer_site` API.
- Mixing them into the registry before refactoring would reduce reliability.
