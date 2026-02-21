# Documentation Index

This directory includes a Sphinx-based documentation tree.

Primary entry page:

- `docs/index.md`

## Build Docs

Activate the project conda environment first:

```bash
conda activate intronmodel
```

Generate architecture figures:

```bash
python src/tools/generate_doc_figures.py
```

Build HTML:

```bash
python -m sphinx -b html docs docs/_build/html
```

Open:

- `docs/_build/html/index.html`

## Main Pages

- `docs/model-architecture.md`: detailed mathematics and layer-level model docs
- `docs/run-wrapper-scripts.md`: wrapper CONFIG options and continue-learning
  flow
- `docs/docs-system.md`: documentation tooling and rationale
- `docs/repo-structure.md`
- `docs/model-integration-contract.md`
- `docs/legacy-model-status.md`
- `docs/data-policy.md`
- `docs/history-rewrite-playbook.md`
