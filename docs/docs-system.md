# Documentation System

## Why Move Details Out of `README.md`

`README.md` should stay as a short entry point. Detailed material such as:

- model equations,
- layer-by-layer explanations,
- architecture figures,
- transcript-level scoring math,

is better maintained in a dedicated documentation tree.

## Selected Python Package

This repository uses **Sphinx** with **MyST Parser**.

- `sphinx`: HTML documentation builder and cross-page navigation.
- `myst-parser`: Markdown authoring with math support.
- `sphinx.ext.mathjax`: LaTeX-style equations in browser-rendered HTML.

Dependency management is centralized in `environment.yml`.

## Alternatives Considered

- MkDocs: good for docs sites, but current repository already has many Markdown
  files and benefits from Sphinx extensions for math-heavy technical pages.
- Notebook-first docs: unnecessary for this codebase because most documentation
  is static architecture and pipeline math.

## Build Commands

Activate the conda environment:

```bash
conda activate intronmodel
```

Generate architecture figures:

```bash
python src/tools/generate_doc_figures.py
```

Build HTML documentation:

```bash
python -m sphinx -b html docs docs/_build/html
```

Open the generated entry page:

`docs/_build/html/index.html`
