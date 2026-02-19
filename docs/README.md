# Documentation Index

This directory contains repository operation documents used before introducing
new model families.

## Files

- `docs/repo-structure.md`: canonical repository structure and ownership
- `docs/model-integration-contract.md`: required interface for model modules
- `docs/legacy-model-status.md`: current legacy model inventory and policy
- `docs/data-policy.md`: data externalization and regeneration policy
- `docs/history-rewrite-playbook.md`: operational steps for Git history cleanup
- `docs/reports/repo_scan_2026-02-19.md`: current consistency scan snapshot

## Environment Baseline

- Python: 3.12+
- Required runtime packages: `torch`, `numpy`, `matplotlib`
- Required test package: `pytest`

Install test dependency:

```bash
python3 -m pip install -r requirements-dev.txt
```

Run tests:

```bash
python3 -m pytest -q
```
