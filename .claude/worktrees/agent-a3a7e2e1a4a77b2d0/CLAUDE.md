# IntronModel — Claude Code Instructions

## Absolute Paths

Never hardcode absolute paths in scripts or source code.

- Shell scripts: use `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` and build paths from there.
- Python: use `Path(__file__).resolve().parent` to derive paths from the file location.
- External paths: pass them in as project-root-relative values instead of hardcoding machine-specific locations.

Data output directories must be derived from the project root at runtime, never hardcoded.
