# IntronModel — Claude Code Instructions

## Absolute Paths

Never hardcode absolute paths in scripts or source code.

- Shell scripts: use `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` and build paths from there.
- Python: use `Path(__file__).resolve().parent` to derive paths from the file location.
- Exception: conda init paths in `run/lib/common.sh` use `/export/${USER}/...` with `${USER}` substitution — this is intentional for the cluster environment.

Data output directories must be derived from the project root at runtime, never hardcoded (e.g. `/export/hibiki/...`).
