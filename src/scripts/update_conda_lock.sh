#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/environment.yml"
LOCK_FILE="${PROJECT_ROOT}/conda-lock.yml"
CACHE_ROOT="${PROJECT_ROOT}/.cache"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export XDG_CACHE_HOME
mkdir -p "${XDG_CACHE_HOME}/conda-lock"

if ! command -v conda-lock >/dev/null 2>&1; then
  echo "[update_conda_lock] conda-lock is not installed." >&2
  echo "[update_conda_lock] Install with: pip install conda-lock" >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[update_conda_lock] Missing ${ENV_FILE}" >&2
  exit 1
fi

echo "[update_conda_lock] Generating ${LOCK_FILE}"
conda-lock lock \
  --file "${ENV_FILE}" \
  --lockfile "${LOCK_FILE}" \
  --platform linux-64 \
  --platform osx-64 \
  --platform osx-arm64 \
  --platform win-64

echo "[update_conda_lock] Done"
