#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/plot_eval.sh [options]

Options:
  --species <name>      Species folder under data/ (default: Dmel)
  --output-png <path>   Output PNG path (default: data/<species>/<species>_snpr.png)
  --interactive         Show plot interactively
  --x-min <float>       X-axis minimum (species default if omitted)
  --x-max <float>       X-axis maximum (species default if omitted)
  --y-min <float>       Y-axis minimum (species default if omitted)
  --y-max <float>       Y-axis maximum (species default if omitted)
  -h, --help            Show this help

Species default ranges:
  Athal: x=[10.0, 52.0], y=[48.0, 75.0]
  Dmel : x=[40.0, 52.0], y=[40.0, 55.0]
  Mmus : x=[10.0, 18.0], y=[40.0, 46.0]
EOT
}

# Ensure conda is available in non-interactive shells.
if command -v conda >/dev/null 2>&1; then
	CONDA_BASE="$(conda info --base 2>/dev/null || true)"
	if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
		# shellcheck source=/dev/null
		source "${CONDA_BASE}/etc/profile.d/conda.sh"
	fi
fi

conda activate intronmodel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SPECIES="Mmus"
OUTPUT_PNG=""
INTERACTIVE="1"
X_MIN=""
X_MAX=""
Y_MIN=""
Y_MAX=""

while [[ $# -gt 0 ]]; do
	case "$1" in
	--species)
		SPECIES="$2"
		shift 2
		;;
	--output-png)
		OUTPUT_PNG="$2"
		shift 2
		;;
	--interactive)
		INTERACTIVE="1"
		shift
		;;
	--x-min)
		X_MIN="$2"
		shift 2
		;;
	--x-max)
		X_MAX="$2"
		shift 2
		;;
	--y-min)
		Y_MIN="$2"
		shift 2
		;;
	--y-max)
		Y_MAX="$2"
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown argument: $1" >&2
		usage
		exit 1
		;;
	esac
done

case "${SPECIES}" in
Athal)
	DEFAULT_X_MIN="10.0"
	DEFAULT_X_MAX="52.0"
	DEFAULT_Y_MIN="48.0"
	DEFAULT_Y_MAX="75.0"
	;;
Dmel)
	DEFAULT_X_MIN="40.0"
	DEFAULT_X_MAX="52.0"
	DEFAULT_Y_MIN="40.0"
	DEFAULT_Y_MAX="55.0"
	;;
Mmus)
	DEFAULT_X_MIN="10.0"
	DEFAULT_X_MAX="18.0"
	DEFAULT_Y_MIN="40.0"
	DEFAULT_Y_MAX="46.0"
	;;
*)
	if [[ -z "${X_MIN}" || -z "${X_MAX}" || -z "${Y_MIN}" || -z "${Y_MAX}" ]]; then
		echo "[plot_eval.sh] Unknown species '${SPECIES}'." >&2
		echo "[plot_eval.sh] Use one of: Athal, Dmel, Mmus; or provide all of" >&2
		echo "[plot_eval.sh] --x-min --x-max --y-min --y-max explicitly." >&2
		exit 1
	fi
	DEFAULT_X_MIN="40.0"
	DEFAULT_X_MAX="50.0"
	DEFAULT_Y_MIN="40.0"
	DEFAULT_Y_MAX="50.0"
	;;
esac

if [[ -z "${X_MIN}" ]]; then
	X_MIN="${DEFAULT_X_MIN}"
fi
if [[ -z "${X_MAX}" ]]; then
	X_MAX="${DEFAULT_X_MAX}"
fi
if [[ -z "${Y_MIN}" ]]; then
	Y_MIN="${DEFAULT_Y_MIN}"
fi
if [[ -z "${Y_MAX}" ]]; then
	Y_MAX="${DEFAULT_Y_MAX}"
fi

RUN_ARGS=(
	plot
	"${SPECIES}"
	--x_min "${X_MIN}"
	--x_max "${X_MAX}"
	--y_min "${Y_MIN}"
	--y_max "${Y_MAX}"
)

if [[ -n "${OUTPUT_PNG}" ]]; then
	RUN_ARGS+=(--output_png "${OUTPUT_PNG}")
fi
if [[ "${INTERACTIVE}" == "1" ]]; then
	RUN_ARGS+=(--interactive)
fi

echo "[plot_eval.sh] species=${SPECIES}"
python3 "${PROJECT_ROOT}/src/evaluate_scores.py" "${RUN_ARGS[@]}"
