#!/usr/bin/env bash

intronmodel_auto_tmux() {
	local script_path="$1"
	local script_name="$2"

	local mode="${INTRONMODEL_AUTO_TMUX:-auto}"
	if [[ "${mode}" != "off" && "${mode}" != "on" && "${mode}" != "auto" ]]; then
		echo "[${script_name}] INTRONMODEL_AUTO_TMUX must be off|on|auto." >&2
		return 1
	fi
	if [[ "${mode}" == "off" ]]; then
		return 0
	fi
	if [[ -n "${TMUX:-}" ]]; then
		return 0
	fi
	if [[ "${INTRONMODEL_TMUX_BOOTSTRAPPED:-0}" == "1" ]]; then
		return 0
	fi
	if [[ "${mode}" == "auto" && -z "${SSH_CONNECTION:-}" ]]; then
		return 0
	fi
	if [[ ! -t 0 || ! -t 1 ]]; then
		return 0
	fi
	if ! command -v tmux >/dev/null 2>&1; then
		echo "[${script_name}] tmux not found; continue without tmux." >&2
		return 0
	fi

	local cwd_q
	local script_q
	printf -v cwd_q "%q" "$(pwd)"
	printf -v script_q "%q" "${script_path}"

	local start_cmd
	start_cmd="cd ${cwd_q} && INTRONMODEL_TMUX_BOOTSTRAPPED=1 bash ${script_q}"
	local session_name="${INTRONMODEL_TMUX_SESSION_NAME:-0}"

	if tmux has-session -t "${session_name}" >/dev/null 2>&1; then
		echo "[${script_name}] tmux session already exists: ${session_name}"
		echo "[${script_name}] attach: tmux attach -t ${session_name}"
		exec tmux attach-session -t "${session_name}"
	fi

	if ! tmux new-session -d -s "${session_name}" "${start_cmd}"; then
		echo "[${script_name}] failed to create tmux session." >&2
		return 1
	fi

	echo "[${script_name}] auto tmux session: ${session_name}"
	echo "[${script_name}] attach: tmux attach -t ${session_name}"
	exec tmux attach-session -t "${session_name}"
}
