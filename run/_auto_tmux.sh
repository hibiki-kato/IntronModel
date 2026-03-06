#!/usr/bin/env bash

_intronmodel_tmux_set_env_var() {
	local key="$1"
	local value="$2"
	tmux set-environment -g "${key}" "${value}" >/dev/null 2>&1 || true
}


_intronmodel_tmux_sync_env() {
	_intronmodel_tmux_set_env_var "PATH" "${PATH}"
	_intronmodel_tmux_set_env_var "HOME" "${HOME}"
	_intronmodel_tmux_set_env_var "SHELL" "${SHELL:-/bin/bash}"
	_intronmodel_tmux_set_env_var "USER" "${USER:-$(id -un)}"
	_intronmodel_tmux_set_env_var "LOGNAME" "${LOGNAME:-${USER:-$(id -un)}}"

	if [[ -n "${CONDA_EXE:-}" ]]; then
		_intronmodel_tmux_set_env_var "CONDA_EXE" "${CONDA_EXE}"
	fi
	if [[ -n "${INTRONMODEL_CONDA_SH:-}" ]]; then
		_intronmodel_tmux_set_env_var "INTRONMODEL_CONDA_SH" \
			"${INTRONMODEL_CONDA_SH}"
	fi

	if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
		_intronmodel_tmux_set_env_var "LD_LIBRARY_PATH" "${LD_LIBRARY_PATH}"
	fi
	if [[ -n "${DISPLAY:-}" ]]; then
		_intronmodel_tmux_set_env_var "DISPLAY" "${DISPLAY}"
	fi
	if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
		_intronmodel_tmux_set_env_var "WAYLAND_DISPLAY" "${WAYLAND_DISPLAY}"
	fi
	if [[ -n "${XAUTHORITY:-}" ]]; then
		_intronmodel_tmux_set_env_var "XAUTHORITY" "${XAUTHORITY}"
	fi
	if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
		_intronmodel_tmux_set_env_var "XDG_RUNTIME_DIR" "${XDG_RUNTIME_DIR}"
	fi
	if [[ -n "${MPLBACKEND:-}" ]]; then
		_intronmodel_tmux_set_env_var "MPLBACKEND" "${MPLBACKEND}"
	fi

	if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
		_intronmodel_tmux_set_env_var "CUDA_VISIBLE_DEVICES" \
			"${CUDA_VISIBLE_DEVICES}"
	else
		tmux set-environment -gru "CUDA_VISIBLE_DEVICES" >/dev/null 2>&1 || true
	fi
}


_intronmodel_tmux_attach_or_exit() {
	local session_name="$1"
	local script_name="$2"

	if tmux attach-session -t "${session_name}"; then
		exit 0
	fi

	local attach_exit="$?"
	echo "[${script_name}] failed to attach tmux session (exit=${attach_exit})." >&2
	echo "[${script_name}] detached session may still be running." >&2
	echo "[${script_name}] attach manually: tmux attach -t ${session_name}" >&2
	exit 0
}


intronmodel_auto_tmux() {
	local script_path="$1"
	local script_name="$2"
	shift 2

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
	local arg_q
	local cwd
	cwd="$(pwd)"
	printf -v cwd_q "%q" "${cwd}"
	printf -v script_q "%q" "${script_path}"
	arg_q=""
	while [[ $# -gt 0 ]]; do
		local next_arg_q
		printf -v next_arg_q "%q" "$1"
		arg_q+=" ${next_arg_q}"
		shift
	done

	local start_cmd
	printf -v start_cmd \
		"cd %s && INTRONMODEL_TMUX_BOOTSTRAPPED=1 bash %s%s" \
		"${cwd_q}" \
		"${script_q}" \
		"${arg_q}"
	local session_default="intronmodel_${script_name%.*}"
	session_default="${session_default//[^a-zA-Z0-9_]/_}"
	local session_name="${INTRONMODEL_TMUX_SESSION_NAME:-${session_default}}"

	_intronmodel_tmux_sync_env

	if tmux has-session -t "${session_name}" >/dev/null 2>&1; then
		echo "[${script_name}] tmux session already exists: ${session_name}"
		echo "[${script_name}] attach: tmux attach -t ${session_name}"
		_intronmodel_tmux_attach_or_exit "${session_name}" "${script_name}"
	fi

	if ! tmux new-session -d -s "${session_name}" -c "${cwd}" "${start_cmd}"; then
		echo "[${script_name}] failed to create tmux session." >&2
		return 1
	fi
	tmux set-option -t "${session_name}:0" remain-on-exit off >/dev/null 2>&1 \
		|| true

	echo "[${script_name}] auto tmux session: ${session_name}"
	echo "[${script_name}] attach: tmux attach -t ${session_name}"
	_intronmodel_tmux_attach_or_exit "${session_name}" "${script_name}"
}
