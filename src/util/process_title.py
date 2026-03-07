from __future__ import annotations

import os
import sys

PROCESS_TITLE_ENV: str = "INTRONMODEL_PROCESS_TITLE"
_LINUX_PR_SET_NAME: int = 15


def _apply_linux_process_name(title: str) -> bool:
    """Apply a short Linux process name via ``prctl(PR_SET_NAME)``.

    Parameters
    ----------
    title : str
        Desired process title. Linux truncates this value to 15 bytes plus the
        terminating null byte.

    Returns
    -------
    bool
        ``True`` when the kernel process name was updated successfully,
        otherwise ``False``.

    Raises
    ------
    None

    Complexity
    ----------
    O(n) time and O(n) memory, where ``n`` is the title length.
    """

    normalized = title.strip()
    if normalized == "" or not sys.platform.startswith("linux"):
        return False

    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        prctl.restype = ctypes.c_int
        name_bytes = normalized.encode("utf-8")[:15]
        name_buffer = ctypes.create_string_buffer(name_bytes + b"\0")
        result = prctl(
            _LINUX_PR_SET_NAME,
            ctypes.cast(name_buffer, ctypes.c_void_p),
            0,
            0,
            0,
        )
    except Exception:
        return False

    return result == 0


def apply_process_title(title: str) -> bool:
    """Apply one optional process title using ``setproctitle`` when available.

    Parameters
    ----------
    title : str
        Desired process title. Empty or whitespace-only strings disable the
        update.

    Returns
    -------
    bool
        ``True`` when a non-empty title was applied successfully. When
        ``setproctitle`` is unavailable, Linux falls back to the shorter
        kernel process name.

    Raises
    ------
    None

    Complexity
    ----------
    O(n) time and O(n) memory, where ``n`` is the title length.
    """

    normalized = title.strip()
    if normalized == "":
        return False
    try:
        import setproctitle

        setproctitle.setproctitle(normalized)
    except Exception:
        return _apply_linux_process_name(normalized)
    return True


def apply_process_title_from_env(env_name: str = PROCESS_TITLE_ENV) -> bool:
    """Apply an optional process title from one environment variable.

    Parameters
    ----------
    env_name : str, default=PROCESS_TITLE_ENV
        Environment variable name that may store the desired process title.

    Returns
    -------
    bool
        ``True`` when the environment variable contained a non-empty title and
        it was applied successfully, otherwise ``False``.

    Raises
    ------
    None

    Complexity
    ----------
    O(n) time and O(n) memory, where ``n`` is the title length.
    """

    return apply_process_title(os.environ.get(env_name, ""))
