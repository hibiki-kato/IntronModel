from __future__ import annotations

from datetime import datetime, timedelta
import os
import sys
import time

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


def format_eta_process_title(remaining_seconds: float) -> str:
    """Format one process title as an ETA wall-clock timestamp.

    Parameters
    ----------
    remaining_seconds : float
        Estimated remaining time in seconds.

    Returns
    -------
    str
        ETA title text in ``ETA:mm/dd HH:MM`` format.

    Raises
    ------
    None

    Complexity
    ----------
    O(1) time and O(1) memory.
    """

    safe_seconds = max(0.0, float(remaining_seconds))
    eta_local = datetime.now().astimezone() + timedelta(seconds=safe_seconds)
    return eta_local.strftime("ETA:%m/%d %H:%M")


def apply_eta_process_title_placeholder() -> bool:
    """Apply one placeholder ETA process title.

    Returns
    -------
    bool
        ``True`` when title application succeeded.

    Raises
    ------
    None

    Complexity
    ----------
    O(1) time and O(1) memory.
    """

    return apply_process_title("ETA:--/-- --:--")


def estimate_eta_remaining_seconds(
    *,
    elapsed_seconds: float,
    completed_epochs: int,
    total_epochs: int,
) -> float:
    """Estimate remaining seconds from epoch progress.

    Parameters
    ----------
    elapsed_seconds : float
        Elapsed training time since task start.
    completed_epochs : int
        Number of completed epochs.
    total_epochs : int
        Planned maximum epochs.

    Returns
    -------
    float
        Non-negative remaining-seconds estimate.

    Raises
    ------
    None

    Complexity
    ----------
    O(1) time and O(1) memory.
    """

    safe_elapsed = max(0.0, float(elapsed_seconds))
    safe_total_epochs = max(1, int(total_epochs))
    safe_completed_epochs = max(1, int(completed_epochs))
    capped_completed = min(safe_completed_epochs, safe_total_epochs)
    avg_epoch_seconds = safe_elapsed / float(capped_completed)
    remaining_epochs = max(0, safe_total_epochs - capped_completed)
    return avg_epoch_seconds * float(remaining_epochs)


def apply_eta_process_title_from_epoch_progress(
    *,
    task_started_at: float,
    completed_epochs: int,
    total_epochs: int,
) -> bool:
    """Estimate and apply ETA process title from epoch progress.

    Parameters
    ----------
    task_started_at : float
        Monotonic timestamp captured when task training started.
    completed_epochs : int
        Number of completed epochs.
    total_epochs : int
        Planned maximum epochs.

    Returns
    -------
    bool
        ``True`` when title application succeeded.

    Raises
    ------
    None

    Complexity
    ----------
    O(1) time and O(1) memory.
    """

    elapsed_seconds = max(0.0, time.perf_counter() - float(task_started_at))
    remaining_seconds = estimate_eta_remaining_seconds(
        elapsed_seconds=elapsed_seconds,
        completed_epochs=completed_epochs,
        total_epochs=total_epochs,
    )
    return apply_process_title(format_eta_process_title(remaining_seconds))
