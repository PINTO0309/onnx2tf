from __future__ import annotations

import atexit
import os
import shutil
import signal
import tempfile
import threading
from typing import Dict, Iterable, Optional, Set

_OWNER_FILE_NAME = ".onnx2tf_tmp_owner_pid"
_MANAGED_DIRS: Set[str] = set()
_LOCK = threading.Lock()
_HANDLERS_REGISTERED = False
_ORIGINAL_SIGNAL_HANDLERS: Dict[int, object] = {}


def _is_pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _write_owner_marker(temp_dir: str) -> None:
    marker_path = os.path.join(temp_dir, _OWNER_FILE_NAME)
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def _read_owner_pid(temp_dir: str) -> Optional[int]:
    marker_path = os.path.join(temp_dir, _OWNER_FILE_NAME)
    if not os.path.exists(marker_path):
        return None
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            text = str(f.read()).strip()
        return int(text)
    except Exception:
        return None


def cleanup_stale_tempdirs(
    *,
    prefixes: Iterable[str],
    root_dir: Optional[str] = None,
) -> None:
    temp_root = root_dir if root_dir is not None else tempfile.gettempdir()
    try:
        names = os.listdir(temp_root)
    except Exception:
        return

    prefix_list = [str(prefix) for prefix in prefixes if str(prefix) != ""]
    if len(prefix_list) == 0:
        return

    for name in names:
        if not any(name.startswith(prefix) for prefix in prefix_list):
            continue
        path = os.path.join(temp_root, name)
        if not os.path.isdir(path):
            continue
        owner_pid = _read_owner_pid(path)
        if owner_pid is None:
            # Skip unknown dirs to avoid removing non-onnx2tf directories
            # that accidentally share a prefix.
            continue
        if _is_pid_alive(owner_pid):
            continue
        shutil.rmtree(path, ignore_errors=True)


def cleanup_managed_tempdir(path: str) -> None:
    path_str = str(path)
    with _LOCK:
        _MANAGED_DIRS.discard(path_str)
    if path_str != "":
        shutil.rmtree(path_str, ignore_errors=True)


def _cleanup_all_managed_tempdirs() -> None:
    with _LOCK:
        targets = list(_MANAGED_DIRS)
        _MANAGED_DIRS.clear()
    for path in targets:
        shutil.rmtree(path, ignore_errors=True)


def _signal_handler(signum: int, _frame) -> None:
    _cleanup_all_managed_tempdirs()
    original = _ORIGINAL_SIGNAL_HANDLERS.get(int(signum), signal.SIG_DFL)
    if callable(original):
        original(signum, _frame)
        return
    if original == signal.SIG_IGN:
        return
    raise SystemExit(128 + int(signum))


def _ensure_handlers_registered() -> None:
    global _HANDLERS_REGISTERED
    with _LOCK:
        if _HANDLERS_REGISTERED:
            return
        atexit.register(_cleanup_all_managed_tempdirs)
        for signum_name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
            signum = getattr(signal, signum_name, None)
            if signum is None:
                continue
            try:
                previous = signal.getsignal(signum)
                _ORIGINAL_SIGNAL_HANDLERS[int(signum)] = previous
                signal.signal(signum, _signal_handler)
            except Exception:
                continue
        _HANDLERS_REGISTERED = True


def make_managed_tempdir(
    *,
    prefix: str,
    root_dir: Optional[str] = None,
    stale_prefixes: Optional[Iterable[str]] = None,
) -> str:
    if stale_prefixes is not None:
        cleanup_stale_tempdirs(
            prefixes=stale_prefixes,
            root_dir=root_dir,
        )
    temp_dir = tempfile.mkdtemp(
        prefix=str(prefix),
        dir=root_dir,
    )
    _write_owner_marker(temp_dir)
    with _LOCK:
        _MANAGED_DIRS.add(temp_dir)
    _ensure_handlers_registered()
    return temp_dir
