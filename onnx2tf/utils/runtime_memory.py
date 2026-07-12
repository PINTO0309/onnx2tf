from __future__ import annotations

import ctypes
import gc
import sys
from typing import Any


def _trim_glibc_heap() -> bool:
    """Return unused glibc arenas to the OS when the platform supports it."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        process_library = ctypes.CDLL(None)
        malloc_trim = getattr(process_library, "malloc_trim")
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except (AttributeError, OSError):
        return False


def reclaim_unused_process_memory() -> dict[str, Any]:
    """Collect unreachable objects and release optional allocator caches.

    Large conversion exports transiently clone constant buffers. CPython drops
    those objects when export returns, but glibc may retain their arenas. That
    retained address space can starve a subsequent isolated inference worker
    even though the buffers are no longer reachable.
    """
    collected_objects = int(gc.collect())
    allocator_trimmed = False
    try:
        allocator_trimmed = bool(_trim_glibc_heap())
    except Exception:
        # Heap trimming is an optional optimization. Conversion correctness
        # must not depend on libc implementation details.
        allocator_trimmed = False
    return {
        "collected_objects": collected_objects,
        "allocator_trimmed": allocator_trimmed,
    }
