from __future__ import annotations

from onnx2tf.utils import runtime_memory


def test_reclaim_unused_process_memory_collects_and_trims(monkeypatch) -> None:
    monkeypatch.setattr(runtime_memory.gc, "collect", lambda: 17)
    monkeypatch.setattr(runtime_memory, "_trim_glibc_heap", lambda: True)

    assert runtime_memory.reclaim_unused_process_memory() == {
        "collected_objects": 17,
        "allocator_trimmed": True,
    }


def test_reclaim_unused_process_memory_tolerates_optional_trim_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_memory.gc, "collect", lambda: 3)

    def _raise() -> bool:
        raise OSError("malloc_trim unavailable")

    monkeypatch.setattr(runtime_memory, "_trim_glibc_heap", _raise)
    assert runtime_memory.reclaim_unused_process_memory() == {
        "collected_objects": 3,
        "allocator_trimmed": False,
    }
