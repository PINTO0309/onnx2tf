import os
import time

import numpy as np
import pytest

from onnx2tf.tflite_builder.accuracy_evaluator import _run_worker_in_subprocess


def _worker_large_payload(payload, result_queue) -> None:
    shape = tuple(int(v) for v in payload["shape"])
    arr = np.zeros(shape, dtype=np.float32)
    result_queue.put({"ok": True, "value": arr})


def _worker_error(payload, result_queue) -> None:
    result_queue.put(
        {
            "ok": False,
            "error": str(payload.get("error", "error")),
            "traceback": "traceback",
        }
    )


def _worker_abort(payload, result_queue) -> None:
    os.abort()


def test_run_worker_in_subprocess_large_payload_returns_without_deadlock() -> None:
    result = _run_worker_in_subprocess(
        worker=_worker_large_payload,
        payload={"shape": [1, 1, 4096, 1024]},  # ~16MiB
        timeout_sec=30,
    )
    assert bool(result.get("ok", False))
    assert tuple(np.asarray(result["value"]).shape) == (1, 1, 4096, 1024)


def test_run_worker_in_subprocess_propagates_worker_error_payload() -> None:
    with pytest.raises(RuntimeError, match="Worker failed"):
        _run_worker_in_subprocess(
            worker=_worker_error,
            payload={"error": "boom"},
            timeout_sec=10,
        )


def test_run_worker_in_subprocess_surfaces_abnormal_exit_without_waiting_full_timeout() -> None:
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="Worker exited abnormally"):
        _run_worker_in_subprocess(
            worker=_worker_abort,
            payload={},
            timeout_sec=30,
        )
    elapsed = float(time.monotonic() - start)
    assert elapsed < 8.0
