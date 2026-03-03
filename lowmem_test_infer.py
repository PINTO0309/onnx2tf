#!/usr/bin/env python3

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run low-memory random test inference for ONNX and/or TFLite models "
            "in isolated worker processes."
        )
    )
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model.")
    parser.add_argument("--tflite", type=str, default=None, help="Path to TFLite model.")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "onnx", "tflite", "both"],
        help="Backend to run. auto: use whichever path(s) are provided.",
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size used when the first dimension is dynamic.",
    )
    parser.add_argument(
        "--dynamic_dim_value",
        type=int,
        default=1,
        help="Fallback value for dynamic dims except batch axis.",
    )
    parser.add_argument(
        "--output_limit",
        type=int,
        default=1,
        help="Number of outputs to fetch/summarize per backend. <=0 means all.",
    )
    parser.add_argument(
        "--fetch_tflite_outputs",
        action="store_true",
        help="Fetch and summarize TFLite outputs after invoke(). Disabled by default to reduce copies.",
    )
    parser.add_argument(
        "--ort_threads",
        type=int,
        default=1,
        help="ONNX Runtime intra-op threads.",
    )
    parser.add_argument(
        "--timeout_sec",
        type=int,
        default=300,
        help="Per-worker timeout in seconds.",
    )
    return parser.parse_args()


def _resolve_backends(args: argparse.Namespace) -> List[str]:
    if args.backend == "onnx":
        return ["onnx"]
    if args.backend == "tflite":
        return ["tflite"]
    if args.backend == "both":
        return ["onnx", "tflite"]

    backends: List[str] = []
    if args.onnx:
        backends.append("onnx")
    if args.tflite:
        backends.append("tflite")
    return backends


def _resolve_shape(
    raw_shape: Sequence[Any],
    *,
    batch_size: int,
    dynamic_dim_value: int,
) -> List[int]:
    shape: List[int] = []
    for axis, dim in enumerate(raw_shape):
        if isinstance(dim, (int, np.integer)) and int(dim) > 0:
            shape.append(int(dim))
            continue
        if axis == 0:
            shape.append(int(batch_size))
        else:
            shape.append(int(dynamic_dim_value))
    return shape


def _ort_dtype_to_numpy(dtype_str: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int8)": np.int8,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(uint32)": np.uint32,
        "tensor(uint64)": np.uint64,
        "tensor(bool)": np.bool_,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported ORT input dtype: {dtype_str}")
    return np.dtype(mapping[dtype_str])


def _generate_input(
    *,
    shape: Sequence[int],
    dtype: np.dtype,
    rng: np.random.Generator,
) -> np.ndarray:
    np_dtype = np.dtype(dtype)
    if np_dtype == np.bool_:
        return rng.integers(0, 2, size=shape, dtype=np.uint8).astype(np.bool_)
    if np.issubdtype(np_dtype, np.floating):
        return rng.standard_normal(size=shape).astype(np_dtype)
    if np.issubdtype(np_dtype, np.signedinteger):
        return rng.integers(-2, 3, size=shape, dtype=np.int64).astype(np_dtype)
    if np.issubdtype(np_dtype, np.unsignedinteger):
        return rng.integers(0, 5, size=shape, dtype=np.uint64).astype(np_dtype)
    raise ValueError(f"Unsupported dtype for random input generation: {np_dtype}")


def _summarize_array(name: str, arr: np.ndarray) -> Dict[str, Any]:
    value = np.asarray(arr)
    summary: Dict[str, Any] = {
        "name": str(name),
        "shape": [int(v) for v in value.shape],
        "dtype": str(value.dtype),
        "numel": int(value.size),
        "bytes": int(value.nbytes),
    }
    if np.issubdtype(value.dtype, np.number):
        finite = np.isfinite(value)
        summary["finite_ratio"] = float(np.mean(finite)) if value.size > 0 else 1.0
        if value.size > 0 and np.any(finite):
            finite_values = value[finite]
            summary["min"] = float(np.min(finite_values))
            summary["max"] = float(np.max(finite_values))
    return summary


def _select_names(names: Sequence[str], limit: int) -> List[str]:
    if int(limit) <= 0:
        return [str(v) for v in names]
    return [str(v) for v in names[: int(limit)]]


def _onnx_worker(payload: Dict[str, Any], result_queue: mp.Queue) -> None:
    try:
        import onnxruntime as ort

        model_path = str(payload["model_path"])
        batch_size = int(payload["batch_size"])
        dynamic_dim_value = int(payload["dynamic_dim_value"])
        seed = int(payload["seed"])
        output_limit = int(payload["output_limit"])
        ort_threads = max(1, int(payload["ort_threads"]))

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = ort_threads
        sess_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        rng = np.random.default_rng(seed)
        feeds: Dict[str, np.ndarray] = {}
        input_summaries: List[Dict[str, Any]] = []
        for inp in session.get_inputs():
            shape = _resolve_shape(
                inp.shape,
                batch_size=batch_size,
                dynamic_dim_value=dynamic_dim_value,
            )
            np_dtype = _ort_dtype_to_numpy(str(inp.type))
            tensor = _generate_input(shape=shape, dtype=np_dtype, rng=rng)
            feeds[str(inp.name)] = tensor
            input_summaries.append(_summarize_array(str(inp.name), tensor))

        output_names = _select_names([str(v.name) for v in session.get_outputs()], output_limit)
        started = time.perf_counter()
        outputs = session.run(output_names, feeds)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        output_summaries = [
            _summarize_array(name, value)
            for name, value in zip(output_names, outputs)
        ]

        result_queue.put(
            {
                "ok": True,
                "backend": "onnx",
                "latency_ms": float(elapsed_ms),
                "inputs": input_summaries,
                "outputs": output_summaries,
            }
        )
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "backend": "onnx",
                "error": traceback.format_exc(),
            }
        )


def _create_tflite_interpreter(model_path: str):
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(model_path=model_path)
    except Exception:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=model_path)


def _tflite_worker(payload: Dict[str, Any], result_queue: mp.Queue) -> None:
    try:
        model_path = str(payload["model_path"])
        batch_size = int(payload["batch_size"])
        dynamic_dim_value = int(payload["dynamic_dim_value"])
        seed = int(payload["seed"])
        output_limit = int(payload["output_limit"])
        fetch_outputs = bool(payload["fetch_outputs"])

        interpreter = _create_tflite_interpreter(model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()

        resized = False
        for detail in input_details:
            shape_signature = detail.get("shape_signature", detail.get("shape"))
            if shape_signature is None:
                continue
            resolved_shape = _resolve_shape(
                list(shape_signature),
                batch_size=batch_size,
                dynamic_dim_value=dynamic_dim_value,
            )
            current_shape = [int(v) for v in np.asarray(detail.get("shape", resolved_shape)).tolist()]
            if current_shape != resolved_shape:
                interpreter.resize_tensor_input(
                    int(detail["index"]),
                    resolved_shape,
                    strict=False,
                )
                resized = True

        if resized:
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()

        rng = np.random.default_rng(seed)
        input_summaries: List[Dict[str, Any]] = []
        for detail in input_details:
            shape = [int(v) for v in np.asarray(detail["shape"]).tolist()]
            dtype = np.dtype(detail["dtype"])
            tensor = _generate_input(shape=shape, dtype=dtype, rng=rng)
            interpreter.set_tensor(int(detail["index"]), tensor)
            input_summaries.append(_summarize_array(str(detail.get("name", "")), tensor))

        started = time.perf_counter()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        output_summaries: List[Dict[str, Any]] = []
        output_details = interpreter.get_output_details()
        selected_output_details = output_details if output_limit <= 0 else output_details[:output_limit]
        if fetch_outputs:
            for detail in selected_output_details:
                value = interpreter.get_tensor(int(detail["index"]))
                output_summaries.append(
                    _summarize_array(str(detail.get("name", "")), value)
                )
        else:
            for detail in selected_output_details:
                output_summaries.append(
                    {
                        "name": str(detail.get("name", "")),
                        "shape": [int(v) for v in np.asarray(detail.get("shape", [])).tolist()],
                        "dtype": str(np.dtype(detail.get("dtype", np.float32))),
                        "numel": int(np.prod(np.asarray(detail.get("shape", [0]), dtype=np.int64))),
                        "bytes": int(
                            np.prod(np.asarray(detail.get("shape", [0]), dtype=np.int64))
                            * np.dtype(detail.get("dtype", np.float32)).itemsize
                        ),
                        "fetched": False,
                    }
                )

        result_queue.put(
            {
                "ok": True,
                "backend": "tflite",
                "latency_ms": float(elapsed_ms),
                "inputs": input_summaries,
                "outputs": output_summaries,
                "fetch_outputs": bool(fetch_outputs),
            }
        )
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "backend": "tflite",
                "error": traceback.format_exc(),
            }
        )


def _run_worker(
    *,
    backend: str,
    payload: Dict[str, Any],
    timeout_sec: int,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    target = _onnx_worker if backend == "onnx" else _tflite_worker
    process = ctx.Process(target=target, args=(payload, queue), daemon=True)
    process.start()
    process.join(timeout=max(1, int(timeout_sec)))

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        return {
            "ok": False,
            "backend": backend,
            "error": f"Worker timed out after {timeout_sec}s.",
        }

    if queue.empty():
        return {
            "ok": False,
            "backend": backend,
            "error": f"Worker exited with code {process.exitcode} and returned no payload.",
        }

    return queue.get()


def _print_result(sample_index: int, result: Dict[str, Any]) -> None:
    backend = str(result.get("backend", ""))
    if not bool(result.get("ok", False)):
        print(f"[sample={sample_index:04d}] [{backend}] FAILED")
        print(result.get("error", "unknown error"))
        return

    latency_ms = float(result.get("latency_ms", 0.0))
    inputs = result.get("inputs", [])
    outputs = result.get("outputs", [])
    total_in_bytes = int(sum(int(v.get("bytes", 0)) for v in inputs if isinstance(v, dict)))
    total_out_bytes = int(sum(int(v.get("bytes", 0)) for v in outputs if isinstance(v, dict)))

    print(
        f"[sample={sample_index:04d}] [{backend}] "
        f"ok latency_ms={latency_ms:.3f} "
        f"input_bytes={total_in_bytes} output_bytes={total_out_bytes}"
    )


def main() -> int:
    args = _parse_args()
    backends = _resolve_backends(args)

    if len(backends) == 0:
        raise ValueError("No backend selected. Provide --onnx and/or --tflite.")
    if "onnx" in backends and (not args.onnx or not os.path.exists(args.onnx)):
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")
    if "tflite" in backends and (not args.tflite or not os.path.exists(args.tflite)):
        raise FileNotFoundError(f"TFLite file not found: {args.tflite}")
    if int(args.num_samples) <= 0:
        raise ValueError(f"num_samples must be > 0. got={args.num_samples}")
    if int(args.batch_size) <= 0:
        raise ValueError(f"batch_size must be > 0. got={args.batch_size}")
    if int(args.dynamic_dim_value) <= 0:
        raise ValueError(f"dynamic_dim_value must be > 0. got={args.dynamic_dim_value}")

    print("lowmem_test_infer started")
    print(f"backends={backends} num_samples={int(args.num_samples)} seed={int(args.seed)}")
    print(
        "config "
        f"batch_size={int(args.batch_size)} "
        f"dynamic_dim_value={int(args.dynamic_dim_value)} "
        f"output_limit={int(args.output_limit)} "
        f"fetch_tflite_outputs={bool(args.fetch_tflite_outputs)}"
    )

    all_results: List[Dict[str, Any]] = []
    started_all = time.perf_counter()

    for sample_index in range(int(args.num_samples)):
        for backend in backends:
            payload: Dict[str, Any] = {
                "batch_size": int(args.batch_size),
                "dynamic_dim_value": int(args.dynamic_dim_value),
                "seed": int(args.seed) + sample_index * 1000 + (0 if backend == "onnx" else 1),
                "output_limit": int(args.output_limit),
            }
            if backend == "onnx":
                payload["model_path"] = str(args.onnx)
                payload["ort_threads"] = int(args.ort_threads)
            else:
                payload["model_path"] = str(args.tflite)
                payload["fetch_outputs"] = bool(args.fetch_tflite_outputs)

            result = _run_worker(
                backend=backend,
                payload=payload,
                timeout_sec=int(args.timeout_sec),
            )
            all_results.append(result)
            _print_result(sample_index, result)

    elapsed_all_ms = (time.perf_counter() - started_all) * 1000.0
    failed = [r for r in all_results if not bool(r.get("ok", False))]
    succeeded = [r for r in all_results if bool(r.get("ok", False))]

    print("summary")
    print(
        f"runs={len(all_results)} ok={len(succeeded)} failed={len(failed)} "
        f"elapsed_ms={elapsed_all_ms:.3f}"
    )

    backend_latency: Dict[str, List[float]] = {}
    for result in succeeded:
        backend = str(result.get("backend", ""))
        backend_latency.setdefault(backend, []).append(float(result.get("latency_ms", 0.0)))
    for backend in sorted(backend_latency.keys()):
        values = backend_latency[backend]
        print(
            f"{backend}: count={len(values)} "
            f"mean_ms={float(np.mean(values)):.3f} "
            f"p95_ms={float(np.percentile(values, 95)):.3f} "
            f"max_ms={float(np.max(values)):.3f}"
        )

    return 1 if len(failed) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
