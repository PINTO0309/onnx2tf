#!/usr/bin/env python3
"""Benchmark TFLite inference latency on CPU/CUDA backends."""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _discover_tflite_models(root: Path) -> List[Path]:
    if not root.exists():
        return []
    if root.is_file() and root.suffix.lower() == ".tflite":
        return [root.resolve()]
    found = sorted({p.resolve() for p in root.rglob("*.tflite")})
    return found


def _resolve_shape(
    shape: Sequence[Any],
    shape_signature: Optional[Sequence[Any]],
    dynamic_dim: int,
) -> List[int]:
    raw = list(shape)
    sig = list(shape_signature) if shape_signature is not None else None
    if sig is not None and len(sig) == len(raw):
        raw = list(sig)

    resolved: List[int] = []
    for dim in raw:
        if dim is None:
            resolved.append(int(dynamic_dim))
            continue
        value = int(dim)
        resolved.append(int(dynamic_dim) if value <= 0 else value)

    if len(resolved) == 0:
        # Scalar tensors are represented as rank-0 for runtime input feeds.
        return []
    return resolved


def _build_input_array(
    *,
    detail: Dict[str, Any],
    rng: np.random.Generator,
    dynamic_dim: int,
) -> np.ndarray:
    shape = _resolve_shape(
        shape=detail.get("shape", []),
        shape_signature=detail.get("shape_signature", None),
        dynamic_dim=dynamic_dim,
    )
    np_dtype = np.dtype(detail["dtype"])

    if np.issubdtype(np_dtype, np.floating):
        return rng.standard_normal(size=shape).astype(np_dtype, copy=False)
    if np.issubdtype(np_dtype, np.integer):
        # Use a narrow integer range to avoid accidental overflow in int8/uint8.
        return rng.integers(low=0, high=10, size=shape, dtype=np_dtype)
    if np.issubdtype(np_dtype, np.bool_):
        return rng.integers(low=0, high=2, size=shape, dtype=np.int32).astype(np.bool_)

    raise ValueError(
        f"Unsupported input dtype for benchmark: dtype={np_dtype} input={detail.get('name', '')}"
    )


def _materialize_outputs(outputs: Any) -> None:
    if isinstance(outputs, dict):
        values = list(outputs.values())
    elif isinstance(outputs, (list, tuple)):
        values = list(outputs)
    else:
        values = [outputs]

    for value in values:
        _ = np.asarray(value)


def _select_signature_key(
    *,
    signature_list: Dict[str, Any],
    requested: str,
) -> Optional[str]:
    if not signature_list:
        return None
    if requested in signature_list:
        return str(requested)
    if "serving_default" in signature_list:
        return "serving_default"
    return str(next(iter(signature_list.keys())))


def _try_create_gpu_delegate() -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    from ai_edge_litert.interpreter import load_delegate  # Delayed import.

    candidate_lib_names = [
        "libtensorflowlite_gpu_delegate.so",
        "libtensorflowlite_gpu_delegate.so.2",
        "libtensorflowlite_gpu_delegate.dylib",
        "tensorflowlite_gpu_delegate.dll",
    ]
    candidate_libs: List[str] = []
    seen: set[str] = set()

    env_delegate = os.environ.get("TFLITE_GPU_DELEGATE_LIBRARY", "").strip()
    if env_delegate != "":
        candidate_lib_names = [env_delegate] + candidate_lib_names

    # First, collect resolvable concrete paths to avoid invoking load_delegate
    # on obviously missing libraries.
    search_dirs = [
        p
        for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if p.strip() != ""
    ] + [
        "/usr/lib",
        "/usr/local/lib",
        "/usr/lib64",
        "/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    for lib_name in candidate_lib_names:
        if os.path.isabs(lib_name):
            if os.path.exists(lib_name) and lib_name not in seen:
                candidate_libs.append(lib_name)
                seen.add(lib_name)
            continue

        found_path = ctypes.util.find_library(lib_name)
        if found_path is not None:
            if found_path not in seen:
                candidate_libs.append(found_path)
                seen.add(found_path)
            continue

        for d in search_dirs:
            full_path = os.path.join(d, lib_name)
            if os.path.exists(full_path) and full_path not in seen:
                candidate_libs.append(full_path)
                seen.add(full_path)
                break

    if len(candidate_libs) == 0:
        return (
            None,
            None,
            "TFLite GPU delegate library was not found. "
            f"checked={candidate_lib_names}",
        )

    errors: List[str] = []

    for lib in candidate_libs:
        try:
            delegate = load_delegate(lib)
            return delegate, lib, None
        except Exception as ex:
            errors.append(f"{lib}: {ex}")

    return None, None, "; ".join(errors)


def _benchmark_single_model(
    *,
    Interpreter: Any,
    model_path: Path,
    backend_label: str,
    signature_key: str,
    warmup: int,
    runs: int,
    dynamic_dim: int,
    seed: int,
    num_threads: Optional[int],
    use_gpu_delegate: bool,
) -> Dict[str, Any]:
    interpreter_kwargs: Dict[str, Any] = {
        "model_path": str(model_path),
    }
    if num_threads is not None:
        interpreter_kwargs["num_threads"] = int(num_threads)

    delegate_name: Optional[str] = None
    if use_gpu_delegate:
        delegate, delegate_name, delegate_error = _try_create_gpu_delegate()
        if delegate is None:
            raise RuntimeError(
                "CUDA backend requested, but TFLite GPU delegate could not be loaded. "
                f"detail={delegate_error}"
            )
        interpreter_kwargs["experimental_delegates"] = [delegate]

    interpreter = Interpreter(**interpreter_kwargs)

    selected_signature: Optional[str] = None
    invoke_mode = "invoke"
    signature_list = interpreter.get_signature_list()
    selected_signature = _select_signature_key(
        signature_list=dict(signature_list),
        requested=str(signature_key),
    )

    rng = np.random.default_rng(seed=seed)

    def _progress(iterable: Any, desc: str) -> Any:
        try:
            if len(iterable) == 0:
                return iterable
        except Exception:
            pass
        if tqdm is None:
            return iterable
        return tqdm(
            iterable,
            desc=desc,
            unit="iter",
            leave=False,
            dynamic_ncols=True,
        )

    warmup_desc = f"{backend_label}:{model_path.name}:warmup"
    run_desc = f"{backend_label}:{model_path.name}:benchmark"

    if selected_signature is not None:
        full_signature_list = interpreter._get_full_signature_list()
        signature_meta = dict(full_signature_list[selected_signature])
        input_index_map = dict(signature_meta.get("inputs", {}))
        tensor_details = {
            int(d["index"]): d
            for d in list(interpreter.get_tensor_details())
        }
        signature_inputs: Dict[str, np.ndarray] = {}
        for input_name, tensor_index in input_index_map.items():
            detail = tensor_details[int(tensor_index)]
            signature_inputs[str(input_name)] = _build_input_array(
                detail=detail,
                rng=rng,
                dynamic_dim=dynamic_dim,
            )

        runner = interpreter.get_signature_runner(selected_signature)
        invoke_mode = "signature_runner"

        for _ in _progress(range(int(warmup)), warmup_desc):
            outputs = runner(**signature_inputs)
            _materialize_outputs(outputs)

        elapsed_ms: List[float] = []
        for _ in _progress(range(int(runs)), run_desc):
            t0 = time.perf_counter()
            outputs = runner(**signature_inputs)
            _materialize_outputs(outputs)
            elapsed_ms.append((time.perf_counter() - t0) * 1000.0)
    else:
        input_details = list(interpreter.get_input_details())
        if len(input_details) == 0:
            raise RuntimeError("No TFLite inputs found.")

        input_arrays: Dict[int, np.ndarray] = {}
        resize_specs: List[Tuple[int, List[int]]] = []
        for detail in input_details:
            arr = _build_input_array(
                detail=detail,
                rng=rng,
                dynamic_dim=dynamic_dim,
            )
            input_arrays[int(detail["index"])] = arr
            target_shape = [int(v) for v in list(arr.shape)]
            current_shape = [int(v) for v in list(detail.get("shape", []))]
            if target_shape != current_shape:
                resize_specs.append((int(detail["index"]), target_shape))

        if len(resize_specs) > 0:
            for tensor_index, target_shape in resize_specs:
                interpreter.resize_tensor_input(
                    tensor_index=tensor_index,
                    tensor_size=target_shape,
                    strict=False,
                )

        interpreter.allocate_tensors()
        output_details = list(interpreter.get_output_details())

        def _invoke_once() -> None:
            for tensor_index, array in input_arrays.items():
                interpreter.set_tensor(tensor_index, array)
            interpreter.invoke()
            for detail in output_details:
                _ = interpreter.get_tensor(int(detail["index"]))

        for _ in _progress(range(int(warmup)), warmup_desc):
            _invoke_once()

        elapsed_ms = []
        for _ in _progress(range(int(runs)), run_desc):
            t0 = time.perf_counter()
            _invoke_once()
            elapsed_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "model_path": str(model_path),
        "signature": (str(selected_signature) if selected_signature is not None else "-"),
        "mode": str(invoke_mode),
        "delegate": (str(delegate_name) if delegate_name is not None else "-"),
        "runs": int(runs),
        "warmup": int(warmup),
        "mean_ms": float(statistics.fmean(elapsed_ms)),
        "median_ms": float(statistics.median(elapsed_ms)),
        "p90_ms": float(np.percentile(np.asarray(elapsed_ms, dtype=np.float64), 90.0)),
        "min_ms": float(np.min(elapsed_ms)),
        "max_ms": float(np.max(elapsed_ms)),
    }


def _run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    if args.backend == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.backend == "cuda" and args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    from ai_edge_litert.interpreter import Interpreter  # Delayed import to honor backend env selection.

    model_paths = _discover_tflite_models(Path(args.model_root))
    if not model_paths:
        return {
            "backend": str(args.backend),
            "available": args.backend == "cpu",
            "error": f"No .tflite found under: {args.model_root}",
            "results": [],
        }

    backend_available = True
    backend_error: Optional[str] = None
    if args.backend == "cuda":
        # Probe delegate load once to report availability up-front.
        delegate, _, delegate_error = _try_create_gpu_delegate()
        if delegate is None:
            backend_available = False
            backend_error = (
                "CUDA backend requested but TFLite GPU delegate is unavailable. "
                f"detail={delegate_error}"
            )

    if not backend_available:
        return {
            "backend": str(args.backend),
            "available": False,
            "error": str(backend_error),
            "results": [],
        }

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for model_path in model_paths:
        try:
            result = _benchmark_single_model(
                Interpreter=Interpreter,
                model_path=model_path,
                backend_label=str(args.backend).upper(),
                signature_key=str(args.signature),
                warmup=int(args.warmup),
                runs=int(args.runs),
                dynamic_dim=int(args.dynamic_dim),
                seed=int(args.seed),
                num_threads=(int(args.num_threads) if args.num_threads is not None else None),
                use_gpu_delegate=(str(args.backend) == "cuda"),
            )
            results.append(result)
        except Exception as ex:
            failures.append({"model_path": str(model_path), "reason": str(ex)})

    return {
        "backend": str(args.backend),
        "available": True,
        "device": ("GPU" if args.backend == "cuda" else "CPU"),
        "model_root": str(Path(args.model_root).resolve()),
        "results": results,
        "failures": failures,
    }


def _print_result_block(payload: Dict[str, Any]) -> None:
    backend = str(payload.get("backend", "unknown")).upper()
    print(f"### {backend}")
    if payload.get("error"):
        print()
        print(f"- error: {payload['error']}")
        print()
        return

    results = list(payload.get("results", []))
    failures = list(payload.get("failures", []))
    if not results:
        print()
        print("- no benchmark results.")
        if failures:
            print()
            print("| failure_model_path | reason |")
            print("|---|---|")
            for f in failures:
                print(f"| {f.get('model_path')} | {f.get('reason')} |")
        print()
        return

    print()
    print("| model_path | mode | signature | delegate | runs | mean_ms | median_ms | p90_ms | min_ms | max_ms |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for item in results:
        print(
            "| {model_path} | {mode} | {signature} | {delegate} | {runs} | {mean_ms:.3f} | {median_ms:.3f} | {p90_ms:.3f} | {min_ms:.3f} | {max_ms:.3f} |".format(
                model_path=item["model_path"],
                mode=item["mode"],
                signature=item["signature"],
                delegate=item["delegate"],
                runs=int(item["runs"]),
                mean_ms=float(item["mean_ms"]),
                median_ms=float(item["median_ms"]),
                p90_ms=float(item["p90_ms"]),
                min_ms=float(item["min_ms"]),
                max_ms=float(item["max_ms"]),
            )
        )

    if failures:
        print()
        print("| failure_model_path | reason |")
        print("|---|---|")
        for f in failures:
            print(f"| {f.get('model_path')} | {f.get('reason')} |")
    print()


def _run_subprocess_worker(args: argparse.Namespace, backend: str) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.model_root),
        "--backend",
        str(backend),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--dynamic-dim",
        str(args.dynamic_dim),
        "--signature",
        str(args.signature),
        "--seed",
        str(args.seed),
        "--worker-json",
    ]
    if args.num_threads is not None:
        cmd.extend(["--num-threads", str(args.num_threads)])
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", str(args.cuda_visible_devices)])

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "backend": str(backend),
            "error": f"worker exited with code {proc.returncode}",
            "results": [],
        }
    stdout = proc.stdout.strip()
    if not stdout:
        return {
            "backend": str(backend),
            "error": "worker produced no JSON output.",
            "results": [],
        }
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as ex:
        return {
            "backend": str(backend),
            "error": f"failed to decode worker JSON: {ex}; raw={stdout[:500]}",
            "results": [],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TFLite inference latency (ms) on CPU/CUDA backends."
    )
    parser.add_argument(
        "model_root",
        type=str,
        help="Path to a .tflite file or a parent directory containing .tflite files.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cpu", "cuda", "both"],
        default="both",
        help="Benchmark backend selection. Default: both",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per model. Default: 10")
    parser.add_argument("--runs", type=int, default=100, help="Measured iterations per model. Default: 100")
    parser.add_argument(
        "--dynamic-dim",
        type=int,
        default=1,
        help="Value used for dynamic input dimensions (None/-1). Default: 1",
    )
    parser.add_argument(
        "--signature",
        type=str,
        default="serving_default",
        help="Signature key to run. Falls back to available signatures, then invoke mode.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dummy input generation.")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Set TFLite interpreter num_threads. Default: runtime default.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Override CUDA_VISIBLE_DEVICES for CUDA backend (e.g., '0').",
    )
    parser.add_argument(
        "--worker-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.warmup < 0 or args.runs <= 0 or args.dynamic_dim <= 0:
        print("warmup must be >=0, runs must be >0, dynamic-dim must be >0", file=sys.stderr)
        return 2
    if args.num_threads is not None and int(args.num_threads) <= 0:
        print("num-threads must be >0 when specified", file=sys.stderr)
        return 2

    if args.worker_json:
        payload = _run_worker(args)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.backend == "both":
        cpu_payload = _run_subprocess_worker(args, backend="cpu")
        cuda_payload = _run_subprocess_worker(args, backend="cuda")
        _print_result_block(cpu_payload)
        _print_result_block(cuda_payload)
        return 0

    payload = _run_worker(args)
    _print_result_block(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
