#!/usr/bin/env python3
"""Benchmark TensorFlow SavedModel inference latency on CPU/CUDA backends."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _discover_saved_models(root: Path) -> List[Path]:
    if not root.exists():
        return []
    if (root / "saved_model.pb").exists():
        return [root.resolve()]
    found = sorted({p.parent.resolve() for p in root.rglob("saved_model.pb")})
    return found


def _resolve_shape(shape: Sequence[Any], dynamic_dim: int) -> List[int]:
    resolved: List[int] = []
    for dim in list(shape):
        if dim is None:
            resolved.append(int(dynamic_dim))
            continue
        value = int(dim)
        resolved.append(int(dynamic_dim) if value < 0 else value)
    return resolved


def _build_input_tensor(tf: Any, spec: Any, rng: np.random.Generator, dynamic_dim: int) -> Any:
    shape = _resolve_shape(spec.shape.as_list(), dynamic_dim=dynamic_dim)
    dtype = spec.dtype
    np_dtype = dtype.as_numpy_dtype

    if dtype.is_floating:
        data = rng.standard_normal(size=shape).astype(np_dtype, copy=False)
    elif dtype.is_integer:
        data = rng.integers(low=0, high=10, size=shape, dtype=np_dtype)
    elif dtype == tf.bool:
        data = rng.integers(low=0, high=2, size=shape, dtype=np.int32).astype(np.bool_)
    elif dtype == tf.string:
        data = np.full(shape=shape, fill_value=b"benchmark", dtype=np.object_)
    else:
        raise ValueError(f"Unsupported input dtype: {dtype!r} (input={spec.name})")

    return tf.convert_to_tensor(data, dtype=dtype)


def _materialize_outputs(tf: Any, outputs: Any) -> None:
    for value in tf.nest.flatten(outputs):
        if hasattr(value, "numpy"):
            _ = value.numpy()


def _select_signature_key(model: Any, requested: str) -> str:
    signatures = getattr(model, "signatures", {})
    if not signatures:
        raise RuntimeError("No signatures found in SavedModel.")
    if requested in signatures:
        return requested
    if "serving_default" in signatures:
        return "serving_default"
    return next(iter(signatures.keys()))


def _benchmark_single_model(
    *,
    tf: Any,
    model_dir: Path,
    backend_label: str,
    signature_key: str,
    warmup: int,
    runs: int,
    dynamic_dim: int,
    seed: int,
    device_name: str,
) -> Dict[str, Any]:
    model = tf.saved_model.load(str(model_dir))
    selected_signature = _select_signature_key(model, requested=signature_key)
    fn = model.signatures[selected_signature]

    positional_specs, keyword_specs = fn.structured_input_signature
    rng = np.random.default_rng(seed=seed)
    args = [
        _build_input_tensor(tf, spec=spec, rng=rng, dynamic_dim=dynamic_dim)
        for spec in list(positional_specs)
    ]
    kwargs = {
        name: _build_input_tensor(tf, spec=spec, rng=rng, dynamic_dim=dynamic_dim)
        for name, spec in dict(keyword_specs).items()
    }

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

    warmup_desc = f"{backend_label}:{model_dir.name}:warmup"
    run_desc = f"{backend_label}:{model_dir.name}:benchmark"

    with tf.device(device_name):
        for _ in _progress(range(int(warmup)), warmup_desc):
            outputs = fn(*args, **kwargs)
            _materialize_outputs(tf, outputs)

        elapsed_ms: List[float] = []
        for _ in _progress(range(int(runs)), run_desc):
            t0 = time.perf_counter()
            outputs = fn(*args, **kwargs)
            _materialize_outputs(tf, outputs)
            elapsed_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "model_dir": str(model_dir),
        "signature": str(selected_signature),
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

    import tensorflow as tf  # Delayed import to honor backend env selection.

    if args.backend == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        device_name = "/CPU:0"
        backend_available = True
    else:
        gpus = tf.config.list_physical_devices("GPU")
        backend_available = len(gpus) > 0
        if backend_available:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        device_name = "/GPU:0"

    model_dirs = _discover_saved_models(Path(args.model_root))
    if not model_dirs:
        return {
            "backend": str(args.backend),
            "available": backend_available,
            "error": f"No saved_model.pb found under: {args.model_root}",
            "results": [],
        }

    if args.backend == "cuda" and not backend_available:
        return {
            "backend": "cuda",
            "available": False,
            "error": "CUDA backend requested but no GPU device is available.",
            "results": [],
        }

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for model_dir in model_dirs:
        try:
            result = _benchmark_single_model(
                tf=tf,
                model_dir=model_dir,
                backend_label=str(args.backend).upper(),
                signature_key=str(args.signature),
                warmup=int(args.warmup),
                runs=int(args.runs),
                dynamic_dim=int(args.dynamic_dim),
                seed=int(args.seed),
                device_name=str(device_name),
            )
            results.append(result)
        except Exception as ex:
            failures.append({"model_dir": str(model_dir), "reason": str(ex)})

    return {
        "backend": str(args.backend),
        "available": backend_available,
        "device": str(device_name),
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
    if not results:
        print()
        print("- no benchmark results.")
        print()
        return

    print()
    print("| model_dir | signature | runs | mean_ms | median_ms | p90_ms | min_ms | max_ms |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for item in results:
        print(
            "| {model_dir} | {signature} | {runs} | {mean_ms:.3f} | {median_ms:.3f} | {p90_ms:.3f} | {min_ms:.3f} | {max_ms:.3f} |".format(
                model_dir=item["model_dir"],
                signature=item["signature"],
                runs=int(item["runs"]),
                mean_ms=float(item["mean_ms"]),
                median_ms=float(item["median_ms"]),
                p90_ms=float(item["p90_ms"]),
                min_ms=float(item["min_ms"]),
                max_ms=float(item["max_ms"]),
            )
        )

    failures = list(payload.get("failures", []))
    if failures:
        print()
        print("| failure_model_dir | reason |")
        print("|---|---|")
        for f in failures:
            print(f"| {f.get('model_dir')} | {f.get('reason')} |")
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
        description="Benchmark TensorFlow SavedModel inference latency (ms) on CPU/CUDA backends."
    )
    parser.add_argument(
        "model_root",
        type=str,
        help="Path to a SavedModel directory or a parent directory containing saved_model.pb files.",
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
        help="Signature key to run. Fallbacks to available signatures if missing.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dummy input generation.")
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
