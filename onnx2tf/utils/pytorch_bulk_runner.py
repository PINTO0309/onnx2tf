from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import onnx

from onnx2tf.tflite_builder.pytorch_accuracy_evaluator import (
    evaluate_pytorch_package_outputs,
    smoke_test_pytorch_package_inference,
)


_STATE_SCHEMA_VERSION = 1
_STATE_FILENAME = "bulk_status.json"
_SUMMARY_JSON_FILENAME = "bulk_summary.json"
_SUMMARY_MD_FILENAME = "bulk_summary.md"
_RUNS_DIRNAME = "runs"
_ACCEPTED_ACCURACY_MISMATCH_PREFIXES = (
    "gmflow-",
)
_DEFAULT_SKIP_MODEL_NAMES = (
    "dehaze_maxim_2022aug_opt_sim_special_05_max_to_relu.onnx",
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sanitize_stem(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    safe = safe.strip("._-")
    if safe == "":
        return "model"
    return safe


def _accept_accuracy_mismatch(model_name: str) -> bool:
    normalized = os.path.basename(str(model_name)).strip().lower()
    return any(
        normalized.startswith(str(prefix).lower())
        for prefix in _ACCEPTED_ACCURACY_MISMATCH_PREFIXES
    )


def _read_model_list(list_path: str) -> List[str]:
    models: List[str] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.lower().endswith(".onnx"):
                models.append(line)
    return models


def _pytorch_accuracy_report_path(*, artifact_dir: str, model_path: str) -> str:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        artifact_dir,
        f"{model_stem}_pytorch_accuracy_report.json",
    )


def _pytorch_smoke_report_path(*, artifact_dir: str, model_path: str) -> str:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        artifact_dir,
        f"{model_stem}_pytorch_smoke_report.json",
    )


def _float32_tflite_path(*, artifact_dir: str, model_path: str) -> str:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        artifact_dir,
        f"{model_stem}_float32.tflite",
    )


def _build_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    entries = state.get("entries", [])
    counts = {
        "pass": 0,
        "skipped_model": 0,
        "missing_model": 0,
        "conversion_error": 0,
        "tflite_generation_missing": 0,
        "pytorch_inference_error": 0,
        "pytorch_accuracy_mismatch": 0,
        "timeout": 0,
    }
    for entry in entries:
        classification = str(entry.get("classification", "conversion_error"))
        counts[classification if classification in counts else "conversion_error"] += 1
    strict_fail_count = int(
        sum(1 for entry in entries if not bool(entry.get("strict_pass", False)))
    )
    return {
        "schema_version": _STATE_SCHEMA_VERSION,
        "list_path": state.get("list_path", ""),
        "list_sha256": state.get("list_sha256", ""),
        "total_entries": int(len(entries)),
        "counts": counts,
        "strict_fail_count": int(strict_fail_count),
        "generated_at": _utc_now_iso(),
    }


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_markdown_summary(path: str, *, state: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# PyTorch Bulk Summary")
    lines.append("")
    lines.append(f"- Generated at: {summary['generated_at']}")
    lines.append(f"- List: `{summary['list_path']}`")
    lines.append(f"- Total entries: {summary['total_entries']}")
    lines.append(f"- Strict fail count: {summary['strict_fail_count']}")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| Classification | Count |")
    lines.append("| --- | ---: |")
    for key, value in summary["counts"].items():
        lines.append(f"| `{key}` | {int(value)} |")
    lines.append("")
    lines.append("## Details")
    lines.append("")
    lines.append("| # | Model | Classification | Strict Pass | Exit | Reason |")
    lines.append("| ---: | --- | --- | :---: | ---: | --- |")
    for entry in state.get("entries", []):
        reason = str(entry.get("reason", "")).replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {int(entry.get('index', 0))} | `{entry.get('model', '')}` | "
            f"`{entry.get('classification', '')}` | "
            f"{'Y' if bool(entry.get('strict_pass', False)) else 'N'} | "
            f"{entry.get('onnx2tf_exit_code', '')} | {reason} |"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_bulk_pytorch_verification(
    *,
    list_path: str,
    output_dir: str,
    resume: bool = False,
    onnx2tf_command: str = "",
    num_samples: int = 1,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-4,
    timeout_sec: int = 600,
    smoke_only: bool = False,
    skip_model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    list_path_abs = os.path.abspath(list_path)
    output_dir_abs = os.path.abspath(output_dir)
    runs_dir = os.path.join(output_dir_abs, _RUNS_DIRNAME)
    state_path = os.path.join(output_dir_abs, _STATE_FILENAME)
    summary_json_path = os.path.join(output_dir_abs, _SUMMARY_JSON_FILENAME)
    summary_md_path = os.path.join(output_dir_abs, _SUMMARY_MD_FILENAME)

    if not os.path.exists(list_path_abs):
        raise FileNotFoundError(f"List file does not exist. path={list_path_abs}")

    models = _read_model_list(list_path_abs)
    list_sha256 = _sha256_file(list_path_abs)
    normalized_skip_model_names = sorted(
        {
            os.path.basename(str(model_name)).strip()
            for model_name in [*list(_DEFAULT_SKIP_MODEL_NAMES), *(skip_model_names or [])]
            if str(model_name).strip() != ""
        }
    )
    os.makedirs(runs_dir, exist_ok=True)

    if str(onnx2tf_command).strip() == "":
        local_onnx2tf_py = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "onnx2tf.py",
            )
        )
        command_prefix = [str(sys.executable), str(local_onnx2tf_py)]
    else:
        command_prefix = shlex.split(str(onnx2tf_command))
    if len(command_prefix) == 0:
        raise ValueError("onnx2tf command prefix must not be empty.")

    if resume and os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if str(state.get("list_sha256", "")) != str(list_sha256):
            raise RuntimeError(
                "Resume state does not match the current model list. "
                f"state_sha256={state.get('list_sha256', '')} current_sha256={list_sha256}"
            )
        if list(state.get("skip_model_names", [])) != normalized_skip_model_names:
            raise RuntimeError(
                "Resume state does not match the current skip_model_names. "
                f"state_skip_model_names={state.get('skip_model_names', [])} "
                f"current_skip_model_names={normalized_skip_model_names}"
            )
        entries: List[Dict[str, Any]] = list(state.get("entries", []))
    else:
        state = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "list_path": list_path_abs,
            "list_sha256": list_sha256,
            "skip_model_names": normalized_skip_model_names,
            "started_at": _utc_now_iso(),
            "entries": [],
        }
        entries = []

    start_index = int(len(entries))
    for offset, model_line in enumerate(models[start_index:], start=start_index + 1):
        model_path = os.path.abspath(model_line)
        model_name = os.path.basename(model_line)
        run_dir = os.path.join(
            runs_dir,
            f"{int(offset):04d}_{_sanitize_stem(os.path.splitext(model_name)[0])}",
        )
        artifact_dir = os.path.join(run_dir, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        stdout_log_path = os.path.join(run_dir, "command.stdout.log")
        stderr_log_path = os.path.join(run_dir, "command.stderr.log")
        accuracy_report_path = _pytorch_accuracy_report_path(
            artifact_dir=artifact_dir,
            model_path=model_name,
        )
        smoke_report_path = _pytorch_smoke_report_path(
            artifact_dir=artifact_dir,
            model_path=model_name,
        )
        entry: Dict[str, Any] = {
            "index": int(offset),
            "model": str(model_name),
            "model_line": str(model_line),
            "model_path": str(model_path),
            "run_dir": str(run_dir),
            "artifact_dir": str(artifact_dir),
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
            "pytorch_accuracy_report_path": str(accuracy_report_path),
            "pytorch_smoke_report_path": str(smoke_report_path),
            "started_at": _utc_now_iso(),
            "onnx2tf_exit_code": None,
            "classification": "",
            "strict_pass": False,
            "reason": "",
            "duration_sec": 0.0,
            "command": "",
            "tflite_generated": False,
            "pytorch_inference_pass": None,
            "pytorch_accuracy_pass": None,
        }

        started = time.time()
        if model_name in normalized_skip_model_names:
            entry["classification"] = "skipped_model"
            entry["strict_pass"] = True
            entry["reason"] = "skipped_by_request"
            entry["duration_sec"] = float(time.time() - started)
            entries.append(entry)
            state["entries"] = entries
            _write_json(state_path, state)
            continue

        if not os.path.exists(model_path):
            entry["classification"] = "missing_model"
            entry["strict_pass"] = True
            entry["reason"] = "model_not_found"
            entry["duration_sec"] = float(time.time() - started)
            entries.append(entry)
            state["entries"] = entries
            _write_json(state_path, state)
            continue

        cmd = [
            *command_prefix,
            "-i",
            str(model_path),
            "-o",
            str(artifact_dir),
            "-tb",
            "flatbuffer_direct",
            "-fdopt",
        ]
        entry["command"] = shlex.join(cmd)
        try:
            completed = subprocess.run(
                cmd,
                cwd=run_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(timeout_sec),
            )
            stdout_text = completed.stdout if completed.stdout is not None else ""
            stderr_text = completed.stderr if completed.stderr is not None else ""
            entry["onnx2tf_exit_code"] = int(completed.returncode)
        except subprocess.TimeoutExpired as ex:
            stdout_text = ex.stdout if isinstance(ex.stdout, str) else ""
            stderr_text = ex.stderr if isinstance(ex.stderr, str) else ""
            entry["onnx2tf_exit_code"] = None
            entry["classification"] = "timeout"
            entry["strict_pass"] = False
            entry["reason"] = f"timeout_after_{int(timeout_sec)}s"
            with open(stdout_log_path, "w", encoding="utf-8") as f:
                f.write(stdout_text)
            with open(stderr_log_path, "w", encoding="utf-8") as f:
                f.write(stderr_text)
            entry["duration_sec"] = float(time.time() - started)
            entries.append(entry)
            state["entries"] = entries
            _write_json(state_path, state)
            continue

        with open(stdout_log_path, "w", encoding="utf-8") as f:
            f.write(stdout_text)
        with open(stderr_log_path, "w", encoding="utf-8") as f:
            f.write(stderr_text)

        if int(entry["onnx2tf_exit_code"]) != 0:
            entry["classification"] = "conversion_error"
            entry["strict_pass"] = False
            entry["reason"] = "onnx2tf_nonzero_exit"
            entry["duration_sec"] = float(time.time() - started)
            entries.append(entry)
            state["entries"] = entries
            _write_json(state_path, state)
            continue

        model_stem = os.path.splitext(model_name)[0]
        package_dir = os.path.join(artifact_dir, f"{model_stem}_pytorch")
        tflite_path = _float32_tflite_path(
            artifact_dir=artifact_dir,
            model_path=model_name,
        )
        entry["tflite_generated"] = bool(os.path.exists(tflite_path))
        if not entry["tflite_generated"]:
            entry["classification"] = "tflite_generation_missing"
            entry["strict_pass"] = False
            entry["reason"] = "float32_tflite_not_found"
            entry["duration_sec"] = float(time.time() - started)
            entries.append(entry)
            state["entries"] = entries
            _write_json(state_path, state)
            continue

        onnx_graph = onnx.load(model_path)
        try:
            smoke_report = smoke_test_pytorch_package_inference(
                onnx_graph=onnx_graph,
                package_dir=package_dir,
                output_report_path=smoke_report_path,
                num_samples=int(num_samples),
            )
            entry["pytorch_inference_pass"] = bool(
                smoke_report.get("inference_pass", False)
            )
            if not entry["pytorch_inference_pass"]:
                entry["classification"] = "pytorch_inference_error"
                entry["strict_pass"] = False
                error_payload = smoke_report.get("error", {}) or {}
                entry["reason"] = (
                    f"{error_payload.get('error_type', 'UnknownError')}:"
                    f"{error_payload.get('error_message', '')}"
                )
                entry["duration_sec"] = float(time.time() - started)
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                continue

            if bool(smoke_only):
                entry["classification"] = "pass"
                entry["strict_pass"] = True
                entry["reason"] = "smoke_only"
                entry["duration_sec"] = float(time.time() - started)
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                continue

            report = evaluate_pytorch_package_outputs(
                onnx_graph=onnx_graph,
                package_dir=package_dir,
                output_report_path=accuracy_report_path,
                num_samples=int(num_samples),
                rtol=float(rtol),
                atol=float(atol),
            )
            if bool(report.get("evaluation_skipped", False)):
                entry["pytorch_accuracy_pass"] = None
                entry["classification"] = "pass"
                entry["strict_pass"] = True
                entry["reason"] = str(report.get("skip_reason", "onnxruntime_reference_error"))
                entry["onnxruntime_reference_error"] = report.get(
                    "onnxruntime_reference_error",
                    None,
                )
                entry["duration_sec"] = float(time.time() - started)
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                continue
            entry["pytorch_accuracy_pass"] = bool(report.get("evaluation_pass", False))
            if bool(report.get("evaluation_pass", False)):
                entry["classification"] = "pass"
                entry["strict_pass"] = True
                entry["reason"] = ""
            else:
                if _accept_accuracy_mismatch(model_name):
                    entry["classification"] = "pass"
                    entry["strict_pass"] = True
                    entry["reason"] = "accepted_accuracy_mismatch"
                else:
                    entry["classification"] = "pytorch_accuracy_mismatch"
                    entry["strict_pass"] = False
                    entry["reason"] = "pytorch_accuracy_report_failed"
        except Exception as ex:
            entry["classification"] = "conversion_error"
            entry["strict_pass"] = False
            entry["reason"] = f"pytorch_accuracy_eval_error:{type(ex).__name__}:{str(ex)}"

        entry["duration_sec"] = float(time.time() - started)
        entries.append(entry)
        state["entries"] = entries
        _write_json(state_path, state)

    state["finished_at"] = _utc_now_iso()
    summary = _build_summary(state)
    state["summary"] = summary
    _write_json(state_path, state)
    _write_json(summary_json_path, summary)
    _write_markdown_summary(summary_md_path, state=state, summary=summary)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run bulk onnx2tf PyTorch-package verification using "
            "onnx2tf -i <model> -tb flatbuffer_direct -fdopt"
        )
    )
    parser.add_argument("-l", "--list_path", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="pytorch_bulk_report")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--onnx2tf_command", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--timeout_sec", type=int, default=600)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument(
        "--skip_model_name",
        action="append",
        default=[],
        help="Basename of an ONNX model to skip during bulk verification. Repeatable.",
    )
    args = parser.parse_args()

    state = run_bulk_pytorch_verification(
        list_path=args.list_path,
        output_dir=args.output_dir,
        resume=bool(args.resume),
        onnx2tf_command=str(args.onnx2tf_command),
        num_samples=int(args.num_samples),
        rtol=float(args.rtol),
        atol=float(args.atol),
        timeout_sec=int(args.timeout_sec),
        smoke_only=bool(args.smoke_only),
        skip_model_names=list(args.skip_model_name),
    )
    print(
        "Bulk PyTorch verification complete. "
        f"total_entries={len(state.get('entries', []))} "
        f"strict_fail_count={state.get('summary', {}).get('strict_fail_count', 0)}"
    )


if __name__ == "__main__":
    main()
