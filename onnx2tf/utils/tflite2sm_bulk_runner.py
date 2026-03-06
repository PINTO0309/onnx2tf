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


_STATE_SCHEMA_VERSION = 1
_STATE_FILENAME = "bulk_status.json"
_SUMMARY_JSON_FILENAME = "bulk_summary.json"
_SUMMARY_MD_FILENAME = "bulk_summary.md"
_RUNS_DIRNAME = "runs"
_POLICY_SKIPPED_MODEL_STEMS = {
    "deformable_detr_one_input_simple",
}
_POLICY_ACCEPTED_MISMATCH_MODEL_STEMS = {
    "mosaic-9",
}


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


def _should_skip_model(model_line: str) -> bool:
    model_stem = os.path.splitext(os.path.basename(str(model_line)))[0]
    return model_stem in _POLICY_SKIPPED_MODEL_STEMS


def _should_accept_saved_model_tflite_mismatch(model_line: str) -> bool:
    model_stem = os.path.splitext(os.path.basename(str(model_line)))[0]
    return model_stem in _POLICY_ACCEPTED_MISMATCH_MODEL_STEMS


def _saved_model_validation_report_path(*, run_dir: str, model_path: str) -> str:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        run_dir,
        "saved_model",
        f"{model_stem}_saved_model_validation_report.json",
    )


def _onnx_tflite_accuracy_report_path(*, run_dir: str, model_path: str) -> str:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        run_dir,
        "saved_model",
        f"{model_stem}_accuracy_report.json",
    )


def _classify_result(
    *,
    exit_code: int,
    report: Optional[Dict[str, Any]],
    accuracy_report: Optional[Dict[str, Any]],
    combined_output: str,
) -> Tuple[str, bool, str]:
    if report is not None:
        inference = report.get("inference", {}) if isinstance(report, dict) else {}
        comparison = report.get("comparison", {}) if isinstance(report, dict) else {}
        inference_status = str(inference.get("status", ""))
        comparison_status = str(comparison.get("status", ""))
        comparison_pass = comparison.get("pass", None)
        if inference_status != "passed":
            return (
                "saved_model_inference_error",
                False,
                str(inference.get("reason", "inference_not_passed")),
            )
        if comparison_status != "passed" or comparison_pass is not True:
            return (
                "saved_model_tflite_mismatch",
                False,
                str(comparison.get("reason", "comparison_not_passed")),
            )
        if int(exit_code) == 0:
            return ("pass", True, "")
        text = combined_output.lower()
        if "flatbuffer_direct" in text:
            return (
                "out_of_scope_flatbuffer_direct_error",
                False,
                "flatbuffer_direct_error_with_report_pass",
            )
        return ("conversion_error", False, "nonzero_exit_with_report_pass")

    if int(exit_code) == 0:
        return ("conversion_error", False, "saved_model_validation_report_missing")

    text = combined_output.lower()
    if "savedmodel inference check failed" in text:
        return ("saved_model_inference_error", False, "saved_model_inference_check_failed")
    if "savedmodel/tflite output comparison complete!" in text and "pass=false" in text:
        return ("saved_model_tflite_mismatch", False, "saved_model_tflite_mismatch")
    if "flatbuffer_direct" in text:
        return ("out_of_scope_flatbuffer_direct_error", False, "flatbuffer_direct_failure")
    return ("conversion_error", False, "conversion_error")


def _build_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    entries = state.get("entries", [])
    counts = {
        "pass": 0,
        "missing_model": 0,
        "conversion_error": 0,
        "saved_model_inference_error": 0,
        "saved_model_tflite_mismatch": 0,
        "out_of_scope_flatbuffer_direct_error": 0,
        "skipped_model": 0,
    }
    for entry in entries:
        classification = str(entry.get("classification", "conversion_error"))
        if classification not in counts:
            counts["conversion_error"] += 1
        else:
            counts[classification] += 1
    in_scope_fail_count = int(counts["saved_model_inference_error"]) + int(
        counts["saved_model_tflite_mismatch"]
    )
    strict_fail_count = int(
        sum(1 for entry in entries if not bool(entry.get("strict_pass", False)))
    )
    return {
        "schema_version": _STATE_SCHEMA_VERSION,
        "list_path": state.get("list_path", ""),
        "list_sha256": state.get("list_sha256", ""),
        "total_entries": int(len(entries)),
        "counts": counts,
        "in_scope_fail_count": int(in_scope_fail_count),
        "strict_fail_count": int(strict_fail_count),
        "generated_at": _utc_now_iso(),
    }


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_markdown_summary(path: str, *, state: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# TFLite2SM Bulk Summary")
    lines.append("")
    lines.append(f"- Generated at: {summary['generated_at']}")
    lines.append(f"- List: `{summary['list_path']}`")
    lines.append(f"- Total entries: {summary['total_entries']}")
    lines.append(f"- In-scope fail count: {summary['in_scope_fail_count']}")
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


def run_bulk_verification(
    *,
    list_path: str,
    output_dir: str,
    resume: bool = False,
    onnx2tf_command: str = "",
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
    os.makedirs(runs_dir, exist_ok=True)

    command_prefix: List[str]
    if str(onnx2tf_command).strip() == "":
        local_onnx2tf_py = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "onnx2tf.py",
            )
        )
        command_prefix = [
            str(sys.executable),
            str(local_onnx2tf_py),
        ]
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
        entries: List[Dict[str, Any]] = list(state.get("entries", []))
    else:
        state = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "list_path": list_path_abs,
            "list_sha256": list_sha256,
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
        os.makedirs(run_dir, exist_ok=True)
        stdout_log_path = os.path.join(run_dir, "command.stdout.log")
        stderr_log_path = os.path.join(run_dir, "command.stderr.log")
        report_path = _saved_model_validation_report_path(
            run_dir=run_dir,
            model_path=model_name,
        )
        accuracy_report_path = _onnx_tflite_accuracy_report_path(
            run_dir=run_dir,
            model_path=model_name,
        )

        entry: Dict[str, Any] = {
            "index": int(offset),
            "model": str(model_name),
            "model_line": str(model_line),
            "model_path": str(model_path),
            "run_dir": str(run_dir),
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
            "saved_model_validation_report_path": str(report_path),
            "onnx_tflite_accuracy_report_path": str(accuracy_report_path),
            "started_at": _utc_now_iso(),
            "onnx2tf_exit_code": None,
            "classification": "",
            "strict_pass": False,
            "reason": "",
            "duration_sec": 0.0,
            "command": "",
        }

        started = time.time()
        if _should_skip_model(model_line):
            entry["onnx2tf_exit_code"] = None
            entry["classification"] = "skipped_model"
            entry["strict_pass"] = True
            entry["reason"] = "policy_skipped_model"
            entry["duration_sec"] = float(time.time() - started)
            entries.append(entry)
            state["entries"] = entries
            _write_json(state_path, state)
            continue

        if not os.path.exists(model_path):
            entry["onnx2tf_exit_code"] = None
            entry["classification"] = "missing_model"
            entry["strict_pass"] = False
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
            "-cotof",
            "-tb",
            "flatbuffer_direct",
            "-fdosm",
        ]
        entry["command"] = shlex.join(cmd)
        completed = subprocess.run(
            cmd,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_text = completed.stdout if completed.stdout is not None else ""
        stderr_text = completed.stderr if completed.stderr is not None else ""
        with open(stdout_log_path, "w", encoding="utf-8") as f:
            f.write(stdout_text)
        with open(stderr_log_path, "w", encoding="utf-8") as f:
            f.write(stderr_text)

        report_payload: Optional[Dict[str, Any]] = None
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_payload = json.load(f)
            except Exception:
                report_payload = None
        accuracy_report_payload: Optional[Dict[str, Any]] = None
        if os.path.exists(accuracy_report_path):
            try:
                with open(accuracy_report_path, "r", encoding="utf-8") as f:
                    accuracy_report_payload = json.load(f)
            except Exception:
                accuracy_report_payload = None

        classification, strict_pass, reason = _classify_result(
            exit_code=int(completed.returncode),
            report=report_payload,
            accuracy_report=accuracy_report_payload,
            combined_output=f"{stdout_text}\n{stderr_text}",
        )
        if (
            classification == "saved_model_tflite_mismatch"
            and _should_accept_saved_model_tflite_mismatch(model_line)
        ):
            entry["policy_override"] = {
                "type": "accept_saved_model_tflite_mismatch",
                "original_classification": str(classification),
                "original_strict_pass": bool(strict_pass),
                "original_reason": str(reason),
            }
            classification = "pass"
            strict_pass = True
            reason = "policy_accepted_saved_model_tflite_mismatch"
        entry["onnx2tf_exit_code"] = int(completed.returncode)
        entry["classification"] = str(classification)
        entry["strict_pass"] = bool(strict_pass)
        entry["reason"] = str(reason)
        entry["saved_model_validation_report_exists"] = bool(os.path.exists(report_path))
        entry["onnx_tflite_accuracy_report_exists"] = bool(
            os.path.exists(accuracy_report_path)
        )
        entry["onnx_tflite_accuracy_pass"] = (
            accuracy_report_payload.get("evaluation_pass", None)
            if isinstance(accuracy_report_payload, dict)
            else None
        )
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
            "Run onnx2tf bulk verification using the list in tflite2sm-plan.md "
            "with fixed command: onnx2tf -i <model> -cotof -tb flatbuffer_direct -fdosm"
        )
    )
    parser.add_argument(
        "-l",
        "--list_path",
        type=str,
        default="tflite2sm-plan.md",
        help="Path to model list file. Default: tflite2sm-plan.md",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="tflite2sm_bulk_report",
        help="Output directory for run logs and summary artifacts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing bulk_status.json when possible.",
    )
    parser.add_argument(
        "--onnx2tf_command",
        type=str,
        default="",
        help=(
            "onnx2tf command prefix. "
            "Default: current Python with local onnx2tf/onnx2tf.py. "
            "Example: \"onnx2tf\" or \"python -m onnx2tf.onnx2tf\""
        ),
    )
    args = parser.parse_args()

    state = run_bulk_verification(
        list_path=args.list_path,
        output_dir=args.output_dir,
        resume=bool(args.resume),
        onnx2tf_command=args.onnx2tf_command,
    )
    summary = state.get("summary", {})
    print(
        "Bulk verification complete. "
        f"total={summary.get('total_entries', 0)} "
        f"in_scope_fail={summary.get('in_scope_fail_count', 0)} "
        f"strict_fail={summary.get('strict_fail_count', 0)} "
        f"output_dir={os.path.abspath(args.output_dir)}"
    )


if __name__ == "__main__":
    main()
