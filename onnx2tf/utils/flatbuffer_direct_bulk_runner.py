from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import onnx
from onnx.external_data_helper import uses_external_data


_STATE_SCHEMA_VERSION = 1
_STATE_FILENAME = "bulk_status.json"
_SUMMARY_JSON_FILENAME = "bulk_summary.json"
_SUMMARY_MD_FILENAME = "bulk_summary.md"
_RUNS_DIRNAME = "runs"
_INTERNAL_PASS_METRICS_PATH_ENV = "ONNX2TF_INTERNAL_PASS_METRICS_PATH"


def _create_progress_bar(
    *,
    total: int,
    initial: int = 0,
    desc: str,
):
    if int(total) <= 0:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(
        total=int(total),
        initial=int(initial),
        desc=str(desc),
        dynamic_ncols=True,
    )


def _update_progress_bar(
    progress_bar: Any,
    *,
    model_name: str,
    classification: str,
) -> None:
    if progress_bar is None:
        return
    model_label = str(model_name)
    if len(model_label) > 48:
        model_label = f"...{model_label[-45:]}"
    progress_bar.set_postfix_str(
        f"{model_label} [{classification}]",
        refresh=True,
    )
    progress_bar.update(1)


class _ProgressSpinner:
    def __init__(self, progress_bar: Any) -> None:
        self._progress_bar = progress_bar
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._label = ""

    def set_context(self, *, model_name: str, status: str = "running") -> None:
        model_label = str(model_name)
        if len(model_label) > 48:
            model_label = f"...{model_label[-45:]}"
        self._label = f"{model_label} [{status}]"

    def start(self) -> None:
        self.stop()
        if self._progress_bar is None:
            return
        self._stop_event = threading.Event()
        self._progress_bar.set_postfix_str(f"{self._label} |", refresh=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is not None:
            self._stop_event.set()
            thread.join(timeout=0.5)
        self._thread = None

    def _run(self) -> None:
        frames = ["|", "/", "-", "\\"]
        frame_index = 0
        while not self._stop_event.wait(0.1):
            if self._progress_bar is None:
                return
            frame_index = (frame_index + 1) % len(frames)
            self._progress_bar.set_postfix_str(
                f"{self._label} {frames[frame_index]}",
                refresh=True,
            )


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_stem(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    safe = safe.strip("._-")
    if safe == "":
        return "model"
    return safe


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_error_signature(
    *,
    classification: str,
    reason: str,
    stdout_text: str = "",
    stderr_text: str = "",
    volatile_paths: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build a stable, readable failure signature without embedding run paths."""

    if str(classification) in {"pass", "skipped_model"}:
        return {"error_signature": "", "error_signature_sha256": ""}
    combined = "\n".join([str(stderr_text or ""), str(stdout_text or "")])
    combined = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", combined)
    for path in sorted(
        {os.path.abspath(path) for path in (volatile_paths or []) if str(path)},
        key=len,
        reverse=True,
    ):
        combined = combined.replace(path, "<PATH>")
    lines = [re.sub(r"\s+", " ", line).strip() for line in combined.splitlines()]
    lines = [line for line in lines if line]
    diagnostic = ""
    for line in reversed(lines):
        if any(
            marker in line
            for marker in ("Error", "Exception", "Traceback", "FAILED", "failed")
        ):
            diagnostic = line
            break
    if not diagnostic and lines:
        diagnostic = lines[-1]
    diagnostic = diagnostic[:500]
    signature = " | ".join(
        part
        for part in [str(classification), str(reason), diagnostic]
        if part
    )
    return {
        "error_signature": signature,
        "error_signature_sha256": _sha256_text(signature),
    }


def _discover_onnx_models(root_dir: str, *, recursive: bool = True) -> List[str]:
    root_dir_abs = os.path.abspath(root_dir)
    pattern = os.path.join(root_dir_abs, "**", "*.onnx") if recursive else os.path.join(root_dir_abs, "*.onnx")
    return sorted(os.path.abspath(path) for path in glob.glob(pattern, recursive=recursive))


def _filter_models_by_node_count(
    model_paths: List[str],
    *,
    min_nodes: Optional[int],
    max_nodes: Optional[int],
) -> List[str]:
    if min_nodes is None and max_nodes is None:
        return list(model_paths)
    selected: List[str] = []
    for model_path in model_paths:
        try:
            model = onnx.load(model_path, load_external_data=False)
        except Exception:
            # Invalid ONNX files belong in the manifest's invalid_onnx class,
            # not in a size tier whose node count cannot be established.
            continue
        node_count = int(len(model.graph.node))
        if min_nodes is not None and node_count < int(min_nodes):
            continue
        if max_nodes is not None and node_count > int(max_nodes):
            continue
        selected.append(model_path)
    return selected


def _models_sha256(model_paths: List[str]) -> str:
    return _sha256_text("\n".join(str(path) for path in model_paths))


def _load_regression_profile(profile_path: str) -> Dict[str, Any]:
    path = os.path.abspath(profile_path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported regression profile schema. path={path}")
    model_entries = payload.get("models", [])
    if not isinstance(model_entries, list) or not model_entries:
        raise ValueError(f"Regression profile contains no models. path={path}")
    model_names: List[str] = []
    active_model_names: List[str] = []
    tiers: List[int] = []
    baseline_classification_counts: Dict[str, int] = {}
    excluded_baseline_classification_counts: Dict[str, int] = {}
    model_options: Dict[str, Dict[str, Any]] = {}
    acceptance_reasons: Dict[str, str] = {}
    for entry in model_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid regression profile model entry. path={path}")
        model_name = os.path.basename(str(entry.get("model", "")).strip())
        tier = int(entry.get("tier", -1))
        if not model_name or tier < 0 or tier > 4:
            raise ValueError(
                "Regression profile may contain only Tier 0-4 models. "
                f"path={path} model={model_name!r} tier={tier}"
            )
        model_names.append(model_name)
        tiers.append(tier)
        normalized_options: Dict[str, Any] = {}
        for option_name in (
            "shape_hints",
            "overwrite_input_shape",
            "keep_shape_absolutely_input_names",
        ):
            raw_values = entry.get(option_name, [])
            if raw_values is None:
                raw_values = []
            if not isinstance(raw_values, list) or any(
                not isinstance(value, str) or str(value).strip() == ""
                for value in raw_values
            ):
                raise ValueError(
                    "Regression profile model options must be lists of "
                    f"non-empty strings. path={path} model={model_name!r} "
                    f"option={option_name}"
                )
            if raw_values:
                normalized_options[option_name] = [
                    str(value) for value in raw_values
                ]
        raw_eval_num_samples = entry.get("eval_num_samples", None)
        if raw_eval_num_samples is not None:
            if (
                isinstance(raw_eval_num_samples, bool)
                or not isinstance(raw_eval_num_samples, int)
                or int(raw_eval_num_samples) <= 0
            ):
                raise ValueError(
                    "Regression profile eval_num_samples must be a positive integer. "
                    f"path={path} model={model_name!r}"
                )
            normalized_options["eval_num_samples"] = int(raw_eval_num_samples)
        raw_accuracy_only = entry.get("accuracy_only", None)
        if raw_accuracy_only is not None:
            if not isinstance(raw_accuracy_only, bool):
                raise ValueError(
                    "Regression profile accuracy_only must be a boolean. "
                    f"path={path} model={model_name!r}"
                )
            normalized_options["accuracy_only"] = bool(raw_accuracy_only)
        if normalized_options:
            model_options[model_name] = normalized_options
        raw_acceptance_reason = entry.get("acceptance_reason", None)
        if raw_acceptance_reason is not None:
            if (
                not isinstance(raw_acceptance_reason, str)
                or str(raw_acceptance_reason).strip() == ""
            ):
                raise ValueError(
                    "Regression profile acceptance_reason must be a non-empty string. "
                    f"path={path} model={model_name!r}"
                )
            acceptance_reasons[model_name] = str(raw_acceptance_reason).strip()
        baseline_classification = str(
            entry.get("baseline_classification", "unspecified")
        )
        baseline_classification_counts[baseline_classification] = int(
            baseline_classification_counts.get(baseline_classification, 0)
        ) + 1
        if baseline_classification == "timeout":
            excluded_baseline_classification_counts[baseline_classification] = int(
                excluded_baseline_classification_counts.get(
                    baseline_classification,
                    0,
                )
            ) + 1
        else:
            active_model_names.append(model_name)
    if len(model_names) != len(set(model_names)):
        raise ValueError(f"Regression profile contains duplicate model names. path={path}")
    declared_model_count = int(payload.get("model_count", len(model_names)))
    if declared_model_count != len(model_names):
        raise ValueError(
            "Regression profile model_count does not match its model entries. "
            f"path={path} declared={declared_model_count} actual={len(model_names)}"
        )
    if not bool(payload.get("root_only", True)):
        raise ValueError("Regression profile must use root-only discovery.")
    if int(payload.get("inference_concurrency", 1)) != 1:
        raise ValueError("Regression profile inference_concurrency must be 1.")
    min_nodes = int(payload.get("min_nodes", 1))
    max_nodes = int(payload.get("max_nodes", 1999))
    if min_nodes < 0 or max_nodes > 1999 or min_nodes > max_nodes:
        raise ValueError(
            "Regression profile must remain within Tier 0-4 (at most 1,999 ONNX nodes). "
            f"path={path} min_nodes={min_nodes} max_nodes={max_nodes}"
        )
    content_sha256 = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "name": str(payload.get("name", os.path.basename(path))),
        "content_sha256": content_sha256,
        "model_names": model_names,
        "active_model_names": active_model_names,
        "model_count": len(model_names),
        "active_model_count": len(active_model_names),
        "excluded_model_count": len(model_names) - len(active_model_names),
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "recursive": False,
        "tiers": sorted(set(tiers)),
        "model_options": model_options,
        "acceptance_reasons": acceptance_reasons,
        "baseline_classification_counts": dict(
            sorted(baseline_classification_counts.items())
        ),
        "excluded_baseline_classification_counts": dict(
            sorted(excluded_baseline_classification_counts.items())
        ),
    }


def _stage_model_for_run(*, model_path: str, run_dir: str) -> str:
    """Copy a model and its external tensors so conversion cannot edit corpus files."""

    source = os.path.abspath(model_path)
    staged = os.path.join(run_dir, os.path.basename(source))
    shutil.copy2(source, staged)
    try:
        model = onnx.load(source, load_external_data=False)
    except Exception:
        return staged

    source_dir = os.path.dirname(source)
    copied_locations: set[str] = set()

    def _graphs(graph: Any):
        yield graph
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from _graphs(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        yield from _graphs(subgraph)

    for graph in _graphs(model.graph):
        for tensor in graph.initializer:
            if not uses_external_data(tensor):
                continue
            fields = {str(item.key): str(item.value) for item in tensor.external_data}
            location = fields.get("location", "")
            if not location or location in copied_locations:
                continue
            source_data = os.path.abspath(os.path.join(source_dir, location))
            if os.path.commonpath([source_dir, source_data]) != source_dir:
                raise ValueError(f"External data escapes model directory: {location}")
            destination_data = os.path.abspath(os.path.join(run_dir, location))
            if os.path.commonpath([os.path.abspath(run_dir), destination_data]) != os.path.abspath(run_dir):
                raise ValueError(f"External data escapes run directory: {location}")
            os.makedirs(os.path.dirname(destination_data), exist_ok=True)
            shutil.copy2(source_data, destination_data)
            copied_locations.add(location)
    return staged


def _accuracy_report_path(*, artifact_dir: str, model_path: str) -> str:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        artifact_dir,
        f"{model_stem}_accuracy_report.json",
    )


def _pytorch_accuracy_report_path(*, artifact_dir: str, model_path: str) -> str:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        artifact_dir,
        f"{model_stem}_pytorch_accuracy_report.json",
    )


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def _classify_reports(
    *,
    tflite_report: Optional[Dict[str, Any]],
    pytorch_report: Optional[Dict[str, Any]],
    require_pytorch_report: bool = True,
) -> Dict[str, Any]:
    def _max_abs(report: Optional[Dict[str, Any]]) -> Optional[float]:
        if report is None:
            return None
        metrics = report.get("overall_metrics", {})
        if not isinstance(metrics, dict) or metrics.get("max_abs") is None:
            return None
        return float(metrics["max_abs"])

    tflite_exists = tflite_report is not None
    pytorch_exists = pytorch_report is not None
    tflite_max_abs = _max_abs(tflite_report)
    pytorch_max_abs = _max_abs(pytorch_report)
    tflite_pass = (
        bool(tflite_report.get("evaluation_pass", False))
        and (tflite_max_abs is None or tflite_max_abs <= 1.0e-1)
        if tflite_exists
        else None
    )
    pytorch_pass = (
        bool(pytorch_report.get("evaluation_pass", False))
        and (pytorch_max_abs is None or pytorch_max_abs <= 1.0e-1)
        if pytorch_exists
        else None
    )

    if not require_pytorch_report:
        if not tflite_exists:
            return {
                "classification": "missing_tflite_report",
                "strict_pass": False,
                "reason": "missing_tflite_report",
                "tflite_accuracy_pass": None,
                "pytorch_accuracy_pass": None,
                "tflite_max_abs": None,
                "pytorch_max_abs": None,
            }
        return {
            "classification": "pass" if tflite_pass else "tflite_fail",
            "strict_pass": bool(tflite_pass),
            "reason": "" if tflite_pass else "tflite_fail",
            "tflite_accuracy_pass": tflite_pass,
            "pytorch_accuracy_pass": None,
            "tflite_max_abs": tflite_max_abs,
            "pytorch_max_abs": None,
        }

    if not tflite_exists and not pytorch_exists:
        return {
            "classification": "missing_both_reports",
            "strict_pass": False,
            "reason": "missing_tflite_report,missing_pytorch_report",
            "tflite_accuracy_pass": None,
            "pytorch_accuracy_pass": None,
            "tflite_max_abs": tflite_max_abs,
            "pytorch_max_abs": pytorch_max_abs,
        }
    if not tflite_exists:
        return {
            "classification": "missing_tflite_report",
            "strict_pass": False,
            "reason": "missing_tflite_report",
            "tflite_accuracy_pass": None,
            "pytorch_accuracy_pass": pytorch_pass,
            "tflite_max_abs": tflite_max_abs,
            "pytorch_max_abs": pytorch_max_abs,
        }
    if not pytorch_exists:
        return {
            "classification": "missing_pytorch_report",
            "strict_pass": False,
            "reason": "missing_pytorch_report",
            "tflite_accuracy_pass": tflite_pass,
            "pytorch_accuracy_pass": None,
            "tflite_max_abs": tflite_max_abs,
            "pytorch_max_abs": pytorch_max_abs,
        }
    if tflite_pass and pytorch_pass:
        return {
            "classification": "pass",
            "strict_pass": True,
            "reason": "",
            "tflite_accuracy_pass": True,
            "pytorch_accuracy_pass": True,
            "tflite_max_abs": tflite_max_abs,
            "pytorch_max_abs": pytorch_max_abs,
        }
    if not tflite_pass and not pytorch_pass:
        return {
            "classification": "both_fail",
            "strict_pass": False,
            "reason": "tflite_fail,pytorch_fail",
            "tflite_accuracy_pass": False,
            "pytorch_accuracy_pass": False,
            "tflite_max_abs": tflite_max_abs,
            "pytorch_max_abs": pytorch_max_abs,
        }
    if not tflite_pass:
        return {
            "classification": "tflite_fail",
            "strict_pass": False,
            "reason": "tflite_fail",
            "tflite_accuracy_pass": False,
            "pytorch_accuracy_pass": True,
            "tflite_max_abs": tflite_max_abs,
            "pytorch_max_abs": pytorch_max_abs,
        }
    return {
        "classification": "pytorch_fail",
        "strict_pass": False,
        "reason": "pytorch_fail",
        "tflite_accuracy_pass": True,
        "pytorch_accuracy_pass": False,
        "tflite_max_abs": tflite_max_abs,
        "pytorch_max_abs": pytorch_max_abs,
    }


def _apply_profile_acceptance(
    classification: Dict[str, Any],
    *,
    acceptance_reason: str,
) -> Dict[str, Any]:
    """Accept a recorded numeric exception without hiding its raw result."""

    reason = str(acceptance_reason).strip()
    if reason == "" or str(classification.get("classification", "")) != "tflite_fail":
        return classification

    accepted = dict(classification)
    accepted.update(
        {
            "classification": "pass",
            "strict_pass": True,
            "reason": f"profile_acceptance:{reason}",
            "accepted_by_profile": True,
            "profile_acceptance_reason": reason,
            "unaccepted_classification": str(
                classification.get("classification", "")
            ),
            "unaccepted_strict_pass": bool(
                classification.get("strict_pass", False)
            ),
            "unaccepted_reason": str(classification.get("reason", "")),
        }
    )
    return accepted


def _build_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    entries = state.get("entries", [])
    counts = {
        "pass": 0,
        "skipped_model": 0,
        "missing_model": 0,
        "conversion_error": 0,
        "timeout": 0,
        "tflite_fail": 0,
        "pytorch_fail": 0,
        "both_fail": 0,
        "missing_tflite_report": 0,
        "missing_pytorch_report": 0,
        "missing_both_reports": 0,
    }
    for entry in entries:
        classification = str(entry.get("classification", "conversion_error"))
        counts[classification if classification in counts else "conversion_error"] += 1
    failed_entries = [
        entry
        for entry in entries
        if not bool(entry.get("strict_pass", False))
    ]
    durations = [float(entry.get("duration_sec", 0.0)) for entry in entries]
    filters: Dict[str, Any] = {
        "min_nodes": state.get("min_nodes"),
        "max_nodes": state.get("max_nodes"),
        "recursive": bool(state.get("recursive", True)),
        "include_pytorch_artifacts": bool(
            state.get("include_pytorch_artifacts", True)
        ),
    }
    if state.get("regression_profile") is not None:
        filters["regression_profile"] = dict(state["regression_profile"])
    summary = {
        "schema_version": _STATE_SCHEMA_VERSION,
        "root_dir": state.get("root_dir", ""),
        "models_sha256": state.get("models_sha256", ""),
        "total_entries": int(len(entries)),
        "filters": filters,
        "timing": {
            "total_duration_sec": float(sum(durations)),
            "median_duration_sec": float(statistics.median(durations))
            if durations
            else 0.0,
            "max_duration_sec": float(max(durations)) if durations else 0.0,
        },
        "counts": counts,
        "strict_fail_count": int(len(failed_entries)),
        "failed_models": [
            {
                "model": str(entry.get("model", "")),
                "model_path": str(entry.get("model_path", "")),
                "classification": str(entry.get("classification", "")),
                "reason": str(entry.get("reason", "")),
                "error_signature": str(entry.get("error_signature", "")),
                "error_signature_sha256": str(
                    entry.get("error_signature_sha256", "")
                ),
            }
            for entry in failed_entries
        ],
        "generated_at": _utc_now_iso(),
    }
    pass_metric_entries = [
        entry.get("pass_metrics")
        for entry in entries
        if isinstance(entry.get("pass_metrics"), dict)
    ]
    if pass_metric_entries:
        aggregate_totals = {
            "preflight_operators_visited": 0,
            "state_build_count": 0,
            "snapshot_count": 0,
            "fingerprint_count": 0,
        }
        total_events = 0
        for metrics in pass_metric_entries:
            total_events += int(metrics.get("event_count", 0))
            totals = metrics.get("totals", {})
            if not isinstance(totals, dict):
                continue
            for key in aggregate_totals:
                aggregate_totals[key] += int(totals.get(key, 0))
        summary["pass_metrics"] = {
            "models_with_metrics": len(pass_metric_entries),
            "event_count": total_events,
            "totals": aggregate_totals,
        }
    return summary


def _write_markdown_summary(path: str, *, state: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Flatbuffer Direct Bulk Summary")
    lines.append("")
    lines.append(f"- Generated at: {summary['generated_at']}")
    lines.append(f"- Root dir: `{summary['root_dir']}`")
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
    lines.append("## Failed Models")
    lines.append("")
    if not summary["failed_models"]:
        lines.append("None")
    else:
        lines.append("| Model | Classification | Reason |")
        lines.append("| --- | --- | --- |")
        for failed in summary["failed_models"]:
            reason = str(failed.get("reason", "")).replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| `{failed.get('model_path', '')}` | "
                f"`{failed.get('classification', '')}` | {reason} |"
            )
    lines.append("")
    lines.append("## Details")
    lines.append("")
    lines.append("| # | Model | Classification | Strict Pass | Exit | Reason |")
    lines.append("| ---: | --- | --- | :---: | ---: | --- |")
    for entry in state.get("entries", []):
        reason = str(entry.get("reason", "")).replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {int(entry.get('index', 0))} | `{entry.get('model_path', '')}` | "
            f"`{entry.get('classification', '')}` | "
            f"{'Y' if bool(entry.get('strict_pass', False)) else 'N'} | "
            f"{entry.get('onnx2tf_exit_code', '')} | {reason} |"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_flatbuffer_direct_bulk_verification(
    *,
    root_dir: str,
    output_dir: str,
    resume: bool = False,
    onnx2tf_command: str = "",
    timeout_sec: int = 600,
    native_pytorch_generation_timeout_sec: int = 0,
    skip_model_names: Optional[List[str]] = None,
    min_nodes: Optional[int] = None,
    max_nodes: Optional[int] = None,
    include_pytorch_artifacts: bool = True,
    recursive: bool = True,
    regression_profile: Optional[str] = None,
) -> Dict[str, Any]:
    root_dir_abs = os.path.abspath(root_dir)
    output_dir_abs = os.path.abspath(output_dir)
    runs_dir = os.path.join(output_dir_abs, _RUNS_DIRNAME)
    state_path = os.path.join(output_dir_abs, _STATE_FILENAME)
    summary_json_path = os.path.join(output_dir_abs, _SUMMARY_JSON_FILENAME)
    summary_md_path = os.path.join(output_dir_abs, _SUMMARY_MD_FILENAME)

    if not os.path.isdir(root_dir_abs):
        raise FileNotFoundError(f"Root directory does not exist. path={root_dir_abs}")

    profile: Optional[Dict[str, Any]] = None
    profile_identity: Optional[Dict[str, Any]] = None
    if regression_profile:
        profile = _load_regression_profile(regression_profile)
        profile_identity = {
            key: value
            for key, value in profile.items()
            if key not in {"model_names", "active_model_names"}
        }
        if min_nodes is not None and int(min_nodes) != int(profile["min_nodes"]):
            raise ValueError("min_nodes conflicts with the regression profile.")
        if max_nodes is not None and int(max_nodes) != int(profile["max_nodes"]):
            raise ValueError("max_nodes conflicts with the regression profile.")
        min_nodes = int(profile["min_nodes"])
        max_nodes = int(profile["max_nodes"])
        recursive = False

    if min_nodes is not None and int(min_nodes) < 0:
        raise ValueError("min_nodes must be >= 0")
    if max_nodes is not None and int(max_nodes) < 0:
        raise ValueError("max_nodes must be >= 0")
    if min_nodes is not None and max_nodes is not None and int(min_nodes) > int(max_nodes):
        raise ValueError("min_nodes must be <= max_nodes")
    discovered_source = (
        _discover_onnx_models(root_dir_abs)
        if recursive
        else _discover_onnx_models(root_dir_abs, recursive=False)
    )
    discovered_models = [
        path
        for path in discovered_source
        if os.path.commonpath([output_dir_abs, os.path.abspath(path)]) != output_dir_abs
    ]
    models = _filter_models_by_node_count(
        discovered_models,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
    )
    if profile is not None:
        allowed_names = set(profile["active_model_names"])
        discovered_names = {os.path.basename(path) for path in models}
        missing_names = sorted(allowed_names - discovered_names)
        if missing_names:
            raise RuntimeError(
                "Regression-profile models are missing from the current root corpus. "
                f"missing_models={missing_names}"
            )
        models = [path for path in models if os.path.basename(path) in allowed_names]
    models_sha256 = _models_sha256(models)
    normalized_skip_model_names = sorted(
        {
            os.path.basename(str(model_name)).strip()
            for model_name in (skip_model_names or [])
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
        if str(state.get("models_sha256", "")) != str(models_sha256):
            raise RuntimeError(
                "Resume state does not match the current discovered models. "
                f"state_sha256={state.get('models_sha256', '')} current_sha256={models_sha256}"
            )
        if list(state.get("skip_model_names", [])) != normalized_skip_model_names:
            raise RuntimeError(
                "Resume state does not match the current skip_model_names. "
                f"state_skip_model_names={state.get('skip_model_names', [])} "
                f"current_skip_model_names={normalized_skip_model_names}"
            )
        if int(state.get("native_pytorch_generation_timeout_sec", 0)) != int(native_pytorch_generation_timeout_sec):
            raise RuntimeError(
                "Resume state does not match the current native_pytorch_generation_timeout_sec. "
                f"state_timeout={state.get('native_pytorch_generation_timeout_sec', 0)} "
                f"current_timeout={int(native_pytorch_generation_timeout_sec)}"
            )
        if state.get("min_nodes") != min_nodes or state.get("max_nodes") != max_nodes:
            raise RuntimeError(
                "Resume state does not match the current node-count tier. "
                f"state=({state.get('min_nodes')},{state.get('max_nodes')}) "
                f"current=({min_nodes},{max_nodes})"
            )
        if bool(state.get("include_pytorch_artifacts", True)) != bool(include_pytorch_artifacts):
            raise RuntimeError(
                "Resume state does not match include_pytorch_artifacts. "
                f"state={state.get('include_pytorch_artifacts')} "
                f"current={include_pytorch_artifacts}"
            )
        if bool(state.get("recursive", True)) != bool(recursive):
            raise RuntimeError(
                "Resume state does not match recursive discovery mode. "
                f"state={state.get('recursive')} current={recursive}"
            )
        if state.get("regression_profile") != profile_identity:
            raise RuntimeError("Resume state does not match regression_profile.")
        entries: List[Dict[str, Any]] = list(state.get("entries", []))
    else:
        state = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "root_dir": root_dir_abs,
            "models_sha256": models_sha256,
            "skip_model_names": normalized_skip_model_names,
            "native_pytorch_generation_timeout_sec": int(native_pytorch_generation_timeout_sec),
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "include_pytorch_artifacts": bool(include_pytorch_artifacts),
            "recursive": bool(recursive),
            "regression_profile": profile_identity,
            "started_at": _utc_now_iso(),
            "entries": [],
        }
        entries = []

    start_index = int(len(entries))
    progress_bar = _create_progress_bar(
        total=len(models),
        initial=start_index,
        desc="flatbuffer_direct bulk",
    )
    spinner = _ProgressSpinner(progress_bar)
    try:
        for offset, model_path in enumerate(models[start_index:], start=start_index + 1):
            model_name = os.path.basename(model_path)
            run_dir = os.path.join(
                runs_dir,
                f"{int(offset):04d}_{_sanitize_stem(os.path.splitext(model_name)[0])}",
            )
            artifact_dir = os.path.join(run_dir, "artifacts")
            os.makedirs(artifact_dir, exist_ok=True)
            stdout_log_path = os.path.join(run_dir, "command.stdout.log")
            stderr_log_path = os.path.join(run_dir, "command.stderr.log")
            tflite_accuracy_report_path = _accuracy_report_path(
                artifact_dir=artifact_dir,
                model_path=model_name,
            )
            pytorch_accuracy_report_path = _pytorch_accuracy_report_path(
                artifact_dir=artifact_dir,
                model_path=model_name,
            )
            pass_metrics_path = os.path.join(run_dir, "pass_metrics.json")

            entry: Dict[str, Any] = {
                "index": int(offset),
                "model": str(model_name),
                "model_path": str(model_path),
                "run_dir": str(run_dir),
                "artifact_dir": str(artifact_dir),
                "stdout_log_path": str(stdout_log_path),
                "stderr_log_path": str(stderr_log_path),
                "tflite_accuracy_report_path": str(tflite_accuracy_report_path),
                "pytorch_accuracy_report_path": str(pytorch_accuracy_report_path),
                "pass_metrics_path": str(pass_metrics_path),
                "started_at": _utc_now_iso(),
                "onnx2tf_exit_code": None,
                "classification": "",
                "strict_pass": False,
                "reason": "",
                "duration_sec": 0.0,
                "command": "",
                "tflite_accuracy_pass": None,
                "pytorch_accuracy_pass": None,
                "tflite_max_abs": None,
                "pytorch_max_abs": None,
                "error_signature": "",
                "error_signature_sha256": "",
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
                _update_progress_bar(
                    progress_bar,
                    model_name=model_name,
                    classification=str(entry["classification"]),
                )
                continue

            if not os.path.exists(model_path):
                entry["classification"] = "missing_model"
                entry["strict_pass"] = False
                entry["reason"] = "model_not_found"
                entry["duration_sec"] = float(time.time() - started)
                entry.update(
                    _normalized_error_signature(
                        classification=str(entry["classification"]),
                        reason=str(entry["reason"]),
                    )
                )
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                _update_progress_bar(
                    progress_bar,
                    model_name=model_name,
                    classification=str(entry["classification"]),
                )
                continue

            staged_model_path = _stage_model_for_run(
                model_path=model_path,
                run_dir=run_dir,
            )

            per_model_options = (
                profile.get("model_options", {}).get(model_name, {})
                if profile is not None
                else {}
            )
            accuracy_only = bool(per_model_options.get("accuracy_only", False))
            cmd = [
                *command_prefix,
                "-i",
                str(staged_model_path),
                "-o",
                str(artifact_dir),
                "-tb",
                "flatbuffer_direct",
                "--eval_with_onnx" if accuracy_only else "-cotof",
            ]
            if profile is not None:
                shape_hints_for_model = per_model_options.get("shape_hints", [])
                if shape_hints_for_model:
                    cmd.extend(["-sh", *shape_hints_for_model])
                overwrite_shapes = per_model_options.get(
                    "overwrite_input_shape",
                    [],
                )
                if overwrite_shapes:
                    cmd.extend(["-ois", *overwrite_shapes])
                keep_shape_names = per_model_options.get(
                    "keep_shape_absolutely_input_names",
                    [],
                )
                if keep_shape_names:
                    cmd.extend(["-kat", *keep_shape_names])
                eval_num_samples = per_model_options.get(
                    "eval_num_samples",
                    None,
                )
                if eval_num_samples is not None:
                    cmd.extend(["-ens", str(int(eval_num_samples))])
            if include_pytorch_artifacts:
                cmd.extend(["-fdopt", "-fdots", "-fdodo", "-fdoep"])
            if int(native_pytorch_generation_timeout_sec) > 0:
                cmd.extend(
                    [
                        "--native_pytorch_generation_timeout_sec",
                        str(int(native_pytorch_generation_timeout_sec)),
                    ]
                )
            entry["command"] = shlex.join(cmd)

            spinner.set_context(
                model_name=model_name,
                status="running",
            )
            spinner.start()
            try:
                os.remove(pass_metrics_path)
            except FileNotFoundError:
                pass
            previous_metrics_path = os.environ.get(
                _INTERNAL_PASS_METRICS_PATH_ENV,
                None,
            )
            os.environ[_INTERNAL_PASS_METRICS_PATH_ENV] = pass_metrics_path
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
                spinner.stop()
                stdout_text = ex.stdout if isinstance(ex.stdout, str) else ""
                stderr_text = ex.stderr if isinstance(ex.stderr, str) else ""
                entry["classification"] = "timeout"
                entry["strict_pass"] = False
                entry["reason"] = f"timeout_after_{int(timeout_sec)}s"
                with open(stdout_log_path, "w", encoding="utf-8") as f:
                    f.write(stdout_text)
                with open(stderr_log_path, "w", encoding="utf-8") as f:
                    f.write(stderr_text)
                entry["duration_sec"] = float(time.time() - started)
                entry.update(
                    _normalized_error_signature(
                        classification=str(entry["classification"]),
                        reason=str(entry["reason"]),
                        stdout_text=stdout_text,
                        stderr_text=stderr_text,
                        volatile_paths=[root_dir_abs, output_dir_abs, run_dir],
                    )
                )
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                _update_progress_bar(
                    progress_bar,
                    model_name=model_name,
                    classification=str(entry["classification"]),
                )
                continue
            finally:
                if previous_metrics_path is None:
                    os.environ.pop(_INTERNAL_PASS_METRICS_PATH_ENV, None)
                else:
                    os.environ[_INTERNAL_PASS_METRICS_PATH_ENV] = (
                        previous_metrics_path
                    )
                spinner.stop()

            with open(stdout_log_path, "w", encoding="utf-8") as f:
                f.write(stdout_text)
            with open(stderr_log_path, "w", encoding="utf-8") as f:
                f.write(stderr_text)

            pass_metrics = _read_json(pass_metrics_path)
            if isinstance(pass_metrics, dict):
                entry["pass_metrics"] = pass_metrics

            if int(entry["onnx2tf_exit_code"]) != 0:
                entry["classification"] = "conversion_error"
                entry["strict_pass"] = False
                entry["reason"] = "onnx2tf_nonzero_exit"
                entry["duration_sec"] = float(time.time() - started)
                entry.update(
                    _normalized_error_signature(
                        classification=str(entry["classification"]),
                        reason=str(entry["reason"]),
                        stdout_text=stdout_text,
                        stderr_text=stderr_text,
                        volatile_paths=[root_dir_abs, output_dir_abs, run_dir],
                    )
                )
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                _update_progress_bar(
                    progress_bar,
                    model_name=model_name,
                    classification=str(entry["classification"]),
                )
                continue

            tflite_report = _read_json(tflite_accuracy_report_path)
            pytorch_report = _read_json(pytorch_accuracy_report_path)
            classification = _classify_reports(
                tflite_report=tflite_report,
                pytorch_report=pytorch_report,
                require_pytorch_report=bool(include_pytorch_artifacts),
            )
            if profile is not None:
                classification = _apply_profile_acceptance(
                    classification,
                    acceptance_reason=profile.get("acceptance_reasons", {}).get(
                        model_name,
                        "",
                    ),
                )
            entry["classification"] = str(classification["classification"])
            entry["strict_pass"] = bool(classification["strict_pass"])
            entry["reason"] = str(classification["reason"])
            entry["tflite_accuracy_pass"] = classification["tflite_accuracy_pass"]
            entry["pytorch_accuracy_pass"] = classification["pytorch_accuracy_pass"]
            entry["tflite_max_abs"] = classification["tflite_max_abs"]
            entry["pytorch_max_abs"] = classification["pytorch_max_abs"]
            for optional_key in (
                "accepted_by_profile",
                "profile_acceptance_reason",
                "unaccepted_classification",
                "unaccepted_strict_pass",
                "unaccepted_reason",
            ):
                if optional_key in classification:
                    entry[optional_key] = classification[optional_key]
            entry["duration_sec"] = float(time.time() - started)
            entry.update(
                _normalized_error_signature(
                    classification=str(entry["classification"]),
                    reason=str(entry["reason"]),
                    stdout_text=stdout_text,
                    stderr_text=stderr_text,
                    volatile_paths=[root_dir_abs, output_dir_abs, run_dir],
                )
            )
            entries.append(entry)
            state["entries"] = entries
            _write_json(state_path, state)
            _update_progress_bar(
                progress_bar,
                model_name=model_name,
                classification=str(entry["classification"]),
            )
    finally:
        spinner.stop()
        if progress_bar is not None:
            progress_bar.close()

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
            "Run bulk flatbuffer_direct verification using recursive ONNX discovery "
            "and onnx2tf -tb flatbuffer_direct -cotof -fdopt -fdots -fdodo -fdoep."
        )
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="flatbuffer_direct_bulk_report")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--onnx2tf_command", type=str, default="")
    parser.add_argument("--timeout_sec", type=int, default=600)
    parser.add_argument("--native_pytorch_generation_timeout_sec", type=int, default=0)
    parser.add_argument("--min_nodes", type=int, default=None)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument(
        "--regression_profile",
        type=str,
        default=None,
        help=(
            "Run the models recorded in a managed Tier 0-4 profile. "
            "The profile fixes root-only discovery and its node-count range; "
            "models whose managed baseline is timeout are excluded."
        ),
    )
    parser.add_argument(
        "--tflite_only",
        action="store_true",
        help="Run the TensorFlow-free ONNX/TFLite check without optional PyTorch artifacts.",
    )
    parser.add_argument(
        "--root_only",
        action="store_true",
        help="Discover only *.onnx directly under root_dir, excluding nested test assets.",
    )
    parser.add_argument(
        "--skip_model_name",
        "--skip_model_names",
        dest="skip_model_name",
        action="append",
        default=[],
        help="Basename of an ONNX model to skip during bulk verification. Repeatable.",
    )
    args = parser.parse_args()

    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(args.root_dir),
        output_dir=str(args.output_dir),
        resume=bool(args.resume),
        onnx2tf_command=str(args.onnx2tf_command),
        timeout_sec=int(args.timeout_sec),
        native_pytorch_generation_timeout_sec=int(args.native_pytorch_generation_timeout_sec),
        skip_model_names=list(args.skip_model_name),
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        include_pytorch_artifacts=not bool(args.tflite_only),
        recursive=not bool(args.root_only),
        regression_profile=args.regression_profile,
    )
    summary = state.get("summary", {}) or {}
    failed_models = list(summary.get("failed_models", []))
    print(
        "Flatbuffer-direct bulk verification complete. "
        f"total_entries={len(state.get('entries', []))} "
        f"pass_count={summary.get('counts', {}).get('pass', 0)} "
        f"fail_count={summary.get('strict_fail_count', 0)}"
    )
    if failed_models:
        print("Failed ONNX models:")
        for failed in failed_models:
            print(
                f"- {failed.get('model_path', '')} "
                f"[{failed.get('classification', '')}] "
                f"{failed.get('reason', '')}"
            )
    raise SystemExit(1 if int(summary.get("strict_fail_count", 0)) > 0 else 0)


if __name__ == "__main__":
    main()
