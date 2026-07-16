import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

from onnx2tf.utils import flatbuffer_direct_bulk_runner as bulk_runner
from onnx2tf.utils.flatbuffer_direct_bulk_runner import (
    run_flatbuffer_direct_bulk_verification,
)


def _write_dummy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy")


def _write_accuracy_report(
    *,
    path: Path,
    evaluation_pass: bool,
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "evaluation_pass": bool(evaluation_pass),
        "overall_metrics": {
            "max_abs": 0.0 if evaluation_pass else 1.0,
            "rmse": 0.0 if evaluation_pass else 1.0,
            "cosine_similarity": 1.0 if evaluation_pass else 0.0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_flatbuffer_direct_bulk_runner_preserves_sorted_discovery_order(
    tmp_path,
    monkeypatch,
) -> None:
    model_b = tmp_path / "z_dir" / "b.onnx"
    model_a = tmp_path / "a_dir" / "a.onnx"
    _write_dummy(model_b)
    _write_dummy(model_a)

    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = str(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(model_path).stem
        calls.append(model_path)
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    assert [entry["model_path"] for entry in state["entries"]] == sorted(
        [str(model_a.resolve()), str(model_b.resolve())]
    )
    assert [Path(path).name for path in calls] == ["a.onnx", "b.onnx"]
    assert all(Path(path).parent.parent.name == "runs" for path in calls)


def test_flatbuffer_direct_bulk_runner_collects_internal_pass_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "model.onnx"
    _write_dummy(model)
    metrics_env = "ONNX2TF_INTERNAL_PASS_METRICS_PATH"
    monkeypatch.delenv(metrics_env, raising=False)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        metrics_path = Path(bulk_runner.os.environ[metrics_env])
        metrics_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "event_count": 3,
                    "status_counts": {"changed": 1, "skipped": 2},
                    "totals": {
                        "preflight_operators_visited": 9,
                        "state_build_count": 1,
                        "snapshot_count": 1,
                        "fingerprint_count": 0,
                    },
                    "by_pass": {},
                }
            ),
            encoding="utf-8",
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )

    assert metrics_env not in bulk_runner.os.environ
    assert state["entries"][0]["pass_metrics"]["event_count"] == 3
    assert state["summary"]["pass_metrics"] == {
        "models_with_metrics": 1,
        "event_count": 3,
        "totals": {
            "preflight_operators_visited": 9,
            "state_build_count": 1,
            "snapshot_count": 1,
            "fingerprint_count": 0,
        },
    }


def test_flatbuffer_direct_bulk_runner_skips_missing_models_cleanly(
    tmp_path,
    monkeypatch,
) -> None:
    existing = tmp_path / "a.onnx"
    missing = tmp_path / "missing.onnx"
    _write_dummy(existing)

    monkeypatch.setattr(
        "onnx2tf.utils.flatbuffer_direct_bulk_runner._discover_onnx_models",
        lambda _root_dir: [str(existing), str(missing)],
    )

    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = str(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(model_path).stem
        calls.append(model_path)
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    assert [entry["classification"] for entry in state["entries"]] == ["pass", "missing_model"]
    assert state["entries"][1]["strict_pass"] is False
    assert [Path(path).name for path in calls] == ["a.onnx"]
    assert Path(calls[0]).parent.parent.name == "runs"


def test_flatbuffer_direct_bulk_runner_resume_skips_completed_entries(
    tmp_path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "a.onnx"
    model_b = tmp_path / "b.onnx"
    _write_dummy(model_a)
    _write_dummy(model_b)
    output_dir = tmp_path / "out"

    first_calls: List[str] = []

    def _fake_run_first(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = str(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(model_path).stem
        first_calls.append(model_path)
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run_first)
    first_state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(output_dir),
    )
    assert len(first_state["entries"]) == 2
    assert len(first_calls) == 2

    second_calls: List[str] = []

    def _fake_run_second(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        second_calls.append("called")
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run_second)
    second_state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(output_dir),
        resume=True,
    )
    assert len(second_state["entries"]) == 2
    assert second_calls == []


def test_flatbuffer_direct_bulk_runner_marks_pass_only_when_both_reports_pass(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "ok.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pass"
    assert entry["strict_pass"] is True


def test_flatbuffer_direct_bulk_runner_records_stable_failure_signature_and_timing(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "broken.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        return type(
            "CP",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Traceback\nValueError: failed in {cwd}/temporary.bin",
            },
        )()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        min_nodes=None,
        max_nodes=None,
        include_pytorch_artifacts=False,
        recursive=False,
    )
    entry = state["entries"][0]
    assert entry["classification"] == "conversion_error"
    assert "ValueError: failed in <PATH>/temporary.bin" in entry["error_signature"]
    assert len(entry["error_signature_sha256"]) == 64

    summary = state["summary"]
    assert summary["filters"] == {
        "min_nodes": None,
        "max_nodes": None,
        "recursive": False,
        "include_pytorch_artifacts": False,
    }
    assert summary["timing"]["total_duration_sec"] >= 0.0
    assert summary["failed_models"][0]["error_signature_sha256"] == entry[
        "error_signature_sha256"
    ]


def test_flatbuffer_direct_bulk_runner_filters_node_count_tier(
    tmp_path,
    monkeypatch,
) -> None:
    from onnx import TensorProto, helper, save

    def _model(path, count):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        previous = "x"
        nodes = []
        for index in range(count):
            output = f"v{index}"
            nodes.append(helper.make_node("Relu", [previous], [output]))
            previous = output
        y = helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1])
        save(helper.make_model(helper.make_graph(nodes, "g", [x], [y])), path)

    _model(tmp_path / "small.onnx", 2)
    _model(tmp_path / "medium.onnx", 55)
    calls = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        calls.append(cmd)
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        min_nodes=50,
        max_nodes=199,
    )
    assert [entry["model"] for entry in state["entries"]] == ["medium.onnx"]
    assert len(calls) == 1


def test_flatbuffer_direct_bulk_runner_enforces_max_abs_one_tenth(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "accuracy.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        for suffix, max_abs in [("accuracy_report", 0.10001), ("pytorch_accuracy_report", 0.01)]:
            path = artifact_dir / f"{stem}_{suffix}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"evaluation_pass": True, "overall_metrics": {"max_abs": max_abs}}),
                encoding="utf-8",
            )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "tflite_fail"
    assert entry["tflite_max_abs"] == 0.10001


def test_flatbuffer_direct_bulk_runner_tflite_only_does_not_require_pytorch(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "core.onnx"
    _write_dummy(model)
    commands = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        commands.append(cmd)
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
    )
    assert state["entries"][0]["classification"] == "pass"
    assert "-fdopt" not in commands[0]


def test_flatbuffer_direct_bulk_runner_stages_model_before_conversion(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "immutable.onnx"
    original = b"original corpus bytes"
    model.write_bytes(original)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        staged = Path(cmd[cmd.index("-i") + 1])
        assert staged != model
        assert staged.parent == Path(cwd)
        staged.write_bytes(b"converter mutation")
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        _write_accuracy_report(
            path=artifact_dir / "immutable_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
    )
    assert state["entries"][0]["classification"] == "pass"
    assert model.read_bytes() == original


def test_flatbuffer_direct_bulk_runner_root_only_excludes_nested_models(
    tmp_path,
    monkeypatch,
) -> None:
    root_model = tmp_path / "root.onnx"
    nested_model = tmp_path / "nested" / "nested.onnx"
    _write_dummy(root_model)
    _write_dummy(nested_model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        _write_accuracy_report(
            path=artifact_dir / "root_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
        recursive=False,
    )
    assert [entry["model"] for entry in state["entries"]] == ["root.onnx"]


def test_flatbuffer_direct_bulk_runner_marks_tflite_failure(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "tflite_fail.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=False,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "tflite_fail"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_marks_pytorch_failure(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "pytorch_fail.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=False,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pytorch_fail"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_marks_both_failures(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "both_fail.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=False,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=False,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "both_fail"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_marks_missing_single_report(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "missing_report.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "missing_pytorch_report"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_marks_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "slow.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        timeout_sec=1,
    )
    entry = state["entries"][0]
    assert entry["classification"] == "timeout"
    assert entry["strict_pass"] is False


def test_process_tree_swap_monitor_records_and_terminates_swapping_descendants(
    monkeypatch,
) -> None:
    killed = []
    swap_by_pid = {101: 32, 102: 16}
    monkeypatch.setattr(
        bulk_runner,
        "_collect_descendant_pids",
        lambda _root_pid: [101, 102],
    )
    monkeypatch.setattr(
        bulk_runner,
        "_read_process_swap_kib",
        lambda pid: swap_by_pid[pid],
    )
    monkeypatch.setattr(
        bulk_runner,
        "_read_process_name",
        lambda pid: {101: "onnx2tf", 102: "delegate"}[pid],
    )
    monkeypatch.setattr(
        bulk_runner.os,
        "kill",
        lambda pid, process_signal: killed.append((pid, process_signal)),
    )

    monitor = bulk_runner._ProcessTreeSwapMonitor(root_pid=100)
    monitor._sample_once()

    assert monitor.result() == {
        "swap_detected": True,
        "peak_swap_kib": 48,
        "swap_processes": [
            {"pid": 101, "name": "onnx2tf", "peak_swap_kib": 32},
            {"pid": 102, "name": "delegate", "peak_swap_kib": 16},
        ],
    }
    assert killed == [
        (102, bulk_runner.signal.SIGTERM),
        (101, bulk_runner.signal.SIGTERM),
    ]

    swap_by_pid.update({101: 0, 102: 0})
    monitor._termination_started_at -= 2.0
    monitor._sample_once()
    assert killed[-2:] == [
        (102, bulk_runner.signal.SIGKILL),
        (101, bulk_runner.signal.SIGKILL),
    ]


def test_flatbuffer_direct_bulk_runner_classifies_detected_swap(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "swapping.onnx"
    _write_dummy(model)

    class _FakeSwapMonitor:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def result(self):
            return {
                "swap_detected": True,
                "peak_swap_kib": 4096,
                "swap_processes": [
                    {"pid": 123, "name": "onnx2tf", "peak_swap_kib": 4096},
                ],
            }

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        return type(
            "CP",
            (),
            {"returncode": -15, "stdout": "", "stderr": "terminated"},
        )()

    monkeypatch.setattr(bulk_runner, "_ProcessTreeSwapMonitor", _FakeSwapMonitor)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
    )

    entry = state["entries"][0]
    assert entry["classification"] == "swap_detected"
    assert entry["strict_pass"] is False
    assert entry["reason"] == "process_tree_swap_detected"
    assert entry["peak_swap_kib"] == 4096
    assert entry["swap_processes"] == [
        {"pid": 123, "name": "onnx2tf", "peak_swap_kib": 4096},
    ]


def test_flatbuffer_direct_bulk_runner_marks_conversion_error(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "error.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        return type("CP", (), {"returncode": 1, "stdout": "", "stderr": "failed"})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "conversion_error"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_passes_native_pytorch_generation_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "ok.onnx"
    _write_dummy(model)
    seen_cmds: List[List[str]] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        seen_cmds.append(list(cmd))
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        native_pytorch_generation_timeout_sec=37,
    )
    assert len(seen_cmds) == 1
    assert "--native_pytorch_generation_timeout_sec" in seen_cmds[0]
    timeout_index = seen_cmds[0].index("--native_pytorch_generation_timeout_sec")
    assert seen_cmds[0][timeout_index + 1] == "37"


def test_flatbuffer_direct_bulk_runner_can_limit_pytorch_to_native_package(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "native.onnx"
    _write_dummy(model)
    seen_cmds: List[List[str]] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        seen_cmds.append(list(cmd))
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        native_pytorch_only=True,
    )

    assert len(seen_cmds) == 1
    assert "-fdopt" in seen_cmds[0]
    assert "-fdots" not in seen_cmds[0]
    assert "-fdodo" not in seen_cmds[0]
    assert "-fdoep" not in seen_cmds[0]
    assert state["native_pytorch_only"] is True
    assert state["summary"]["filters"]["pytorch_artifact_mode"] == "native"


def test_flatbuffer_direct_bulk_runner_updates_tqdm_progress(
    tmp_path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "a.onnx"
    model_b = tmp_path / "b.onnx"
    _write_dummy(model_a)
    _write_dummy(model_b)

    class _FakeProgressBar:
        def __init__(self) -> None:
            self.postfixes: List[str] = []
            self.updates: List[int] = []
            self.closed = False

        def set_postfix_str(self, text, refresh=True):
            self.postfixes.append(str(text))

        def update(self, value):
            self.updates.append(int(value))

        def close(self):
            self.closed = True

    created_progress_bars: List[_FakeProgressBar] = []

    def _fake_create_progress_bar(*, total, initial=0, desc):
        assert total == 2
        assert initial == 0
        assert desc == "flatbuffer_direct bulk"
        bar = _FakeProgressBar()
        created_progress_bars.append(bar)
        return bar

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(
        "onnx2tf.utils.flatbuffer_direct_bulk_runner._create_progress_bar",
        _fake_create_progress_bar,
    )
    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    assert len(created_progress_bars) == 1
    assert created_progress_bars[0].updates == [1, 1]
    assert created_progress_bars[0].closed is True
    assert created_progress_bars[0].postfixes[-1].endswith("[pass]")


def test_flatbuffer_direct_bulk_runner_starts_spinner_while_running(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "spin.onnx"
    _write_dummy(model)

    class _FakeSpinner:
        instances: List["_FakeSpinner"] = []

        def __init__(self, progress_bar) -> None:
            self.progress_bar = progress_bar
            self.contexts: List[tuple[str, str]] = []
            self.starts = 0
            self.stops = 0
            _FakeSpinner.instances.append(self)

        def set_context(self, *, model_name: str, status: str = "running") -> None:
            self.contexts.append((str(model_name), str(status)))

        def start(self) -> None:
            self.starts += 1

        def stop(self) -> None:
            self.stops += 1

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(bulk_runner, "_ProgressSpinner", _FakeSpinner)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    assert len(_FakeSpinner.instances) == 1
    spinner = _FakeSpinner.instances[0]
    assert spinner.contexts == [("spin.onnx", "running")]
    assert spinner.starts == 1
    assert spinner.stops >= 2


def test_flatbuffer_direct_bulk_runner_summary_lists_failed_models_only(
    tmp_path,
    monkeypatch,
) -> None:
    ok_model = tmp_path / "ok.onnx"
    fail_model = tmp_path / "fail.onnx"
    _write_dummy(ok_model)
    _write_dummy(fail_model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = Path(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        if model_path.stem == "ok":
            _write_accuracy_report(
                path=artifact_dir / "ok_accuracy_report.json",
                evaluation_pass=True,
            )
            _write_accuracy_report(
                path=artifact_dir / "ok_pytorch_accuracy_report.json",
                evaluation_pass=True,
            )
        else:
            _write_accuracy_report(
                path=artifact_dir / "fail_accuracy_report.json",
                evaluation_pass=False,
            )
            _write_accuracy_report(
                path=artifact_dir / "fail_pytorch_accuracy_report.json",
                evaluation_pass=True,
            )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    failed_models = state["summary"]["failed_models"]
    assert len(failed_models) == 1
    assert failed_models[0]["model"] == "fail.onnx"
    assert failed_models[0]["classification"] == "tflite_fail"


def test_regression_profile_runs_recorded_tier_zero_to_three_successes_and_failures(
    tmp_path,
    monkeypatch,
) -> None:
    from onnx import TensorProto, helper, save

    def _model(path: Path, count: int) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        previous = "x"
        nodes = []
        for index in range(count):
            output = f"v{index}"
            nodes.append(helper.make_node("Relu", [previous], [output]))
            previous = output
        y = helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1])
        save(helper.make_model(helper.make_graph(nodes, "g", [x], [y])), path)

    _model(tmp_path / "baseline_pass.onnx", 2)
    _model(tmp_path / "known_failure.onnx", 2)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test-tier0-3-all-models",
                "min_nodes": 1,
                "max_nodes": 999,
                "models": [
                    {
                        "tier": 0,
                        "model": "baseline_pass.onnx",
                        "baseline_classification": "pass",
                    },
                    {
                        "tier": 0,
                        "model": "known_failure.onnx",
                        "baseline_classification": "conversion_error",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = Path(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        calls.append(model_path.name)
        _write_accuracy_report(
            path=artifact_dir / f"{model_path.stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
        regression_profile=str(profile_path),
    )

    assert calls == ["baseline_pass.onnx", "known_failure.onnx"]
    assert [entry["model"] for entry in state["entries"]] == [
        "baseline_pass.onnx",
        "known_failure.onnx",
    ]
    profile_filter = state["summary"]["filters"]["regression_profile"]
    assert profile_filter["model_count"] == 2
    assert profile_filter["tiers"] == [0]
    assert profile_filter["baseline_classification_counts"] == {
        "conversion_error": 1,
        "pass": 1,
    }
    assert "model_names" not in profile_filter
    assert state["summary"]["filters"]["max_nodes"] == 999
    assert state["summary"]["filters"]["recursive"] is False


def test_regression_profile_excludes_recorded_timeouts_from_future_runs(
    tmp_path,
    monkeypatch,
) -> None:
    from onnx import TensorProto, helper, save

    def _model(path: Path) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        node = helper.make_node("Relu", ["x"], ["y"])
        save(helper.make_model(helper.make_graph([node], "g", [x], [y])), path)

    _model(tmp_path / "active.onnx")
    _model(tmp_path / "timed_out.onnx")
    _model(tmp_path / "excluded.onnx")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 1,
                "max_nodes": 49,
                "models": [
                    {
                        "tier": 0,
                        "model": "active.onnx",
                        "baseline_classification": "pass",
                        "shape_hints": ["input:0:1,16,16,3"],
                        "overwrite_input_shape": ["image:1,3,16,16"],
                        "keep_shape_absolutely_input_names": ["state:0"],
                        "eval_num_samples": 1,
                        "accuracy_only": True,
                    },
                    {
                        "tier": 0,
                        "model": "timed_out.onnx",
                        "baseline_classification": "timeout",
                    },
                    {
                        "tier": 0,
                        "model": "excluded.onnx",
                        "baseline_classification": "excluded",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: List[str] = []
    commands: List[List[str]] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = Path(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        calls.append(model_path.name)
        commands.append(list(cmd))
        _write_accuracy_report(
            path=artifact_dir / f"{model_path.stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
        regression_profile=str(profile_path),
    )

    assert calls == ["active.onnx"]
    assert commands[0][commands[0].index("-sh") + 1] == "input:0:1,16,16,3"
    assert commands[0][commands[0].index("-ois") + 1] == "image:1,3,16,16"
    assert commands[0][commands[0].index("-kat") + 1] == "state:0"
    assert commands[0][commands[0].index("-ens") + 1] == "1"
    assert "--eval_with_onnx" in commands[0]
    assert "-cotof" not in commands[0]
    assert [entry["model"] for entry in state["entries"]] == ["active.onnx"]
    profile_filter = state["summary"]["filters"]["regression_profile"]
    assert profile_filter["model_count"] == 3
    assert profile_filter["active_model_count"] == 1
    assert profile_filter["excluded_model_count"] == 2
    assert profile_filter["excluded_baseline_classification_counts"] == {
        "excluded": 1,
        "timeout": 1,
    }


def test_regression_profile_accepts_recorded_tflite_numeric_exception(
    tmp_path,
    monkeypatch,
) -> None:
    from onnx import TensorProto, helper, save

    model_path = tmp_path / "accepted.onnx"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Relu", ["x"], ["y"])
    save(helper.make_model(helper.make_graph([node], "g", [x], [y])), model_path)

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 1,
                "max_nodes": 49,
                "models": [
                    {
                        "tier": 0,
                        "model": model_path.name,
                        "baseline_classification": "pass",
                        "acceptance_reason": "approved_unstable_topk_indices",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        _write_accuracy_report(
            path=artifact_dir / "accepted_accuracy_report.json",
            evaluation_pass=False,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
        regression_profile=str(profile_path),
    )

    entry = state["entries"][0]
    assert entry["classification"] == "pass"
    assert entry["strict_pass"] is True
    assert entry["accepted_by_profile"] is True
    assert entry["profile_acceptance_reason"] == "approved_unstable_topk_indices"
    assert entry["unaccepted_classification"] == "tflite_fail"
    assert entry["unaccepted_strict_pass"] is False
    assert entry["unaccepted_reason"] == "tflite_fail"
    assert entry["tflite_accuracy_pass"] is False
    assert entry["tflite_max_abs"] == 1.0
    assert state["summary"]["counts"]["pass"] == 1
    assert state["summary"]["strict_fail_count"] == 0


def test_regression_profile_runs_models_in_declared_tier_order(
    tmp_path,
    monkeypatch,
) -> None:
    from onnx import TensorProto, helper, save

    def _model(path: Path, *, node_count: int) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        nodes = []
        current = "x"
        for index in range(node_count):
            output = "y" if index == node_count - 1 else f"v{index}"
            nodes.append(helper.make_node("Relu", [current], [output]))
            current = output
        save(helper.make_model(helper.make_graph(nodes, "g", [x], [y])), path)

    tier_zero = tmp_path / "z_tier_zero.onnx"
    tier_one = tmp_path / "a_tier_one.onnx"
    _model(tier_zero, node_count=1)
    _model(tier_one, node_count=50)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 1,
                "max_nodes": 199,
                "models": [
                    {"tier": 0, "model": tier_zero.name},
                    {"tier": 1, "model": tier_one.name},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = Path(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        calls.append(model_path.name)
        _write_accuracy_report(
            path=artifact_dir / f"{model_path.stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
        regression_profile=str(profile_path),
    )

    assert calls == [tier_zero.name, tier_one.name]
    assert [entry["model"] for entry in state["entries"]] == calls


def test_regression_profile_accepts_tier_four_models(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 1000,
                "max_nodes": 1999,
                "models": [{"tier": 4, "model": "large.onnx"}],
            }
        ),
        encoding="utf-8",
    )

    profile = bulk_runner._load_regression_profile(str(profile_path))
    assert profile["tiers"] == [4]
    assert profile["max_nodes"] == 1999


def test_regression_profile_rejects_tier_five_models(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 2000,
                "max_nodes": 4000,
                "models": [{"tier": 5, "model": "huge.onnx"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only Tier 0-4 models"):
        bulk_runner._load_regression_profile(str(profile_path))


def test_regression_profile_rejects_decreasing_tier_order(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 1,
                "max_nodes": 199,
                "models": [
                    {"tier": 1, "model": "later.onnx"},
                    {"tier": 0, "model": "earlier.onnx"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-decreasing tier"):
        bulk_runner._load_regression_profile(str(profile_path))


@pytest.mark.parametrize(
    ("model_option", "expected_error"),
    [
        ({"eval_num_samples": 0}, "positive integer"),
        ({"eval_num_samples": True}, "positive integer"),
        ({"accuracy_only": "true"}, "must be a boolean"),
        ({"acceptance_reason": ""}, "non-empty string"),
        ({"acceptance_reason": 1}, "non-empty string"),
    ],
)
def test_regression_profile_rejects_invalid_evaluation_options(
    tmp_path,
    model_option,
    expected_error,
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 1,
                "max_nodes": 49,
                "models": [
                    {
                        "tier": 0,
                        "model": "model.onnx",
                        **model_option,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_error):
        bulk_runner._load_regression_profile(str(profile_path))


def test_managed_regression_profile_includes_all_tier_zero_to_four_models() -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "baselines"
        / "flatbuffer_direct_active_tier0_4.json"
    )
    profile = bulk_runner._load_regression_profile(str(profile_path))

    assert profile["model_count"] == 420
    assert profile["active_model_count"] == 381
    assert profile["excluded_model_count"] == 39
    assert profile["excluded_baseline_classification_counts"] == {
        "excluded": 12,
        "timeout": 27,
    }
    assert profile["tiers"] == [0, 1, 2, 3, 4]
    assert profile["min_nodes"] == 1
    assert profile["max_nodes"] == 1999
    assert profile["baseline_classification_counts"] == {
        "missing_tflite_report": 6,
        "pass": 355,
        "tflite_fail": 20,
        "timeout": 27,
        "excluded": 12,
    }
    assert profile["acceptance_reasons"] == {
        "deim_hgnetv2_n_wholebody28_1250query_fp16.onnx": (
            "user_approved_topk_index_instability_from_near_tied_scores"
        ),
        "deim_hgnetv2_s_wholebody28_ft_1250query_fixed.onnx": (
            "user_approved_topk_index_instability_from_near_tied_scores"
        ),
    }
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    encoder_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "encoder.onnx"
    )
    assert encoder_entry == {
        "tier": 3,
        "model": "encoder.onnx",
        "baseline_classification": "timeout",
        "baseline_reason": "user_approved_timeout_after_600s",
        "error_signature_sha256": (
            "2347ce95c7a93231a34b5d76d4387d3de16b1a7b51e339ebbc3e8c1f1c3357f1"
        ),
    }
    assert profile["model_options"]["silero_vad.onnx"] == {
        "keep_shape_absolutely_input_names": ["input", "state", "sr"],
    }
    for model_name in (
        "anime-gan-v2.onnx",
        "anime-gan-v2_org.onnx",
        "face_paint_512_v2_0.onnx",
        "model_paint_v2_test.onnx",
    ):
        paint_entry = next(
            entry
            for entry in profile_payload["models"]
            if entry["model"] == model_name
        )
        assert paint_entry == {
            "tier": 1,
            "model": model_name,
            "baseline_classification": "pass",
            "tflite_max_abs": 0.0017458945512771606,
        }
    for model_name in (
        "onnx_dense_optimized.onnx",
        "onnx_dense_optimized_org.onnx",
    ):
        inverse_entry = next(
            entry
            for entry in profile_payload["models"]
            if entry["model"] == model_name
        )
        assert inverse_entry == {
            "tier": 1,
            "model": model_name,
            "baseline_classification": "pass",
            "tflite_max_abs": 0.00015753507614135742,
        }
    modnet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "modnet_old.onnx"
    )
    assert modnet_entry == {
        "tier": 2,
        "model": "modnet_old.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 2.3931264877319336e-05,
    }
    linea_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "LINEA.onnx"
    )
    assert linea_entry == {
        "tier": 3,
        "model": "LINEA.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.002297189086675644,
    }
    for model_name, expected_signature in (
        (
            "best.onnx",
            "9990c93cc0bb978faa93f402b92e3799d0a2f720adbbba481346987fc955a277",
        ),
        (
            "best_org.onnx",
            "634981e4d0f8ac09f0f8251ce1454356e7dfc708abb8b42de784cbb0afe6c40b",
        ),
    ):
        best_entry = next(
            entry
            for entry in profile_payload["models"]
            if entry["model"] == model_name
        )
        assert best_entry["baseline_classification"] == "tflite_fail"
        assert best_entry["baseline_reason"] == (
            "qdq_rounding_outliers_amplified_by_detector_decode"
        )
        assert best_entry["tflite_max_abs"] == 58.7506103515625
        assert best_entry["error_signature_sha256"] == expected_signature
    for model_name, expected_max_abs in (
        (
            "deim_hgnetv2_n_wholebody28_1250query_fp16.onnx",
            27.0,
        ),
        (
            "deim_hgnetv2_s_wholebody28_ft_1250query_fixed.onnx",
            20.0,
        ),
    ):
        deim_entry = next(
            entry
            for entry in profile_payload["models"]
            if entry["model"] == model_name
        )
        assert deim_entry == {
            "tier": 3,
            "model": model_name,
            "baseline_classification": "pass",
            "tflite_max_abs": expected_max_abs,
            "acceptance_reason": (
                "user_approved_topk_index_instability_from_near_tied_scores"
            ),
        }
    for model_name, expected_max_abs, expected_signature in (
        (
            "model_70_2023_0220_32_2_1_grid_sample_bilinear_sim.onnx",
            0.296916950494051,
            "8b74d3b46e6a1b361bb4cc872e621174731e12ff1fecf609475dedf7e6a7463f",
        ),
        (
            "model_grid_sample.onnx",
            0.2830471396446228,
            "1ce0c24d87ab41013650754287d6fa1dc9e55a779453168a587fd2d0d737d7c6",
        ),
    ):
        grid_sample_entry = next(
            entry
            for entry in profile_payload["models"]
            if entry["model"] == model_name
        )
        assert grid_sample_entry["baseline_classification"] == "tflite_fail"
        assert grid_sample_entry["baseline_reason"] == (
            "grid_coordinate_rounding_amplified_at_zero_padding_boundary"
        )
        assert grid_sample_entry["tflite_max_abs"] == expected_max_abs
        assert grid_sample_entry["error_signature_sha256"] == expected_signature
    inverse_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "inverse_11.onnx"
    )
    assert inverse_entry["baseline_classification"] == "missing_tflite_report"
    assert inverse_entry["baseline_reason"] == (
        "unsupported_exact_inverse_matrix_size_224"
    )
    assert inverse_entry["error_signature_sha256"] == (
        "c1bdaa58f8b1dda9b86c24ce66f509320a6201e08f6ebfd35b4ecb058109d885"
    )
    string_normalizer_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "string_normalizer_11.onnx"
    )
    assert string_normalizer_entry["baseline_classification"] == (
        "missing_tflite_report"
    )
    assert string_normalizer_entry["baseline_reason"] == (
        "unsupported_stock_tflite_string_normalizer"
    )
    assert string_normalizer_entry["error_signature_sha256"] == (
        "2623d1f687770e4aade4749af32963fca446f80b3dd131e7061bcb13ae225722"
    )
    maskrcnn_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "maskrcnn_resnet50_fpn.onnx"
    )
    assert maskrcnn_entry == {
        "tier": 3,
        "model": "maskrcnn_resnet50_fpn.onnx",
        "baseline_classification": "excluded",
        "baseline_reason": "user_excluded_from_future_validation",
    }
    for model_name in (
        "fast_acvnet_generalization_opset16_192x320.onnx",
        "htdemucs_ft_onnx_1sec.onnx",
    ):
        excluded_entry = next(
            entry
            for entry in profile_payload["models"]
            if entry["model"] == model_name
        )
        assert excluded_entry == {
            "tier": 3,
            "model": model_name,
            "baseline_classification": "excluded",
            "baseline_reason": "user_excluded_from_future_validation",
        }
    for model_name, tier in (
        ("conv_tasnet_dnn_ins.onnx", 0),
        ("model1.onnx", 1),
        ("paddlepaddle_26_ocr.onnx", 1),
        ("bread_180x320.onnx", 2),
        ("bread_nonfm_180x320.onnx", 2),
        ("double_gru.onnx", 2),
        ("gtcrn_simple.onnx", 2),
        ("spkrec-resnet-voxceleb.onnx", 2),
    ):
        excluded_entry = next(
            entry
            for entry in profile_payload["models"]
            if entry["model"] == model_name
        )
        assert excluded_entry == {
            "tier": tier,
            "model": model_name,
            "baseline_classification": "excluded",
            "baseline_reason": "user_excluded_from_future_validation",
        }
    hybridnets_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "hybridnets_384x640_sim.onnx"
    )
    assert hybridnets_entry == {
        "tier": 4,
        "model": "hybridnets_384x640_sim.onnx",
        "baseline_classification": "excluded",
        "baseline_reason": "repeated_quick_ceiling_timeout",
    }
    tiny_decoder_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "tiny_decoder_11.onnx"
    )
    assert tiny_decoder_entry == {
        "tier": 3,
        "model": "tiny_decoder_11.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 5.048513412475586e-05,
        "shape_hints": [
            "tokens:1,1",
            "audio_features:1,1500,384",
            "kv_cache:8,1,1,384",
            "offset:1",
        ],
        "keep_shape_absolutely_input_names": [
            "tokens",
            "audio_features",
            "kv_cache",
            "offset",
        ],
    }
    vit_b_encoder_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "vit_b_encoder.onnx"
    )
    assert vit_b_encoder_entry == {
        "tier": 3,
        "model": "vit_b_encoder.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 2.6226043701171875e-06,
    }
    ssd_mobilenet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "ssd_mobilenet_v1_12-int8.onnx"
    )
    assert ssd_mobilenet_entry["baseline_classification"] == (
        "missing_tflite_report"
    )
    assert ssd_mobilenet_entry["baseline_reason"] == (
        "invalid_onnx_missing_loop_captures_186"
    )
    assert ssd_mobilenet_entry["error_signature_sha256"] == (
        "a2db2a961bcee7b5af536f985dff9ceee1ed8ef8fc8115f8a57dbf64b5f1fe26"
    )
    assert profile["model_options"]["ssd_mobilenet_v1_12-int8.onnx"] == {
        "overwrite_input_shape": ["inputs:1,300,300,3"],
        "keep_shape_absolutely_input_names": ["inputs"],
    }
    efficientnet_lite4_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "efficientnet-lite4-11-int8.onnx"
    )
    assert efficientnet_lite4_entry["baseline_classification"] == "tflite_fail"
    assert efficientnet_lite4_entry["baseline_reason"] == (
        "onnxruntime_u8s8_saturating_pair_accumulation"
    )
    assert efficientnet_lite4_entry["error_signature_sha256"] == (
        "da92ac2e2f0e69a6b68654c2152fadfadb079a20d73aaf8b33fc3a81af4afc15"
    )
    arcfaceresnet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "arcfaceresnet100-11-int8.onnx"
    )
    assert arcfaceresnet_entry["baseline_classification"] == "tflite_fail"
    assert arcfaceresnet_entry["baseline_reason"] == (
        "onnxruntime_u8s8_saturating_pair_accumulation"
    )
    assert arcfaceresnet_entry["error_signature_sha256"] == (
        "f1495fb3c58cf0f48521281b92113fa31d9dd301329f552c0cd80dcbc1687c3a"
    )
    assert arcfaceresnet_entry["tflite_max_abs"] == 0.3681950643658638
    afhq_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "afhq_generator.v11.quant.onnx"
    )
    assert afhq_entry == {
        "tier": 2,
        "model": "afhq_generator.v11.quant.onnx",
        "baseline_classification": "tflite_fail",
        "baseline_reason": (
            "instance_normalization_drift_amplified_by_dynamic_quantization_decoder"
        ),
        "error_signature_sha256": (
            "516b1d24be24fbe12ff074541d5800538fb37be131282c52c85c6db8edf48e50"
        ),
        "tflite_max_abs": 0.21375656127929688,
    }
    dynamics_rife_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "dynamics_rife_sim.onnx"
    )
    assert dynamics_rife_entry["baseline_classification"] == (
        "missing_tflite_report"
    )
    assert dynamics_rife_entry["baseline_reason"] == (
        "invalid_onnx_concat_spatial_mismatch_64_128"
    )
    assert dynamics_rife_entry["error_signature_sha256"] == (
        "603ca474b8eee210cb3bb2df39bb20791becc95c66d60a1ec68e1a8a2744c109"
    )
    yolov3_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolov3-12-int8.onnx"
    )
    assert yolov3_entry["baseline_classification"] == "tflite_fail"
    assert yolov3_entry["baseline_reason"] == (
        "u8s8_detector_strict_metric_mismatch"
    )
    assert yolov3_entry["error_signature_sha256"] == (
        "042bc0ec6020004d0f33640af6ad3197daf7d2e012c2306e64ef42a999b29de9"
    )
    assert yolov3_entry["tflite_max_abs"] == 0.09563881158828735
    assert profile["model_options"]["yolov3-12-int8.onnx"] == {
        "overwrite_input_shape": [
            "input_1:1,3,416,416",
            "image_shape:1,2",
        ],
    }
    nanodet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "object_detection_nanodet_2022nov_int8.onnx"
    )
    assert nanodet_entry == {
        "tier": 2,
        "model": "object_detection_nanodet_2022nov_int8.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 4.76837158203125e-07,
    }
    yolox_int8_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "object_detection_yolox_2022nov_int8.onnx"
    )
    assert yolox_int8_entry == {
        "tier": 2,
        "model": "object_detection_yolox_2022nov_int8.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 2.384185791015625e-07,
    }
    ppocrv3_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "text_detection_en_ppocrv3_2023may_int8.onnx"
    )
    assert ppocrv3_entry["baseline_classification"] == "tflite_fail"
    assert ppocrv3_entry["baseline_reason"] == (
        "int8_requantization_outliers_amplified_by_transpose_conv"
    )
    assert ppocrv3_entry["error_signature_sha256"] == (
        "70b5455f472dbbc6067111117087dfd042b249969d88555128fe4aaf1ae9c64a"
    )
    assert ppocrv3_entry["overwrite_input_shape"] == ["x:1,3,480,640"]
    assert ppocrv3_entry["tflite_max_abs"] == 0.7411765307188034
    yolov5s_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolov5s.onnx"
    )
    assert yolov5s_entry["baseline_classification"] == "tflite_fail"
    assert yolov5s_entry["baseline_reason"] == (
        "float16_decode_rounding_boundary"
    )
    assert yolov5s_entry["error_signature_sha256"] == (
        "3d19e30742aaefcf03909adb23b0cfcbe24641b67bd5183d4c50f5998d8cb8ed"
    )
    assert yolov5s_entry["tflite_max_abs"] == 0.32965087890625
    dequantize_linear_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "dequantize_linear.onnx"
    )
    assert dequantize_linear_entry["baseline_classification"] == "tflite_fail"
    assert dequantize_linear_entry["baseline_reason"] == (
        "onnxruntime_qdq_fusion_and_float_conv_decode_amplification"
    )
    assert dequantize_linear_entry["error_signature_sha256"] == (
        "aca9525029ba7a608c692da1d3944c7ba88e2c55d50b7b54a6bc35b237aa2a42"
    )
    assert dequantize_linear_entry["tflite_max_abs"] == 81.25048828125
    conv_tasnet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "conv_tasnet.onnx"
    )
    assert conv_tasnet_entry["baseline_classification"] == (
        "missing_tflite_report"
    )
    assert conv_tasnet_entry["baseline_reason"] == (
        "invalid_onnx_scatterelements_rank_mismatch_4_6"
    )
    assert conv_tasnet_entry["error_signature_sha256"] == (
        "1a12a88b8f42185e545973a18cc67b274f6b1c7a01b21319ffeaeecf4fdb6052"
    )
    rtdetrv4_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "rtdetrv4_s.onnx"
    )
    assert rtdetrv4_entry["baseline_classification"] == "tflite_fail"
    assert rtdetrv4_entry["baseline_reason"] == (
        "builtin_conv_accumulation_amplified_by_topk"
    )
    assert rtdetrv4_entry["error_signature_sha256"] == (
        "743900046d8e38c14684c26f9df0c0fcd60a95f014698a135d2162ef77b8ae05"
    )
    assert rtdetrv4_entry["tflite_max_abs"] == 79.0
    rf_detr_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "rf-detr-nano.onnx"
    )
    assert rf_detr_entry == {
        "tier": 3,
        "model": "rf-detr-nano.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.000102996826171875,
    }
    new_encoder_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "new_encoder.onnx"
    )
    assert new_encoder_entry == {
        "tier": 3,
        "model": "new_encoder.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.0008774623274803162,
        "eval_num_samples": 1,
        "accuracy_only": True,
    }
    assert profile["model_options"]["new_encoder.onnx"] == {
        "eval_num_samples": 1,
        "accuracy_only": True,
    }
    fasterrcnn_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "fasterrcnn_resnet50_fpn.onnx"
    )
    assert fasterrcnn_entry == {
        "tier": 3,
        "model": "fasterrcnn_resnet50_fpn.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.0,
        "eval_num_samples": 1,
        "accuracy_only": True,
    }
    assert profile["model_options"]["fasterrcnn_resnet50_fpn.onnx"] == {
        "eval_num_samples": 1,
        "accuracy_only": True,
    }
    fasterrcnn_test4_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "fasterrcnn_test4_new.onnx"
    )
    assert fasterrcnn_test4_entry == {
        "tier": 3,
        "model": "fasterrcnn_test4_new.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.0,
        "eval_num_samples": 1,
        "accuracy_only": True,
    }
    assert profile["model_options"]["fasterrcnn_test4_new.onnx"] == {
        "eval_num_samples": 1,
        "accuracy_only": True,
    }
    yolo11x_obb_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolo11x-obb.onnx"
    )
    assert yolo11x_obb_entry == {
        "tier": 3,
        "model": "yolo11x-obb.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 6.103515625e-05,
        "overwrite_input_shape": ["input_image:1,3,1024,1024"],
    }
    assert profile["model_options"]["yolo11x-obb.onnx"] == {
        "overwrite_input_shape": ["input_image:1,3,1024,1024"],
    }
    yolov9_n_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolov9_n_wholebody15_Nx3HxW.onnx"
    )
    assert yolov9_n_entry == {
        "tier": 3,
        "model": "yolov9_n_wholebody15_Nx3HxW.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.0008544921875,
        "overwrite_input_shape": ["images:1,3,640,640"],
    }
    assert profile["model_options"]["yolov9_n_wholebody15_Nx3HxW.onnx"] == {
        "overwrite_input_shape": ["images:1,3,640,640"],
    }
    yolov9_t_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolov9_t_wholebody28_Nx3HxW.onnx"
    )
    assert yolov9_t_entry == {
        "tier": 3,
        "model": "yolov9_t_wholebody28_Nx3HxW.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.0081634521484375,
        "overwrite_input_shape": ["images:1,3,640,640"],
    }
    assert profile["model_options"]["yolov9_t_wholebody28_Nx3HxW.onnx"] == {
        "overwrite_input_shape": ["images:1,3,640,640"],
    }
    version_rfb_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "version-RFB-320-int8.onnx"
    )
    assert version_rfb_entry["baseline_classification"] == "tflite_fail"
    assert version_rfb_entry["baseline_reason"] == (
        "onnxruntime_u8s8_saturating_pair_accumulation"
    )
    assert version_rfb_entry["error_signature_sha256"] == (
        "da92ac2e2f0e69a6b68654c2152fadfadb079a20d73aaf8b33fc3a81af4afc15"
    )
    assert version_rfb_entry["tflite_max_abs"] == 0.14972958900034428
    crnn_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "text_recognition_CRNN_CN_2021nov_int8.onnx"
    )
    assert crnn_entry["baseline_classification"] == "tflite_fail"
    assert crnn_entry["baseline_reason"] == (
        "lstm_float_drift_crosses_quantization_boundary_before_qlinear_matmul"
    )
    assert crnn_entry["error_signature_sha256"] == (
        "42f6b758e04d423511002b64e18281c40f1c2fe6eb72f20065608fb10bca90a1"
    )
    assert crnn_entry["tflite_max_abs"] == 0.14842605590820312
    fcn_resnet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "fcn-resnet50-12-int8.onnx"
    )
    assert fcn_resnet_entry["baseline_classification"] == "tflite_fail"
    assert fcn_resnet_entry["baseline_reason"] == (
        "onnxruntime_u8s8_saturating_pair_accumulation"
    )
    assert fcn_resnet_entry["error_signature_sha256"] == (
        "da92ac2e2f0e69a6b68654c2152fadfadb079a20d73aaf8b33fc3a81af4afc15"
    )
    assert fcn_resnet_entry["tflite_max_abs"] == 0.5471203327178955
    yolox_nano_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolox_nano.onnx"
    )
    assert yolox_nano_entry["baseline_classification"] == "tflite_fail"
    assert yolox_nano_entry["baseline_reason"] == (
        "float_conv_accumulation_amplified_by_exp_stride"
    )
    assert yolox_nano_entry["error_signature_sha256"] == (
        "9516f4001a1361f52f4218a2585bedb5371d3be64116a8182cccfd91a3c0d5c0"
    )
    assert yolox_nano_entry["tflite_max_abs"] == 0.1362457275390625
    alike_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "alike_l_opset11_192x320_post.onnx"
    )
    assert alike_entry["baseline_classification"] == "tflite_fail"
    assert alike_entry["baseline_reason"] == (
        "topk_index_instability_from_float_ties"
    )
    assert alike_entry["error_signature_sha256"] == (
        "9a77ab8f63d543c221797af2d444b22c59a1cbb4a483e4c4caed79f45fcc8c9b"
    )
    assert alike_entry["tflite_max_abs"] == 290.0000057220459
    for debug_model_name, expected_signature in (
        (
            "tmp_alike_debug3.onnx",
            "224de9b141fb9878c1b1ef289b511c6f87d8844c4eac617b0e897cf0ade38f75",
        ),
        (
            "tmp_alike_debug4.onnx",
            "55596567db57cb322d3d8364fd9a668b24e93b882334c6d876a1b3af07506601",
        ),
    ):
        debug_entry = next(
            entry
            for entry in profile_payload["models"]
            if entry["model"] == debug_model_name
        )
        assert debug_entry["baseline_classification"] == "tflite_fail"
        assert debug_entry["baseline_reason"] == (
            "exact_equality_mask_instability_from_float_accumulation"
        )
        assert debug_entry["error_signature_sha256"] == expected_signature
        assert debug_entry["tflite_max_abs"] == 1.0
    assert profile["model_options"]["tiny_decoder_11.onnx"] == {
        "shape_hints": [
            "tokens:1,1",
            "audio_features:1,1500,384",
            "kv_cache:8,1,1,384",
            "offset:1",
        ],
        "keep_shape_absolutely_input_names": [
            "tokens",
            "audio_features",
            "kv_cache",
            "offset",
        ],
    }
    assert profile["model_options"]["d3net_dnn_double_44.onnx"] == {
        "keep_shape_absolutely_input_names": ["input"],
    }
    assert profile["model_options"]["G_180000.onnx"] == {
        "overwrite_input_shape": ["specs:1,257,73"],
        "keep_shape_absolutely_input_names": [
            "specs",
            "lengths",
            "sid_src",
            "sid_tgt",
        ],
    }
    assert profile["model_options"]["LibreRFDETRn.onnx"] == {
        "overwrite_input_shape": ["input:1,3,384,384"],
    }
    libre_rfdetr_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "LibreRFDETRn.onnx"
    )
    assert libre_rfdetr_entry == {
        "tier": 4,
        "model": "LibreRFDETRn.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.0001087188720703125,
        "overwrite_input_shape": ["input:1,3,384,384"],
    }
    bertsquad_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "bertsquad-12-int8.onnx"
    )
    assert bertsquad_entry["baseline_classification"] == "tflite_fail"
    assert bertsquad_entry["baseline_reason"] == (
        "onnxruntime_u8s8_matmulinteger_cpu_saturation"
    )
    assert bertsquad_entry["error_signature_sha256"] == (
        "6ff24c95a66de18bc32f4c9ad1ab3d41d714213b5849f3fb3a156aa652896105"
    )
    assert bertsquad_entry["tflite_max_abs"] == 2.001576066017151
    campp_vin_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "campp_vin.onnx"
    )
    assert campp_vin_entry == {
        "tier": 4,
        "model": "campp_vin.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 3.3020973205566406e-05,
    }
    assert profile["model_options"]["conv_tasnet.onnx"] == {
        "keep_shape_absolutely_input_names": ["onnx::Unsqueeze_0"],
    }
    for model_name in ["best.onnx", "best_org.onnx"]:
        assert profile["model_options"][model_name] == {
            "overwrite_input_shape": ["images:1,3,512,640"],
        }


def test_measured_quick_profile_excludes_repeated_timeout_model() -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "baselines"
        / "flatbuffer_direct_quick_tier0_4_2026-07-16.json"
    )
    profile = bulk_runner._load_regression_profile(str(profile_path))

    assert profile["model_count"] == 49
    assert profile["active_model_count"] == 48
    assert profile["excluded_model_count"] == 1
    assert profile["excluded_baseline_classification_counts"] == {
        "excluded": 1,
    }
    assert profile["baseline_classification_counts"] == {
        "missing_tflite_report": 1,
        "pass": 43,
        "tflite_fail": 4,
        "excluded": 1,
    }
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    hybridnets_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "hybridnets_384x640_sim.onnx"
    )
    assert hybridnets_entry["baseline_classification"] == "excluded"
    assert hybridnets_entry["baseline_reason"] == (
        "repeated_quick_ceiling_timeout"
    )
