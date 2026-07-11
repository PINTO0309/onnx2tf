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


def test_managed_regression_profile_includes_all_tier_zero_to_four_models() -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "baselines"
        / "flatbuffer_direct_active_tier0_4.json"
    )
    profile = bulk_runner._load_regression_profile(str(profile_path))

    assert profile["model_count"] == 420
    assert profile["tiers"] == [0, 1, 2, 3, 4]
    assert profile["min_nodes"] == 1
    assert profile["max_nodes"] == 1999
    assert profile["baseline_classification_counts"] == {
        "conversion_error": 38,
        "missing_tflite_report": 65,
        "pass": 267,
        "tflite_fail": 26,
        "timeout": 24,
    }
